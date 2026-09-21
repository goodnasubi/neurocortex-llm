"""海馬モジュール（12.6.4節・ステップ1）。

皮質バックボーン `SpikingLM` の上に後付けする、勾配を用いない連想メモリ。

  x_t（残差ストリーム）→ DG: パターン分離（k-WTA疎SDR）→ CA3: 連想ストア → m_t
  logits = head(ln_f(x_t) + g * m_t)

12.6.4節の主指標は「想起できること」ではなく「**パターン分離が効くこと**」である。
`softmax(βKᵀq)V` は β→∞ で厳密な最近傍探索（＝dict）に収束するため、想起できた
だけでは dict との区別がつかない。したがって比較可能な形で分離層を差し替えられる
ことが、このモジュールの設計上もっとも重要な性質である（`PatternSeparator.mode`）。

書き込みは推論時オンラインで行い、勾配を一切通さない（12.3.2節）。この性質は
`tests/test_path_audit.py` で機械的に保証する。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


class PatternSeparator(nn.Module):
    """歯状回（DG）に対応するパターン分離層（6.1節）。

    3つのモードを持つ。これは利便性ではなく**アブレーションそのもの**であり、
    「効いているのは高次元化なのか疎化なのか」を分離するために必要である
    （12.6.4節のアブレーション3水準）。

      - ``"sdr"``      : 固定ランダム射影 + k-WTA による二値疎表現（本設計）
      - ``"dense"``    : 同次元の密なランダム射影（疎化だけを取り除いた対照）
      - ``"identity"`` : 皮質表現をそのままキーにする（分離層なしの対照）

    射影は固定であり学習しない（ステップ1）。STDPによる学習はステップ5。
    出力は常にL2正規化されているので、内積がそのままコサイン類似度になる。
    k-WTAの出力は非ゼロ成分が常にk個なので、内積は「重なり数 / k」に一致する。
    """

    def __init__(
        self,
        d_model: int,
        n_units: int = 2048,
        k: int = 40,
        mode: str = "sdr",
        seed: int = 0,
    ) -> None:
        super().__init__()
        if mode not in ("sdr", "dense", "identity"):
            raise ValueError(f"未知のモード: {mode}")
        if mode != "identity" and not 1 <= k <= n_units:
            raise ValueError(f"k は 1..{n_units} の範囲（受領: {k}）")
        self.mode = mode
        self.k = k
        self.n_units = n_units
        self.out_dim = d_model if mode == "identity" else n_units
        if mode == "identity":
            self.register_buffer("weight", torch.empty(0), persistent=False)
        else:
            g = torch.Generator().manual_seed(seed)
            w = torch.randn(n_units, d_model, generator=g) / d_model**0.5
            # 学習しないのでパラメータではなくバッファとして持つ。
            # これにより requires_grad を持つテンソルがこの層に一切存在しなくなる。
            self.register_buffer("weight", w, persistent=True)

    @property
    def sparsity(self) -> float:
        """非ゼロ成分の割合。identity では定義できないので 1.0 を返す。"""
        return 1.0 if self.mode == "identity" else self.k / self.n_units

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[..., d_model] → [..., out_dim]（L2正規化済み）。"""
        x = _l2_normalize(x)
        if self.mode == "identity":
            return x
        h = x @ self.weight.T
        if self.mode == "dense":
            return _l2_normalize(h)
        # k-WTA: 上位k個だけを1にする。固定閾値ではなく順位で切ることで、
        # 入力ノルムによらず疎度がkに固定される（12.6.4節）。
        idx = h.topk(self.k, dim=-1).indices
        sdr = torch.zeros_like(h).scatter_(-1, idx, 1.0)
        return sdr / self.k**0.5


@dataclass
class StoreStats:
    """読み出し1回分の診断値。"""

    max_score: torch.Tensor  # [B] 最良一致のコサイン類似度
    top1_index: torch.Tensor  # [B] 最良一致のスロット番号（-1 は空ストア）


class AssociativeStore:
    """CA3に対応する自己連想メモリ（追記のみのキー・バリュー表）。

    キーがL2正規化されているので ``scores = K @ q`` はコサイン類似度になる。
    ``sdr`` モードではキーが二値であるため、この積和は**乗算を伴わない**
    （12.11.3節のイベント駆動実装と同じ論法が適用できる。ただし本実装は
    PyTorchの密な行列積で計算しており、その利点は実測していない）。

    ``exact=True`` にすると softmax を使わず最良一致の値をそのまま返す。これは
    β→∞ の極限であり、12.6.4節の統制3「dict 対照群」に対応する。
    """

    def __init__(self, key_dim: int, value_dim: int, beta: float = 50.0,
                 exact: bool = False, device: torch.device | None = None) -> None:
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.beta = beta
        self.exact = exact
        self._device = device or torch.device("cpu")
        self._keys = torch.zeros(0, key_dim, device=self._device)
        self._values = torch.zeros(0, value_dim, device=self._device)
        self.write_count = 0

    def __len__(self) -> int:
        return self._keys.shape[0]

    @torch.no_grad()
    def write(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """追記する。勾配は通さない（12.3.2節「勾配を通さない局所更新」）。"""
        if keys.shape[0] != values.shape[0]:
            raise ValueError("キーと値の件数が一致しない")
        if keys.shape[-1] != self.key_dim or values.shape[-1] != self.value_dim:
            raise ValueError("次元が一致しない")
        self._keys = torch.cat([self._keys, keys.detach().to(self._device)])
        self._values = torch.cat([self._values, _l2_normalize(values.detach()).to(self._device)])
        self.write_count += keys.shape[0]

    @torch.no_grad()
    def read(self, keys: torch.Tensor, chunk: int = 512) -> tuple[torch.Tensor, StoreStats]:
        """[B, key_dim] → ([B, value_dim], 診断値)。空ストアではゼロを返す。"""
        b = keys.shape[0]
        out = torch.zeros(b, self.value_dim, device=keys.device)
        best = torch.zeros(b, device=keys.device)
        arg = torch.full((b,), -1, dtype=torch.long, device=keys.device)
        if len(self) == 0:
            return out, StoreStats(best, arg)
        for i in range(0, b, chunk):
            q = keys[i : i + chunk].to(self._device)
            scores = q @ self._keys.T  # コサイン類似度 [b', N]
            top = scores.max(dim=-1)
            best[i : i + chunk] = top.values.to(keys.device)
            arg[i : i + chunk] = top.indices.to(keys.device)
            if self.exact:
                v = self._values[top.indices]
            else:
                v = torch.softmax(self.beta * scores, dim=-1) @ self._values
            out[i : i + chunk] = v.to(keys.device)
        return out, StoreStats(best, arg)

    def clear(self) -> None:
        """統制2「因果的除去」用。ストアを空にする。"""
        self._keys = torch.zeros(0, self.key_dim, device=self._device)
        self._values = torch.zeros(0, self.value_dim, device=self._device)


class HippocampalMemory(nn.Module):
    """分離層と連想ストアを束ね、皮質の残差ストリームに読み出しを注入する。

    ゲート ``g = σ(gate_slope * (max_score - gate_bias))`` は、最良一致の類似度が
    低い（＝知らない事実である）ときに注入を遮断する。6.1節がCA1に帰している
    新規性検出に対応する。ステップ1では学習せず定数で置く（12.6.4節）。
    """

    def __init__(
        self,
        d_model: int,
        value_dim: int,
        n_units: int = 2048,
        k: int = 40,
        mode: str = "sdr",
        beta: float = 50.0,
        exact: bool = False,
        gain: float = 4.0,
        gate_slope: float = 20.0,
        gate_bias: float = 0.3,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.separator = PatternSeparator(d_model, n_units, k, mode, seed)
        self.store = AssociativeStore(self.separator.out_dim, value_dim, beta, exact)
        self.gain = gain
        self.gate_slope = gate_slope
        self.gate_bias = gate_bias

    @torch.no_grad()
    def write(self, x: torch.Tensor, values: torch.Tensor) -> None:
        """皮質表現 x [N, d_model] をキーに変換して値を書き込む。"""
        self.store.write(self.separator(x), values)

    @torch.no_grad()
    def read(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """皮質表現 x [B, d_model] から注入ベクトル g*m と ゲート値 g を返す。"""
        m, stats = self.store.read(self.separator(x))
        g = torch.sigmoid(self.gate_slope * (stats.max_score - self.gate_bias))
        return self.gain * g[:, None] * m, g

    def clear(self) -> None:
        self.store.clear()
