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


# --- 分離層の学習（12.6.6節・ロードマップ段階2） -----------------------------
#
# 【重要な但し書き】側方抑制（k-WTA）付きのSTDPは、レート符号化された入力に対して
# 「勝者の重みを入力ベクトルの方へ動かす」更新に帰着し、オンライン競合学習とほぼ
# 等価になることが知られている（Diehl & Cook 2015 系の定式化）。したがって
# `fit_stdp` が `fit_hebbian` に勝てなければ「STDPである必然性」は何も示せない。
# 12.9節で整数スパイク方式が階段関数に縮退したのと同じ構図なので、両方を実装して
# 明示的に比較する。


@torch.no_grad()
def fit_hebbian(sep: PatternSeparator, x: torch.Tensor, epochs: int = 5,
                eta: float = 0.05, batch_size: int = 256,
                generator: torch.Generator | None = None) -> dict:
    """競合ヘブ学習。勝者の重みを入力へ寄せ、行ごとにL2正規化する（Oja型の安定化）。

    STDP + 側方抑制がレート符号化入力に対して縮退する先そのものであり、
    「STDPだから効いた」と「クラスタリングしたから効いた」を分離する対照群。
    """
    if sep.mode == "identity":
        raise ValueError("identity には学習する重みがない")
    n = x.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, generator=generator)
        lr = eta * (1.0 - ep / max(1, epochs))  # 線形に冷ます
        for i in range(0, n, batch_size):
            xb = _l2_normalize(x[perm[i : i + batch_size]])
            h = xb @ sep.weight.T
            idx = h.topk(sep.k, dim=-1).indices  # [b, k] 勝者
            # 勝者 i について W[i] += lr * (x - W[i])。同一勝者が複数回選ばれる
            # 場合は加算される（発火頻度の高いユニットほど強く引かれる）。
            flat = idx.reshape(-1)
            src = xb[:, None, :].expand(-1, sep.k, -1).reshape(-1, xb.shape[-1])
            delta = torch.zeros_like(sep.weight)
            delta.index_add_(0, flat, src)
            cnt = torch.zeros(sep.weight.shape[0], device=xb.device)
            cnt.index_add_(0, flat, torch.ones_like(flat, dtype=xb.dtype))
            hit = cnt > 0
            delta[hit] = delta[hit] / cnt[hit, None] - sep.weight[hit]
            sep.weight[hit] += lr * delta[hit]
            sep.weight.copy_(_l2_normalize(sep.weight))
    return {"rule": "hebbian", "epochs": epochs, "eta": eta, "n_samples": n}


@torch.no_grad()
def fit_stdp(sep: PatternSeparator, spikes: torch.Tensor, epochs: int = 5,
             a_plus: float = 0.01, a_minus: float = 0.008, tau: float = 0.9,
             batch_size: int = 256, generator: torch.Generator | None = None) -> dict:
    """ペア型STDP（8.1節の指数窓）。時間軸はトークン位置そのもの（A案）。

    前シナプス入力は皮質バックボーンの二値スパイク列 `spikes` [N, T, d]、
    後シナプスは本層の k-WTA 出力である。指数トレースを用いた標準形::

        trace_pre  <- tau * trace_pre  + s_pre
        trace_post <- tau * trace_post + s_post
        dW += a_plus * s_post^T @ trace_pre - a_minus * trace_post^T @ s_pre

    第1項が「前が先に発火 → 増強」、第2項が「後が先 → 抑圧」に対応する。
    """
    if sep.mode == "identity":
        raise ValueError("identity には学習する重みがない")
    n, t_len, d = spikes.shape
    for ep in range(epochs):
        perm = torch.randperm(n, generator=generator)
        scale = 1.0 - ep / max(1, epochs)
        for i in range(0, n, batch_size):
            sb = spikes[perm[i : i + batch_size]]  # [b, T, d]
            b = sb.shape[0]
            tr_pre = torch.zeros(b, d)
            tr_post = torch.zeros(b, sep.weight.shape[0])
            dw = torch.zeros_like(sep.weight)
            for t in range(t_len):
                s_pre = sb[:, t]                      # [b, d]
                h = s_pre @ sep.weight.T              # [b, M]
                idx = h.topk(sep.k, dim=-1).indices
                s_post = torch.zeros_like(h).scatter_(-1, idx, 1.0)
                tr_pre = tau * tr_pre + s_pre
                tr_post = tau * tr_post + s_post
                dw += a_plus * (s_post.T @ tr_pre) - a_minus * (tr_post.T @ s_pre)
            sep.weight += scale * dw / b
            sep.weight.copy_(_l2_normalize(sep.weight))
    return {"rule": "stdp", "epochs": epochs, "a_plus": a_plus,
            "a_minus": a_minus, "tau": tau, "n_samples": n}


@torch.no_grad()
def separation_diagnostics(sep: PatternSeparator, x: torch.Tensor,
                           x_noisy: torch.Tensor) -> dict:
    """12.6.5節が特定した敗因（k-WTAの勝者反転）を直接測る。

    - `margin`  : 第k位と第k+1位の活性の差（大きいほど反転しにくい）
    - `winner_retention`: 手がかりを劣化させたとき、勝者集合が保たれる割合
    """
    if sep.mode != "sdr":
        # k-WTA を持たない条件では「勝者」が定義できない（identity には重みがなく、
        # dense は全ユニットが非ゼロ）。無意味な値を返さず空にする。
        return {}
    h = _l2_normalize(x) @ sep.weight.T
    top = h.topk(sep.k + 1, dim=-1).values
    margin = (top[:, sep.k - 1] - top[:, sep.k]).mean()
    a = sep(x) != 0
    b = sep(x_noisy) != 0
    retention = (a & b).sum(dim=-1).float() / sep.k
    return {"margin": float(margin), "winner_retention": float(retention.mean())}
