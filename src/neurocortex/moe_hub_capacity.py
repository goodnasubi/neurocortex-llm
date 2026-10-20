"""ハブ専門家に負荷に見合う容量を与える（12.6.205節）。

12.6.202節では、領域ごとに分かれた土台（`moe_modular.ModularRoutedLM`）で目標域の恒常性が生んだハブは
4言語から入力を受ける接続者になったが、統合課題でだけ不利になり（交互作用 +0.100）、ボトルネックとして
働いた。専門家は全部同じ大きさで、負荷だけが偏っていた。皮質の rich club のハブは、ほかの領域より大きく、
多くのシナプスと高い代謝コストを持つ（van den Heuvel et al., 2012; Collin et al., 2014;
Bullmore & Sporns, 2012）。そこで、学習の途中で専門家ごとの隠れ層の幅を負荷の比に応じて配り直す。

- 幅の総和（= 専門家の有効なパラメータ数）は均等な幅 E·d_ff に固定する。ハブを大きくした分、
  ハブ以外を小さくする（活動に依存した構造的可塑性。シナプスの総量は保たれる）。
- 再配分は realloc_step ステップ目の恒常性の更新の後に一度だけ行う。負荷は選択割合の移動平均 u_e
  （`usage_ema`）。幅 w_e ∝ u_e を [w_min, w_max] に収め、和を E·d_ff に合わせる（`allocate_widths`）。
- alloc = "load": 負荷の大きい専門家ほど広い。alloc = "random": 同じ幅の並びを、負荷と無関係な
  乱数の置換（alloc_seed）で専門家に割り当てる対照。
- 専門家 e は隠れ層の先頭 w_e 個の単位だけを使う。新しく使い始める単位は出力側の重み（w2 の行）を 0 に
  してから加える（再配分の瞬間に出力は変わらない。縮む専門家は末尾の単位を失う）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .moe_modular import ModularRoutedLM


def allocate_widths(load, total: int, w_min: int, w_max: int) -> list[int]:
    """負荷に比例した整数の幅。各幅は [w_min, w_max]、和は total（水位合わせ＋最大剰余で丸める）。"""
    v = np.asarray(load, dtype=np.float64)
    E = len(v)
    if not E * w_min <= total <= E * w_max:
        raise ValueError((total, w_min, w_max))
    v = np.clip(v, 0, None)
    v = v / v.sum() if v.sum() > 0 else np.full(E, 1.0 / E)
    lo, hi = 0.0, w_max / v[v > 0].min()  # w = clip(c·v) の和が total になる c を二分探索
    for _ in range(200):
        c = (lo + hi) / 2
        if np.clip(c * v, w_min, w_max).sum() < total:
            lo = c
        else:
            hi = c
    w = np.clip(hi * v, w_min, w_max)
    base = np.floor(w).astype(int)
    rest = total - int(base.sum())
    order = sorted(range(E), key=lambda e: (-(w[e] - base[e]), -v[e], e))
    for e in order:
        if rest == 0:
            break
        if base[e] < w_max:
            base[e] += 1
            rest -= 1
    assert base.sum() == total and base.min() >= w_min and base.max() <= w_max
    return base.tolist()


class HubCapacityLM(ModularRoutedLM):
    """`ModularRoutedLM` の専門家の隠れ層の幅を、学習の途中で負荷に応じて配り直す。

    realloc_step 以前は全専門家が幅 d_ff（`ModularRoutedLM` と同じ初期値・乱数の消費）。w_max − d_ff 個の
    予備の単位は、実行の乱数（torch.initial_seed() + 1）とは別の生成器で初期化する。
    """

    def __init__(self, vocab_size: int, d: int = 48, n_experts: int = 8, d_ff: int = 64, n_layers: int = 1,
                 kind: str = "select", eta: float = 0.0, alpha: float = 0.05, r_lo: float = 1.0,
                 r_hi: float = 1.0, r_floor: float = 0.01, region_of_token: list[int] | None = None,
                 d_mod: int = 24, realloc_step: int = 200, alloc: str = "load", alloc_seed: int = 0,
                 w_min: int = 16, w_max: int = 192) -> None:
        super().__init__(vocab_size, d=d, n_experts=n_experts, d_ff=d_ff, n_layers=n_layers, kind=kind, eta=eta,
                         alpha=alpha, r_lo=r_lo, r_hi=r_hi, r_floor=r_floor, region_of_token=region_of_token,
                         d_mod=d_mod)
        if alloc not in ("load", "random"):
            raise ValueError(alloc)
        if not w_min <= d_ff <= w_max:
            raise ValueError((w_min, d_ff, w_max))
        self.d_ff, self.realloc_step, self.alloc, self.alloc_seed = d_ff, realloc_step, alloc, alloc_seed
        self.w_min, self.w_max = w_min, w_max
        KD, D, extra = self.n_modules * d_mod, d_mod, w_max - d_ff
        g = torch.Generator().manual_seed(torch.initial_seed() + 1)
        w1x = torch.randn(n_experts, KD, extra, generator=g) * KD ** -0.5
        w2x = torch.randn(n_experts, extra, D, generator=g) * d_ff ** -0.5
        self.w1 = nn.ParameterList([nn.Parameter(torch.cat([self.w1[0].data, w1x], dim=2))])
        self.w2 = nn.ParameterList([nn.Parameter(torch.cat([self.w2[0].data, w2x], dim=1))])
        self.register_buffer("widths", torch.full((n_experts,), d_ff, dtype=torch.long))
        self.register_buffer("steps_seen", torch.zeros((), dtype=torch.long))
        self.realloc_load: list[float] | None = None

    def active_expert_params(self) -> int:
        """使われている単位の重みの数（w1 の列と w2 の行）。均等な幅の `ModularRoutedLM` と同じになる。"""
        return int(self.widths.sum()) * (self.w1[0].shape[1] + self.w2[0].shape[2])

    def expert_ffn(self, z_in: torch.Tensor, choice: torch.Tensor) -> torch.Tensor:
        B, T, KD = z_in.shape
        z = z_in.reshape(B * T, KD)
        c = choice.reshape(-1)
        ys, idxs = [], []
        for e in range(self.n_experts):
            idx = (c == e).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            w = int(self.widths[e])
            h = (z[idx] @ self.w1[0][e, :, :w]).relu()
            ys.append(h @ self.w2[0][e, :w])
            idxs.append(idx)
        out = z.new_zeros(B * T, self.w2[0].shape[2])
        if ys:
            out = out.index_copy(0, torch.cat(idxs), torch.cat(ys))
        return out.reshape(B, T, -1)

    @torch.no_grad()
    def reallocate(self, load) -> list[int]:
        """幅を負荷（または乱数の置換）で配り直す。新しく使う単位の w2 の行を 0 にする。"""
        w = allocate_widths(load, self.n_experts * self.d_ff, self.w_min, self.w_max)
        if self.alloc == "random":
            prof = sorted(w, reverse=True)
            perm = np.random.default_rng(self.alloc_seed).permutation(self.n_experts)
            w = [0] * self.n_experts
            for rank, e in enumerate(perm):
                w[int(e)] = prof[rank]
        for e, new in enumerate(w):
            old = int(self.widths[e])
            if new > old:
                self.w2[0][e, old:new] = 0
        self.widths.copy_(torch.as_tensor(w, dtype=torch.long))
        return w

    @torch.no_grad()
    def homeostatic_step(self, choice: torch.Tensor) -> None:
        super().homeostatic_step(choice)
        self.steps_seen += 1
        if int(self.steps_seen) == self.realloc_step:
            self.realloc_load = self.usage_ema[0].tolist()
            self.reallocate(self.realloc_load)
