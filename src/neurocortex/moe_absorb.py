"""大きな規模の恒常性 MoE で重みへの取り込みを促す（12.6.141節）。

`moe_scale.py` には手を入れない。ここに置く `AnnealedScaledRoutedCharLM` は `ScaledRoutedCharLM` の継承で、
閾値の状態 θ を別のバッファ `threshold_state` に持ち、選択に効く閾値を s(t)·θ とする（手当て (c)、臨界期の終わり）。
s(t) は学習の前半は 1、開始割合 `anneal_start` 以降に直線的に 0 へ下がる。`anneal_start=None`（既定）では s ≡ 1 で、
`ScaledRoutedCharLM` と同じ乱数の消費・同じ学習経過になる（`tests/test_moe_absorb.py`）。学習は `moe_scale.train_scaled`
をそのまま使う（`homeostatic_step` の呼び出し回数でステップを数える）。
"""

from __future__ import annotations

import torch

from .moe_scale import ScaledRoutedCharLM


def anneal_strength(step: int, total: int, start: float | None) -> float:
    """恒常性の効き s。step ≦ start·total で 1、step = total で 0 の区分線形。start=None なら常に 1。"""
    if start is None:
        return 1.0
    t0 = start * total
    if step <= t0:
        return 1.0
    return max(0.0, 1.0 - (step - t0) / max(total - t0, 1e-12))


class AnnealedScaledRoutedCharLM(ScaledRoutedCharLM):
    """選択に効く閾値 = s(t)·θ。θ は 12.6.127・133節と同じ積分則で更新し続ける。"""

    def __init__(self, *args, total_steps: int = 0, anneal_start: float | None = None, **kw) -> None:
        super().__init__(*args, **kw)
        self.total_steps, self.anneal_start = total_steps, anneal_start
        self.n_steps = 0
        self.strength = 1.0
        self.register_buffer("threshold_state", torch.zeros_like(self.threshold))

    @torch.no_grad()
    def homeostatic_step(self, choice: torch.Tensor) -> None:
        if self.kind == "plain":
            return
        self.n_steps += 1
        for l in range(self.n_layers):
            frac = torch.bincount(choice[l].reshape(-1), minlength=self.n_experts).float() / choice[l].numel()
            self.usage_ema[l].mul_(1 - self.alpha).add_(self.alpha * frac)
            self.threshold_state[l].add_(self.eta * (self.n_experts * self.usage_ema[l] - 1))
            self.threshold_state[l].sub_(self.threshold_state[l].mean())
        self.strength = anneal_strength(self.n_steps, self.total_steps, self.anneal_start)
        self.threshold.copy_(self.strength * self.threshold_state)
