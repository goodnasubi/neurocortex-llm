"""ハブを許す目標域つきの乗法的な恒常性スケーリングで、ハブ専門家への負荷集中を抑える（12.6.195節）。

皮質のハブ（rich club）は代謝コストが高いが、発火率は恒常性シナプススケーリング（Turrigiano & Nelson, 2004;
Turrigiano, 2008）と内因性可塑性で目標の範囲に保たれる。これを模して、専門家ごとの使用率を目標域
[r_lo/E, r_hi/E] に保つ局所則をルータの選択に入れる（補助損失ではない。勾配を流さない）。

- 選択専用（12.6.133節）: 閾値 θ は argmax の選択だけに効き、ゲートの確率・ルータの勾配には入らない。
- 乗法的な更新: 専門家の「駆動」exp(z_e) に掛かる利得 s_e = exp(−θ_e) を、シナプススケーリングと同じく
  使用率の目標比で掛け算的に直す: s_e ← s_e · (r*_e / r_e)^η、すなわち θ_e ← θ_e + η log(r_e / r*_e)。
  r_e = E u_e（u_e は選択割合の移動平均）、r*_e = clip(r_e, r_lo, r_hi)（目標域の最も近い端）。
  域内（r_lo ≤ r_e ≤ r_hi）では θ は動かない。r_lo = r_hi = 1 は一点の目標（従来の恒常性）に当たる。
  12.6.127節の線形の積分則 θ ← θ + η(E u − 1) は、目標比の対数の一次近似に当たる。
- 入力重みの乗法的な縮小（ルータの行 w_e を s_e 倍）は採らない: ルータの対数は正負をとり、行を縮めると
  負の対数は大きくなって補正の向きが入力で逆になる（12.6.127節）。駆動 exp(z) は正なので、その利得を
  掛け算で直す形にした。
"""

from __future__ import annotations

import numpy as np
import torch

from .moe_scale import ScaledRoutedCharLM, eval_ce


class BandHomeostaticLM(ScaledRoutedCharLM):
    """選択専用の閾値を目標域つきの乗法則（対数比）で動かす `ScaledRoutedCharLM`。

    r_lo = r_hi = 1 で一点の目標。r_floor は使われない専門家の r を下から抑える値（log 0 を避ける）。
    """

    def __init__(self, vocab_size: int, d: int = 48, n_experts: int = 8, d_ff: int = 64, n_layers: int = 1,
                 kind: str = "select", eta: float = 0.0, alpha: float = 0.05, r_lo: float = 1.0,
                 r_hi: float = 1.0, r_floor: float = 0.01) -> None:
        super().__init__(vocab_size, d=d, n_experts=n_experts, d_ff=d_ff, n_layers=n_layers, kind=kind,
                         eta=eta, alpha=alpha)
        if not 0 < r_lo <= r_hi:
            raise ValueError((r_lo, r_hi))
        self.r_lo, self.r_hi, self.r_floor = r_lo, r_hi, r_floor

    def drive_error(self) -> torch.Tensor:
        """log(r / r*) [L,E]。域内で 0、使われすぎで正、使われなさすぎで負。"""
        r = (self.n_experts * self.usage_ema).clamp_min(self.r_floor)
        return torch.log(r / r.clamp(self.r_lo, self.r_hi))

    @torch.no_grad()
    def homeostatic_step(self, choice: torch.Tensor) -> None:
        if self.kind == "plain":
            return
        for l in range(self.n_layers):
            frac = torch.bincount(choice[l].reshape(-1), minlength=self.n_experts).float() / choice[l].numel()
            self.usage_ema[l].mul_(1 - self.alpha).add_(self.alpha * frac)
        self.threshold.add_(self.eta * self.drive_error())
        self.threshold.sub_(self.threshold.mean(dim=1, keepdim=True))  # 選択（argmax）には効かない


@torch.no_grad()
def single_lesions(model: ScaledRoutedCharLM, x: torch.Tensor, y: torch.Tensor, mode: str = "zero") -> list[float]:
    """1層目の専門家を1個ずつ除去したときの ΔCE（`hub_load` の1個除去の多層モデル版。L=1 で使う）。"""
    base = eval_ce(model, x, y)
    out = []
    for e in range(model.n_experts):
        m = torch.zeros(model.n_layers, model.n_experts, dtype=torch.bool)
        m[0, e] = True
        out.append(eval_ce(model, x, y, disabled=m, mode=mode) - base)
    return out


def topk_share(load, k: int) -> float:
    """負荷の上位 k 個の割合（一様で k/E）。"""
    v = np.sort(np.asarray(load, dtype=np.float64))[::-1]
    return float(v[:k].sum() / v.sum()) if v.sum() > 0 else float("nan")


def band_violation(load, r_lo: float, r_hi: float) -> float:
    """目標域 [r_lo/E, r_hi/E] から外れた負荷の割合の和（域内で 0）。"""
    v = np.asarray(load, dtype=np.float64)
    r = len(v) * v / v.sum()
    return float((np.clip(r_lo - r, 0, None) + np.clip(r - r_hi, 0, None)).sum() / len(v))

