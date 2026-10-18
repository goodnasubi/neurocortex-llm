"""臨界期型の閉じ方とゲートの大きさに左右されない取り込みの指標（12.6.145節）。

`moe_scale.py`・`moe_absorb.py` の挙動は変えない。ここに置くもの:

- `close_strength`: 恒常性の効き s(t)。開いている期間（t ≦ open·T）は 1、閉じる期間（open·T < t ≦ (open+ramp)·T）に
  直線で 0 へ下がり、その後（固定化期間）は 0。open=0.5・ramp=0.5 は 12.6.141節 (c) の `anneal_strength` と同じ。
- `CriticalPeriodLM`: `ScaledRoutedCharLM` の継承。閾値の状態 θ を `threshold_state` に持ち、12.6.127・133節と同じ
  積分則で更新し続け、選択に効く閾値を s(t)·θ とする。open=None（既定）では s ≡ 1 で `ScaledRoutedCharLM` と同じ
  乱数の消費・同じ学習経過になる（`tests/test_moe_critical.py`）。probe を与えると、学習中の閾値・ルータの対数・
  偏り・選択の一致率の時系列を記録する（測定だけで、学習の計算には触れない）。
- `absorption_metrics`: 閾値を外したときの選択の一致率 A と、ΔCE_off を ‖g·y‖ の平均で割った正規化した悪化。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .hub_load import gini
from .moe_scale import ScaledRoutedCharLM


def close_strength(step: int, total: int, open_frac: float | None, ramp_frac: float = 0.0) -> float:
    """恒常性の効き s。step ≦ open·total で 1、(open+ramp)·total で 0 に達する区分線形。open=None なら常に 1。"""
    if open_frac is None:
        return 1.0
    t0 = open_frac * total
    if step <= t0:
        return 1.0
    return max(0.0, 1.0 - (step - t0) / max(ramp_frac * total, 1e-12))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else 0.0


class CriticalPeriodLM(ScaledRoutedCharLM):
    """選択に効く閾値 = s(t)·θ（s は `close_strength`）。θ は 12.6.127・133節と同じ積分則で更新し続ける。"""

    def __init__(self, *args, total_steps: int = 0, open_frac: float | None = None, ramp_frac: float = 0.0,
                 probe: torch.Tensor | None = None, log_every: int = 20, vec_every: int = 100, **kw) -> None:
        super().__init__(*args, **kw)
        self.total_steps, self.open_frac, self.ramp_frac = total_steps, open_frac, ramp_frac
        self.n_steps, self.strength = 0, 1.0
        self.register_buffer("threshold_state", torch.zeros_like(self.threshold))
        self.probe, self.log_every, self.vec_every = probe, log_every, vec_every
        self.series: list[dict] = []
        self.logit0: np.ndarray | None = None
        if probe is not None and self.kind != "plain":
            self.logit0 = self._mean_logits()

    @torch.no_grad()
    def _mean_logits(self) -> np.ndarray:
        """プローブの全トークンで平均したルータの対数 [L,E]（閾値ありで積んだ残差の上）。"""
        return self.run(self.probe)["router"].mean(dim=(1, 2)).numpy()

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
        self.strength = close_strength(self.n_steps, self.total_steps, self.open_frac, self.ramp_frac)
        self.threshold.copy_(self.strength * self.threshold_state)
        if self.probe is not None and (self.n_steps % self.log_every == 0 or self.n_steps == 1):
            self.series.append(self.snapshot(self.n_steps % self.vec_every == 0 or self.n_steps == self.total_steps))

    @torch.no_grad()
    def snapshot(self, with_vectors: bool = False) -> dict:
        """学習中の時系列の1点。gini_on/off は閾値あり／なしの選択の偏り（層の平均）、agree は両者の選択の一致率、
        rho は「ルータの対数の平均の変化 Δc_e」と「−θ_e（状態）」の専門家間の相関（層の平均。重みが閾値の
        割り当てを取り込むと正になる）。"""
        on, off = self.run(self.probe), self.run(self.probe, use_threshold=False)
        c_on, c_off = on["choice"].numpy(), off["choice"].numpy()
        e = self.n_experts
        g_on = [gini(np.bincount(c_on[l].ravel(), minlength=e).astype(float)) for l in range(self.n_layers)]
        g_off = [gini(np.bincount(c_off[l].ravel(), minlength=e).astype(float)) for l in range(self.n_layers)]
        st = self.threshold_state.numpy()
        logits = self._mean_logits()
        dc = logits - self.logit0
        rho = [_corr(dc[l], -st[l]) for l in range(self.n_layers)]
        th = self.threshold
        p = {"step": self.n_steps, "s": self.strength,
             "th_range": float((th.max(1).values - th.min(1).values).mean()),
             "state_range": float((st.max(1) - st.min(1)).mean()),
             "gini_on": float(np.mean(g_on)), "gini_off": float(np.mean(g_off)),
             "agree": float((c_on == c_off).mean()), "rho": float(np.mean(rho)),
             "logit_spread": float((dc.max(1) - dc.min(1)).mean())}
        if with_vectors:
            p["theta_state"] = np.round(st, 4).tolist()
            p["d_logit"] = np.round(dc, 4).tolist()
        return p


@torch.no_grad()
def absorption_metrics(model: ScaledRoutedCharLM, x: torch.Tensor, y: torch.Tensor) -> dict:
    """ゲートの大きさに左右されない取り込みの指標（12.6.145節）。

    agree: 閾値あり／なしで選ばれる専門家が同じトークンの割合（層の平均）。contrib: 閾値ありの ‖g·y‖ の平均
    （トークン・層の平均）。d_ce_off: 閾値を外したときの CE の悪化。d_ce_off_norm = d_ce_off ÷ contrib。
    d_ce_per_shift = d_ce_off ÷（選択が変わったトークンの割合）は参考。閾値の状態 θ を持つ模型では、
    閾値を全部の強さ（s=1）で効かせたときとの一致率 agree_state も返す（閉じた後に恒常性が残す要求の大きさ）。
    """
    on, off = model.run(x), model.run(x, use_threshold=False)

    def ce(r):
        return float(F.cross_entropy(r["logits"].reshape(-1, r["logits"].shape[-1]), y.reshape(-1)))

    agree = float((on["choice"] == off["choice"]).float().mean())
    contrib = float(on["contrib"].mean())
    d = ce(off) - ce(on)
    out = {"agree": agree, "contrib": contrib, "contrib_off": float(off["contrib"].mean()), "d_ce_off": d,
           "d_ce_off_norm": d / contrib if contrib > 0 else float("nan"),
           "d_ce_per_shift": d / (1 - agree) if agree < 1 else 0.0}
    if hasattr(model, "threshold_state") and model.kind != "plain":
        saved = model.threshold.clone()
        model.threshold.copy_(model.threshold_state)
        full = model.run(x)
        model.threshold.copy_(saved)
        out["agree_state"] = float((full["choice"] == off["choice"]).float().mean())
    return out
