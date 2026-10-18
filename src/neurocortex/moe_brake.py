"""臨界期を閉じた後の可塑性のブレーキと、トークンごとの取り込みの指標（12.6.149節）。

`moe_scale.py`・`moe_critical.py` の挙動は変えない。ここに置くもの:

- `BrakedCriticalLM`: `CriticalPeriodLM` の継承。
  - `freeze=True`: 閉じ始め（s < 1）と同時に閾値の状態 θ の積分を止める（活動センサーは更新し続ける）。12.6.146節の
    積分のワインドアップ（閉じる間に θ が 11〜30 まで育つ）を除く。成体の視覚野4層でシナプススケーリングが
    起きなくなること（Desai et al., 2002）に当たる。
  - `brake=β`: ルータ（競合の回路）の学習率の倍率を m(t) = β + (1−β)·s(t) にする（PNN・Lynx1 のような可塑性の
    ブレーキが、臨界期が閉じるのにつれて付く。Pizzorusso et al., 2002; Morishita et al., 2010）。専門家の学習率は変えない。
  - `router_hidden=h`: ルータを線形から2層（d→h→E、ReLU）にする（ルータの表現力の切り分け）。
  既定（freeze=False・brake=None・router_hidden=0）では `CriticalPeriodLM` と同じ乱数の消費・同じ学習経過になる
  （`tests/test_moe_brake.py`）。
- `train_braked`: `moe_scale.train_scaled` と同じ手順・乱数の消費で、ルータのパラメータを別の群にし、各ステップの前に
  その学習率を `router_lr_scale` 倍にする（Adam の更新は要素ごとなので、倍率 1 では `train_scaled` と一致する）。
- トークンごとの取り込み（`adoption`）: プローブの時系列の各点 t で閾値が選択を変えたトークン（閾値あり ≠ なし）について、
  後の時点 t'（t+Δ と学習の終わり）の閾値なしのルータが、t の閾値ありの割り当てを選ぶ割合 adopt と、t の閾値なしの
  選択を保つ割合 keep。偶然の水準 = (1 − adopt − keep)/(E − 2)（他の専門家1つあたり）。閾値ありの割り当てを教師と
  みなしたときの、後のルータの一致率に当たる。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe_critical import CriticalPeriodLM, close_strength


class BrakedCriticalLM(CriticalPeriodLM):
    """閉じると同時に θ の積分を止める（freeze）、ルータの学習率のブレーキ（brake）、2層ルータ（router_hidden）。"""

    def __init__(self, *args, freeze: bool = False, brake: float | None = None, router_hidden: int = 0,
                 lag: int = 100, **kw) -> None:
        super().__init__(*args, **kw)
        self.freeze, self.brake, self.router_hidden, self.lag = freeze, brake, router_hidden, lag
        self.router_lr_scale = 1.0
        self._choices: list[tuple[int, np.ndarray, np.ndarray]] = []
        if router_hidden > 0:  # 専門家・出力層の初期値は線形ルータの場合と同じ（ルータは後から作り直す）
            d = self.routers[0].in_features
            self.routers = nn.ModuleList(nn.Sequential(nn.Linear(d, router_hidden), nn.ReLU(),
                                                       nn.Linear(router_hidden, self.n_experts))
                                         for _ in range(self.n_layers))
            if self.probe is not None and self.kind != "plain":
                self.logit0 = self._mean_logits()

    def router_parameters(self) -> list[nn.Parameter]:
        return list(self.routers.parameters())

    @torch.no_grad()
    def homeostatic_step(self, choice: torch.Tensor) -> None:
        if self.kind == "plain":
            return
        self.n_steps += 1
        s = close_strength(self.n_steps, self.total_steps, self.open_frac, self.ramp_frac)
        for l in range(self.n_layers):
            frac = torch.bincount(choice[l].reshape(-1), minlength=self.n_experts).float() / choice[l].numel()
            self.usage_ema[l].mul_(1 - self.alpha).add_(self.alpha * frac)
            if not (self.freeze and s < 1.0):
                self.threshold_state[l].add_(self.eta * (self.n_experts * self.usage_ema[l] - 1))
                self.threshold_state[l].sub_(self.threshold_state[l].mean())
        self.strength = s
        self.threshold.copy_(self.strength * self.threshold_state)
        if self.brake is not None:
            self.router_lr_scale = self.brake + (1.0 - self.brake) * s
        if self.probe is not None and (self.n_steps % self.log_every == 0 or self.n_steps == 1):
            self.series.append(self.snapshot(self.n_steps % self.vec_every == 0 or self.n_steps == self.total_steps))

    @torch.no_grad()
    def snapshot(self, with_vectors: bool = False) -> dict:
        p = super().snapshot(with_vectors)
        on, off = self.run(self.probe), self.run(self.probe, use_threshold=False)
        self._choices.append((self.n_steps, on["choice"].numpy().astype(np.int16), off["choice"].numpy().astype(np.int16)))
        p["shift"] = float((on["choice"] != off["choice"]).float().mean())
        p["router_lr_scale"] = self.router_lr_scale
        return p

    def adoption(self) -> None:
        """時系列の各点に、トークンごとの取り込みの指標（Δ 後と学習の終わり）を足す。測るトークンがなければ None。"""
        if not self._choices:
            return
        steps = [c[0] for c in self._choices]
        final_off = self._choices[-1][2]
        for p, (t, on, off) in zip(self.series, self._choices):
            p.update(adoption_stats(on, off, final_off, self.n_experts, "T"))
            j = steps.index(t + self.lag) if t + self.lag in steps else None
            p.update(adoption_stats(on, off, None if j is None else self._choices[j][2], self.n_experts, "lag"))


def adoption_stats(on: np.ndarray, off: np.ndarray, later_off: np.ndarray | None, n_experts: int,
                   tag: str) -> dict:
    """閾値が選択を変えたトークン（on ≠ off）について、後の閾値なしの選択が on を選ぶ割合（adopt）・off を保つ割合（keep）・
    偶然の水準（他の専門家1つあたり）と、その比 R = adopt / 偶然。"""
    mask = on != off
    if later_off is None or not mask.any():
        return {f"adopt_{tag}": None, f"keep_{tag}": None, f"chance_{tag}": None, f"R_{tag}": None}
    lo = later_off[mask]
    adopt, keep = float((lo == on[mask]).mean()), float((lo == off[mask]).mean())
    chance = (1.0 - adopt - keep) / max(n_experts - 2, 1)
    return {f"adopt_{tag}": adopt, f"keep_{tag}": keep, f"chance_{tag}": chance,
            f"R_{tag}": adopt / max(chance, 1e-3)}


def train_braked(model: BrakedCriticalLM, train_data: torch.Tensor, steps: int, batch_size: int, seq_len: int,
                 lr: float, generator: torch.Generator, balance_coef: float = 0.0) -> float:
    """`moe_scale.train_scaled` と同じ手順・乱数の消費。ルータの学習率だけ `router_lr_scale` 倍にする。"""
    rp = model.router_parameters()
    ids = {id(p) for p in rp}
    opt = torch.optim.Adam([{"params": rp}, {"params": [p for p in model.parameters() if id(p) not in ids]}], lr=lr)
    ce = float("nan")
    for _ in range(steps):
        idx = torch.randint(0, len(train_data) - seq_len - 1, (batch_size,), generator=generator)
        x = torch.stack([train_data[i:i + seq_len] for i in idx.tolist()])
        y = torch.stack([train_data[i + 1:i + seq_len + 1] for i in idx.tolist()])
        logits, balance, choice = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        ce = float(loss.detach())
        opt.zero_grad(set_to_none=True)
        (loss + balance_coef * balance).backward()
        opt.param_groups[0]["lr"] = lr * model.router_lr_scale
        opt.step()
        model.homeostatic_step(choice)
    return ce
