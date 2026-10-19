"""教師の割り当ての恒常性の閾値と、臨界期を閉じる前の使われない専門家の再学習（12.6.171節）。

12.6.166節の部分評価の反実仮想の教師（`moe_cfe.EstimatedCFLM` の "top{k}"・"rand{m}"）でも、E=32 では閾値を外した
ときの選択の一致率が 0.85 に届かず、閉じた後に使われない専門家が残った。ここでは2つの手当てを足す。
`moe_cfe.py` ほかの既存の模型の挙動は変えない。ここに置くもの:

- `ReactCFLM`: `EstimatedCFLM` の継承。
  - `teacher_homeo=True`（(a)）: 教師の割り当てにも恒常性の閾値 θ_T を掛ける。部分評価の教師の目標を
    argmin_e(CE_e/τ + θ_T,e)（評価した専門家の中）にし、θ_T は教師の目標の負荷の移動平均 u_T から選択の閾値と同じ
    積分則 θ_T += η(E·u_T − 1)（平均を引く）で更新する（η・α は選択の閾値と同じ値。新しい定数を足さない）。
    θ_T は臨界期を閉じた後も更新し続ける（選択の θ の `freeze` とは別）。
  - `react=True`（(b)）: 閾値が完全に閉じるまで（s > 0 の間）、閾値なしのルータの選択の負荷の移動平均 u_off が
    ρ/E 未満の専門家（使われない専門家）を、その専門家が最も得意なトークン（実際に選ばれた専門家 c との CE の差
    CE_c − CE_e の上位 N/E 個、自分の取り分）に強制的に送って学習させる。ゲートは c のゲート（勾配を切る）、
    勾配はその専門家の重みにだけ届く（h・出力層・ルータには届かない）。損失は μ·Σ CE / N（取り分のトークンを
    実際に割り当てられたときと同じ重み）。
  teacher_homeo=False・react=False では `EstimatedCFLM` と同じ学習経過（`tests/test_moe_react.py`）。
  教師の目標の負荷の偏り（閾値を掛けた後と掛ける前のジニ）を `teacher_stats` に、使われない専門家の数と
  再学習の評価数を `react_stats`・`cost` に残す。
- `train_react`: `moe_cf.train_cf` と同じ手順・乱数の消費で、react があれば損失に `react_loss` を足す。
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from .hub_load import gini
from .moe_cfe import EstimatedCFLM, _dup_mask, _probs, _random_others, subset_ce
from .moe_hier import GROUPS


def _gini_of(target: torch.Tensor, n_experts: int) -> float:
    return float(gini(np.bincount(target.numpy(), minlength=n_experts).astype(float)))


class ReactCFLM(EstimatedCFLM):
    """部分評価の教師に恒常性の閾値（teacher_homeo）と、閉じる前の使われない専門家の再学習（react）を足す。"""

    def __init__(self, *args, teacher_homeo: bool = False, react: bool = False, react_rho: float = 0.25,
                 react_mu: float = 1.0, **kw) -> None:
        super().__init__(*args, **kw)
        if teacher_homeo and self.kind_t not in ("top", "rand"):
            raise ValueError("教師の閾値は部分評価の教師（top・rand）だけ")
        if react and self.n_layers != 1:
            raise ValueError("再学習は1層の模型だけ")
        self.teacher_homeo, self.react = teacher_homeo, react
        self.react_rho, self.react_mu = react_rho, react_mu
        self.register_buffer("teacher_threshold", torch.zeros(self.n_experts))
        self.register_buffer("teacher_usage", torch.full((self.n_experts,), 1.0 / self.n_experts))
        self.register_buffer("off_usage", torch.full((self.n_experts,), 1.0 / self.n_experts))
        self.react_stats: list[dict] = []
        self.cost["react_evals"] = 0.0

    def teacher_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.kind_t not in ("top", "rand"):
            return super().teacher_loss(x, y)
        t0 = time.perf_counter()
        with torch.no_grad():  # `EstimatedCFLM.teacher_loss` の部分評価と同じ計算・同じ乱数の消費
            h, _ = self.rnn(self.embed(x))
            raw0 = self.routers[0](h).reshape(-1, self.n_experts)
            chosen = (raw0 - self.threshold[0]).argmax(-1) if self.kind == "select" else _probs(self, raw0).argmax(-1)
            extra = (raw0.topk(self.m, dim=-1).indices if self.kind_t == "top"
                     else _random_others(chosen, self.n_experts, self.m, self.sample_gen))
            idx = torch.cat([chosen.unsqueeze(-1), extra], dim=-1)
            ce = subset_ce(self, h, y, idx)
            dup = _dup_mask(idx)
            pos_raw = ce.masked_fill(dup, float("inf")).argmin(-1)
            if self.teacher_homeo:
                score = ce / self.tau + self.teacher_threshold[idx]
                pos = score.masked_fill(dup, float("inf")).argmin(-1)
            else:
                pos = pos_raw
            hard = idx.gather(-1, pos.unsqueeze(-1)).squeeze(-1)
            hard_raw = idx.gather(-1, pos_raw.unsqueeze(-1)).squeeze(-1)
        n = y.numel()
        evals = sum(len(set(r)) for r in idx.tolist())
        raw = self.routers[0](h).reshape(-1, self.n_experts)
        logits = raw.gather(-1, idx).masked_fill(dup, float("-inf"))
        loss = self.lam * F.cross_entropy(logits, pos)
        with torch.no_grad():
            self.teacher_stats.append({"agree": float((raw.argmax(-1) == hard).float().mean()),
                                       "t_gini": _gini_of(hard, self.n_experts),
                                       "t_gini_raw": _gini_of(hard_raw, self.n_experts)})
            if self.teacher_homeo:  # 選択の閾値と同じ積分則（η・α も同じ）
                frac = torch.bincount(hard, minlength=self.n_experts).float() / n
                self.teacher_usage.mul_(1 - self.alpha).add_(self.alpha * frac)
                self.teacher_threshold.add_(self.eta * (self.n_experts * self.teacher_usage - 1))
                self.teacher_threshold.sub_(self.teacher_threshold.mean())
        self._count(evals, n, t0)
        return loss

    def react_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """閾値が完全に閉じるまで、使われない専門家を得意なトークン（自分の取り分）で学習させる損失。"""
        t0 = time.perf_counter()
        e_n = self.n_experts
        with torch.no_grad():
            h, _ = self.rnn(self.embed(x))
            hf, yf = h.reshape(-1, h.shape[-1]), y.reshape(-1)
            raw0 = self.routers[0](hf)
            frac = torch.bincount(raw0.argmax(-1), minlength=e_n).float() / yf.numel()
            self.off_usage.mul_(1 - self.alpha).add_(self.alpha * frac)
            active = self.strength > 0.0
            dormant = (self.off_usage < self.react_rho / e_n).nonzero().squeeze(-1).tolist() if active else []
            self.react_stats.append({"step": self.n_steps + 1, "active": active, "dormant": len(dormant)})
        loss = hf.new_zeros(())
        if not dormant:
            self.cost["sec"] += time.perf_counter() - t0
            return loss
        n = yf.numel()
        k = max(1, n // e_n)
        with torch.no_grad():
            chosen = (raw0 - self.threshold[0]).argmax(-1) if self.kind == "select" else _probs(self, raw0).argmax(-1)
            gate = _probs(self, raw0).gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
            ce_c = subset_ce(self, h, y, chosen.unsqueeze(-1)).squeeze(-1)
            w_head, b_head = self.head.weight.detach(), self.head.bias.detach()
        for e in dormant:
            with torch.no_grad():
                ye = (hf @ self.w1[0][e]).relu() @ self.w2[0][e]
                ce_e = F.cross_entropy(F.linear(hf + gate.unsqueeze(-1) * ye, w_head, b_head), yf, reduction="none")
                niche = (ce_c - ce_e).topk(k).indices
            ye = (hf[niche] @ self.w1[0][e]).relu() @ self.w2[0][e]  # 勾配はこの専門家の重みにだけ届く
            z = F.linear(hf[niche] + gate[niche].unsqueeze(-1) * ye, w_head, b_head)
            loss = loss + F.cross_entropy(z, yf[niche], reduction="sum") / n
        self.cost["react_evals"] += len(dormant) * n
        self.cost["sec"] += time.perf_counter() - t0
        return self.react_mu * loss


def train_react(model: ReactCFLM, train_data: torch.Tensor, steps: int, batch_size: int, seq_len: int,
                lr: float, generator: torch.Generator, balance_coef: float = 0.0) -> float:
    """`moe_cf.train_cf` と同じ手順・乱数の消費。react があれば損失に `react_loss` を足す。"""
    gp = model.group_parameters()
    braked = [g for g in GROUPS if g in model.brake_sched]
    ids = {id(p) for g in braked for p in gp[g]}
    rp = [p for p in model.router_parameters() if id(p) not in ids]
    ids |= {id(p) for p in rp}
    groups = [{"params": gp[g], "name": g} for g in braked]
    groups += [{"params": rp, "name": "_router"}, {"params": [p for p in model.parameters() if id(p) not in ids],
                                                   "name": "_rest"}]
    opt = torch.optim.Adam([g for g in groups if g["params"]], lr=lr)
    ce = float("nan")
    for _ in range(steps):
        idx = torch.randint(0, len(train_data) - seq_len - 1, (batch_size,), generator=generator)
        x = torch.stack([train_data[i:i + seq_len] for i in idx.tolist()])
        y = torch.stack([train_data[i + 1:i + seq_len + 1] for i in idx.tolist()])
        logits, balance, choice = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        ce = float(loss.detach())
        opt.zero_grad(set_to_none=True)
        total = loss + balance_coef * balance
        if model.teacher is not None:
            total = total + model.teacher_loss(x, y)
        if model.react and model.kind != "plain":
            total = total + model.react_loss(x, y)
        total.backward()
        for pg in opt.param_groups:
            name = pg["name"]
            scale = model.group_lr_scale.get(name, 1.0) if name in GROUPS else 1.0
            if name == "_router" or (name == "router" and model.brake is not None):
                scale *= model.router_lr_scale
            pg["lr"] = lr * scale
        opt.step()
        model.homeostatic_step(choice)
    return ce
