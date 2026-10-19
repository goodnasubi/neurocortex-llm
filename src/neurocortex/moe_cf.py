"""選ばれなかった専門家の反実仮想の学習信号でルータを学習する（12.6.157節）。

`moe_scale.py`・`moe_critical.py`・`moe_brake.py`・`moe_hier.py` の挙動は変えない。ここに置くもの:

- `forced_ce`: 1層の模型で、各トークンを各専門家に強制的に送ったときの CE [N,E]（ゲートは e の確率）。
  `moe_hier.oracle_assignment` と同じ計算（最小値は同関数の ce_oracle と一致する。`tests/test_moe_cf.py`）。
- `CounterfactualLM`: `HierBrakeLM` の継承。ルータに教師つきの補助損失 λ·CE(router(h), 教師) を足す。h は GRU の出力を
  切り離したもので、補助損失の勾配はルータのパラメータにだけ届く（埋め込み・GRU・専門家には届かない）。教師:
  - "cf": CE を下げる割り当て（`forced_ce` の argmin。選ばれなかった専門家の結果 = 反実仮想の結果を使う）
  - "cfs": 反実仮想の結果のソフトな教師 softmax(−CE_e/τ)（全選択肢の価値を同時に更新する）
  - "dist": 閾値ありの選択（argmax(raw − θ)）。閾値の割り当ての蒸留（価値によらず選んだ行動を強める習慣の学習）
  - "shuf": "cf" の教師をバッチ内のトークンで並べ替えたもの（教師の負荷の分布は同じで、トークンごとの情報がない対照）
  - "self": 閾値なしのルータ自身の選択（argmax raw）。新しい情報を足さずにルータの確率を鋭くするだけの対照（12.6.157節の修正1）
  teacher=None（既定）では `HierBrakeLM` と同じ乱数の消費・同じ学習経過（同テスト）。
- `train_cf`: `moe_hier.train_hier` と同じ手順・乱数の消費で、損失に補助損失を足す（"shuf" の並べ替えは別の乱数列）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .moe_hier import GROUPS, HierBrakeLM

TEACHERS = ("cf", "cfs", "dist", "shuf", "self")


def forced_ce(model, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """h [B,T,d]（GRU の出力）で各トークンを専門家 e に送ったときの CE [N,E]。ゲートは e の確率（`oracle_assignment` と同じ）。"""
    raw = model.routers[0](h)
    probs = F.softmax(raw - model.threshold[0] if model.kind == "gate" else raw, dim=-1)
    ye = torch.einsum("btd,edf->btef", h, model.w1[0]).relu()
    ye = torch.einsum("btef,efd->bted", ye, model.w2[0])
    logits = model.head(h.unsqueeze(2) + probs.unsqueeze(-1) * ye)  # [B,T,E,V]
    v = logits.shape[-1]
    ce = F.cross_entropy(logits.reshape(-1, v), y.unsqueeze(-1).expand(*y.shape, model.n_experts).reshape(-1),
                         reduction="none")
    return ce.reshape(-1, model.n_experts)


class CounterfactualLM(HierBrakeLM):
    """ルータに教師つきの補助損失を足す（teacher: `TEACHERS`、lam: 係数 λ、tau: "cfs" の温度）。"""

    def __init__(self, *args, teacher: str | None = None, lam: float = 0.0, tau: float = 0.05,
                 shuffle_seed: int = 0, **kw) -> None:
        super().__init__(*args, **kw)
        if teacher is not None and teacher not in TEACHERS:
            raise ValueError(teacher)
        if teacher is not None and self.n_layers != 1:
            raise ValueError("教師つきのルータは1層の模型だけ")
        self.teacher, self.lam, self.tau = teacher, lam, tau
        self.shuffle_gen = torch.Generator().manual_seed(shuffle_seed)
        self.teacher_stats: list[dict] = []

    def teacher_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """λ·CE(router(h), 教師)。h と教師は勾配を切る。教師と閾値なしのルータの一致率を `teacher_stats` に残す。"""
        with torch.no_grad():
            h, _ = self.rnn(self.embed(x))
            raw0 = self.routers[0](h)
            if self.teacher == "dist":
                target = (raw0 - self.threshold[0]).argmax(-1).reshape(-1)
            elif self.teacher == "self":
                target = raw0.argmax(-1).reshape(-1)
            else:
                ce = forced_ce(self, h, y)
                if self.teacher == "cfs":
                    target = F.softmax(-ce / self.tau, dim=-1)
                else:
                    target = ce.argmin(-1)
                    if self.teacher == "shuf":
                        target = target[torch.randperm(target.numel(), generator=self.shuffle_gen)]
        raw = self.routers[0](h).reshape(-1, self.n_experts)
        with torch.no_grad():
            hard = target if target.dim() == 1 else target.argmax(-1)
            self.teacher_stats.append({"agree": float((raw.argmax(-1) == hard).float().mean())})
        return self.lam * F.cross_entropy(raw, target)


def train_cf(model: CounterfactualLM, train_data: torch.Tensor, steps: int, batch_size: int, seq_len: int,
             lr: float, generator: torch.Generator, balance_coef: float = 0.0) -> float:
    """`moe_hier.train_hier` と同じ手順・乱数の消費。teacher があれば損失に `teacher_loss` を足す。"""
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
