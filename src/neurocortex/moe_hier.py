"""可塑性のブレーキの範囲と時期、臨界期の階層的な順序、CE を下げる割り当てと均等な割り当てのずれ（12.6.153節）。

`moe_scale.py`・`moe_critical.py`・`moe_brake.py` の挙動は変えない。ここに置くもの:

- `HierBrakeLM`: `BrakedCriticalLM` の継承。ブレーキを部品の群（埋め込み `embed`・GRU `rnn`・ルータ `router`・専門家
  `experts`）ごとに、それぞれの閉じ方（開いている期間, 閉じる期間）で掛ける。群 g の学習率の倍率は
  m_g(t) = β + (1−β)·close_strength(t, T, o_g, r_g)。出力層 `head` には掛けない。閾値の閉じ方（open_frac・ramp_frac）
  と θ の積分を止めるか（freeze）は `BrakedCriticalLM` のまま。ブレーキの時期を閾値の閉じ方と独立に動かせるので、
  下位の表現から順に閉じる階層的な順序（感覚野 → 連合野、Hensch, 2005; Reh et al., 2020）と、逆順・同時を比べられる。
  brake_sched=None（既定）では `BrakedCriticalLM` と同じ学習経過（`tests/test_moe_hier.py`）。ルータだけを閾値と
  同じ閉じ方で braked にすると 12.6.149節の `brake=β` と同じ学習経過になる（同テスト）。
- `train_hier`: `moe_brake.train_braked` と同じ手順・乱数の消費で、群ごとに学習率を m_g 倍にする。
- `oracle_assignment`: 1層の模型で、各トークンを各専門家に強制的に送ったときの CE を測り、CE を最小にする専門家
  （CE を下げる割り当て）を求める。その負荷の偏り（ジニ）と、閾値あり／なしの選択との一致、閾値がずらしたトークンの
  うち閾値なしの選択が CE 最小だった割合（閾値が CE に逆らってずらした割合）などを返す（12.6.150節の残課題1）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hub_load import gini
from .moe_brake import BrakedCriticalLM
from .moe_critical import close_strength

GROUPS = ("embed", "rnn", "router", "experts")


class HierBrakeLM(BrakedCriticalLM):
    """群ごとの閉じ方で学習率のブレーキを掛ける（brake_sched: 群 → (open, ramp)、beta: 閉じ終わりの倍率）。"""

    def __init__(self, *args, brake_sched: dict[str, tuple[float, float]] | None = None, beta: float = 0.1,
                 **kw) -> None:
        super().__init__(*args, **kw)
        bad = set(brake_sched or {}) - set(GROUPS)
        if bad:
            raise ValueError(f"未知の群: {sorted(bad)}")
        self.brake_sched, self.beta = dict(brake_sched or {}), beta
        self.group_lr_scale = {g: 1.0 for g in self.brake_sched}

    def group_parameters(self) -> dict[str, list[nn.Parameter]]:
        return {"embed": list(self.embed.parameters()), "rnn": list(self.rnn.parameters()),
                "router": list(self.routers.parameters()), "experts": [*self.w1, *self.w2]}

    @torch.no_grad()
    def homeostatic_step(self, choice: torch.Tensor) -> None:
        super().homeostatic_step(choice)
        if self.kind == "plain":
            return
        for g, (o, r) in self.brake_sched.items():  # 次のステップで使う倍率（`train_braked` のルータの倍率と同じ時期）
            self.group_lr_scale[g] = self.beta + (1.0 - self.beta) * close_strength(self.n_steps, self.total_steps, o, r)

    @torch.no_grad()
    def snapshot(self, with_vectors: bool = False) -> dict:
        p = super().snapshot(with_vectors)
        for g in self.brake_sched:
            p[f"lr_{g}"] = self.group_lr_scale.get(g, 1.0)
        return p


def train_hier(model: HierBrakeLM, train_data: torch.Tensor, steps: int, batch_size: int, seq_len: int,
               lr: float, generator: torch.Generator, balance_coef: float = 0.0) -> float:
    """`moe_brake.train_braked` と同じ手順・乱数の消費。群ごとに学習率を `group_lr_scale` 倍にする
    （ルータの `router_lr_scale` も従来どおり掛かる。Adam の更新は要素ごとなので、群の分け方は倍率 1 では結果を変えない）。"""
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
        (loss + balance_coef * balance).backward()
        for pg in opt.param_groups:
            name = pg["name"]
            scale = model.group_lr_scale.get(name, 1.0) if name in GROUPS else 1.0
            if name == "_router" or (name == "router" and model.brake is not None):
                scale *= model.router_lr_scale
            pg["lr"] = lr * scale
        opt.step()
        model.homeostatic_step(choice)
    return ce


@torch.no_grad()
def oracle_assignment(model, x: torch.Tensor, y: torch.Tensor) -> dict:
    """1層の模型で、各トークンを専門家 e に強制的に送ったときの CE（ゲートは e の確率）から、CE 最小の専門家を求める。

    返り値: oracle_gini・oracle_n_used（CE を下げる割り当ての負荷の偏り）、ce_on / ce_off / ce_oracle（トークン平均）、
    agree_on / agree_off（閾値あり／なしの選択と CE 最小の専門家の一致率）、shift_frac（閾値が選択を変えた割合）、
    shift_cost（ずらしたトークンのうち閾値なしの選択が CE 最小だった割合 = 閾値が CE に逆らった割合）、
    shift_benefit（ずらしたトークンのうち閾値ありの選択が CE 最小だった割合）、regret_on / regret_off（選択の CE − 最小 CE）。
    """
    if model.n_layers != 1:
        raise ValueError("oracle_assignment は1層の模型だけ")
    h, _ = model.rnn(model.embed(x))
    raw = model.routers[0](h)
    th = model.threshold[0]
    probs = F.softmax(raw - th if model.kind == "gate" else raw, dim=-1)
    ce = []
    for e in range(model.n_experts):
        ye = torch.einsum("btd,df->btf", h, model.w1[0][e]).relu()
        ye = torch.einsum("btf,fd->btd", ye, model.w2[0][e])
        logits = model.head(h + probs[..., e:e + 1] * ye)
        ce.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none"))
    ce = torch.stack(ce, dim=-1)  # [N, E]
    best = ce.argmin(-1)
    on = model.run(x)["choice"][0].reshape(-1)
    off = model.run(x, use_threshold=False)["choice"][0].reshape(-1)
    load = np.bincount(best.numpy(), minlength=model.n_experts).astype(float)
    shifted = on != off
    n_sh = int(shifted.sum())
    pick = ce.gather(-1, on.unsqueeze(-1)).squeeze(-1), ce.gather(-1, off.unsqueeze(-1)).squeeze(-1)
    cmin = ce.min(-1).values
    return {"oracle_gini": float(gini(load)), "oracle_n_used": int((load > 0).sum()),
            "ce_on": float(pick[0].mean()), "ce_off": float(pick[1].mean()), "ce_oracle": float(cmin.mean()),
            "agree_on": float((on == best).float().mean()), "agree_off": float((off == best).float().mean()),
            "shift_frac": n_sh / on.numel(),
            "shift_cost": float((off[shifted] == best[shifted]).float().mean()) if n_sh else None,
            "shift_benefit": float((on[shifted] == best[shifted]).float().mean()) if n_sh else None,
            "regret_on": float((pick[0] - cmin).mean()), "regret_off": float((pick[1] - cmin).mean())}
