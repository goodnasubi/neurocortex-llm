"""閉じた後に偏りが戻る原因の切り分けと、ルータのシナプス固定化（12.6.177節）。

12.6.172節で、E=32 の臨界期型（cp_rand3 ほか）は閉じた後に使われない専門家が残り、偏りは教師の偏りでも専門家の
能力の欠如でも説明できなかった。ここでは (1) 閉じ始めから一つの部品を固定して原因を切り分け、(2) 臨界期に作った
割り当てをルータのシナプス固定化で守れるかを調べる。`moe_react.py` ほかの既存の模型の挙動は変えない。ここに置くもの:

- `ConsolLM`: `ReactCFLM` の継承（教師の閾値・再学習は使わないが、同じ教師 rand3 の経路を使う）。
  - 切り分け（閉じ始め = s < 1 になった最初のステップから学習の終わりまで）:
    `fix=("router",)` などで部品の群（router・front（埋め込みと GRU）・experts）の更新を止める（可塑性 0）。
    `gate_detach=True` はゲート（選ばれた専門家の確率）から LM の損失の勾配をルータに送らない（ゲートの値は同じ）。
    `teacher_off=True` は閉じ始めから反実仮想の教師の補助損失を外す。
  - 固定化（ルータの重みと偏りだけ。可塑性 f の要素ごとの倍率を、閉じる強さ s に合わせて f_eff = s + (1−s)·f にする）:
    "pnn": 一様な f = β（周囲神経網による一様な安定化。Pizzorusso et al., 2002）。
    "meta": 重要度に応じた f = 1/(1 + c·Ω/mean Ω)（メタ可塑性。Abraham & Bear, 1996）。Ω は開いている期間の
      シナプスの寄与の経路積分 Σ max(0, −g·Δw)（Zenke, Poole & Ganguli, 2017 の synaptic intelligence と同じ量）。
    "stc": タグ（開いている期間の |Δw| の移動平均）が中央値以上のシナプスを、閉じ始め（捕獲の合図）で固定する
      （f = 0、他は f = 1。シナプスのタグ付けと捕獲。Frey & Morris, 1997）。
    "bf": 閉じ始めから、見える重み u1 を隠れた変数の連鎖 u2・u3 に結ぶ（Benna & Fusi, 2016 のカスケード）。
      各ステップの後に u1 += g12(u2 − u1)、u2 += [g12(u1 − u2) + g23(u3 − u2)]/C2、u3 += g23(u2 − u3)/C3。
      結合は (1 − s) 倍。
    固定化は各ステップの最適化の後に w ← w_old + f_eff·(w_new − w_old) で掛ける（Adam の状態はそのまま）。
  - 診断（学習の計算には触れない）: 開いている期間の終わり（閉じ始めの直前）・閉じ終わり・学習の終わりの
    ルータと前段（埋め込み・GRU）の状態を残し、`cross_assignment` で組み合わせを入れ替えた割り当ての偏りを測る。
  既定（fix=()・gate_detach=False・teacher_off=False・consol=None）では `ReactCFLM`＋`train_react` と同じ学習経過
  （`tests/test_moe_consol.py`）。
- `train_consol`: `moe_react.train_react` と同じ手順・乱数の消費で、上の切り分けと固定化を掛ける。
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .hub_load import gini
from .moe_critical import close_strength
from .moe_hier import GROUPS
from .moe_react import ReactCFLM

PARTS = ("router", "front", "experts")
CONSOL = ("pnn", "meta", "stc", "bf")


class ConsolLM(ReactCFLM):
    """閉じ始めからの部品の固定（fix）・ゲートの勾配の切断・教師の停止と、ルータのシナプス固定化（consol）。"""

    def __init__(self, *args, fix: tuple[str, ...] = (), gate_detach: bool = False, teacher_off: bool = False,
                 consol: str | None = None, pnn_beta: float = 0.1, meta_c: float = 9.0,
                 bf_g12: float = 0.1, bf_g23: float = 0.025, bf_c: tuple[float, float] = (4.0, 16.0), **kw) -> None:
        super().__init__(*args, **kw)
        bad = set(fix) - set(PARTS)
        if bad or (consol is not None and consol not in CONSOL):
            raise ValueError(f"未知の部品・固定化: {sorted(bad)} {consol}")
        if (fix or gate_detach or teacher_off or consol) and self.n_layers != 1:
            raise ValueError("切り分けと固定化は1層の模型だけ")
        if gate_detach and self.kind != "select":
            raise ValueError("gate_detach は選択専用（select）の模型だけ")
        self.fix, self.gate_detach, self.teacher_off, self.consol = tuple(fix), gate_detach, teacher_off, consol
        self.pnn_beta, self.meta_c = pnn_beta, meta_c
        self.bf_g12, self.bf_g23, self.bf_c = bf_g12, bf_g23, bf_c
        self.closing = False  # 閉じ始め（s < 1）以降か
        self.f_mask: dict[str, torch.Tensor] = {}  # ルータのパラメータ名 → 可塑性の倍率 f（閉じ始めで決める）
        self.omega = {n: torch.zeros_like(p) for n, p in self.routers[0].named_parameters()}
        self.tag = {n: torch.zeros_like(p) for n, p in self.routers[0].named_parameters()}
        self.chain: dict[str, list[torch.Tensor]] = {}
        self.snap: dict[str, dict] = {}

    def run(self, x: torch.Tensor, disabled=None, mode: str = "zero", gate_one: bool = False,
            use_threshold: bool = True) -> dict:
        if not (getattr(self, "gate_detach", False) and self.closing and self.training):
            return super().run(x, disabled, mode, gate_one, use_threshold)
        # 1層・閾値は選択だけ（select）の学習中の順伝播。ゲートは確率の値を使い、ルータへの勾配だけ切る
        h, _ = self.rnn(self.embed(x))
        raw = self.routers[0](h)
        th = self.threshold[0] if use_threshold else torch.zeros_like(self.threshold[0])
        probs = F.softmax(raw.detach(), dim=-1)
        choice = (raw - th).argmax(-1)
        gate = probs.gather(-1, choice.unsqueeze(-1))
        y = torch.einsum("btd,btdf->btf", h, self.w1[0][choice]).relu()
        y = torch.einsum("btf,btfd->btd", y, self.w2[0][choice])
        gy = gate * y
        h = h + gy
        frac = F.one_hot(choice, self.n_experts).float().mean(dim=(0, 1))
        balance = self.n_experts * (frac * probs.mean(dim=(0, 1))).sum()
        return {"logits": self.head(h), "choice": choice.unsqueeze(0), "gate": gate.squeeze(-1).unsqueeze(0),
                "contrib": gy.norm(dim=-1).unsqueeze(0), "router": raw.unsqueeze(0), "balance": balance}

    def part_parameters(self) -> dict[str, list[torch.nn.Parameter]]:
        return {"router": list(self.routers.parameters()), "front": [*self.embed.parameters(), *self.rnn.parameters()],
                "experts": [*self.w1, *self.w2]}

    def save_snapshot(self, tag: str) -> None:
        self.snap[tag] = {"router": copy.deepcopy(self.routers[0].state_dict()),
                          "embed": copy.deepcopy(self.embed.state_dict()), "rnn": copy.deepcopy(self.rnn.state_dict()),
                          "w1": self.w1[0].detach().clone(), "w2": self.w2[0].detach().clone(),
                          "threshold": self.threshold.clone(), "strength": self.strength, "step": self.n_steps}

    @torch.no_grad()
    def begin_closing(self) -> None:
        """閉じ始め（捕獲の合図）。固定化の可塑性の倍率 f と、カスケードの隠れた変数を決める。"""
        self.closing = True
        params = dict(self.routers[0].named_parameters())
        if self.consol == "pnn":
            self.f_mask = {n: torch.full_like(p, self.pnn_beta) for n, p in params.items()}
        elif self.consol == "meta":
            om = {n: o.clamp(min=0.0) for n, o in self.omega.items()}
            mean = torch.cat([o.reshape(-1) for o in om.values()]).mean().clamp(min=1e-12)
            self.f_mask = {n: 1.0 / (1.0 + self.meta_c * o / mean) for n, o in om.items()}
        elif self.consol == "stc":
            med = torch.cat([t.reshape(-1) for t in self.tag.values()]).median()
            self.f_mask = {n: (t < med).float() for n, t in self.tag.items()}
        elif self.consol == "bf":
            self.chain = {n: [p.detach().clone(), p.detach().clone()] for n, p in params.items()}

    def plasticity_mean(self) -> float | None:
        """固定化の可塑性の倍率 f の平均（閉じ終わりの値。bf は None）。"""
        if not self.f_mask:
            return None
        return float(torch.cat([f.reshape(-1) for f in self.f_mask.values()]).mean())


def train_consol(model: ConsolLM, train_data: torch.Tensor, steps: int, batch_size: int, seq_len: int,
                 lr: float, generator: torch.Generator, balance_coef: float = 0.0) -> float:
    """`moe_react.train_react` と同じ手順・乱数の消費。閉じ始めから部品の固定・固定化を掛ける。"""
    gp = model.group_parameters()
    braked = [g for g in GROUPS if g in model.brake_sched]
    ids = {id(p) for g in braked for p in gp[g]}
    rp = [p for p in model.router_parameters() if id(p) not in ids]
    ids |= {id(p) for p in rp}
    groups = [{"params": gp[g], "name": g} for g in braked]
    groups += [{"params": rp, "name": "_router"}, {"params": [p for p in model.parameters() if id(p) not in ids],
                                                   "name": "_rest"}]
    opt = torch.optim.Adam([g for g in groups if g["params"]], lr=lr)
    parts = model.part_parameters()
    rnamed = dict(model.routers[0].named_parameters())
    t_open = None if model.open_frac is None else round(model.open_frac * model.total_steps)
    t_close = None if model.open_frac is None else round(min(1.0, model.open_frac + model.ramp_frac)
                                                          * model.total_steps)
    track = model.consol in ("meta", "stc")
    ce = float("nan")
    for _ in range(steps):
        idx = torch.randint(0, len(train_data) - seq_len - 1, (batch_size,), generator=generator)
        x = torch.stack([train_data[i:i + seq_len] for i in idx.tolist()])
        y = torch.stack([train_data[i + 1:i + seq_len + 1] for i in idx.tolist()])
        if not model.closing and model.kind != "plain" and model.strength < 1.0:
            model.begin_closing()
        logits, balance, choice = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        ce = float(loss.detach())
        opt.zero_grad(set_to_none=True)
        total = loss + balance_coef * balance
        if model.teacher is not None and not (model.teacher_off and model.closing):
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
        frozen = [p for g in model.fix for p in parts[g]] if model.closing else []
        old = {id(p): p.detach().clone() for p in frozen}
        if model.closing and model.f_mask:
            old.update({id(p): p.detach().clone() for p in rnamed.values()})
        if track and not model.closing:
            prev = {n: p.detach().clone() for n, p in rnamed.items()}
            grads = {n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
                     for n, p in rnamed.items()}
        opt.step()
        with torch.no_grad():
            for p in frozen:  # 部品の固定（可塑性 0）
                p.copy_(old[id(p)])
            if model.closing and model.f_mask:  # 固定化: w ← w_old + f_eff·(w_new − w_old)
                s = model.strength
                for n, p in rnamed.items():
                    f = s + (1.0 - s) * model.f_mask[n]
                    p.copy_(old[id(p)] + f * (p - old[id(p)]))
            if model.closing and model.chain:  # Benna & Fusi のカスケード（結合は (1 − s) 倍）
                k = 1.0 - model.strength
                g12, g23 = k * model.bf_g12, k * model.bf_g23
                c2, c3 = model.bf_c
                for n, p in rnamed.items():
                    u2, u3 = model.chain[n]
                    d12, d23 = u2 - p, u3 - u2
                    p.add_(g12 * d12)
                    u2.add_((-g12 * d12 + g23 * d23) / c2)
                    u3.add_(-g23 * d23 / c3)
            if track and not model.closing:  # 開いている期間の重要度（経路積分）とタグ（|Δw| の移動平均）
                for n, p in rnamed.items():
                    dw = p - prev[n]
                    model.omega[n].add_((-grads[n] * dw).clamp(min=0.0))
                    model.tag[n].mul_(1 - model.alpha).add_(model.alpha * dw.abs())
        model.homeostatic_step(choice)
        if model.kind != "plain" and model.n_steps in (t_open, t_close):
            model.save_snapshot("open" if model.n_steps == t_open else "close")
    if model.kind != "plain":
        model.save_snapshot("end")
    return ce


@torch.no_grad()
def cross_assignment(model: ConsolLM, x: torch.Tensor, rho: float = 0.25) -> dict:
    """保存したルータ（R）と前段（F、埋め込みと GRU）の組 (R_a, F_b) で閾値なしの割り当てを作り、偏りを測る。

    a, b ∈ {open, close, end}。返り値の "gini"[a][b] はジニ、"used"[a][b] は使われる専門家の数。"on_open" は開いている
    期間の終わりの閾値ありの割り当て（臨界期に作った割り当て）のジニ。"dormant_end_from_open" は学習の終わりに
    使われない（負荷 < ρ/E）専門家のうち、開いている期間の終わりの閾値なしの割り当てでもすでに負荷 < ρ/E だった割合。
    "gate" は各時点のゲートの平均、"expert_change" は専門家の重みの開いている期間の終わりからの相対変化。
    """
    tags = [t for t in ("open", "close", "end") if t in model.snap]
    if "open" not in tags:
        return {}
    cur = copy.deepcopy(model.state_dict())
    e = model.n_experts
    hs, routers = {}, {}
    for t in tags:
        model.embed.load_state_dict(model.snap[t]["embed"])
        model.rnn.load_state_dict(model.snap[t]["rnn"])
        hs[t] = model.rnn(model.embed(x))[0]
        routers[t] = model.snap[t]["router"]
    out: dict = {"gini": {}, "used": {}}
    loads = {}
    for a in tags:
        model.routers[0].load_state_dict(routers[a])
        out["gini"][a], out["used"][a] = {}, {}
        for b in tags:
            c = model.routers[0](hs[b]).argmax(-1)
            load = np.bincount(c.reshape(-1).numpy(), minlength=e).astype(float)
            loads[(a, b)] = load
            out["gini"][a][b], out["used"][a][b] = float(gini(load)), int((load > 0).sum())
        if a == "open":
            raw = model.routers[0](hs["open"])
            c_on = (raw - model.snap["open"]["threshold"][0]).argmax(-1)
            load_on = np.bincount(c_on.reshape(-1).numpy(), minlength=e).astype(float)
            out["on_open"] = float(gini(load_on))
            out["agree_open"] = float((c_on == raw.argmax(-1)).float().mean())
    model.load_state_dict(cur)
    if "end" in tags:
        n = loads[("open", "open")].sum()
        dorm_end = loads[("end", "end")] < rho * n / e
        dorm_open = loads[("open", "open")] < rho * n / e
        out["dormant_end"] = int(dorm_end.sum())
        out["dormant_open_off"] = int(dorm_open.sum())
        out["dormant_end_from_open"] = float((dorm_end & dorm_open).sum() / dorm_end.sum()) if dorm_end.any() else None
        out["expert_change"] = float(torch.cat([(model.snap["end"][k] - model.snap["open"][k]).reshape(-1)
                                                for k in ("w1", "w2")]).norm()
                                     / torch.cat([model.snap["open"][k].reshape(-1) for k in ("w1", "w2")]).norm())
    gates = {}
    for t in tags:
        model.embed.load_state_dict(model.snap[t]["embed"])
        model.rnn.load_state_dict(model.snap[t]["rnn"])
        model.routers[0].load_state_dict(model.snap[t]["router"])
        raw = model.routers[0](model.rnn(model.embed(x))[0])
        th = model.snap[t]["threshold"][0]
        c = (raw - th).argmax(-1)
        gates[t] = float(F.softmax(raw, -1).gather(-1, c.unsqueeze(-1)).mean())
    model.load_state_dict(cur)
    out["gate"] = gates
    return out
