"""臨界期を閉じる直前の睡眠に似たオフラインの段階で、恒常性の閾値が作った割り当てをルータのシナプスに写す（12.6.221節）。

12.6.178節の分解で、E=32 の臨界期型（cp_rand3）の学習の終わりの偏りの主因は、閉じる時点でルータが臨界期の割り当て
（閾値ありの選択）を取り込みきれていないこと（取り込みの不足 A = 0.105、閉じた後の漂流 D = 0.034）だった。固定化は
守る対象（閉じる時点のルータ）がまだ偏っているので効かず、「固定化の前に、閾値が作った割り当てを重みに写す段階が要る」と
結論した。ここではその段階を、覚醒中の入力の再生（海馬のバッファ）による睡眠中の蒸留として足す。`moe_consol.py` ほか
既存の模型の挙動は変えない。ここに置くもの:

- `SleepLM`: `moe_consol.ConsolLM` の継承。
  - `sleep="w"`・`"wb"`・`"perm"`: 閉じ始め（s < 1 になった最初のステップの順伝播の前）に、開いている期間の最後の
    `sleep_buffer` 個の学習の入力 x（覚醒中の経験。リングバッファ）を再生する。再生した x から今の前段（埋め込みと GRU、
    勾配なし）で h を作り、睡眠に入る時点のルータ（凍結した写し）の対数 r から目標 q = softmax(r − θ) を作る（θ は
    閾値の状態。睡眠に入る時点の閾値ありの選択 argmax(r − θ) を、同じ鋭さのまま閾値なしの対数に移す目標）。
    ルータの重みを KL(q ‖ softmax(router(h))) で `sleep_steps` 回、別の Adam（学習率 `sleep_lr`）で学ぶ。
    "w" は重み W だけ（シナプス。偏り b = 内在的な興奮性は動かさない）、"wb" は W と b、"perm" は W だけで、θ を
    専門家の間で並べ替えた目標（同じ大きさのずれで内容のない対照）。覚醒中の学習の Adam の状態・乱数には触れない
    （再生の順は別の乱数 `sleep_seed`）。睡眠の直後のルータと前段を `snap["slept"]` に残す。
  - `fold=True`: 閉じる間、閾値の強さ s が下がった分の θ をルータの偏りに移す（b ← b − (s_prev − s)·θ）。閉じた後も
    閾値ありの選択が保たれる（内在的な興奮性の恒久化）。睡眠と組にはしない。
  既定（sleep=None・fold=False）では `ConsolLM`＋`train_consol` と同じ学習経過（`tests/test_moe_sleep.py`）。
- `slept_assignment`: 睡眠の直後のルータ R_slept と開いている期間の終わりの前段 F_open の組での閾値なしの割り当ての
  偏りと、開いている期間の終わりの閾値ありの選択との一致率（睡眠で写せた量）。
- 学習は `moe_consol.train_consol` をそのまま使う（`begin_closing` と `homeostatic_step` を上書きする）。
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .hub_load import gini
from .moe_consol import ConsolLM

SLEEP = ("w", "wb", "perm")


class SleepLM(ConsolLM):
    """閉じ始めの睡眠に似た再生による蒸留（sleep）と、閾値のルータの偏りへの恒久化（fold）。"""

    def __init__(self, *args, sleep: str | None = None, sleep_buffer: int = 16, sleep_steps: int = 200,
                 sleep_lr: float = 3e-3, sleep_seed: int = 0, fold: bool = False, **kw) -> None:
        super().__init__(*args, **kw)
        if sleep is not None and sleep not in SLEEP:
            raise ValueError(f"未知の睡眠: {sleep}")
        if (sleep or fold) and (self.n_layers != 1 or self.kind != "select"):
            raise ValueError("睡眠と恒久化は1層の選択専用（select）の模型だけ")
        if sleep and fold:
            raise ValueError("睡眠と恒久化は組にしない")
        self.sleep, self.sleep_buffer, self.sleep_steps, self.sleep_lr = sleep, sleep_buffer, sleep_steps, sleep_lr
        self.sleep_seed, self.fold = sleep_seed, fold
        self.replay: list[torch.Tensor] = []
        self.sleep_stats: dict = {}
        self._s_prev = 1.0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.sleep and self.training and not self.closing:  # 覚醒中の経験を海馬のバッファに残す
            self.replay.append(x.detach().clone())
            if len(self.replay) > self.sleep_buffer:
                self.replay.pop(0)
        return super().forward(x)

    @torch.no_grad()
    def begin_closing(self) -> None:
        super().begin_closing()
        if self.sleep:
            with torch.enable_grad():
                self._sleep()
            self.snap["slept"] = {"router": copy.deepcopy(self.routers[0].state_dict()),
                                  "embed": copy.deepcopy(self.embed.state_dict()),
                                  "rnn": copy.deepcopy(self.rnn.state_dict()), "step": self.n_steps}

    def _sleep(self) -> None:
        router = self.routers[0]
        frozen = copy.deepcopy(router)
        theta = self.threshold_state[0].clone()
        g = torch.Generator().manual_seed(self.sleep_seed)
        if self.sleep == "perm":
            theta = theta[torch.randperm(self.n_experts, generator=g)]
        with torch.no_grad():
            hs = [self.rnn(self.embed(x))[0].reshape(-1, router.in_features) for x in self.replay]
            qs = [F.softmax(frozen(h) - theta, dim=-1) for h in hs]
        params = [router.weight] if self.sleep in ("w", "perm") else [router.weight, router.bias]
        opt = torch.optim.Adam(params, lr=self.sleep_lr)
        losses = []
        for _ in range(self.sleep_steps):
            i = int(torch.randint(0, len(hs), (1,), generator=g))
            logp = F.log_softmax(router(hs[i]), dim=-1)
            loss = F.kl_div(logp, qs[i], reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        for p in router.parameters():
            p.grad = None
        with torch.no_grad():
            h = torch.cat(hs)
            on = (frozen(h) - self.threshold_state[0]).argmax(-1)
            before, after = frozen(h).argmax(-1), router(h).argmax(-1)
            self.sleep_stats = {"loss_first": losses[0], "loss_last": losses[-1], "n_replay": len(hs),
                                "agree_before": float((before == on).float().mean()),
                                "agree_after": float((after == on).float().mean()),
                                "bias_change": float((router.bias - frozen.bias).norm()),
                                "weight_rel_change": float((router.weight - frozen.weight).norm() / frozen.weight.norm())}

    @torch.no_grad()
    def homeostatic_step(self, choice: torch.Tensor) -> None:
        super().homeostatic_step(choice)
        if self.fold and self.kind != "plain":
            ds = self._s_prev - self.strength
            if ds > 0:  # 下がった分の閾値を偏りに移す（θ は閉じ始めから積分を止めている）
                self.routers[0].bias.sub_(ds * self.threshold_state[0])
            self._s_prev = self.strength


@torch.no_grad()
def slept_assignment(model: SleepLM, x: torch.Tensor) -> dict:
    """睡眠の直後のルータ（R_slept）と開いている期間の終わりの前段（F_open）で作る閾値なしの割り当ての偏りと、
    開いている期間の終わりの閾値ありの選択（R_open, F_open, θ_open）との一致率。睡眠がなければ空。"""
    if "slept" not in model.snap or "open" not in model.snap:
        return {}
    cur = copy.deepcopy(model.state_dict())
    o = model.snap["open"]
    model.embed.load_state_dict(o["embed"])
    model.rnn.load_state_dict(o["rnn"])
    h = model.rnn(model.embed(x))[0]
    model.routers[0].load_state_dict(o["router"])
    on = (model.routers[0](h) - o["threshold"][0]).argmax(-1)
    model.routers[0].load_state_dict(model.snap["slept"]["router"])
    c = model.routers[0](h).argmax(-1)
    model.load_state_dict(cur)
    load = np.bincount(c.reshape(-1).numpy(), minlength=model.n_experts).astype(float)
    return {"gini_slept_off": float(gini(load)), "used": int((load > 0).sum()),
            "agree_with_on_open": float((c == on).float().mean())}
