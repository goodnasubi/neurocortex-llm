"""専門家の選択（ルータ）を線条体の直接路（G、Go、D1）と間接路（N、NoGo、D2）の対立で表す OpAL 型ルータ（12.6.207節）。

12.6.186節で、ルータの学習則を臨界期の前後で切り替えても、閉じる前の取り込みの不足 A と閉じた後の漂流 D は独立で
なかった（D ≈ −0.71·A）。ここではルータを OpAL（Collins & Frank, 2014）の2系統の重みで表す。選択の強さ（ルータの
対数）は act_j(h) = β·(G_j(h) − N_j(h))、G_j(h) = σ(w^G_j·h + b^G_j)、N_j(h) = σ(w^N_j·h + b^N_j)（どちらも
0〜1 の発火率。中型有棘ニューロンの発火率の飽和。上限のない softplus は予備調整で発散した。12.6.207節）。学習はドーパミンの報酬予測誤差 δ を第三因子とする逆向きのヘッブ則だけで行う（下流の LM の誤差を
ゲートからルータへ送らない。`moe_tf` の tf と同じ順伝播）:

    Δw^G_k = α · mean_n[ δ_nk · G_k(h_n) · h_n ]      （D1: δ > 0 で LTP、δ < 0 で LTD）
    Δw^N_k = α · mean_n[ −δ_nk · N_k(h_n) · h_n ]     （D2: 逆向き）
    （偏り b も同じ式で h_n の代わりに 1）

シナプス前の活動 h × シナプス後の活動 G（または N）× ドーパミンの三要素で、OpAL の G ← G + α·G·δ、
N ← N + α·N·(−δ) を線形の入力から読む形にしたもの。δ は 12.6.181節の部分評価（rand3: 閾値が実際に選んだ専門家 c と、
c 以外から無作為な 3 個）で、評価した専門家 k ごとに δ_nk = V_n − CE_nk、V_n = 評価した専門家の CE の平均
（クリティックの状態価値をその場の標本で置き換えたもの。報酬 = −CE。Kishida et al., 2016 の反実仮想の結果の誤差）。
評価する専門家・乱数の消費・CE の計算は tf_srand3（`moe_tf.ThreeFactorLM`、soft）と同じで、学習則だけが違う。

- `OpponentRouter`: G と N の2つの読み出し。forward は β·(G − N)（`routers[0]` と同じ形の対数）。
- `OpALLM`: `moe_sw.SwitchLM`（`ThreeFactorLM` の継承）の継承。`opal="gn"` は G と N の両方、`opal="g"` は G だけ（N の重みを初期値に固定する対照）。
  重みの更新は教師の計算（`teacher_loss`）で求め、最適化の後（`homeostatic_step` の初め）に足す。ルータの重みは
  LM の勾配も Adam も受けない。
  `opal_delta="critic"`（修正1、12.6.208節）: OpAL の元の形に合わせ、実際に選んだ専門家 c だけを δ_n = −CE_nc − V(h_n) で
  更新する。V(h) = v·h + b は同じ δ で学ぶ線形のクリティック（ゼロで初期化）。評価の組と乱数の消費は "cf" と同じ。
  `opal_delta="cfcritic"`（修正3）: 評価したすべての専門家 k を δ_nk = −CE_nk − V(h_n) で更新する（選ばなかった専門家の
  結果も、実際に選んだ行動の期待との差として第三因子にする。Kishida et al., 2016）。クリティックは c の δ だけで学ぶ。
  `critic_lr`（修正2）: クリティックを正規化 LMS（ステップ critic_lr / mean(|h|² + 1)）で学ぶ。None は行為者と同じ α（修正1）。`opal=None` では `SwitchLM` と同じ（`tests/test_moe_opal.py`）。
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe_cfe import _dup_mask, _random_others, subset_ce
from .moe_sw import SwitchLM

OPAL = ("gn", "g")


class OpponentRouter(nn.Module):
    """act = β·(G − N)。act="sigmoid"（既定）は G・N = σ(線形)（発火率の飽和。初期は G ≈ N ≈ 0.5）、
    "softplus" は G・N = softplus(線形)（上限なし。初期は G ≈ N ≈ 1 で OpAL の初期値。12.6.207節の予備調整1）。w は小さな乱数。"""

    def __init__(self, d: int, n_experts: int, init_std: float, gen: torch.Generator, act: str = "sigmoid",
                 beta: float = 1.0) -> None:
        super().__init__()
        if act not in ("sigmoid", "softplus"):
            raise ValueError(f"未知の act: {act}")
        self.act, self.beta = act, beta
        b0 = math.log(math.e - 1.0) if act == "softplus" else 0.0  # softplus(b0) = 1、σ(0) = 0.5
        self.wg = nn.Parameter(torch.randn(n_experts, d, generator=gen) * init_std)
        self.wn = nn.Parameter(torch.randn(n_experts, d, generator=gen) * init_std)
        self.bg = nn.Parameter(torch.full((n_experts,), b0))
        self.bn = nn.Parameter(torch.full((n_experts,), b0))

    def pathways(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = torch.sigmoid if self.act == "sigmoid" else F.softplus
        return f(h @ self.wg.T + self.bg), f(h @ self.wn.T + self.bn)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        g, n = self.pathways(h)
        return self.beta * (g - n)


class OpALLM(SwitchLM):
    """OpAL 型ルータ（G − N）を、反実仮想の報酬予測誤差を第三因子とする逆向きのヘッブ則だけで学ぶ。"""

    def __init__(self, *args, opal: str | None = None, opal_lr: float = 0.1, opal_init: float | None = None,
                 opal_seed: int = 0, opal_act: str = "sigmoid", opal_beta: float = 4.0,
                 opal_delta: str = "cf", critic_lr: float | None = None, **kw) -> None:
        super().__init__(*args, **kw)
        if opal is None:
            return
        if opal not in OPAL:
            raise ValueError(f"未知の opal: {opal}")
        if not self.tf or self.soft or self.eq_gate or self.pg or self.switch or self.soft_after or self.kind_t != "rand":
            raise ValueError("opal は tf=True と部分評価の教師 rand だけ（soft・eq_gate・pg とは組まない）")
        if opal_delta not in ("cf", "critic", "cfcritic"):
            raise ValueError(f"未知の opal_delta: {opal_delta}")
        self.opal, self.opal_lr, self.opal_delta, self.critic_lr = opal, opal_lr, opal_delta, critic_lr
        d = self.routers[0].in_features
        if opal_init is None:  # 初期の対数の散らばりを既定の線形ルータ（重み U(±1/√d)）にそろえる（σ の傾き 1/4 の近似）
            opal_init = 4.0 / (math.sqrt(6.0 * d) * opal_beta)
        self.routers[0] = OpponentRouter(d, self.n_experts, opal_init, torch.Generator().manual_seed(opal_seed),
                                         opal_act, opal_beta)
        for p in self.routers[0].parameters():
            p.requires_grad_(False)
        self.omega = {n: torch.zeros_like(p) for n, p in self.routers[0].named_parameters()}
        self.tag = {n: torch.zeros_like(p) for n, p in self.routers[0].named_parameters()}
        if self.probe is not None:
            self.logit0 = self._mean_logits()
        self.register_buffer("critic_w", torch.zeros(d))  # 修正1 のクリティック V(h) = critic_w·h + critic_b
        self.register_buffer("critic_b", torch.zeros(()))
        self.pending: dict[str, torch.Tensor] | None = None
        self.opal_stats: list[dict] = []

    def teacher_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if getattr(self, "opal", None) is None:
            return super().teacher_loss(x, y)
        with torch.no_grad():
            return self._opal_update(x, y)

    def _opal_update(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """rand3 の評価から報酬予測誤差 δ を作り、G・N の更新を pending に置く（足すのは最適化の後）。"""
        t0 = time.perf_counter()
        h, _ = self.rnn(self.embed(x))  # `ThreeFactorLM.teacher_loss` と同じ評価の組・同じ乱数の消費
        hf = h.reshape(-1, h.shape[-1])
        router = self.routers[0]
        raw0 = router(hf)
        chosen = (raw0 - self.threshold[0]).argmax(-1)
        idx = torch.cat([chosen.unsqueeze(-1), _random_others(chosen, self.n_experts, self.m, self.sample_gen)], -1)
        ce = subset_ce(self, h, y, idx)
        dup = _dup_mask(idx)
        ok = (~dup).float()
        crit = None
        if self.opal_delta == "cf":
            v = (ce * ok).sum(-1, keepdim=True) / ok.sum(-1, keepdim=True)  # 状態価値（評価した結果の平均）
            delta = (v - ce) * ok  # 報酬予測誤差（報酬 = −CE）。重複は 0
        else:  # 修正1: 実際に選んだ専門家だけ、学習するクリティックとの差 δ = −CE_c − V(h)
            vh = hf @ self.critic_w + self.critic_b
            d_c = -ce[:, 0] - vh
            if self.opal_delta == "critic":
                delta = torch.zeros_like(ce)
                delta[:, 0] = d_c
            else:  # 修正3: 評価したすべての専門家を同じクリティックとの差 δ_k = −CE_k − V(h) で（クリティックは c だけで学ぶ）
                delta = (-ce - vh.unsqueeze(-1)) * ok
            if self.critic_lr is None:  # 修正1: 行為者と同じ α
                lr_w = lr_b = self.opal_lr
            else:  # 修正2: 正規化 LMS（入力の大きさで割る。ステップ critic_lr ∈ (0, 2) で安定）
                lr_w = lr_b = self.critic_lr / float((hf.pow(2).sum(-1) + 1.0).mean())
            crit = {"critic_w": lr_w * (d_c.unsqueeze(-1) * hf).mean(0), "critic_b": lr_b * d_c.mean()}
        g, nn_ = router.pathways(hf)
        n = hf.shape[0]
        dg = torch.zeros_like(g).scatter_add_(1, idx, delta * g.gather(1, idx)) / n  # δ·G（D1）
        dn = torch.zeros_like(nn_).scatter_add_(1, idx, -delta * nn_.gather(1, idx)) / n  # −δ·N（D2）
        a = self.opal_lr
        upd = {"wg": a * dg.T @ hf, "bg": a * dg.sum(0)}
        if self.opal == "gn":
            upd |= {"wn": a * dn.T @ hf, "bn": a * dn.sum(0)}
        self.pending = upd
        self.pending_critic = crit
        hard = idx.gather(-1, ce.masked_fill(dup, float("inf")).argmin(-1, keepdim=True)).squeeze(-1)
        self.teacher_stats.append({"agree": float((raw0.argmax(-1) == hard).float().mean())})
        self.opal_stats.append({"delta_chosen": float(delta[:, 0].mean()), "G": float(g.mean()),
                                "N": float(nn_.mean())})
        self._count(sum(len(set(r)) for r in idx.tolist()), y.numel(), t0)
        return hf.new_zeros(())

    @torch.no_grad()
    def homeostatic_step(self, choice: torch.Tensor) -> None:
        if getattr(self, "pending", None):
            params = dict(self.routers[0].named_parameters())
            for k, dw in self.pending.items():
                params[k].add_(dw)
            self.pending = None
        if getattr(self, "pending_critic", None):
            self.critic_w.add_(self.pending_critic["critic_w"])
            self.critic_b.add_(self.pending_critic["critic_b"])
            self.pending_critic = None
        super().homeostatic_step(choice)
