"""ルータの学習を局所の活動 × 反実仮想の教える信号の三要素則だけにする（12.6.181節）。

12.6.178節で、E=32 の学習の終わりの偏りは主に閉じる前の取り込みの不足（A）で、教師なしの漂流はゲート（選ばれた
専門家の確率）からルータへ流れる LM の勾配の正のフィードバックによった。ここでは学習の初めから、その勾配をルータに
送らない。ルータの重みが受けるのは反実仮想の教師の補助損失の勾配だけになる。線形のルータでは、その勾配は
ΔW_j ∝ Σ_n (t_nj − p_nj)·h_n（シナプス前の活動 h × シナプス後の出力 p と教える信号 t）で、選択肢 k ごとの適格性
h·(1[j=k] − p_j) に第三因子 t_k を掛けて足した三要素則の形である（Frémaux & Gerstner, 2016）。
`moe_consol.py` ほかの既存の模型の挙動は変えない。ここに置くもの:

- `ThreeFactorLM`: `ConsolLM` の継承。
  - `tf=True`: 学習の初めから、ゲートは確率の値を使うが、LM の損失の勾配をルータに送らない（ルータを通って GRU へ
    流れる勾配もなくなる）。12.6.177節の d_gate（閉じ始めから）を学習の初めから掛けるのと同じ順伝播。
  - `soft=True`: 部分評価の教師（top・rand）の目標を、評価した選択肢の中のソフトな分布 softmax(−CE/τ) にする
    （評価する専門家と乱数の消費はハードな教師と同じ。評価した選択肢ごとの相対的な結果を第三因子にする）。
  - `eq_gate=True`（修正1、12.6.181節）: 部分評価の教師で、評価したすべての専門家を同じゲート（選ばれた専門家の
    実際のゲート）で評価する。第三因子がルータ自身の確信度（評価に使うゲートの大きさ）に依らず、専門家の出力の
    比較だけで決まる。
  - `pg`（修正2、12.6.181節）: 実際に選ばれた専門家 c の適格性 h·(1[j=c] − p_j)（シナプス前 × 選んだ行動の
    シナプス後の活動）に、反実仮想の報酬予測誤差 δ = mean_alt CE − CE_c（評価した他の専門家の結果の平均との差。
    同じゲートで評価する）を掛ける古典的な三要素則（損失 −λ·mean δ·log p_c）。"only" はこれだけ、"add" は
    教師の補助損失に足す。
  既定（tf=False・soft=False・eq_gate=False）では `ConsolLM`＋`train_consol` と同じ学習経過（`tests/test_moe_tf.py`）。
  学習は `moe_consol.train_consol` を使う（閉じる時点の保存と `cross_assignment` の分解もそのまま）。
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .moe_cfe import _dup_mask, _probs, _random_others, subset_ce
from .moe_consol import ConsolLM


def subset_ce_eq(model, h: torch.Tensor, y: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """`moe_cfe.subset_ce` と同じ CE [N,m] だが、ゲートはどの専門家も idx[:,0]（選ばれた専門家）の確率にそろえる。"""
    hf = h.reshape(-1, h.shape[-1])
    yf = y.reshape(-1)
    probs = _probs(model, model.routers[0](hf))
    n, m = idx.shape
    tok = torch.arange(n).unsqueeze(-1).expand(n, m).reshape(-1)
    ex = idx.reshape(-1)
    ye = hf.new_zeros(n * m, hf.shape[-1])
    for e in ex.unique().tolist():
        sel = (ex == e).nonzero().squeeze(-1)
        ye[sel] = (hf[tok[sel]] @ model.w1[0][e]).relu() @ model.w2[0][e]
    g = probs.gather(-1, idx[:, :1]).squeeze(-1)[tok]
    ce = F.cross_entropy(model.head(hf[tok] + g.unsqueeze(-1) * ye), yf[tok], reduction="none")
    return ce.reshape(n, m)


class ThreeFactorLM(ConsolLM):
    """学習の初めからゲートの勾配を切り（tf）、ルータを反実仮想の教師の三要素則だけで学習する。soft はソフトな目標。"""

    def __init__(self, *args, tf: bool = False, soft: bool = False, eq_gate: bool = False,
                 pg: str | None = None, **kw) -> None:
        super().__init__(*args, **kw)
        if tf and (self.kind != "select" or self.n_layers != 1):
            raise ValueError("tf は1層の選択専用（select）の模型だけ")
        if pg not in (None, "only", "add"):
            raise ValueError(f"未知の pg: {pg}")
        if (soft or eq_gate or pg) and (self.kind_t not in ("top", "rand") or self.teacher_homeo or self.react_probe):
            raise ValueError("soft・eq_gate・pg は閾値・再学習の評価を足さない部分評価の教師（top・rand）だけ")
        self.tf, self.soft, self.eq_gate, self.pg = tf, soft, eq_gate, pg

    def detach_gate(self) -> bool:
        return getattr(self, "tf", False) or super().detach_gate()

    def teacher_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not (self.soft or self.eq_gate or self.pg):
            return super().teacher_loss(x, y)
        t0 = time.perf_counter()
        with torch.no_grad():  # `ReactCFLM.teacher_loss` の部分評価と同じ評価の組・同じ乱数の消費
            h, _ = self.rnn(self.embed(x))
            raw0 = self.routers[0](h).reshape(-1, self.n_experts)
            chosen = (raw0 - self.threshold[0]).argmax(-1) if self.kind == "select" else _probs(self, raw0).argmax(-1)
            extra = (raw0.topk(self.m, dim=-1).indices if self.kind_t == "top"
                     else _random_others(chosen, self.n_experts, self.m, self.sample_gen))
            idx = torch.cat([chosen.unsqueeze(-1), extra], dim=-1)
            ce_eq = subset_ce_eq(self, h, y, idx) if (self.eq_gate or self.pg) else None
            ce = ce_eq if self.eq_gate else subset_ce(self, h, y, idx)
            dup = _dup_mask(idx)
            if self.pg:  # 反実仮想の報酬予測誤差（代わりの専門家の結果の平均 − 選んだ専門家の結果）
                alt = (~dup[:, 1:]).float()
                delta = (ce_eq[:, 1:] * alt).sum(-1) / alt.sum(-1).clamp(min=1.0) - ce_eq[:, 0]
            if self.soft:
                target = F.softmax((-ce / self.tau).masked_fill(dup, float("-inf")), dim=-1)
            else:
                target = F.one_hot(ce.masked_fill(dup, float("inf")).argmin(-1), idx.shape[1]).float()
            hard = idx.gather(-1, target.argmax(-1, keepdim=True)).squeeze(-1)
        n = y.numel()
        evals = sum(len(set(r)) for r in idx.tolist())
        raw = self.routers[0](h).reshape(-1, self.n_experts)
        loss = raw.new_zeros(())
        if self.pg != "only":
            logits = raw.gather(-1, idx).masked_fill(dup, float("-inf"))
            logp = F.log_softmax(logits, dim=-1).masked_fill(dup, 0.0)
            loss = loss - self.lam * (target * logp).sum(-1).mean()
        if self.pg:  # 選んだ行動の適格性 × 第三因子 δ
            logp_c = F.log_softmax(raw, dim=-1).gather(-1, idx[:, :1]).squeeze(-1)
            loss = loss - self.lam * (delta * logp_c).mean()
        with torch.no_grad():
            self.teacher_stats.append({"agree": float((raw.argmax(-1) == hard).float().mean())})
        self._count(evals, n, t0)
        return loss
