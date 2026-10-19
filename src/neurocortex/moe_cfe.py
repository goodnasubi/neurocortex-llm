"""反実仮想の結果を一部の専門家だけで推定する教師と、価値を予測するルータ（12.6.165節）。

12.6.157節の `moe_cf.CounterfactualLM` の教師は全専門家の順伝播（E 回の評価）を要した。ここでは評価する専門家を減らす。
`moe_cf.py` ほかの既存の模型の挙動は変えない。ここに置くもの:

- `subset_ce`: 1層の模型で、トークンごとに指定した専門家 idx [N,m] に送ったときの CE [N,m]（ゲートは e の確率、
  gate_one なら 1）。m 回の評価だけを行う（`moe_cf.forced_ce` の該当列と一致する。`tests/test_moe_cfe.py`）。
- `forced_ce_gate1`: 全専門家をゲート 1 で評価した CE [N,E]（教師の CE をゲートに依らず測る）。
- `EstimatedCFLM`: `CounterfactualLM` の継承。teacher（`moe_cf.TEACHERS` に加えて）:
  - "top{k}": 実際に選ばれた専門家 c と、閾値なしのルータの上位 k 個だけを評価し、その中の CE 最小を教師にする
  - "rand{m}": c と、c 以外から無作為に選んだ m 個だけを評価し、その中の CE 最小を教師にする
    （top・rand の補助損失は評価した専門家の対数だけの softmax の CE。評価しなかった専門家には勾配を送らない）
  - "val{m}" / "valall": 価値の予測。ルータの対数を各専門家の CE の予測として学ぶ: 予測 CE_e = τ·(b(h) − raw_e)。
    b(h) はトークンごとの基準の価値（状態の価値、線形の読み出し、ゼロで初期化）。評価した専門家（c と無作為な m 個、
    "valall" は全専門家）だけで、価値の予測誤差 raw_e − (b(h) − CE_e/τ) に smooth L1 を掛ける。m=0 は実際の結果だけ
  - "dval{m}" / "dvalall"（12.6.165節の修正1）: 実際の結果との差の予測。評価した専門家 e について
    smooth L1((raw_e − raw_c) − (−(CE_e − CE_c)/τ))（c は実際に選ばれた専門家。トークンの難しさが打ち消し合う）
  - "cfg": 全専門家をゲート 1 で評価した CE の argmin（ゲートに依らない反実仮想の教師）
  費用として、教師が評価した専門家の数（トークンあたり、重複を除く）と教師の計算時間を `cost` に積む。
  teacher が既存の教師なら `CounterfactualLM` と同じ（同テスト）。
"""

from __future__ import annotations

import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe_cf import TEACHERS, CounterfactualLM, forced_ce

_PAT = re.compile(r"^(top|rand|val|dval)(\d+)$|^(valall|dvalall|cfg)$")


def parse_teacher(t: str) -> tuple[str, int]:
    """"top3" → ("top", 3)、"valall" → ("valall", 0)。既存の教師は (t, 0)。不明なら ValueError。"""
    if t in TEACHERS:
        return t, 0
    m = _PAT.match(t)
    if m is None or (m.group(1) in ("top", "dval") and int(m.group(2)) < 1):
        raise ValueError(t)
    return (m.group(1), int(m.group(2))) if m.group(1) else (m.group(3), 0)


def _probs(model, raw):
    return F.softmax(raw - model.threshold[0] if model.kind == "gate" else raw, dim=-1)


def subset_ce(model, h: torch.Tensor, y: torch.Tensor, idx: torch.Tensor, gate_one: bool = False) -> torch.Tensor:
    """h [B,T,d]、y [B,T]、idx [N,m]（N=B·T）。トークン n を専門家 idx[n,j] に送ったときの CE [N,m]。"""
    hf = h.reshape(-1, h.shape[-1])
    yf = y.reshape(-1)
    probs = _probs(model, model.routers[0](hf))
    n, m = idx.shape
    tok = torch.arange(n).unsqueeze(-1).expand(n, m).reshape(-1)
    ex = idx.reshape(-1)
    ye = hf.new_zeros(n * m, hf.shape[-1])
    for e in ex.unique().tolist():  # 専門家ごとにまとめて計算する（評価する組の数 n·m に比例する計算）
        sel = (ex == e).nonzero().squeeze(-1)
        ye[sel] = (hf[tok[sel]] @ model.w1[0][e]).relu() @ model.w2[0][e]
    g = torch.ones(n * m, dtype=hf.dtype) if gate_one else probs[tok, ex]
    ce = F.cross_entropy(model.head(hf[tok] + g.unsqueeze(-1) * ye), yf[tok], reduction="none")
    return ce.reshape(n, m)


def forced_ce_gate1(model, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """全専門家をゲート 1 で評価した CE [N,E]。"""
    ye = torch.einsum("btd,edf->btef", h, model.w1[0]).relu()
    ye = torch.einsum("btef,efd->bted", ye, model.w2[0])
    logits = model.head(h.unsqueeze(2) + ye)
    v = logits.shape[-1]
    ce = F.cross_entropy(logits.reshape(-1, v), y.unsqueeze(-1).expand(*y.shape, model.n_experts).reshape(-1),
                         reduction="none")
    return ce.reshape(-1, model.n_experts)


def _random_others(chosen: torch.Tensor, n_experts: int, m: int, gen: torch.Generator) -> torch.Tensor:
    """各トークンで chosen 以外から重複なしに m 個 [N,m]。"""
    r = torch.rand(chosen.numel(), n_experts, generator=gen)
    r.scatter_(1, chosen.unsqueeze(-1), -1.0)
    return r.topk(m, dim=-1).indices


def _dup_mask(idx: torch.Tensor) -> torch.Tensor:
    """idx [N,m] で、同じ行の前の列と重なる位置を True（重複を1つに数える）。"""
    eq = idx.unsqueeze(-1) == idx.unsqueeze(-2)  # [N,m,m]
    return torch.tril(eq, diagonal=-1).any(-1)


class EstimatedCFLM(CounterfactualLM):
    """一部の専門家だけを評価する反実仮想の教師と、価値を予測するルータ（teacher は `parse_teacher` の形）。"""

    def __init__(self, *args, teacher: str | None = None, sample_seed: int = 0, **kw) -> None:
        base = None if teacher is None else parse_teacher(teacher)[0]
        super().__init__(*args, teacher=teacher if base in TEACHERS else None, **kw)
        if teacher is not None and self.n_layers != 1:
            raise ValueError("教師つきのルータは1層の模型だけ")
        self.teacher = teacher
        self.kind_t, self.m = (None, 0) if teacher is None else parse_teacher(teacher)
        self.sample_gen = torch.Generator().manual_seed(sample_seed)
        if self.kind_t in ("val", "valall"):  # 基準の価値 b(h)（ゼロで初期化、乱数を消費しない）
            d = self.head.in_features
            self.value_w = nn.Parameter(torch.zeros(d))
            self.value_b = nn.Parameter(torch.zeros(()))
        self.cost = {"evals": 0.0, "tokens": 0, "sec": 0.0}

    def teacher_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.kind_t in TEACHERS:
            t0 = time.perf_counter()
            loss = super().teacher_loss(x, y)
            ev = {"cf": self.n_experts, "cfs": self.n_experts, "shuf": self.n_experts}.get(self.kind_t, 0)
            self._count(ev * y.numel(), y.numel(), t0)
            return loss
        t0 = time.perf_counter()
        with torch.no_grad():
            h, _ = self.rnn(self.embed(x))
            raw0 = self.routers[0](h).reshape(-1, self.n_experts)
            chosen = (raw0 - self.threshold[0]).argmax(-1) if self.kind == "select" else _probs(self, raw0).argmax(-1)
            if self.kind_t == "cfg":
                idx, ce = None, forced_ce_gate1(self, h, y)
            elif self.kind_t in ("valall", "dvalall"):
                idx, ce = None, forced_ce(self, h, y)
            else:
                extra = (raw0.topk(self.m, dim=-1).indices if self.kind_t == "top"
                         else _random_others(chosen, self.n_experts, self.m, self.sample_gen))
                idx = torch.cat([chosen.unsqueeze(-1), extra], dim=-1)
                ce = subset_ce(self, h, y, idx)
        n = y.numel()
        evals = n * self.n_experts if idx is None else sum(len(set(r)) for r in idx.tolist())
        raw = self.routers[0](h).reshape(-1, self.n_experts)
        if self.kind_t in ("val", "valall"):
            base = h.reshape(-1, h.shape[-1]) @ self.value_w + self.value_b  # [N]
            pred = raw if idx is None else raw.gather(-1, idx)
            target = base.unsqueeze(-1) - ce / self.tau
            loss = self.lam * F.smooth_l1_loss(pred, target)
            hard = ce.argmin(-1) if idx is None else idx.gather(-1, ce.argmin(-1, keepdim=True)).squeeze(-1)
        elif self.kind_t in ("dval", "dvalall"):  # 修正1: 実際の結果との差（fictive な予測誤差）
            if idx is None:  # 全専門家（c の項は 0）
                idx = torch.arange(self.n_experts).expand(n, -1)
                c_ce, oi, oce = ce.gather(-1, chosen.unsqueeze(-1)), idx, ce
            else:  # 評価した c 以外の専門家
                c_ce, oi, oce = ce[:, :1], idx[:, 1:], ce[:, 1:]
            c_raw = raw.gather(-1, chosen.unsqueeze(-1))
            loss = self.lam * F.smooth_l1_loss(raw.gather(-1, oi) - c_raw, -(oce - c_ce) / self.tau)
            hard = idx.gather(-1, ce.argmin(-1, keepdim=True)).squeeze(-1)
        elif idx is None:
            hard = ce.argmin(-1)
            loss = self.lam * F.cross_entropy(raw, hard)
        else:  # 評価した専門家の中だけの比較（評価しなかった専門家の対数には勾配を送らない）
            pos = ce.masked_fill(_dup_mask(idx), float("inf")).argmin(-1)
            hard = idx.gather(-1, pos.unsqueeze(-1)).squeeze(-1)
            logits = raw.gather(-1, idx).masked_fill(_dup_mask(idx), float("-inf"))
            loss = self.lam * F.cross_entropy(logits, pos)
        with torch.no_grad():
            self.teacher_stats.append({"agree": float((raw.argmax(-1) == hard).float().mean())})
        self._count(evals, n, t0)
        return loss

    def _count(self, evals: float, n: int, t0: float) -> None:
        self.cost["evals"] += evals
        self.cost["tokens"] += n
        self.cost["sec"] += time.perf_counter() - t0
