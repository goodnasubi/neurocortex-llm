"""文脈推定機構（12.6.105節）。タスク ID を与えずに、予測誤差の驚きから文脈（潜在原因）を推定し、
推定した文脈ごとにリプレイする。

  oracle   … 真のタスク ID を文脈として入力する（上限）
  none     … 文脈を入力しない（下限）
  inferred … 驚き（確立済みの文脈での損失が閾値 τ を超えたこと）を合図に、既存の文脈から最も合うものに
             切り替えるか、合うものがなければ新しい文脈を割り当てる（Gershman, Blei & Niv 2010 の潜在原因推論）
  random   … 文脈スロットを毎回ランダムに選ぶ（統制。文脈が情報を持たない場合）

テスト時、inferred は各タスクの少数のラベル付き手がかりを最もよく説明する文脈を選ぶ。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from dataclasses import dataclass

from .basal_ganglia_core import ParitySpec
from .basal_ganglia_core import make_batch as _parity_batch
from .hippocampus_replay import make_model

CONDITIONS = ("oracle", "none", "inferred", "random")


@dataclass(frozen=True)
class FlipSpec:
    """似たタスク（12.6.108節）。正解は b[a] XOR b[b] を基本とし、flip の全ビットが +1 の領域でだけ反転する。
    反転領域の割合は 2^-len(flip)。flip が空なら基本タスクそのもの。"""

    n_inputs: int = 8
    relevant: tuple[int, int] = (0, 1)
    flip: tuple[int, ...] = ()


def make_batch(spec, batch_size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """ParitySpec はステップ10と同じ乱数消費で、FlipSpec は同じ入力に反転領域を適用して生成する。"""
    if isinstance(spec, ParitySpec):
        return _parity_batch(spec, batch_size, generator)
    bits = torch.randint(0, 2, (batch_size, spec.n_inputs), generator=generator).float() * 2 - 1
    a, b = spec.relevant
    labels = (bits[:, a] > 0) ^ (bits[:, b] > 0)
    if spec.flip:
        labels = labels ^ (bits[:, list(spec.flip)] > 0).all(dim=1)
    return bits, labels.long()


def run_context_sequence(condition: str, specs: list, hidden_dim: int, generator: torch.Generator,
                         eval_generator: torch.Generator, n_slots: int | None = None, tau: float = 0.4,
                         ema: float = 0.9, pretrain_steps: int = 500, pretrain_weight_decay: float = 0.01,
                         steps: int = 500, batch: int = 64, lr: float = 0.05, buffer_size: int = 50,
                         cue_n: int = 16, eval_n: int = 2000,
                         selection_trials: int = 0, transition_steps: int = 0) -> dict:
    """戻り値: {"R": 正確度行列, "assign": 学習中の (タスク, 文脈) の列, "n_contexts": 使った文脈数,
    "selection_acc": 文脈の選択精度（inferred かつ selection_trials > 0 のとき）}

    selection_acc は最終モデルで、cue_n 個の手がかりから選んだ文脈が、そのタスクに学習中最も多く割り当てられた
    文脈と一致する割合（タスクごとに selection_trials 回、評価用の乱数で試行）。12.6.107節。

    transition_steps = L > 0 では、タスク k（k ≥ 1）の最初の L ステップを移行期間とし、各サンプルを確率
    p = (s+1)/(L+1) でタスク k から、残りをタスク k−1 から取る（少しずつ変わる文脈, 12.6.109節）。oracle は
    サンプルごとに本当の出どころのタスクを文脈として受け取る。L = 0 では以前と同じ乱数消費になる。
    """
    if condition not in CONDITIONS:
        raise ValueError(condition)
    T = len(specs)
    C = 0 if condition == "none" else (T if condition == "oracle" else (n_slots or T + 2))
    model = make_model(specs[0].n_inputs + C, hidden_dim)

    def with_ctx(bits: torch.Tensor, ctx: torch.Tensor | int) -> torch.Tensor:
        if C == 0:
            return bits
        oh = torch.zeros(bits.shape[0], C)
        if isinstance(ctx, int):
            oh[:, ctx] = 1.0
        else:
            oh[torch.arange(bits.shape[0]), ctx] = 1.0
        return torch.cat([bits, oh], dim=1)

    @torch.no_grad()
    def loss_under(bits: torch.Tensor, labels: torch.Tensor, ctx: torch.Tensor | int) -> float:
        return float(F.cross_entropy(model(with_ctx(bits, ctx)), labels))

    buf_bits: list[torch.Tensor] = []
    buf_labels: list[torch.Tensor] = []
    buf_ctx: list[torch.Tensor] = []
    buf_count: dict[int, int] = {}
    ema_loss: dict[int, float] = {}
    cur = 0
    used = [0]
    assign: list[tuple[int, int]] = []

    def pick_ctx(k: int, bits: torch.Tensor, labels: torch.Tensor) -> int:
        nonlocal cur
        if condition == "oracle":
            return k
        if condition == "random":
            return int(torch.randint(C, (1,), generator=generator))
        if condition == "none":
            return 0
        established = ema_loss.get(cur, math.inf) < tau
        if established and loss_under(bits, labels, cur) > tau:
            losses = {c: loss_under(bits, labels, c) for c in used}
            best = min(losses, key=losses.get)
            if losses[best] < tau:
                cur = best
            elif len(used) < C:
                cur = len(used)
                used.append(cur)
            else:
                cur = best
        return cur

    @torch.no_grad()
    def accuracy(j: int) -> float:
        if condition == "inferred":
            cb, cl = make_batch(specs[j], cue_n, eval_generator)
            ctx = min(used, key=lambda c: loss_under(cb, cl, c))
        elif condition == "random":
            ctx = int(torch.randint(C, (1,), generator=eval_generator))
        else:
            ctx = j if condition == "oracle" else 0
        bits, labels = make_batch(specs[j], eval_n, eval_generator)
        return float((model(with_ctx(bits, ctx)).argmax(-1) == labels).float().mean())

    R: list[list[float]] = []
    for k, spec in enumerate(specs):
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=pretrain_weight_decay if k == 0 else 0.0)
        for step in range(pretrain_steps if k == 0 else steps):
            bits, labels = make_batch(spec, batch, generator)
            src = None
            if k > 0 and step < transition_steps:
                old_bits, old_labels = make_batch(specs[k - 1], batch, generator)
                take_new = torch.rand(batch, generator=generator) < (step + 1) / (transition_steps + 1)
                bits = torch.where(take_new[:, None], bits, old_bits)
                labels = torch.where(take_new, labels, old_labels)
                src = torch.where(take_new, k, k - 1)
            ctx = pick_ctx(k, bits, labels)
            assign.append((k, ctx))
            ctx_in = src if (condition == "oracle" and src is not None) else ctx
            x = with_ctx(bits, ctx_in)
            y = labels
            if buf_bits and k > 0:
                bb, bl, bc = torch.cat(buf_bits), torch.cat(buf_labels), torch.cat(buf_ctx)
                idx = torch.randint(bb.shape[0], (batch,), generator=generator)
                x = torch.cat([x, with_ctx(bb[idx], bc[idx])])
                y = torch.cat([y, bl[idx]])
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                cur_loss = loss_under(bits, labels, ctx_in)
            ema_loss[ctx] = cur_loss if ctx not in ema_loss else ema * ema_loss[ctx] + (1 - ema) * cur_loss
            # 観測したサンプルを文脈ラベル付きで保存する。保存枠は inferred では推定した文脈ごと
            # （タスクの区切りを知らない）、他の条件では真のタスクごとに buffer_size 個まで
            key = ctx if condition == "inferred" else k
            n = min(buffer_size - buf_count.get(key, 0), batch)
            if n > 0:
                buf_count[key] = buf_count.get(key, 0) + n
                buf_bits.append(bits[:n])
                buf_labels.append(labels[:n])
                buf_ctx.append(ctx_in[:n].long() if isinstance(ctx_in, torch.Tensor)
                               else torch.full((n,), ctx, dtype=torch.long))
        R.append([accuracy(j) for j in range(T)])
    selection_acc = None
    if condition == "inferred" and selection_trials > 0:
        majority = assignment_majority(assign, T)
        hits = 0
        for j in range(T):
            for _ in range(selection_trials):
                cb, cl = make_batch(specs[j], cue_n, eval_generator)
                hits += int(min(used, key=lambda c: loss_under(cb, cl, c)) == majority[j])
        selection_acc = hits / (T * selection_trials)
    return {"R": R, "assign": assign, "n_contexts": len(used) if condition == "inferred" else C,
            "selection_acc": selection_acc}


def assignment_majority(assign: list[tuple[int, int]], n_tasks: int) -> list[int]:
    """各タスクで学習中に最も多く割り当てられた文脈。"""
    out = []
    for k in range(n_tasks):
        counts: dict[int, int] = {}
        for t, c in assign:
            if t == k:
                counts[c] = counts.get(c, 0) + 1
        out.append(max(counts, key=counts.get))
    return out


def assignment_purity(assign: list[tuple[int, int]], n_tasks: int) -> dict[str, float]:
    """各タスクのバッチのうち、そのタスクで最も多く使われた文脈に割り当てられた割合（純度）と、
    タスクごとの最頻文脈が互いに異なるかどうか。"""
    majority: list[int] = []
    pure = []
    for k in range(n_tasks):
        ctxs = [c for t, c in assign if t == k]
        counts: dict[int, int] = {}
        for c in ctxs:
            counts[c] = counts.get(c, 0) + 1
        m = max(counts, key=counts.get)
        majority.append(m)
        pure.append(counts[m] / len(ctxs))
    return {"purity": sum(pure) / n_tasks, "distinct": float(len(set(majority)) == n_tasks)}
