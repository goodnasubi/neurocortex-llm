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

CONDITIONS = ("oracle", "none", "inferred", "random", "inferred_gated")


@dataclass(frozen=True)
class FlipSpec:
    """似たタスク（12.6.108節）。正解は b[a] XOR b[b] を基本とし、flip の全ビットが +1 の領域でだけ反転する。
    反転領域の割合は 2^-len(flip)。flip が空なら基本タスクそのもの。"""

    n_inputs: int = 8
    relevant: tuple[int, int] = (0, 1)
    flip: tuple[int, ...] = ()


@dataclass(frozen=True)
class CueSpec:
    """入力分布がタスクで異なる課題（12.6.111節）。正解は b[a] XOR b[b]（最初の n_parity ビット）。
    残りの手がかりビットのうち cue に含まれるビットは確率 q で +1、他は一様。"""

    n_inputs: int = 16
    relevant: tuple[int, int] = (0, 1)
    cue: tuple[int, ...] = ()
    q: float = 0.9


def make_batch(spec, batch_size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """ParitySpec はステップ10と同じ乱数消費で、FlipSpec は同じ入力に反転領域を適用して生成する。"""
    if isinstance(spec, ParitySpec):
        return _parity_batch(spec, batch_size, generator)
    bits = torch.randint(0, 2, (batch_size, spec.n_inputs), generator=generator).float() * 2 - 1
    if isinstance(spec, CueSpec) and spec.cue:
        u = torch.rand(batch_size, len(spec.cue), generator=generator)
        bits[:, list(spec.cue)] = torch.where(u < spec.q, 1.0, -1.0)
    a, b = spec.relevant
    labels = (bits[:, a] > 0) ^ (bits[:, b] > 0)
    if isinstance(spec, FlipSpec) and spec.flip:
        labels = labels ^ (bits[:, list(spec.flip)] > 0).all(dim=1)
    return bits, labels.long()


def run_context_sequence(condition: str, specs: list, hidden_dim: int, generator: torch.Generator,
                         eval_generator: torch.Generator, n_slots: int | None = None, tau: float = 0.4,
                         ema: float = 0.9, pretrain_steps: int = 500, pretrain_weight_decay: float = 0.01,
                         steps: int = 500, batch: int = 64, lr: float = 0.05, buffer_size: int = 50,
                         cue_n: int = 16, eval_n: int = 2000,
                         selection_trials: int = 0, transition_steps: int = 0,
                         novel_patience: int = 20, store_consistent: bool = False, train_consistent: bool = False,
                         diagnostics: bool = False, trace_every: int = 0, diag_oracle_filter: str = "",
                         reset_opt_on_new: bool = False, plasticity_cap: float = 0.0, plasticity_ema: float = 0.99,
                         plasticity_ref: str = "capped", optimizer: str = "adam", split_window: int = 0,
                         split_ctx_gain: float = 1.0, split_shared_gain: float = 1.0) -> dict:
    """戻り値: {"R": 正確度行列, "assign": 学習中の (タスク, 文脈) の列, "n_contexts": 使った文脈数,
    "selection_acc": 文脈の選択精度（inferred かつ selection_trials > 0 のとき）}

    selection_acc は最終モデルで、cue_n 個の手がかりから選んだ文脈が、そのタスクに学習中最も多く割り当てられた
    文脈と一致する割合（タスクごとに selection_trials 回、評価用の乱数で試行）。12.6.107節。

    transition_steps = L > 0 では、タスク k（k ≥ 1）の最初の L ステップを移行期間とし、各サンプルを確率
    p = (s+1)/(L+1) でタスク k から、残りをタスク k−1 から取る（少しずつ変わる文脈, 12.6.109節）。oracle は
    サンプルごとに本当の出どころのタスクを文脈として受け取る。L = 0 では以前と同じ乱数消費になる。

    store_consistent = True（inferred_gated のみ）では、保存する文脈で正しく予測できるサンプルだけをバッファに
    入れる（12.6.116節）。train_consistent = True（inferred_gated のみ）では、確立した文脈で驚きのなかったバッチ
    のうち、その文脈で正しく予測できるサンプルだけで学習する。どちらも既定の False では 12.6.115節と同じ挙動。

    diagnostics = True では原因の切り分け用（12.6.116節）に "diag" を加える。学習・評価の乱数は消費しない
    （診断専用の乱数を使う）ので、R と assign は diagnostics = False と同じになる。
      ctx_acc[i][j][c] … タスク i の学習後、タスク j を文脈 c で評価した正確度（used の順）
      chosen[i][j]     … 同じ時点で R の計算が手がかりから選んだ文脈
      buffer_src[c]    … 文脈 c のバッファのサンプルの本当の出どころのタスクごとの数
      buffer           … バッファの中身（bits, labels, ctx, src のテンソル）
      trained_src[i]   … タスク i の期間に、新しい入力として学習に使ったサンプルの (文脈, 出どころ) ごとの数
      trace            … trace_every > 0 のとき、trace_every ステップごとの (タスク, ステップ, 現在の文脈, 学習したか,
                         保存したか, 各タスク j を各文脈で評価した正確度) の列（12.6.121節）

    reset_opt_on_new = True では、新しい文脈を作った時点で最適化器（Adam）の状態を作り直す（12.6.121節）。
    作成の直前はリプレイだけの学習で勾配が極小になり、2次モーメントの推定が縮んでいるため、作成直後の大きな
    勾配で更新が過大になり、前の文脈を壊す。既定の False では 12.6.116節と同じ挙動。

    diag_oracle_filter は原因の切り分け専用の反実仮想（12.6.121節。機構ではない）。本当の出どころを使い、
    "new" ではタスク k の期間に作った文脈の学習から前のタスクのサンプルを、"old" ではそれ以外の文脈の学習から
    タスク k のサンプルを除く（"both" は両方）。
    diagnostics = True では "upd"（各最適化ステップの (タスク, ステップ, 文脈, 勾配ノルム, 更新ノルム)）も加える。
    更新ノルムは plasticity_cap による縮小の後の値。

    plasticity_cap = ρ > 0 では、驚きに応じた可塑性の調節（12.6.125節。reset_opt_on_new の代わり）を行う。
    新しい入力を学習したステップの更新ノルムの指数移動平均 ū（減衰 plasticity_ema、平常時の可塑性の大きさ）を
    保持し、更新ノルムが ρ·ū を超える（平常時から予期せぬほど大きい）ときは、更新を ρ·ū に縮める。ū の更新には
    縮めた後の値を使うので、持続的な変化には1ステップあたり最大 (ρ − 1)(1 − plasticity_ema) の割合で追従する。
    最適化器の内部状態は変えない。既定の 0 では従来と同じ挙動。plasticity_ref = "raw" では、ū の更新に縮める前の
    値を使う（修正2。縮めた値を使うと ū が自分の縮小で下がり続け、平常時の可塑性まで抑えるため）。

    optimizer は "adam"（既定）か "sgd"（モーメントなし。Adam 以外でも同じ問題が起きるかの確認用, 12.6.125節）。

    split_window = W > 0（inferred / inferred_gated のみ）では、文脈固有と共有で可塑性を分ける（12.6.129節）。
    新しい文脈を作った時点から W ステップの間、各最適化ステップの更新（パラメータの変化）に利得をかける。
    新しい文脈の one-hot 入力からの重み（fc1.weight の該当列）には split_ctx_gain、それ以外（共有の重みと
    既存の文脈の列）には split_shared_gain。最適化器の内部状態は変えない。既定の 0 では従来と同じ挙動。
    """
    if condition not in CONDITIONS:
        raise ValueError(condition)
    if optimizer not in ("adam", "sgd"):
        raise ValueError(optimizer)
    if plasticity_ref not in ("capped", "raw"):
        raise ValueError(plasticity_ref)
    if split_window < 0 or split_ctx_gain < 0 or split_shared_gain < 0:
        raise ValueError("split_window / split_ctx_gain / split_shared_gain は 0 以上")
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
    buf_src: list[torch.Tensor] = []
    diag_gen = torch.Generator().manual_seed(1234)
    trace_gen = torch.Generator().manual_seed(4321)  # trace 専用（diag_gen の消費を変えない）
    diag: dict = {"ctx_acc": [], "chosen": [], "trained_src": [], "trace": [], "upd": []}
    buf_count: dict[int, int] = {}
    ema_loss: dict[int, float] = {}
    cur = 0
    used = [0]
    assign: list[tuple[int, int]] = []

    gate = {"train": True, "store": True}
    novel_streak = [0]

    def pick_ctx(k: int, bits: torch.Tensor, labels: torch.Tensor) -> int:
        nonlocal cur
        gate["train"], gate["store"] = True, True
        if condition == "oracle":
            return k
        if condition == "random":
            return int(torch.randint(C, (1,), generator=generator))
        if condition == "none":
            return 0
        established = ema_loss.get(cur, math.inf) < tau
        if condition == "inferred_gated":
            # 12.6.115節: 驚きのバッチ（混合の疑い）は学習・保存に使わない。新しい文脈は、現在の文脈での正答率が
            # 完全とチャンスの中間（0.75）を下回ったとき、つまり新しい原因が多数派になってから作る。たまたまの1バッチで作らないよう、それが novel_patience 回続くことを条件にする
            gate["train"], gate["store"] = True, False
            if not established:
                return cur
            if loss_under(bits, labels, cur) <= tau:
                gate["store"] = True
                return cur
            losses = {c: loss_under(bits, labels, c) for c in used}
            best = min(losses, key=losses.get)
            if best != cur and losses[best] < tau:
                cur = best
                return cur
            with torch.no_grad():
                accs = {c: float((model(with_ctx(bits, c)).argmax(-1) == labels).float().mean()) for c in used}
            novel_streak[0] = novel_streak[0] + 1 if max(accs.values()) < 0.75 else 0
            if novel_streak[0] >= novel_patience and len(used) < C:
                novel_streak[0] = 0
                cur = len(used)
                used.append(cur)
                return cur
            gate["train"] = False
            return cur
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
    def accuracy(j: int, chosen: list[int] | None = None) -> float:
        if condition in ("inferred", "inferred_gated"):
            cb, cl = make_batch(specs[j], cue_n, eval_generator)
            ctx = min(used, key=lambda c: loss_under(cb, cl, c))
            if chosen is not None:
                chosen.append(ctx)
        elif condition == "random":
            ctx = int(torch.randint(C, (1,), generator=eval_generator))
        else:
            ctx = j if condition == "oracle" else 0
        bits, labels = make_batch(specs[j], eval_n, eval_generator)
        return float((model(with_ctx(bits, ctx)).argmax(-1) == labels).float().mean())

    def make_opt(k: int) -> torch.optim.Optimizer:
        wd = pretrain_weight_decay if k == 0 else 0.0
        if optimizer == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    upd_ref: list[float] = []  # plasticity_cap の ū（平常時の更新ノルムの移動平均）
    split = {"left": 0, "ctx": -1}  # split_window の残りステップと、対象の新しい文脈
    split_on = split_window > 0 and condition in ("inferred", "inferred_gated")
    fc1_w = model.backbone.fc1.weight
    n_bits = specs[0].n_inputs
    R: list[list[float]] = []
    for k, spec in enumerate(specs):
        trained: dict[tuple[int, int], int] = {}
        n_used_at_start = len(used)
        opt = make_opt(k)
        for step in range(pretrain_steps if k == 0 else steps):
            bits, labels = make_batch(spec, batch, generator)
            src = None
            if k > 0 and step < transition_steps:
                old_bits, old_labels = make_batch(specs[k - 1], batch, generator)
                take_new = torch.rand(batch, generator=generator) < (step + 1) / (transition_steps + 1)
                bits = torch.where(take_new[:, None], bits, old_bits)
                labels = torch.where(take_new, labels, old_labels)
                src = torch.where(take_new, k, k - 1)
            n_used_before = len(used)
            ctx = pick_ctx(k, bits, labels)
            assign.append((k, ctx))
            if reset_opt_on_new and len(used) > n_used_before:
                opt = make_opt(k)
            if split_on and len(used) > n_used_before:
                split["left"], split["ctx"] = split_window, ctx
            split_now = split["left"] > 0
            split["left"] = max(split["left"] - 1, 0)
            ctx_in = src if (condition == "oracle" and src is not None) else ctx
            x = with_ctx(bits, ctx_in)
            y = labels
            s_ = src if src is not None else torch.full((batch,), k)
            if not gate["train"]:
                x, y = x[:0], y[:0]
            elif train_consistent and condition == "inferred_gated" and gate["store"]:
                # 12.6.116節: 確立した文脈で驚きのなかったバッチでも、文脈の規則と食い違うサンプルでは学習しない
                # （移行の初期に混ざる次のタスクのサンプルで、確立した文脈が上書きされるのを防ぐ）
                with torch.no_grad():
                    keep = model(x).argmax(-1) == y
                x, y, s_ = x[keep], y[keep], s_[keep]
            if diag_oracle_filter and gate["train"] and src is not None and condition == "inferred_gated":
                new_ctx = ctx >= n_used_at_start
                if (new_ctx and diag_oracle_filter in ("new", "both")) or (not new_ctx and diag_oracle_filter in ("old", "both")):
                    keep = s_ == (k if new_ctx else k - 1)
                    x, y, s_ = x[keep], y[keep], s_[keep]
            if gate["train"] and diagnostics:
                for t in s_.unique().tolist():
                    trained[(ctx, t)] = trained.get((ctx, t), 0) + int((s_ == t).sum())
            n_new = y.shape[0]
            if buf_bits and k > 0:
                bb, bl, bc = torch.cat(buf_bits), torch.cat(buf_labels), torch.cat(buf_ctx)
                idx = torch.randint(bb.shape[0], (batch,), generator=generator)
                x = torch.cat([x, with_ctx(bb[idx], bc[idx])])
                y = torch.cat([y, bl[idx]])
            if y.numel() > 0:
                loss = F.cross_entropy(model(x), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if diagnostics or plasticity_cap > 0 or split_now:
                    before = [q.detach().clone() for q in model.parameters()]
                if diagnostics:
                    gnorm = float(torch.sqrt(sum((q.grad ** 2).sum() for q in model.parameters() if q.grad is not None)))
                opt.step()
                if split_now:
                    # 12.6.129節: 新しい原因を立てた直後は、その文脈の入力重みの可塑性を上げ（新しい情報からの学習を
                    # 速める）、共有の重みの可塑性を下げる（前の文脈への干渉を抑える）
                    with torch.no_grad():
                        for q, b in zip(model.parameters(), before):
                            d = (q - b) * split_shared_gain
                            if q is fc1_w:
                                col = n_bits + split["ctx"]
                                d[:, col] = (q[:, col] - b[:, col]) * split_ctx_gain
                            q.copy_(b + d)
                if diagnostics or plasticity_cap > 0:
                    unorm = float(torch.sqrt(sum(((q.detach() - b) ** 2).sum() for q, b in zip(model.parameters(), before))))
                if plasticity_cap > 0:
                    # 12.6.125節: 平常時の可塑性 ū から予期せぬほど大きい更新を縮める（ノルアドレナリンによる利得調節の
                    # 単純化）。リプレイだけの静かな期間の後、最適化器の古い推定のまま大きな誤差に反応するのを抑える
                    raw_unorm = unorm
                    if upd_ref and unorm > plasticity_cap * upd_ref[0]:
                        scale = plasticity_cap * upd_ref[0] / unorm
                        with torch.no_grad():
                            for q, b in zip(model.parameters(), before):
                                q.copy_(b + scale * (q - b))
                        unorm *= scale
                    if n_new > 0:
                        u_ = raw_unorm if plasticity_ref == "raw" else unorm
                        upd_ref[:] = [u_ if not upd_ref else plasticity_ema * upd_ref[0] + (1 - plasticity_ema) * u_]
                if diagnostics:
                    diag["upd"].append((k, step, ctx, gnorm, unorm))
            if gate["train"]:
                with torch.no_grad():
                    cur_loss = loss_under(bits, labels, ctx_in)
                ema_loss[ctx] = cur_loss if ctx not in ema_loss else ema * ema_loss[ctx] + (1 - ema) * cur_loss
            if diagnostics and trace_every > 0 and step % trace_every == 0 and condition in ("inferred", "inferred_gated"):
                with torch.no_grad():
                    accs_t = []
                    for j in range(T):
                        tb, tl = make_batch(specs[j], 500, trace_gen)
                        accs_t.append([float((model(with_ctx(tb, c)).argmax(-1) == tl).float().mean()) for c in used])
                diag["trace"].append({"task": k, "step": step, "ctx": ctx, "train": gate["train"],
                                      "store": gate["store"], "acc": accs_t})
            if not gate["store"]:
                continue
            # 観測したサンプルを文脈ラベル付きで保存する。保存枠は inferred では推定した文脈ごと
            # （タスクの区切りを知らない）、他の条件では真のタスクごとに buffer_size 個まで
            key = ctx if condition in ("inferred", "inferred_gated") else k
            if store_consistent and condition == "inferred_gated":
                # 12.6.116節: 文脈の今の規則と整合する（正しく予測できる）サンプルだけを保存する。移行の終盤に
                # 混ざる前のタスクのサンプルのうち、規則と食い違うものがバッファに入るのを防ぐ（スキーマ整合的な固定化）
                with torch.no_grad():
                    keep = model(with_ctx(bits, ctx)).argmax(-1) == labels
                bits, labels = bits[keep], labels[keep]
                src = src[keep] if src is not None else None
            n = min(buffer_size - buf_count.get(key, 0), labels.shape[0])
            if n > 0:
                buf_count[key] = buf_count.get(key, 0) + n
                buf_bits.append(bits[:n])
                buf_labels.append(labels[:n])
                buf_ctx.append(ctx_in[:n].long() if isinstance(ctx_in, torch.Tensor)
                               else torch.full((n,), ctx, dtype=torch.long))
                buf_src.append(src[:n].long() if src is not None else torch.full((n,), k, dtype=torch.long))
        chosen: list[int] = []
        R.append([accuracy(j, chosen) for j in range(T)])
        if diagnostics:
            with torch.no_grad():
                acc_row = []
                for j in range(T):
                    bits, labels = make_batch(specs[j], eval_n, diag_gen)
                    acc_row.append([float((model(with_ctx(bits, c)).argmax(-1) == labels).float().mean())
                                    for c in (used if condition in ("inferred", "inferred_gated") else [0])])
            diag["ctx_acc"].append(acc_row)
            diag["chosen"].append(chosen)
            diag["trained_src"].append({f"{c}<-{t}": n for (c, t), n in sorted(trained.items())})
    selection_acc = None
    if condition in ("inferred", "inferred_gated") and selection_trials > 0:
        majority = assignment_majority(assign, T)
        hits = 0
        for j in range(T):
            for _ in range(selection_trials):
                cb, cl = make_batch(specs[j], cue_n, eval_generator)
                hits += int(min(used, key=lambda c: loss_under(cb, cl, c)) == majority[j])
        selection_acc = hits / (T * selection_trials)
    out = {"R": R, "assign": assign, "n_contexts": len(used) if condition in ("inferred", "inferred_gated") else C,
           "selection_acc": selection_acc}
    if diagnostics:
        bc = torch.cat(buf_ctx) if buf_ctx else torch.zeros(0, dtype=torch.long)
        bs = torch.cat(buf_src) if buf_src else torch.zeros(0, dtype=torch.long)
        bb = torch.cat(buf_bits) if buf_bits else torch.zeros(0)
        bl = torch.cat(buf_labels) if buf_labels else torch.zeros(0, dtype=torch.long)
        diag["buffer"] = {"bits": bb, "labels": bl, "ctx": bc, "src": bs}
        diag["buffer_src"] = {int(c): {int(t): int(((bc == c) & (bs == t)).sum()) for t in bs[bc == c].unique()}
                              for c in bc.unique()}
        out["diag"] = diag
    return out


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
