"""境界の推定が難しい設定での継続学習：戻ってくる言語と1断片ずつの割り当て（12.6.249〜250節）。

12.6.245〜246節では、言語の間の驚き（≧1.04 nats/文字）が言語の中の揺らぎ（≦0.21）の約5倍あり、64 断片のバッチ
ごとの割り当てでは境界の推定がほぼ自明だった。また同じ言語が戻ってこないので、以前の文脈に戻る再活性化
（COIN の文脈の想起: Heald, Lengyel & Wolpert 2021）は一度も使われなかった。ここでは
- 流れ: 各言語を同じ長さの区間 visits 個に分け、言語が戻ってくる順で流す（1巡目は order の順、2巡目以降は
  無作為な順で、同じ言語が続かない）。1言語の総ステップ数は 12.6.245節と同じ。
- 観測を小さくする（assign="frag"）: バッチの各断片（32 文字）を1つずつ順に割り当てる。驚き s_c = L_c − b_c は
  断片1つの損失で測り、普段の損失 b_c は断片ごとの移動平均（最初は標本平均）で更新する。規則は 12.6.245節と
  同じ（持続・再活性化・新しい文脈）。断片ごとに割り当てた文脈のヘッドで学び、断片はその文脈のバッファに入れる。
- 証拠の蓄積（assign="ev"）: frag と同じく断片ごとに割り当てるが、規則に使う驚きを各文脈の漏れのある積分
  m_c ← (1 − λ)·m_c + λ·s_c（λ = 1/8）にする。新しい文脈を作ったら、その m を 0 から始める。1断片の驚きの揺らぎを
  数断片ぶん平均してから判断する（皮質の証拠の蓄積: Gold & Shadlen 2007, Annu Rev Neurosci 30:535。変化の検出の
  漏れのある積分: Nassar et al. 2010, J Neurosci 30:12366）。
対照:
- assign="batch": 12.6.245節と同じバッチごとの割り当て（戻ってくる流れの再活性化だけを試す）。
- assign="id": 言語の番号を文脈とする（境界も言語の同定も与える）。
- assign="blk": 区間ごとに新しい文脈（境界は与えるが、戻ってきた言語を同定しない＝再活性化しない、分けすぎの極端）。
潜在的な原因の推定の誤り（統合しすぎ・分けすぎ）が干渉と記憶の分割に効くという主張（Gershman, Blei & Niv 2010;
Gershman, Monfils, Norman & Niv 2017, eLife 6:e23763）を、実言語の忘却で測る。

評価は 12.6.239節の推定の読み出し（作った全文脈の事後分布でヘッドを混ぜる）で、各区間の終わりに全言語を測る。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .context_realtext import LangData, TransformerContextLM, _passages
from .realtext_boundary_free import ari, assign
from .realtext_forgetting import eval_segments, infer_logp

RULES = ("naive", "replay")
ASSIGNS = ("frag", "ev", "batch", "id", "blk")


@dataclass
class HardConfig:
    rule: str = "naive"
    assign: str = "frag"
    steps_per_lang: int = 1500
    visits: int = 3  # 1言語の区間の数（区間の長さは steps_per_lang / visits）
    batch: int = 64
    replay_batch: int = 32
    seq_len: int = 32
    buffer: int = 2000  # 文脈ごとのバッファの断片の数
    lr: float = 3e-3
    d: int = 64
    layers: int = 2
    heads: int = 4
    hidden: int = 128
    eval_segments: int = 400
    max_ctx: int = 24  # 用意するヘッドの数（blk は 6 言語 × 3 区間 = 18 個を使う）
    delta: float = 0.3  # 驚きの閾値（nats/文字、12.6.245節と同じ）
    leak: float = 0.125  # ev の驚きの漏れのある積分の速さ（断片ごと。約 8 断片の時定数）
    ema: float = 0.05  # バッチごとの普段の損失の移動平均の速さ（frag は断片ごとに 1 − (1 − ema)^(1/batch)）
    trace_every: int = 25  # 割り当ての記録の間引き（ステップ）
    sample_every: int = 500  # 驚きの標本の間引き（断片）


def stream_blocks(n_l: int, order: list[int], visits: int, generator: torch.Generator) -> list[int]:
    """区間の言語の列。1巡目は order、2巡目以降は無作為な順（直前の区間と同じ言語で始まらない）。"""
    blocks = list(order)
    for _ in range(visits - 1):
        while True:
            p = torch.randperm(n_l, generator=generator).tolist()
            if p[0] != blocks[-1]:
                break
        blocks += p
    return blocks


def frag_nll(model: TransformerContextLM, seg: torch.Tensor, k: int) -> torch.Tensor:
    """断片 seg [B, T + 1] を最初の k 個の文脈で読んだ、断片ごとの平均の負の対数尤度 [B, k]（勾配なし）。"""
    with torch.no_grad():
        h = model.trunk(seg[:, :-1])
        z = torch.tanh(h[:, :, None, :] + model.ctx.weight[None, None, :k])
        lp = torch.log_softmax(torch.einsum("btkh,khv->btkv", z, model.head_w[:k]) + model.head_b[:k], -1)
        y = seg[:, 1:, None, None].expand(-1, -1, k, 1)
        return -lp.gather(-1, y)[..., 0].mean(1)


def assignment_stats(ctx: list[int], lang: list[int], block: list[int], blocks: list[int], n_l: int,
                     batch: int) -> dict:
    """断片ごとの割り当て ctx と真の言語 lang・区間 block から、割り当ての指標。
    - ARI（言語に対して）、純度、文脈の数、余分な文脈（文脈の数 − 言語の数）
    - 統合しすぎ: 断片の最も多くが入る文脈の主な言語が自分でない言語の数
    - 再活性化: 戻ってきた区間（2回目以降）の断片のうち、その区間より前からある文脈で主な言語がその言語のものに
      入った割合。新しく分けた割合: その区間の中で作られた文脈に入った割合
    - 切り替えの遅れ: 区間の始めから、バッチの最多の文脈の主な言語が区間の言語になるまでのステップ数"""
    n_ctx = max(ctx) + 1
    counts = [[0] * n_l for _ in range(n_ctx)]
    for a, j in zip(ctx, lang):
        counts[a][j] += 1
    major = [max(range(n_l), key=lambda p: row[p]) for row in counts]
    born = {}
    for i, a in enumerate(ctx):
        born.setdefault(a, block[i])
    merged = 0
    for j in range(n_l):
        top = max(range(n_ctx), key=lambda a: counts[a][j])
        merged += major[top] != j
    react, split, n_ret = 0, 0, 0
    first = {}
    for b, j in enumerate(blocks):
        first.setdefault(j, b)
    for i, (a, b) in enumerate(zip(ctx, block)):
        if first[blocks[b]] == b:
            continue
        n_ret += 1
        react += born[a] < b and major[a] == blocks[b]
        split += born[a] == b
    steps = len(ctx) // batch
    per_block = steps // len(blocks)
    delays = []
    for b in range(1, len(blocks)):
        d = per_block
        for t in range(b * per_block, (b + 1) * per_block):
            row = ctx[t * batch:(t + 1) * batch]
            top = max(set(row), key=row.count)
            if major[top] == blocks[b]:
                d = t - b * per_block
                break
        delays.append(d)
    return {"n_ctx": n_ctx, "extra": n_ctx - n_l, "ARI": ari(ctx, lang),
            "purity": sum(max(r) for r in counts) / len(ctx), "merged": merged, "major": major,
            "react": react / n_ret if n_ret else None, "split": split / n_ret if n_ret else None,
            "delays": delays, "ctx_sizes": [sum(r) for r in counts]}


def train_hard(data: list[LangData], vocab: int, cfg: HardConfig, order: list[int],
               generator: torch.Generator, eval_generator: torch.Generator) -> dict:
    """言語を戻ってくる区間の列で流し、文脈を割り当てながら学ぶ。各区間の終わりに推定の読み出しで全言語を測る
    （R[b][j] は b 番目の区間の後の、学習の順の j 番目の言語 order[j] の値）。"""
    if cfg.rule not in RULES or cfg.assign not in ASSIGNS:
        raise ValueError(f"rule は {RULES}、assign は {ASSIGNS}")
    n_l, t_len, k_max, bs = len(data), cfg.seq_len, cfg.max_ctx, cfg.batch
    flat = [_passages(d.train, 400).reshape(-1) for d in data]
    segs = eval_segments(data, cfg.eval_segments, t_len, eval_generator)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator)))
        model = TransformerContextLM(vocab, k_max, t_len, cfg.d, cfg.layers, cfg.heads, hidden=cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    blocks = stream_blocks(n_l, order, cfg.visits, generator)
    acc_s = torch.zeros(k_max)  # ev の驚きの漏れのある積分
    if cfg.assign == "blk" and len(blocks) > k_max:
        raise ValueError("blk には区間の数以上の max_ctx が要る")
    per_block = cfg.steps_per_lang // cfg.visits
    ema = cfg.ema if cfg.assign == "batch" else 1 - (1 - cfg.ema) ** (1 / bs)
    buf = torch.empty(k_max, cfg.buffer, t_len + 1, dtype=torch.long)
    n_buf, n_seen, n_asg = [0] * k_max, [0] * k_max, [0] * k_max
    base = torch.zeros(k_max)
    n_ctx, prev = 0, None
    ctx_all: list[int] = []
    lang_all: list[int] = []
    block_all: list[int] = []
    trace: list[list] = []
    s_within: list[float] = []  # 直前と同じ言語の断片での直前の文脈の驚き（間引き）
    s_switch: list[float] = []  # 区間の最初の断片での直前の文脈の驚き
    rows_ce, rows_acc, rows_id = [], [], []
    pos_of = {j: p for p, j in enumerate(order)}

    def batch_from(j: int, n: int) -> torch.Tensor:
        s = torch.randint(0, flat[j].shape[0] - t_len - 1, (n,), generator=generator)
        return flat[j][s[:, None] + torch.arange(t_len + 1)[None]]

    def store(a: int, seg: torch.Tensor) -> None:
        n_seen[a] += 1
        if n_buf[a] < cfg.buffer:
            buf[a, n_buf[a]] = seg
            n_buf[a] += 1
        else:
            k = int(torch.rand(1, generator=generator) * n_seen[a])
            if k < cfg.buffer:
                buf[a, k] = seg

    def update_base(a: int, x: float, rate: float) -> None:
        n_asg[a] += 1
        r = max(rate, 1 / n_asg[a])  # 最初は標本平均、その後は移動平均
        base[a] = (1 - r) * base[a] + r * x

    def record() -> None:
        st = assignment_stats(ctx_all, lang_all, block_all, blocks, n_l, bs)
        major = st["major"]
        seen = list(range(n_ctx))
        acc, ce, ident = [], [], []
        with torch.no_grad():
            for p in range(n_l):
                s = segs[order[p]]
                y = s[:, 1:]
                mix, post = infer_logp(model, s[:, :-1], y, seen)
                acc.append(float((mix.argmax(-1) == y).float().mean()))
                ce.append(float(nn.functional.nll_loss(mix.reshape(-1, mix.shape[-1]), y.reshape(-1))))
                hit = torch.tensor([major[c] for c in seen])[post.argmax(-1)] == order[p]
                ident.append(float(hit.float().mean()))
        rows_acc.append(acc)
        rows_ce.append(ce)
        rows_id.append(ident)

    losses = []
    prev_lang = None
    for b, j in enumerate(blocks):
        for step_i in range(per_block):
            model.train()
            seg = batch_from(j, bs)
            if cfg.assign in ("id", "blk"):
                a = pos_of[j] if cfg.assign == "id" else b  # 文脈は初めて現れた順に作る
                n_ctx = max(n_ctx, a + 1)
                c = torch.full((bs,), a)
            else:
                nll = frag_nll(model, seg, min(n_ctx + bs, k_max))
                if cfg.assign == "batch":
                    m = nll.mean(0)
                    s = m[:n_ctx] - base[:n_ctx]
                    if prev is not None:
                        if j != prev_lang:
                            s_switch.append(float(s[prev]))
                        elif step_i % cfg.trace_every == 0:
                            s_within.append(float(s[prev]))
                    a = assign(s, prev, n_ctx, cfg.delta, k_max)
                    if a == n_ctx:
                        n_ctx += 1
                    update_base(a, float(m[a]), ema)
                    c = torch.full((bs,), a)
                    prev = a
                else:
                    cs = []
                    for i in range(bs):
                        s = nll[i, :n_ctx] - base[:n_ctx]
                        if cfg.assign == "ev":
                            acc_s[:n_ctx] = (1 - cfg.leak) * acc_s[:n_ctx] + cfg.leak * s
                            s = acc_s[:n_ctx].clone()
                        if prev is not None:
                            if j != prev_lang:
                                s_switch.append(float(s[prev]))
                            elif len(ctx_all) % cfg.sample_every == 0:
                                s_within.append(float(s[prev]))
                        a = assign(s, prev, n_ctx, cfg.delta, k_max)
                        if a == n_ctx:
                            n_ctx += 1
                            acc_s[a] = 0.0
                        update_base(a, float(nll[i, a]), ema)
                        cs.append(a)
                        prev = a
                        prev_lang = j
                        ctx_all.append(a)
                        lang_all.append(j)
                        block_all.append(b)
                    c = torch.tensor(cs)
            if cfg.assign not in ("frag", "ev"):
                ctx_all += c.tolist()
                lang_all += [j] * bs
                block_all += [b] * bs
                prev_lang = j
            present = set(c.tolist())
            past = [k for k in range(n_ctx) if k not in present and n_buf[k] > 0]
            if step_i % cfg.trace_every == 0:
                cl = c.tolist()
                top = max(set(cl), key=cl.count)
                trace.append([top, round(cl.count(top) / bs, 2)])
            x, cc = seg, c
            if cfg.rule == "replay" and past:
                rc = torch.tensor(past)[torch.randint(0, len(past), (cfg.replay_batch,), generator=generator)]
                ri = (torch.rand(cfg.replay_batch, generator=generator)
                      * torch.tensor([n_buf[k] for k in rc.tolist()])).long()
                x, cc = torch.cat([seg, buf[rc, ri]]), torch.cat([c, rc])
            opt.zero_grad()
            lp = model.seq_logp(x[:, :-1], cc)
            loss = nn.functional.nll_loss(lp.reshape(-1, lp.shape[-1]), x[:, 1:].reshape(-1))
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            for i in range(bs):
                store(int(c[i]), seg[i])
        model.eval()
        record()
    out = {"R_ce": rows_ce, "R_acc": rows_acc, "R_id": rows_id, "final_loss": sum(losses[-100:]) / 100,
           "order": order, "blocks": blocks, "blocks_pos": [pos_of[j] for j in blocks],
           "seg": assignment_stats(ctx_all, lang_all, block_all, blocks, n_l, bs), "trace": trace,
           "surprise": {"within": [round(x, 3) for x in s_within], "switch": [round(x, 3) for x in s_switch]}}
    out["metrics"] = stream_metrics(rows_ce, [pos_of[j] for j in blocks])
    return out


def stream_metrics(rows_ce: list[list[float]], blocks_pos: list[int]) -> dict:
    """区間の終わりごとの −CE の行列（rows[b][p]、p は学習の順の言語）から、戻ってくる流れの指標。
    - ACC: 最後の区間の後の全言語の −CE の平均
    - FGT: 区間 b の終わりの、今の区間の言語でなく以前に学んだ言語 p について、−CE_p(b) − −CE_p(p を最後に学んだ区間)
      の平均（負なら、学ばない間の忘却）
    - AVG: 区間の終わりごとの、それまでに学んだ言語の −CE の平均の、区間についての平均（いつでもの性能）
    - LA1: 各言語の最初の区間の終わりの −CE の平均（初めて学んだときの習得）"""
    R = [[-x for x in r] for r in rows_ce]
    n_b = len(R)
    last: dict[int, int] = {}
    drops, avg = [], []
    first: dict[int, float] = {}
    for b in range(n_b):
        p_now = blocks_pos[b]
        for p, lb in last.items():
            if p != p_now:
                drops.append(R[b][p] - R[lb][p])
        last[p_now] = b
        first.setdefault(p_now, R[b][p_now])
        avg.append(sum(R[b][p] for p in last) / len(last))
    return {"ACC": sum(R[-1]) / len(R[-1]), "FGT": sum(drops) / len(drops), "AVG": sum(avg) / n_b,
            "LA1": sum(first.values()) / len(first)}
