"""言語の境界を与えない継続学習（12.6.245〜246節）。

12.6.239〜240節は評価のときに言語の番号を与えなかったが、学習のときには「新しい言語が始まった時点」を与えていた
（言語ごとに新しいヘッドを使い、バッファも言語の終わりに作った）。脳は経験の流れに境界の印を受け取らず、新しい文脈を
作るかどうか自体を推定する（潜在的な原因の推定: Gershman, Blei & Niv 2010; 文脈の生成と想起の COIN: Heald, Lengyel &
Wolpert 2021。機械学習での対応は Dirichlet 過程の専門家の混合 CN-DPM: Lee et al. 2020）。

ここでは学習の各ステップで、流れから来たバッチ（1つの言語の断片）を、学習の前に既存の各文脈 c で読んだ平均の負の
対数尤度 L_c と、その文脈の「普段の損失」b_c（その文脈に割り当てたバッチの L の移動平均）を比べた驚き
s_c = L_c − b_c で割り当てる:
  1. 直前の文脈の驚きが δ 以下ならそのまま（文脈の持続、COIN の粘着性）。
  2. そうでなければ、驚きが最小の既存の文脈が δ 以下ならそれに戻る（文脈の再活性化）。
  3. どれも δ を超えるなら新しい文脈（新しいヘッド）を作る（max_ctx に達したら驚きが最小の文脈）。
δ は、新しい文脈の事前予測の尤度を「普段の損失 + δ」とみなす中華料理店過程の新しい卓の重みに当たる。
バッファ（海馬の痕跡）は文脈ごとに流れのバッチから貯水池抽出で貯め、replay は今の文脈以外の文脈の痕跡を
その文脈のヘッドで混ぜて学ぶ。境界を与える対照（boundary="given"）は、言語が変わった時点で新しい文脈を作るだけで
他は同じ（割り当てが一致すれば、推定の系と同じ乱数の流れでまったく同じ学習になる）。

評価は 12.6.239節の推定の読み出し（作った全文脈の事後分布でヘッドを混ぜる）で、各言語の学習の後に測る。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .context_realtext import LangData, TransformerContextLM, _passages
from .continual_metrics import forgetting_metrics
from .realtext_forgetting import eval_segments, infer_logp

RULES = ("naive", "replay")
BOUNDARIES = ("infer", "given")


@dataclass
class BoundaryConfig:
    rule: str = "naive"
    boundary: str = "infer"
    steps_per_lang: int = 1500
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
    max_ctx: int = 12  # 用意するヘッドの数（作れる文脈の上限）
    delta: float = 0.3  # 驚きの閾値（nats/文字）
    ema: float = 0.05  # 普段の損失の移動平均の速さ
    trace_every: int = 10  # 驚きの記録の間引き


def assign(s: torch.Tensor, prev: int | None, n_ctx: int, delta: float, max_ctx: int) -> int:
    """驚き s [n_ctx]（既存の文脈）から割り当てる文脈。n_ctx を返すと新しい文脈。"""
    if n_ctx == 0:
        return 0
    if prev is not None and float(s[prev]) <= delta:
        return prev
    c = int(s.argmin())
    if float(s[c]) <= delta or n_ctx >= max_ctx:
        return c
    return n_ctx


def batch_nll(model: TransformerContextLM, seg: torch.Tensor, k: int) -> torch.Tensor:
    """断片 seg [B, T + 1] を最初の k 個の文脈で読んだ、全位置の平均の負の対数尤度 [k]（勾配なし）。"""
    with torch.no_grad():
        h = model.trunk(seg[:, :-1])  # [B, T, H]
        z = torch.tanh(h[:, :, None, :] + model.ctx.weight[None, None, :k])
        lp = torch.log_softmax(torch.einsum("btkh,khv->btkv", z, model.head_w[:k]) + model.head_b[:k], -1)
        y = seg[:, 1:, None, None].expand(-1, -1, k, 1)
        return -lp.gather(-1, y)[..., 0].mean((0, 1))


def ari(a: list[int], b: list[int]) -> float:
    """調整ランド指数（Hubert & Arabie 1985）。"""
    from math import comb
    n = len(a)
    pairs: dict = {}
    for x, y in zip(a, b):
        pairs[(x, y)] = pairs.get((x, y), 0) + 1
    ra: dict = {}
    rb: dict = {}
    for (x, y), m in pairs.items():
        ra[x] = ra.get(x, 0) + m
        rb[y] = rb.get(y, 0) + m
    idx = sum(comb(m, 2) for m in pairs.values())
    sa, sb = sum(comb(m, 2) for m in ra.values()), sum(comb(m, 2) for m in rb.values())
    exp = sa * sb / comb(n, 2)
    mx = (sa + sb) / 2
    return 1.0 if mx == exp else (idx - exp) / (mx - exp)


def segmentation(assigned: list[int], steps: int, n_l: int) -> dict:
    """割り当ての列（学習の順の言語 pos の steps 個ずつ）から、文脈の数・ARI・境界の検出の遅れ・文脈と言語の対応。"""
    truth = [p for p in range(n_l) for _ in range(steps)]
    delays = []
    for pos in range(1, n_l):
        t0 = pos * steps
        if t0 + steps > len(assigned):  # 学習の途中（記録の時点）ではまだ終わっていない境界を数えない
            break
        old = set(assigned[:t0])
        d = next((t - t0 for t in range(t0, t0 + steps) if assigned[t] not in old), steps)
        delays.append(d)
    n_ctx = max(assigned) + 1
    counts = [[0] * n_l for _ in range(n_ctx)]
    for a, p in zip(assigned, truth):
        counts[a][p] += 1
    major = [max(range(n_l), key=lambda p: row[p]) for row in counts]
    runs, start = [], 0
    for t in range(1, len(assigned) + 1):
        if t == len(assigned) or assigned[t] != assigned[start]:
            runs.append([assigned[start], start, t - start])
            start = t
    return {"n_ctx": n_ctx, "ARI": ari(assigned, truth), "delays": delays, "major": major, "runs": runs,
            "purity": sum(max(r) for r in counts) / len(assigned), "missed": sum(d >= steps for d in delays)}


def train_boundary_free(data: list[LangData], vocab: int, cfg: BoundaryConfig, order: list[int],
                        generator: torch.Generator, eval_generator: torch.Generator) -> dict:
    """言語を order の順に流し、文脈を割り当てながら学ぶ。各言語の流れの終わりに推定の読み出しで正確度行列の行を測る
    （R[i][j] は学習の順の i 番目の言語の後の j 番目の言語の値）。"""
    if cfg.rule not in RULES or cfg.boundary not in BOUNDARIES:
        raise ValueError(f"rule は {RULES}、boundary は {BOUNDARIES}")
    n_l, t_len, k_max = len(data), cfg.seq_len, cfg.max_ctx
    flat = [_passages(d.train, 400).reshape(-1) for d in data]
    segs = eval_segments(data, cfg.eval_segments, t_len, eval_generator)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator)))
        model = TransformerContextLM(vocab, k_max, t_len, cfg.d, cfg.layers, cfg.heads, hidden=cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    buf = torch.empty(k_max, cfg.buffer, t_len + 1, dtype=torch.long)
    n_buf = [0] * k_max  # バッファに入っている数
    n_seen = [0] * k_max  # 貯水池抽出で見た数
    base = torch.zeros(k_max)
    n_ctx, prev = 0, None
    assigned: list[int] = []
    trace: list[list[float]] = []

    def batch_from(j: int, n: int) -> torch.Tensor:
        s = torch.randint(0, flat[j].shape[0] - t_len - 1, (n,), generator=generator)
        return flat[j][s[:, None] + torch.arange(t_len + 1)[None]]

    def store(a: int, seg: torch.Tensor) -> None:
        r = torch.rand(seg.shape[0], generator=generator)
        for i in range(seg.shape[0]):
            n_seen[a] += 1
            if n_buf[a] < cfg.buffer:
                buf[a, n_buf[a]] = seg[i]
                n_buf[a] += 1
            else:
                k = int(r[i] * n_seen[a])
                if k < cfg.buffer:
                    buf[a, k] = seg[i]

    rows_acc, rows_ce, rows_id = [], [], []

    def record() -> None:
        seen = list(range(n_ctx))
        major = segmentation(assigned, cfg.steps_per_lang, n_l)["major"] if assigned else []
        acc, ce, ident = [], [], []
        with torch.no_grad():
            for pos in range(n_l):
                s = segs[order[pos]]
                y = s[:, 1:]
                mix, post = infer_logp(model, s[:, :-1], y, seen)
                acc.append(float((mix.argmax(-1) == y).float().mean()))
                ce.append(float(nn.functional.nll_loss(mix.reshape(-1, mix.shape[-1]), y.reshape(-1))))
                hit = torch.tensor([major[c] for c in seen])[post.argmax(-1)] == pos
                ident.append(float(hit.float().mean()))
        rows_acc.append(acc)
        rows_ce.append(ce)
        rows_id.append(ident)

    losses = []
    for pos, j in enumerate(order):
        for step_i in range(cfg.steps_per_lang):
            model.train()
            seg = batch_from(j, cfg.batch)
            nll = batch_nll(model, seg, min(n_ctx + 1, k_max))
            s = nll[:n_ctx] - base[:n_ctx]
            if cfg.boundary == "given":
                a = pos
            else:
                a = assign(s, prev, n_ctx, cfg.delta, k_max)
            if step_i % cfg.trace_every == 0:  # 直前の文脈の驚き（割り当ての前）と割り当て
                trace.append([round(float(s[prev]), 3) if prev is not None else None, a])
            fresh = a == n_ctx
            if fresh:
                n_ctx += 1
                base[a] = nll[a]
            else:
                base[a] = (1 - cfg.ema) * base[a] + cfg.ema * nll[a]
            c = torch.full((cfg.batch,), a)
            past = [k for k in range(n_ctx) if k != a and n_buf[k] > 0]
            if cfg.rule == "replay" and past:
                rc = torch.tensor(past)[torch.randint(0, len(past), (cfg.replay_batch,), generator=generator)]
                ri = (torch.rand(cfg.replay_batch, generator=generator)
                      * torch.tensor([n_buf[k] for k in rc.tolist()])).long()
                seg, c = torch.cat([seg, buf[rc, ri]]), torch.cat([c, rc])
            opt.zero_grad()
            lp = model.seq_logp(seg[:, :-1], c)
            loss = nn.functional.nll_loss(lp.reshape(-1, lp.shape[-1]), seg[:, 1:].reshape(-1))
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            store(a, seg[:cfg.batch])
            assigned.append(a)
            prev = a
        model.eval()
        record()
    out = {"R_acc": rows_acc, "R_ce": rows_ce, "R_id": rows_id, "final_loss": sum(losses[-100:]) / 100,
           "order": order, "seg": segmentation(assigned, cfg.steps_per_lang, n_l), "trace": trace}
    out["metrics_acc"] = forgetting_metrics(rows_acc)
    out["metrics_nce"] = forgetting_metrics([[-x for x in r] for r in rows_ce])
    return out
