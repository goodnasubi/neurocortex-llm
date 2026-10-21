"""継続学習の忘却指標（ACC・BWT・F・LA）を、実テキストの多言語を順に学ぶ自己回帰の文字 Transformer で測る（12.6.229〜230節）。

12.6.101〜102節で定義した指標（continual_metrics）は、XOR の合成課題でしか測っていない（12.7節「評価指標の欠如」の
「実言語モデルでの測定は未着手」）。ここでは 12.6.213節の基盤（Europarl の実テキスト、文脈ごとのヘッドを持つ因果的な
文字 Transformer、context_realtext）で、言語を順に学ぶときの正確度行列 R[i][j]（言語 i の学習後の言語 j のテスト文の
次の文字の正確度・CE。ヘッドは言語 j のもの＝タスクの番号を与える形）を測る。

規則（rule）:
- naive: 言語を順に学ぶだけ。
- replay: 相補的学習系（McClelland et al. 1995; Kumaran et al. 2016）の海馬のリプレイ。各言語の学習の終わりに断片を
  バッファへ保存し、以後の各ステップで過去の言語の断片を新しい言語のバッチに混ぜて学ぶ（12.6.213節の seq と同じ）。
- ewc: シナプスの固定化（Kirkpatrick et al. 2017。シナプスの固定化の程度を重みごとに持つ形: Benna & Fusi 2016）。
  各言語の学習の終わりに、バッファの断片で対角のフィッシャー情報量を求めて足し合わせ、その時点の重みからのずれに
  λ/2·Σ F (θ − θ*)² の罰則をかける（タスクを跨いだ online EWC）。F は全パラメータの平均で割って正規化する。
- replay_ewc: 両方。
- joint: 全言語を毎ステップ混ぜて学ぶ（同じ総ステップ数。順に学ばないので最後の行だけを測る参照）。

正確度・CE は、テスト文（学習・話し手とは別の日）の無作為な断片（長さ T + 1）の全位置の次の文字で測る。評価の断片は
評価専用の乱数で一度だけ取り、全規則・全時点で同じものを使う（学習の過程は評価で変わらない）。

タスクの番号を与えない形（12.6.239〜240節）:
- 推定（infer）: 同じモデルを、言語の番号を与えずに読む。それまでに学んだ言語の文脈（ヘッド）について、断片の前の
  文字から求めた事後分布（一様な事前分布から逐次のベイズの更新）で各ヘッドの予測を混ぜる（measure_infer）。
- 共有（shared_head）: 文脈を持たない1つのヘッドだけのモデルを同じ規則で学ぶ（文脈の系のない対照）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .context_realtext import LangData, TransformerContextLM, _passages
from .continual_metrics import forgetting_metrics

RULES = ("naive", "replay", "ewc", "replay_ewc", "joint")


@dataclass
class ForgetConfig:
    rule: str = "naive"
    steps_per_lang: int = 1500
    batch: int = 64
    replay_batch: int = 32
    seq_len: int = 32
    buffer: int = 2000  # 言語ごとのバッファの断片の数
    lr: float = 3e-3
    d: int = 64
    layers: int = 2
    heads: int = 4
    hidden: int = 128
    ewc_lambda: float = 0.0
    fisher_batches: int = 64  # フィッシャー情報量を求めるミニバッチの数
    fisher_batch: int = 8
    eval_segments: int = 400  # 評価の断片の数（言語ごと）
    shared_head: bool = False  # True なら文脈を持たない1つのヘッド（全言語が文脈 0）


def eval_segments(data: list[LangData], n: int, t_len: int, generator: torch.Generator) -> list[torch.Tensor]:
    """言語ごとのテスト文の無作為な断片 [n, T + 1]。"""
    out = []
    for d in data:
        s = torch.randint(0, d.test.shape[0] - t_len - 1, (n,), generator=generator)
        out.append(d.test[s[:, None] + torch.arange(t_len + 1)[None]])
    return out


def measure(model: TransformerContextLM, segs: list[torch.Tensor],
            shared: bool = False) -> tuple[list[float], list[float]]:
    """各言語 j の断片を言語 j のヘッドで読み（shared なら唯一のヘッド）、次の文字の正確度と CE（nats）の全位置の平均。"""
    acc, ce = [], []
    with torch.no_grad():
        for j, s in enumerate(segs):
            lp = model.seq_logp(s[:, :-1], torch.full((s.shape[0],), 0 if shared else j))
            y = s[:, 1:]
            acc.append(float((lp.argmax(-1) == y).float().mean()))
            ce.append(float(nn.functional.nll_loss(lp.reshape(-1, lp.shape[-1]), y.reshape(-1))))
    return acc, ce


def infer_logp(model: TransformerContextLM, x: torch.Tensor, y: torch.Tensor,
               seen: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """タスクの番号なしの予測。位置 t の文脈の事後分布 w_c(t) ∝ Π_{s<t} p(y_s | x_≤s, c)（seen の上で一様な事前分布）で
    各ヘッドの予測を混ぜる: log p(·|x_≤t) = logsumexp_c(log w_c(t) + log p(·|x_≤t, c))。

    返り値は混ぜた対数確率 [N, T, V] と、最後の位置の事後分布 [N, len(seen)]（全位置を見た後）。"""
    lps = torch.stack([model.seq_logp(x, torch.full((x.shape[0],), c)) for c in seen])  # [K, N, T, V]
    g = lps.gather(-1, y[None, :, :, None].expand(len(seen), -1, -1, 1))[..., 0]  # [K, N, T]
    cum = torch.cumsum(g, -1)
    prior = torch.cat([torch.zeros_like(cum[..., :1]), cum[..., :-1]], -1)  # 位置 t までの前の文字（t を含まない）
    logw = prior - torch.logsumexp(prior, 0, keepdim=True)
    mix = torch.logsumexp(logw[..., None] + lps, 0)
    post = torch.softmax(cum[..., -1], 0).T
    return mix, post


def measure_infer(model: TransformerContextLM, segs: list[torch.Tensor],
                  seen: list[int]) -> tuple[list[float], list[float], list[float]]:
    """各言語 j の断片を、言語の番号を与えずに seen の文脈の事後分布で混ぜて読む。正確度・CE と、全位置を見た後の
    事後分布の最大が言語 j である割合（文脈の同定の正確度。j が seen にないときは 0）。"""
    acc, ce, ident = [], [], []
    with torch.no_grad():
        for j, s in enumerate(segs):
            y = s[:, 1:]
            mix, post = infer_logp(model, s[:, :-1], y, seen)
            acc.append(float((mix.argmax(-1) == y).float().mean()))
            ce.append(float(nn.functional.nll_loss(mix.reshape(-1, mix.shape[-1]), y.reshape(-1))))
            hit = torch.tensor(seen)[post.argmax(-1)] == j
            ident.append(float(hit.float().mean()))
    return acc, ce, ident


def fisher_diag(model: TransformerContextLM, segs: torch.Tensor, c: int, batches: int, bs: int,
                generator: torch.Generator) -> list[torch.Tensor]:
    """断片 segs [N, T + 1] の文脈 c での対角の経験フィッシャー情報量（ミニバッチの勾配の2乗の平均）。"""
    fs = [torch.zeros_like(p) for p in model.parameters()]
    for _ in range(batches):
        idx = torch.randint(0, segs.shape[0], (bs,), generator=generator)
        seg = segs[idx]
        model.zero_grad()
        lp = model.seq_logp(seg[:, :-1], torch.full((bs,), c))
        nn.functional.nll_loss(lp.reshape(-1, lp.shape[-1]), seg[:, 1:].reshape(-1)).backward()
        for f, p in zip(fs, model.parameters()):
            if p.grad is not None:
                f += p.grad.detach() ** 2 / batches
    model.zero_grad()
    return fs


def train_forgetting(data: list[LangData], vocab: int, cfg: ForgetConfig, order: list[int],
                     generator: torch.Generator, eval_generator: torch.Generator) -> dict:
    """言語を order の順に学び、各言語の学習の後に正確度行列の行を測る（joint は最後の行だけ）。

    R[i][j] は order の i 番目の言語の学習後の、order の j 番目の言語の値（学習の順に並べ替えた行列）。"""
    if cfg.rule not in RULES:
        raise ValueError(f"rule は {RULES}")
    n_l, t_len = len(data), cfg.seq_len
    flat = [_passages(d.train, 400).reshape(-1) for d in data]
    segs = eval_segments(data, cfg.eval_segments, t_len, eval_generator)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator)))
        model = TransformerContextLM(vocab, 1 if cfg.shared_head else n_l, t_len, cfg.d, cfg.layers, cfg.heads, hidden=cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    use_replay = cfg.rule in ("replay", "replay_ewc")
    use_ewc = cfg.rule in ("ewc", "replay_ewc") and cfg.ewc_lambda > 0
    sh = cfg.shared_head
    ctx = (lambda j: 0) if sh else (lambda j: j)  # noqa: E731  共有のヘッドなら全言語が文脈 0
    buf: dict[int, torch.Tensor] = {}
    fisher_raw = [torch.zeros_like(p) for p in model.parameters()]
    fisher = fisher_raw
    anchor: list[torch.Tensor] | None = None

    def batch_from(j: int, n: int) -> torch.Tensor:
        s = torch.randint(0, flat[j].shape[0] - t_len - 1, (n,), generator=generator)
        return flat[j][s[:, None] + torch.arange(t_len + 1)[None]]

    def step(seg: torch.Tensor, c: torch.Tensor) -> float:
        opt.zero_grad()
        lp = model.seq_logp(seg[:, :-1], c)
        loss = nn.functional.nll_loss(lp.reshape(-1, lp.shape[-1]), seg[:, 1:].reshape(-1))
        total = loss
        if use_ewc and anchor is not None:
            pen = sum((f * (p - a) ** 2).sum() for f, p, a in zip(fisher, model.parameters(), anchor))
            total = loss + 0.5 * cfg.ewc_lambda * pen
        total.backward()
        opt.step()
        return float(loss.detach())

    rows_acc, rows_ce, losses = [], [], []
    inf_acc, inf_ce, inf_id = [], [], []

    def record(seen: list[int]) -> None:
        a, e = measure(model, segs, sh)
        rows_acc.append(perm(a))
        rows_ce.append(perm(e))
        if not sh:  # 共有のヘッドはもともと番号を使わない（推定の行は持たない）
            a, e, i = measure_infer(model, segs, seen)
            inf_acc.append(perm(a))
            inf_ce.append(perm(e))
            inf_id.append(perm(i))

    perm = lambda row: [row[j] for j in order]  # noqa: E731  言語の番号 → 学習の順
    if cfg.rule == "joint":
        for _ in range(cfg.steps_per_lang * n_l):
            c = torch.randint(0, n_l, (cfg.batch,), generator=generator)
            seg = torch.empty(cfg.batch, t_len + 1, dtype=torch.long)
            for j in range(n_l):
                sel = c == j
                if int(sel.sum()):
                    seg[sel] = batch_from(j, int(sel.sum()))
            losses.append(step(seg, torch.zeros_like(c) if sh else c))
        model.eval()
        record(list(range(n_l)))
    else:
        for pos, j in enumerate(order):
            past = order[:pos]
            model.train()
            for _ in range(cfg.steps_per_lang):
                seg, c = batch_from(j, cfg.batch), torch.full((cfg.batch,), ctx(j))
                if use_replay and past:
                    rc = torch.tensor(past)[torch.randint(0, len(past), (cfg.replay_batch,), generator=generator)]
                    ri = torch.randint(0, cfg.buffer, (cfg.replay_batch,), generator=generator)
                    rseg = torch.stack([buf[a][b] for a, b in zip(rc.tolist(), ri.tolist())])
                    seg, c = torch.cat([seg, rseg]), torch.cat([c, rc if not sh else torch.zeros_like(rc)])
                losses.append(step(seg, c))
            model.eval()
            buf[j] = batch_from(j, cfg.buffer)
            if use_ewc:
                f_j = fisher_diag(model, buf[j], ctx(j), cfg.fisher_batches, cfg.fisher_batch, generator)
                fisher_raw = [f + g for f, g in zip(fisher_raw, f_j)]
                scale = torch.cat([f.reshape(-1) for f in fisher_raw]).mean().clamp_min(1e-20)
                fisher = [f / scale for f in fisher_raw]  # 罰則に使うのは全パラメータの平均が 1 になる形
                anchor = [p.detach().clone() for p in model.parameters()]
            record([order[k] for k in range(pos + 1)])
    out = {"R_acc": rows_acc, "R_ce": rows_ce, "final_loss": sum(losses[-100:]) / 100,
           "order": order}
    if not sh:
        out.update({"R_acc_inf": inf_acc, "R_ce_inf": inf_ce, "R_id_inf": inf_id})
    for suf, ra, rc in [("", rows_acc, rows_ce)] + ([] if sh else [("_inf", inf_acc, inf_ce)]):
        if cfg.rule != "joint":
            out[f"metrics_acc{suf}"] = forgetting_metrics(ra)
            out[f"metrics_nce{suf}"] = forgetting_metrics([[-x for x in r] for r in rc])  # −CE（大きいほど良い）
        else:
            out[f"metrics_acc{suf}"] = {"ACC": sum(ra[0]) / n_l}
            out[f"metrics_nce{suf}"] = {"ACC": -sum(rc[0]) / n_l}
    return out
