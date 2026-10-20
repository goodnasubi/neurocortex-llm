"""小脳の順モデルを先読みの帰結（遅れて届く登上線維の誤差）で学習させる（12.6.211節の設計）。

12.6.209節の構成（8専門家・1層のルーティング列、予測器は並列、読み込み経路2本）で、予測器の教師と
先読みの方針を変えて壁時計時間と無駄 I/O を測る:
none、last、freq、oracle、cerebellum（正解ラベル・ゲートなし）、cerebellum_fixed{θ}（正解ラベル＋固定閾値）、
cerebellum_efixed{θ}（正解ラベル＋固定閾値、常駐でない最上位を選ぶ）、cerebellum_delayed2.5（12.6.209節の
遅延 CF ゲート）、outcome{θ}（`OutcomeCerebellum`、帰結で学ぶ）、outcomeshuf{θ}（帰結を並べ替えた対照）。

使い方:
    OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python -m neurocortex.experiments.run_expert_prefetch_outcome \\
        --seeds 100 101 102 103 104 --out results/expert_prefetch_outcome/dev_rev0.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..data import CharCorpus, synthetic_corpus
from ..expert_prefetch import RoutedCharLM, make_predictor, routing_trace, simulate_prefetch_v2, train_routed_lm
from ..expert_prefetch_delayed import CFTaggedStore, DelayedCFGate, curve_residual, warm_delayed
from ..expert_prefetch_outcome import EligibleTopGate, OutcomeCerebellum, OutcomeTaggedStore, warm_outcome
from ..run_env import env_info
from .run_expert_prefetch_adaptive import SIZES, h_pass, select_on_dev, summarize, warm

DELAYED = "cerebellum_delayed2.5"
FAMILIES = ("cerebellum_fixed", "cerebellum_efixed", "outcome", "outcomeshuf")


def conditions(grid: list[float]) -> list[str]:
    out = ["none", "last", "freq", "cerebellum"]
    for fam in FAMILIES:
        out += [f"{fam}{th:g}" for th in grid]
    return out + [DELAYED, "oracle"]


def _parse(cond: str) -> tuple[str, float | None]:
    for pre in ("cerebellum_delayed", "cerebellum_efixed", "cerebellum_fixed", "outcomeshuf", "outcome"):
        if cond.startswith(pre):
            return pre, float(cond[len(pre):])
    return cond, None


def _top1(model, choices: np.ndarray, hidden: np.ndarray) -> float:
    """次の時刻の正解に対する Top-1 的中率（学習なし、参考）。"""
    hits = sum(model.rank(int(choices[t]), hidden[t])[0] == int(choices[t + 1]) for t in range(len(choices) - 1))
    return hits / max(len(choices) - 1, 1)


def run_seed(seed: int, args, conds: list[str]) -> dict:
    torch.manual_seed(seed)
    corpus = CharCorpus(synthetic_corpus(n_sentences=args.n_sentences, seed=seed))
    model = RoutedCharLM(corpus.vocab_size, d=args.d, n_experts=args.n_experts)
    g = torch.Generator().manual_seed(seed)
    ce = train_routed_lm(model, corpus.train, args.train_steps, 32, 64, 3e-3, args.balance_coef, g)
    choices, hidden = routing_trace(model, corpus.val[: args.trace_len])
    w0, w1 = args.warm_len, args.warm_len + args.speed_len
    wc, wh = choices[: w0 + 1], hidden[: w0 + 1]
    sc, sh = choices[w0:w1], hidden[w0:w1]

    rng = np.random.default_rng(seed)
    small = [np.concatenate([model.w1[e].detach().numpy().ravel(), model.w2[e].detach().numpy().ravel()])
             for e in range(args.n_experts)]
    large = [rng.standard_normal((2, args.big_d, args.big_dff)).astype(np.float32) * 0.02
             for _ in range(args.n_experts)]
    x_big = rng.standard_normal((args.big_batch, args.big_d)).astype(np.float32)
    order_rng = random.Random(seed)
    raw = {size: {cond: [] for cond in conds} for size in SIZES}
    info = {size: {} for size in SIZES}
    for rep in range(args.reps):
        jobs = [(size, cond) for size in SIZES for cond in conds]
        order_rng.shuffle(jobs)  # 時間の揺らぎが条件に偏らないよう毎回順を並べ替える
        for size, cond in jobs:
            big = size == "large"
            kind, val = _parse(cond)
            tmp = Path(tempfile.mkdtemp(prefix="expert_prefetch_outcome_"))
            try:
                pred = agate = None
                if kind in ("outcome", "outcomeshuf"):
                    pred = OutcomeCerebellum(args.n_experts, args.d, args.lr, theta=val, window=args.window,
                                             y_late=args.y_late, shuffle=(kind == "outcomeshuf"),
                                             shuffle_buf=args.shuffle_buf, seed=seed * 1000 + rep)
                    warm_outcome(pred, wc, wh, args.capacity)
                    agate = pred
                elif kind in ("last", "freq") or kind.startswith("cerebellum"):
                    pred = make_predictor("cerebellum" if kind.startswith("cerebellum") else kind,
                                          args.n_experts, args.d, args.lr)
                    if kind == "cerebellum_delayed":
                        agate = DelayedCFGate(args.n_experts, eta_up=val * args.eta_down, eta_down=args.eta_down,
                                              theta0=args.theta0, window=args.window)
                        warm_delayed(pred, wc, wh, agate, args.capacity)
                    else:
                        warm(pred, wc, wh, None, args.capacity)
                        if kind == "cerebellum_efixed":
                            agate = EligibleTopGate(val)
                # 遅延 CF ゲートは 12.6.209節と同じ `CFTaggedStore`（`on_prefetch_used(c)` に late を渡さない）
                store_cls = CFTaggedStore if isinstance(agate, DelayedCFGate) else OutcomeTaggedStore
                store = store_cls(large if big else small, tmp, args.capacity, args.big_io_ms if big else args.io_ms,
                                  n_paths=args.paths,
                                  gate=agate if isinstance(agate, (OutcomeCerebellum, DelayedCFGate)) else None)
                res = simulate_prefetch_v2(sc, sh, store, pred, args.k_pf,
                                           args.big_compute_ms if big else args.compute_ms,
                                           oracle=(cond == "oracle"),
                                           gate=val if kind == "cerebellum_fixed" else None,
                                           parallel_predictor=True, x_batch=x_big if big else None,
                                           adaptive_gate=agate)
                raw[size][cond].append(asdict(res))
                if rep == 0 and isinstance(pred, OutcomeCerebellum):
                    info[size][cond] = pred.stats() | {"top1_after": _top1(pred, sc, sh)}
                elif rep == 0 and cond == "cerebellum":
                    info[size][cond] = {"top1_after": _top1(pred, sc, sh)}
                elif rep == 0 and isinstance(agate, DelayedCFGate):
                    info[size][cond] = {"end": agate.stats()}
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    speed = {size: {cond: {m: statistics.median(r[m] for r in raw[size][cond]) for m in raw[size][cond][0]}
                    | {"wall_s_reps": [r["wall_s"] for r in raw[size][cond]]} for cond in conds}
             for size in SIZES}
    return {"seed": seed, "train_ce": ce, "speed": speed, "info": info}


def curve_conds(conds: list[str]) -> list[str]:
    """正解ラベルの予測器の点（2つの選び方の和集合）。"""
    return ["none", "cerebellum"] + [c for c in conds if _parse(c)[0] in ("cerebellum_fixed", "cerebellum_efixed")]


def residuals(runs: list[dict], conds: list[str], size: str, cond: str, metric: str = "wall_s") -> list[float]:
    """シードごとの、正解ラベルの交換曲線に対する残差。metric="stalls" は待ちの回数で数えた版（参考）。"""
    out = []
    for r in runs:
        sp = r["speed"][size]
        base = sp["none"][metric]

        def red(c):
            return 1 - sp[c][metric] / base if base > 0 else 0.0

        pts = [(sp[c]["wasted_loads"], red(c)) for c in curve_conds(conds)]
        out.append(curve_residual(pts, sp[cond]["wasted_loads"], red(cond)))
    return out


def _test(vals: list[float], thr: float = 0.01) -> dict:
    mean = statistics.mean(vals)
    n_pos = sum(v > 0 for v in vals)
    return {"per_seed": vals, "mean": mean, "n_pos": n_pos, "pass": mean >= thr and n_pos >= 4}


def select_all(s: dict) -> dict:
    sel = {}
    for size in SIZES:
        d = {fam: select_on_dev({c: v for c, v in s[size].items() if _parse(c)[0] in (fam, "freq")} | {
            "freq": s[size]["freq"]}, fam) for fam in FAMILIES}
        union = {c: v for c, v in s[size].items() if _parse(c)[0] in ("cerebellum_fixed", "cerebellum_efixed")}
        d["label_best"] = select_on_dev(union | {"freq": s[size]["freq"]}, "cerebellum_")
        sel[size] = d
    return sel


def judge(runs: list[dict], s: dict, conds: list[str], sel: dict) -> dict:
    out = {}
    for size in SIZES:
        d = sel[size]
        o, sh = d["outcome"], d["outcomeshuf"]
        ro = residuals(runs, conds, size, o)
        grid_o = [c for c in conds if _parse(c)[0] == "outcome"]
        per_theta = [residuals(runs, conds, size, c) for c in grid_o]
        curve_mean = [statistics.mean(v) for v in zip(*per_theta)]
        rs = residuals(runs, conds, size, sh)
        rd = residuals(runs, conds, size, DELAYED)
        out[f"H1_{size}"] = {"cond": o} | _test(ro)
        out[f"H2_{size}"] = {"conds": grid_o, "per_theta_mean": [statistics.mean(v) for v in per_theta]} | _test(
            curve_mean, 0.005)
        out[f"H3_{size}"] = {"cond": o, "shuf": sh, "resid_shuf": statistics.mean(rs)} | _test(
            [x - y for x, y in zip(ro, rs)])
        out[f"H4_{size}"] = {"cond": o, "resid_delayed": statistics.mean(rd)} | _test(
            [x - y for x, y in zip(ro, rd)])
        out[f"H5_{size}"] = h_pass(s[size], o)
        grid_s = [c for c in conds if _parse(c)[0] == "outcomeshuf"]
        out[f"ref_{size}"] = {
            "label_best": h_pass(s[size], d["label_best"]) | {"resid": _test(residuals(runs, conds, size,
                                                                                        d["label_best"]))},
            "delayed": h_pass(s[size], DELAYED) | {"resid": _test(rd)},
            "shuf_curve_mean": statistics.mean(statistics.mean(residuals(runs, conds, size, c)) for c in grid_s),
            "resid_stalls_outcome": _test(residuals(runs, conds, size, o, "stalls")),
            "resid_stalls_delayed": _test(residuals(runs, conds, size, DELAYED, "stalls")),
            "outcome_all": {c: {"resid": statistics.mean(residuals(runs, conds, size, c)),
                                "reduction": s[size][c]["reduction_vs_none"]["mean"],
                                "wasted": s[size][c]["wasted_loads"]["mean"]} for c in grid_o + grid_s},
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    p.add_argument("--n-sentences", type=int, default=3000)
    p.add_argument("--d", type=int, default=48)
    p.add_argument("--n-experts", type=int, default=8)
    p.add_argument("--train-steps", type=int, default=600)
    p.add_argument("--balance-coef", type=float, default=0.05)
    p.add_argument("--trace-len", type=int, default=8000)
    p.add_argument("--warm-len", type=int, default=2000)
    p.add_argument("--speed-len", type=int, default=600)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.2, help="予測器の学習率（正解ラベル・帰結の両方）")
    p.add_argument("--capacity", type=int, default=2)
    p.add_argument("--k-pf", type=int, default=1)
    p.add_argument("--paths", type=int, default=2)
    p.add_argument("--io-ms", type=float, default=2.0)
    p.add_argument("--compute-ms", type=float, default=1.0)
    p.add_argument("--big-d", type=int, default=512)
    p.add_argument("--big-dff", type=int, default=1024)
    p.add_argument("--big-batch", type=int, default=32)
    p.add_argument("--big-io-ms", type=float, default=4.0)
    p.add_argument("--big-compute-ms", type=float, default=2.0)
    p.add_argument("--grid", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--y-late", type=float, default=0.5)
    p.add_argument("--shuffle-buf", type=int, default=64)
    p.add_argument("--eta-down", type=float, default=0.01)
    p.add_argument("--theta0", type=float, default=0.5)
    p.add_argument("--select", type=str, default=None, help="評価用: 開発で確定した選択（JSON）")
    p.add_argument("--out", type=str, default="results/expert_prefetch_outcome/dev.json")
    args = p.parse_args()
    torch.set_num_threads(1)

    conds = conditions(args.grid)
    runs = []
    for seed in args.seeds:
        r = run_seed(seed, args, conds)
        runs.append(r)
        print(f"seed {seed}: ce={r['train_ce']:.3f}", flush=True)
        for size in SIZES:
            print(f"  {size}: " + " ".join(f"{c}={v['wall_s']:.3f}s/w{v['wasted_loads']:.0f}"
                                          for c, v in r["speed"][size].items()), flush=True)
    s = summarize(runs, conds)
    sel = json.loads(args.select) if args.select else select_all(s)
    verdict = {"selected": sel} | judge(runs, s, conds, sel)
    print(json.dumps(verdict, ensure_ascii=False, indent=1))
    for size in SIZES:
        print(size + ": " + " ".join(f"{c}={v['wall_s']['mean']:.3f}s({v['reduction_vs_none']['mean']*100:+.1f}%)"
                                     f"/w{v['wasted_loads']['mean']:.0f}" for c, v in s[size].items()))
    out = {"環境": env_info() | {"OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")},
           "設定": vars(args), "条件": conds, "要約": s, "判定": verdict, "シード別": runs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
