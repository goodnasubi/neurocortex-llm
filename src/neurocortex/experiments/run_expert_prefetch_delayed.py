"""遅れて届く登上線維の誤差と適格性の窓による先読みの閾値の適応（12.6.209節の設計）。

12.6.193節の構成（8専門家・1層のルーティング列、小脳の順モデル、予測器は並列、読み込み経路2本）で、
先読みの方針だけを変えて壁時計時間と無駄 I/O を測る:
none、last、freq、cerebellum（ゲートなし）、cerebellum_fixed{θ}（固定閾値。交換曲線を引く）、
cerebellum_adapt2（12.6.193節の次の時刻の誤差）、cerebellum_delayed{ρ}（`DelayedCFGate`、W=2）、
cerebellum_delayedw1{ρ}（参考、W=1）、oracle。

開発用シードでは ρ の候補をすべて測り、事前に決めた規則で1つ選ぶ。評価用シードでは選んだ ρ だけを測る。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_expert_prefetch_delayed --seeds 100 101 102 103 104 \\
        --rho-grid 1 2 4 --out results/expert_prefetch_delayed/dev_rev0.json
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
from ..expert_prefetch import AdaptiveThresholdGate, RoutedCharLM, make_predictor, routing_trace, \
    simulate_prefetch_v2, train_routed_lm
from ..expert_prefetch_delayed import CFTaggedStore, DelayedCFGate, curve_residual, warm_delayed
from ..run_env import env_info
from .run_expert_prefetch_adaptive import SIZES, h_pass, summarize, warm

ADAPT = "cerebellum_adapt2"


def conditions(fixed: list[float], rhos: list[float], with_w1: bool) -> list[str]:
    out = ["none", "last", "freq", "cerebellum"]
    out += [f"cerebellum_fixed{th:g}" for th in fixed]
    out += [ADAPT]
    out += [f"cerebellum_delayed{r:g}" for r in rhos]
    if with_w1:
        out += [f"cerebellum_delayedw1{r:g}" for r in rhos]
    return out + ["oracle"]


def _parse(cond: str) -> tuple[str, float | None]:
    for pre in ("cerebellum_delayedw1", "cerebellum_delayed", "cerebellum_adapt", "cerebellum_fixed"):
        if cond.startswith(pre):
            return pre, float(cond[len(pre):])
    return cond, None


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
    gate_info = {size: {} for size in SIZES}
    for rep in range(args.reps):
        jobs = [(size, cond) for size in SIZES for cond in conds]
        order_rng.shuffle(jobs)  # 時間の揺らぎが条件に偏らないよう毎回順を並べ替える
        for size, cond in jobs:
            big = size == "large"
            kind, val = _parse(cond)
            tmp = Path(tempfile.mkdtemp(prefix="expert_prefetch_delayed_"))
            try:
                pred = agate = dgate = None
                if kind in ("last", "freq") or kind.startswith("cerebellum"):
                    pred = make_predictor("cerebellum" if kind.startswith("cerebellum") else kind,
                                          args.n_experts, args.d, args.cb_lr)
                if kind == "cerebellum_adapt":
                    agate = AdaptiveThresholdGate(args.n_experts, eta_up=val * args.eta_down,
                                                  eta_down=args.eta_down, theta0=args.theta0)
                if kind in ("cerebellum_delayed", "cerebellum_delayedw1"):
                    dgate = DelayedCFGate(args.n_experts, eta_up=val * args.eta_down, eta_down=args.eta_down,
                                          theta0=args.theta0,
                                          window=1 if kind == "cerebellum_delayedw1" else args.window)
                if dgate is not None:
                    warm_delayed(pred, wc, wh, dgate, args.capacity)
                elif pred is not None:
                    warm(pred, wc, wh, agate, args.capacity)
                theta_start = None
                if agate is not None:
                    theta_start = agate.theta.tolist()
                if dgate is not None:
                    theta_start = dgate.theta.tolist()
                    warm_stats = dgate.stats()
                store = CFTaggedStore(large if big else small, tmp, args.capacity,
                                      args.big_io_ms if big else args.io_ms, n_paths=args.paths, gate=dgate)
                res = simulate_prefetch_v2(sc, sh, store, pred, args.k_pf,
                                           args.big_compute_ms if big else args.compute_ms,
                                           oracle=(cond == "oracle"),
                                           gate=val if kind == "cerebellum_fixed" else None,
                                           parallel_predictor=True, x_batch=x_big if big else None,
                                           adaptive_gate=dgate if dgate is not None else agate)
                raw[size][cond].append(asdict(res))
                if rep == 0 and agate is not None:
                    gate_info[size][cond] = {"theta_after_warm": theta_start, "theta_end": agate.theta.tolist(),
                                             "issued": agate.n_issued, "issued_miss": agate.n_issued_miss,
                                             "skipped_hit": agate.n_skipped_hit}
                if rep == 0 and dgate is not None:
                    gate_info[size][cond] = {"theta_after_warm": theta_start, "warm": warm_stats,
                                             "end": dgate.stats()}
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    speed = {size: {cond: {m: statistics.median(r[m] for r in raw[size][cond]) for m in raw[size][cond][0]}
                    | {"wall_s_reps": [r["wall_s"] for r in raw[size][cond]]} for cond in conds}
             for size in SIZES}
    return {"seed": seed, "train_ce": ce, "speed": speed, "gate": gate_info}


def select_rho(s_size: dict) -> str:
    """事前の規則（12.6.193節と同じ）: 無駄 I/O が freq 以下の候補のうち壁時計の最も短いもの。
    なければ無駄 I/O の最小のもの。候補は W=2 の delayed だけ。"""
    cands = [c for c in s_size if c.startswith("cerebellum_delayed") and not c.startswith("cerebellum_delayedw1")]
    ok = [c for c in cands if s_size[c]["wasted_loads"]["mean"] <= s_size["freq"]["wasted_loads"]["mean"]]
    if ok:
        return min(ok, key=lambda c: s_size[c]["wall_s"]["mean"])
    return min(cands, key=lambda c: s_size[c]["wasted_loads"]["mean"])


def curve_conds(conds: list[str]) -> list[str]:
    return ["none", "cerebellum"] + [c for c in conds if c.startswith("cerebellum_fixed")]


def residuals(runs: list[dict], conds: list[str], size: str, cond: str, metric: str = "wall_s") -> list[float]:
    """シードごとの、固定閾値の交換曲線に対する残差。metric="stalls" は待ちの回数で数えた版（参考）。"""
    out = []
    for r in runs:
        sp = r["speed"][size]
        base = sp["none"][metric]

        def red(c):
            return 1 - sp[c][metric] / base if base > 0 else 0.0

        pts = [(sp[c]["wasted_loads"], red(c)) for c in curve_conds(conds)]
        out.append(curve_residual(pts, sp[cond]["wasted_loads"], red(cond)))
    return out


def _resid_test(vals: list[float]) -> dict:
    mean = statistics.mean(vals)
    n_pos = sum(v > 0 for v in vals)
    return {"per_seed": vals, "mean": mean, "n_pos": n_pos, "pass": mean >= 0.01 and n_pos >= 4}


def judge(runs: list[dict], s: dict, conds: list[str], sel: dict) -> dict:
    out = {}
    for size, hname in (("small", "H1_small"), ("large", "H2_large")):
        d = sel[size]
        out[hname] = h_pass(s[size], d)
        rd = residuals(runs, conds, size, d)
        ra = residuals(runs, conds, size, ADAPT)
        out[f"H3_{size}"] = {"cond": d} | _resid_test(rd)
        out[f"H4_{size}"] = {"cond": d, "resid_adapt": statistics.mean(ra)} | _resid_test(
            [x - y for x, y in zip(rd, ra)])
        ref = {"resid_adapt": _resid_test(ra),
               "resid_stalls_delayed": _resid_test(residuals(runs, conds, size, d, "stalls")),
               "resid_stalls_adapt": _resid_test(residuals(runs, conds, size, ADAPT, "stalls")),
               "fixed_pass": {c: h_pass(s[size], c)["pass"] for c in conds if c.startswith("cerebellum_fixed")},
               "adapt_pass": h_pass(s[size], ADAPT)}
        w1 = d.replace("cerebellum_delayed", "cerebellum_delayedw1")
        if w1 in conds:
            rw = residuals(runs, conds, size, w1)
            ref["w1"] = {"cond": w1, "h": h_pass(s[size], w1)} | _resid_test(rw)
            ref["w2_minus_w1"] = _resid_test([x - y for x, y in zip(rd, rw)])
        out[f"ref_{size}"] = ref
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
    p.add_argument("--cb-lr", type=float, default=0.2)
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
    p.add_argument("--fixed-grid", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    p.add_argument("--rho-grid", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--eta-down", type=float, default=0.01)
    p.add_argument("--theta0", type=float, default=0.5)
    p.add_argument("--no-w1", action="store_true", help="W=1 の参考条件を測らない")
    p.add_argument("--delayed-small", type=str, default=None, help="評価用: H1 の条件（開発で確定）")
    p.add_argument("--delayed-large", type=str, default=None)
    p.add_argument("--out", type=str, default="results/expert_prefetch_delayed/dev.json")
    args = p.parse_args()
    torch.set_num_threads(1)

    conds = conditions(args.fixed_grid, args.rho_grid, not args.no_w1)
    runs = []
    for seed in args.seeds:
        r = run_seed(seed, args, conds)
        runs.append(r)
        print(f"seed {seed}: ce={r['train_ce']:.3f}", flush=True)
        for size in SIZES:
            print(f"  {size}: " + " ".join(f"{c}={v['wall_s']:.3f}s/w{v['wasted_loads']:.0f}"
                                          for c, v in r["speed"][size].items()), flush=True)
    s = summarize(runs, conds)
    sel = {"small": args.delayed_small or select_rho(s["small"]),
           "large": args.delayed_large or select_rho(s["large"])}
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
