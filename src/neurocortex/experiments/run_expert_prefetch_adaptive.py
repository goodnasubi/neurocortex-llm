"""登上線維の誤差による先読みの閾値のオンライン適応（12.6.193節の設計）。

12.6.189〜190節の構成（8専門家・1層のルーティング列、小脳の順モデルによる予測、読み込み経路2本、
予測器は並列）で、先読みの方針だけを変えて壁時計時間と無駄 I/O を測る:
none、last、freq、cerebellum（ゲートなし）、cerebellum_fixed{θ}（固定閾値）、
cerebellum_adapt{ρ}（`AdaptiveThresholdGate`、専門家ごとの閾値、eta_up = ρ·eta_down）、
cerebellum_adaptglobal{ρ}（閾値を全候補で共有する参考条件）、oracle。

開発用シードでは θ と ρ の候補をすべて測り、事前に決めた規則で1つずつ選ぶ。評価用シードでは
選んだ値だけを測る（`--fixed-theta`・`--adapt-rho`）。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_expert_prefetch_adaptive --seeds 100 101 102 103 104 \\
        --fixed-grid 0.2 0.3 0.4 0.5 --rho-grid 1 2 4 --out results/expert_prefetch_adaptive/dev_rev0.json
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import tempfile
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..data import CharCorpus, synthetic_corpus
from ..expert_prefetch import (AdaptiveThresholdGate, MultiPathExpertStore, RoutedCharLM, make_predictor,
                               routing_trace, simulate_prefetch_v2, train_routed_lm)
from ..run_env import env_info

METRICS = ("wall_s", "stalls", "stall_s", "demand_loads", "prefetch_loads", "wasted_loads", "wasted_bytes",
           "predictor_s")
SIZES = ("small", "large")


def conditions(fixed: list[float], rhos: list[float], with_global: bool) -> list[str]:
    out = ["none", "last", "freq", "cerebellum"]
    out += [f"cerebellum_fixed{th:g}" for th in fixed]
    out += [f"cerebellum_adapt{r:g}" for r in rhos]
    if with_global:
        out += [f"cerebellum_adaptglobal{r:g}" for r in rhos]
    return out + ["oracle"]


def warm(predictor, choices: np.ndarray, hidden: np.ndarray, gate: AdaptiveThresholdGate | None,
         capacity: int) -> None:
    """速度を測る区間の前の列で予測器（と適応閾値）を学習させる。I/O の時間は使わず、
    常駐は容量 `capacity` の LRU で模す（閾値の更新に要る「先読みすれば I/O が起きるか」のため）。"""
    resident: "OrderedDict[int, None]" = OrderedDict()

    def touch(e: int) -> None:
        resident[e] = None
        resident.move_to_end(e)
        while len(resident) > capacity:
            resident.popitem(last=False)

    for t in range(len(choices) - 1):
        e = int(choices[t])
        touch(e)
        if gate is not None:
            if t > 0:
                gate.feedback(e)
            c = gate.decide(predictor.prob(e, hidden[t]), e, lambda x: x in resident)
            if c is not None:
                touch(c)
        predictor.observe(e, hidden[t], int(choices[t + 1]))
    if gate is not None:
        gate.feedback(int(choices[-1]))


def _parse(cond: str) -> tuple[str, float | None]:
    for pre in ("cerebellum_adaptglobal", "cerebellum_adapt", "cerebellum_fixed"):
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
            tmp = Path(tempfile.mkdtemp(prefix="expert_prefetch_adapt_"))
            try:
                store = MultiPathExpertStore(large if big else small, tmp, args.capacity,
                                             args.big_io_ms if big else args.io_ms, n_paths=args.paths)
                pred = gate = None
                if kind in ("last", "freq") or kind.startswith("cerebellum"):
                    pred = make_predictor("cerebellum" if kind.startswith("cerebellum") else kind,
                                          args.n_experts, args.d, args.cb_lr)
                if kind in ("cerebellum_adapt", "cerebellum_adaptglobal"):
                    gate = AdaptiveThresholdGate(args.n_experts, eta_up=val * args.eta_down,
                                                 eta_down=args.eta_down, theta0=args.theta0,
                                                 per_expert=(kind == "cerebellum_adapt"))
                if pred is not None:
                    warm(pred, wc, wh, gate, args.capacity)
                theta_start = None if gate is None else gate.theta.tolist()
                res = simulate_prefetch_v2(sc, sh, store, pred, args.k_pf,
                                           args.big_compute_ms if big else args.compute_ms,
                                           oracle=(cond == "oracle"),
                                           gate=val if kind == "cerebellum_fixed" else None,
                                           parallel_predictor=True, x_batch=x_big if big else None,
                                           adaptive_gate=gate)
                raw[size][cond].append(asdict(res))
                if gate is not None and rep == 0:
                    gate_info[size][cond] = {"theta_after_warm": theta_start, "theta_end": gate.theta.tolist(),
                                             "issued": gate.n_issued, "issued_miss": gate.n_issued_miss,
                                             "skipped_hit": gate.n_skipped_hit}
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    speed = {size: {cond: {m: statistics.median(r[m] for r in raw[size][cond]) for m in METRICS}
                    | {"wall_s_reps": [r["wall_s"] for r in raw[size][cond]]} for cond in conds}
             for size in SIZES}
    return {"seed": seed, "train_ce": ce, "speed": speed, "gate": gate_info}


def _mean_sd(vals):
    return {"mean": statistics.mean(vals), "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0}


def summarize(runs: list[dict], conds: list[str]) -> dict:
    s: dict = {}
    for size in SIZES:
        s[size] = {}
        for cond in conds:
            d = {m: _mean_sd([r["speed"][size][cond][m] for r in runs]) for m in METRICS}
            red = [1 - r["speed"][size][cond]["wall_s"] / r["speed"][size]["none"]["wall_s"] for r in runs]
            d["reduction_vs_none"] = _mean_sd(red)
            d["reduction_per_seed"] = red
            s[size][cond] = d
    return s


def h_pass(s_size: dict, cb: str) -> dict:
    """12.6.189節の H1 と同じ4条件。"""
    sp = s_size
    red = sp[cb]["reduction_vs_none"]["mean"]
    wall_cb = sp[cb]["wall_s"]["mean"]
    simple = min(sp["last"]["wall_s"]["mean"], sp["freq"]["wall_s"]["mean"])
    n_pos = sum(x > 0 for x in sp[cb]["reduction_per_seed"])
    wasted_cb, wasted_freq = sp[cb]["wasted_loads"]["mean"], sp["freq"]["wasted_loads"]["mean"]
    return {"cond": cb, "reduction": red, "wall_cb": wall_cb, "wall_simple_min": simple,
            "seeds_faster_than_none": n_pos, "wasted_cb": wasted_cb, "wasted_freq": wasted_freq,
            "c1_reduction": red >= 0.10, "c2_faster_than_simple": wall_cb < simple, "c3_seeds": n_pos >= 4,
            "c4_wasted": wasted_cb <= wasted_freq,
            "pass": red >= 0.10 and wall_cb < simple and n_pos >= 4 and wasted_cb <= wasted_freq}


def select_on_dev(s_size: dict, prefix: str) -> str | None:
    """事前の規則: 無駄 I/O が freq 以下の候補のうち壁時計の最も短いもの。なければ無駄 I/O の最小のもの。"""
    cands = [c for c in s_size if c.startswith(prefix) and not c.startswith(prefix + "global")]
    if not cands:
        return None
    ok = [c for c in cands if s_size[c]["wasted_loads"]["mean"] <= s_size["freq"]["wasted_loads"]["mean"]]
    if ok:
        return min(ok, key=lambda c: s_size[c]["wall_s"]["mean"])
    return min(cands, key=lambda c: s_size[c]["wasted_loads"]["mean"])


def judge(s: dict, fixed_cond: dict, adapt_cond: dict) -> dict:
    out = {}
    for size, hname in (("small", "H1_small"), ("large", "H2_large")):
        a, f = adapt_cond[size], fixed_cond[size]
        out[hname] = h_pass(s[size], a)
        out[f"fixed_{size}"] = h_pass(s[size], f)
        ra, rf = s[size][a]["reduction_vs_none"]["mean"], s[size][f]["reduction_vs_none"]["mean"]
        out[f"H3_{size}"] = {"adapt": a, "fixed": f, "red_adapt": ra, "red_fixed": rf,
                             "wasted_adapt": s[size][a]["wasted_loads"]["mean"],
                             "wasted_fixed": s[size][f]["wasted_loads"]["mean"], "pass": ra >= rf}
        rc = s[size]["cerebellum"]["reduction_vs_none"]["mean"]
        out[f"H4_{size}"] = {"red_cerebellum": rc, "red_adapt": ra, "keep_ratio": ra / rc if rc > 0 else None,
                             "pass": rc > 0 and ra >= 0.8 * rc}
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
    p.add_argument("--reps", type=int, default=3)
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
    p.add_argument("--fixed-grid", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5])
    p.add_argument("--rho-grid", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    p.add_argument("--eta-down", type=float, default=0.01)
    p.add_argument("--theta0", type=float, default=0.5)
    p.add_argument("--with-global", action="store_true", help="閾値を共有する参考条件も測る")
    p.add_argument("--fixed-small", type=str, default=None, help="評価用: H1 の固定閾値の条件（開発で確定）")
    p.add_argument("--fixed-large", type=str, default=None)
    p.add_argument("--adapt-small", type=str, default=None, help="評価用: H1 の適応閾値の条件（開発で確定）")
    p.add_argument("--adapt-large", type=str, default=None)
    p.add_argument("--out", type=str, default="results/expert_prefetch_adaptive/dev.json")
    args = p.parse_args()
    torch.set_num_threads(1)

    conds = conditions(args.fixed_grid, args.rho_grid, args.with_global)
    runs = []
    for seed in args.seeds:
        r = run_seed(seed, args, conds)
        runs.append(r)
        print(f"seed {seed}: ce={r['train_ce']:.3f}", flush=True)
        for size in SIZES:
            print(f"  {size}: " + " ".join(f"{c}={v['wall_s']:.2f}s/w{v['wasted_loads']:.0f}"
                                          for c, v in r["speed"][size].items()), flush=True)
    s = summarize(runs, conds)
    fixed = {"small": args.fixed_small or select_on_dev(s["small"], "cerebellum_fixed"),
             "large": args.fixed_large or select_on_dev(s["large"], "cerebellum_fixed")}
    adapt = {"small": args.adapt_small or select_on_dev(s["small"], "cerebellum_adapt"),
             "large": args.adapt_large or select_on_dev(s["large"], "cerebellum_adapt")}
    verdict = {"selected": {"fixed": fixed, "adapt": adapt}} | judge(s, fixed, adapt)
    print(json.dumps(verdict, ensure_ascii=False, indent=1))
    for size in SIZES:
        print(size + ": " + " ".join(f"{c}={v['wall_s']['mean']:.3f}s({v['reduction_vs_none']['mean']*100:+.1f}%)"
                                     f"/w{v['wasted_loads']['mean']:.0f}" for c, v in s[size].items()))
    out = {"環境": env_info(), "設定": vars(args), "条件": conds, "要約": s, "判定": verdict, "シード別": runs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
