"""先読みの帰結で学ぶ小脳の順モデルの利得の源の切り分け（12.6.217節の設計）。

12.6.211節の構成のまま、正解ラベルの交換曲線（none・cerebellum・cerebellum_fixed{θ}・cerebellum_efixed{θ}）と、
帰結の予測器の対照（`fam@θ`）を測る:
outcome（12.6.211節と同じ、W=2・1/0.5/0）、outbin（段階づけを除く、1/0）、outltd・outltp（片側の可塑性）、
outw1・outw4（窓 W=1・4）、supw1・supw2（常駐でない候補が W 以内に使われるかの即時の教師あり学習）。

使い方（シードを分けて2本並列に流し、`--merge` でまとめる）:
    OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python -m neurocortex.experiments.run_expert_prefetch_outcome_ablation \\
        --seeds 100 101 102 --out results/expert_prefetch_outcome_ablation/dev_a.json
    PYTHONPATH=src python -m neurocortex.experiments.run_expert_prefetch_outcome_ablation \\
        --merge results/.../dev_a.json results/.../dev_b.json --out results/expert_prefetch_outcome_ablation/dev.json
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
from ..expert_prefetch_delayed import curve_residual
from ..expert_prefetch_outcome import EligibleTopGate, OutcomeCerebellum, OutcomeTaggedStore, warm_outcome
from ..expert_prefetch_outcome_ablation import OneSidedOutcomeCerebellum, SupervisedEligibleCerebellum
from ..run_env import env_info
from .run_expert_prefetch_adaptive import SIZES, h_pass, select_on_dev, summarize, warm
from .run_expert_prefetch_outcome import _test, _top1

LABEL_FAMS = ("cerebellum_fixed", "cerebellum_efixed")
# 族 → (クラスの種類, 窓 W, 待った帰結の値)
FAMS = {
    "outcome": ("outcome", 2, 0.5),
    "outbin": ("outcome", 2, 1.0),
    "outltd": ("ltd", 2, 0.5),
    "outltp": ("ltp", 2, 0.5),
    "outw1": ("outcome", 1, 0.5),
    "outw4": ("outcome", 4, 0.5),
    "supw1": ("sup", 1, 0.5),
    "supw2": ("sup", 2, 0.5),
}


def conditions(label_grid: list[float], grid: list[float], fams=tuple(FAMS)) -> list[str]:
    out = ["none", "last", "freq", "cerebellum"]
    for fam in LABEL_FAMS:
        out += [f"{fam}{th:g}" for th in label_grid]
    for fam in fams:
        out += [f"{fam}@{th:g}" for th in grid]
    return out + ["oracle"]


def parse(cond: str) -> tuple[str, float | None]:
    if "@" in cond:
        fam, th = cond.split("@")
        return fam, float(th)
    for pre in ("cerebellum_efixed", "cerebellum_fixed"):
        if cond.startswith(pre):
            return pre, float(cond[len(pre):])
    return cond, None


def make_outcome_model(fam: str, theta: float, n_experts: int, d: int, lr: float, seed: int) -> OutcomeCerebellum:
    kind, window, y_late = FAMS[fam]
    kw = dict(theta=theta, window=window, y_late=y_late, seed=seed)
    if kind == "sup":
        return SupervisedEligibleCerebellum(n_experts, d, lr, **kw)
    if kind in ("ltd", "ltp"):
        return OneSidedOutcomeCerebellum(n_experts, d, lr, plasticity=kind, **kw)
    return OutcomeCerebellum(n_experts, d, lr, **kw)


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
            kind, val = parse(cond)
            tmp = Path(tempfile.mkdtemp(prefix="expert_prefetch_ablation_"))
            try:
                pred = agate = None
                if kind in FAMS:
                    pred = make_outcome_model(kind, val, args.n_experts, args.d, args.lr, seed * 1000 + rep)
                    if isinstance(pred, SupervisedEligibleCerebellum):
                        pred.set_trace(wc)
                    warm_outcome(pred, wc, wh, args.capacity)
                    if isinstance(pred, SupervisedEligibleCerebellum):
                        pred.set_trace(sc)
                    agate = pred
                elif kind in ("last", "freq") or kind.startswith("cerebellum"):
                    pred = make_predictor("cerebellum" if kind.startswith("cerebellum") else kind,
                                          args.n_experts, args.d, args.lr)
                    warm(pred, wc, wh, None, args.capacity)
                    if kind == "cerebellum_efixed":
                        agate = EligibleTopGate(val)
                store = OutcomeTaggedStore(large if big else small, tmp, args.capacity,
                                           args.big_io_ms if big else args.io_ms, n_paths=args.paths,
                                           gate=agate if isinstance(agate, OutcomeCerebellum) else None)
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
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    speed = {size: {cond: {m: statistics.median(r[m] for r in raw[size][cond]) for m in raw[size][cond][0]}
                    | {"wall_s_reps": [r["wall_s"] for r in raw[size][cond]]} for cond in conds}
             for size in SIZES}
    return {"seed": seed, "train_ce": ce, "speed": speed, "info": info}


def curve_conds(conds: list[str]) -> list[str]:
    """正解ラベルの予測器の点（12.6.211節と同じ、2つの選び方の和集合）。"""
    return ["none", "cerebellum"] + [c for c in conds if parse(c)[0] in LABEL_FAMS]


def residuals(runs: list[dict], conds: list[str], size: str, cond: str, metric: str = "wall_s") -> list[float]:
    """シードごとの、正解ラベルの交換曲線に対する残差（12.6.211節と同じ定義）。"""
    out = []
    for r in runs:
        sp = r["speed"][size]
        base = sp["none"][metric]

        def red(c):
            return 1 - sp[c][metric] / base if base > 0 else 0.0

        pts = [(sp[c]["wasted_loads"], red(c)) for c in curve_conds(conds)]
        out.append(curve_residual(pts, sp[cond]["wasted_loads"], red(cond)))
    return out


def families(conds: list[str]) -> list[str]:
    return [f for f in FAMS if any(parse(c)[0] == f for c in conds)]


def select_all(s: dict, conds: list[str]) -> dict:
    """開発用シードで族ごとに θ を1つ選ぶ（12.6.193節の規則）。label_best は fixed と efixed の和集合から。"""
    sel = {}
    for size in SIZES:
        freq = {"freq": s[size]["freq"]}
        d = {fam: select_on_dev({c: v for c, v in s[size].items() if parse(c)[0] == fam} | freq, fam + "@")
             for fam in families(conds)}
        union = {c: v for c, v in s[size].items() if parse(c)[0] in LABEL_FAMS}
        d["label_best"] = select_on_dev(union | freq, "cerebellum_")
        sel[size] = d
    return sel


# 仮説 → (引かれる族, 引く族)。引く族が None なら残差そのもの
HYPS = {
    "R0": ("outcome", None),
    "H1": ("supw2", None),
    "H2": ("outcome", "supw2"),
    "H2b": ("outbin", "supw2"),
    "H3": ("outcome", "outbin"),
    "H4a": ("outcome", "outltd"),
    "H4b": ("outcome", "outltp"),
    "H4c": ("outltd", "outltp"),
    "H5a": ("outcome", "outw1"),
    "H5b": ("outcome", "outw4"),
}
REFS = {"ref_a_actionable": ("supw1", "label_best"), "ref_b_multistep": ("supw2", "supw1")}


def judge(runs: list[dict], s: dict, conds: list[str], sel: dict) -> dict:
    out = {}
    for size in SIZES:
        d = sel[size]
        res = {k: residuals(runs, conds, size, c) for k, c in d.items() if c is not None}
        res_st = {k: residuals(runs, conds, size, c, "stalls") for k, c in d.items() if c is not None}
        for name, (a, b) in (HYPS | REFS).items():
            if a not in res or (b is not None and b not in res):
                continue
            vals = res[a] if b is None else [x - y for x, y in zip(res[a], res[b])]
            st = res_st[a] if b is None else [x - y for x, y in zip(res_st[a], res_st[b])]
            out[f"{name}_{size}"] = {"a": d[a], "b": d.get(b)} | _test(vals) | {"stalls": _test(st)}
        h5 = [out.get(f"H5a_{size}"), out.get(f"H5b_{size}")]
        if all(h5):
            out[f"H5_{size}"] = {"pass": h5[0]["pass"] and h5[1]["pass"]}
        out[f"ref_{size}"] = {
            "resid_mean": {k: statistics.mean(v) for k, v in res.items()},
            "resid_stalls_mean": {k: statistics.mean(v) for k, v in res_st.items()},
            "speed": {k: h_pass(s[size], c) for k, c in d.items() if c is not None},
            "curve_mean": {fam: statistics.mean(statistics.mean(residuals(runs, conds, size, c))
                                                for c in conds if parse(c)[0] == fam) for fam in families(conds)},
            "all": {c: {"resid": statistics.mean(residuals(runs, conds, size, c)),
                        "reduction": s[size][c]["reduction_vs_none"]["mean"],
                        "wasted": s[size][c]["wasted_loads"]["mean"]}
                    for c in conds if parse(c)[0] in FAMS},
        }
    return out


def finalize(runs: list[dict], conds: list[str], args_dict: dict, select: dict | None, out_path: str) -> dict:
    s = summarize(runs, conds)
    sel = select if select else select_all(s, conds)
    verdict = {"selected": sel} | judge(runs, s, conds, sel)
    print(json.dumps({k: v for k, v in verdict.items() if not k.startswith("ref_")}, ensure_ascii=False, indent=1))
    for size in SIZES:
        print(size + ": " + " ".join(f"{c}={v['wall_s']['mean']:.3f}s({v['reduction_vs_none']['mean']*100:+.1f}%)"
                                     f"/w{v['wasted_loads']['mean']:.0f}" for c, v in s[size].items()))
    out = {"環境": env_info() | {"OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")},
           "設定": args_dict, "条件": conds, "要約": s, "判定": verdict, "シード別": runs}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return verdict


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
    p.add_argument("--lr", type=float, default=0.2)
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
    p.add_argument("--label-grid", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    p.add_argument("--grid", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    p.add_argument("--fams", type=str, nargs="+", default=list(FAMS))
    p.add_argument("--select", type=str, default=None, help="評価用: 開発で確定した選択（JSON）")
    p.add_argument("--merge", type=str, nargs="+", default=None, help="シードを分けて流した結果をまとめる")
    p.add_argument("--no-judge", action="store_true", help="シードを分けて流すとき判定を省く")
    p.add_argument("--out", type=str, default="results/expert_prefetch_outcome_ablation/dev.json")
    args = p.parse_args()
    torch.set_num_threads(1)
    select = json.loads(args.select) if args.select else None

    if args.merge:
        parts = [json.loads(Path(f).read_text()) for f in args.merge]
        conds = parts[0]["条件"]
        if any(pt["条件"] != conds for pt in parts):
            raise ValueError("条件が揃っていない")
        runs = sorted((r for pt in parts for r in pt["シード別"]), key=lambda r: r["seed"])
        cfg = parts[0]["設定"] | {"seeds": [r["seed"] for r in runs], "merged_from": args.merge}
        finalize(runs, conds, cfg, select, args.out)
        return

    conds = conditions(args.label_grid, args.grid, tuple(args.fams))
    runs = []
    for seed in args.seeds:
        r = run_seed(seed, args, conds)
        runs.append(r)
        print(f"seed {seed}: ce={r['train_ce']:.3f}", flush=True)
        for size in SIZES:
            print(f"  {size}: " + " ".join(f"{c}={v['wall_s']:.3f}s/w{v['wasted_loads']:.0f}"
                                          for c, v in r["speed"][size].items()), flush=True)
    if args.no_judge:
        out = {"環境": env_info() | {"OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")},
               "設定": vars(args), "条件": conds, "シード別": runs}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        return
    finalize(runs, conds, vars(args), select, args.out)


if __name__ == "__main__":
    main()
