"""ハブ専門家への負荷集中（12.6.123節の設計）の主実験。

負荷分散損失の係数ごとに `RoutedCharLM` を学習し、(a) 専門家ごとの負荷の偏り、
(b) ハブ（負荷上位 k 個）の除去と非ハブ k 個の除去での検証 CE の悪化、を測る。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_hub_load --seeds 100 101 102 103 104 \
        --out results/hub_load/dev.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from ..data import CharCorpus, synthetic_corpus
from ..expert_prefetch import RoutedCharLM, train_routed_lm
from ..hub_load import (expert_choices, functional_degree, gini, lesion_study, make_windows,
                        max_over_mean, normalized_entropy, spearman, transition_matrix)
from ..run_env import env_info

MODES = ("zero", "reroute")


def run_one(seed: int, coef: float, corpus: CharCorpus, args: argparse.Namespace) -> dict:
    torch.manual_seed(seed)  # 係数が違っても初期値・データ順は同じ
    model = RoutedCharLM(corpus.vocab_size, d=args.d, n_experts=args.n_experts)
    g = torch.Generator().manual_seed(seed)
    train_ce = train_routed_lm(model, corpus.train, args.train_steps, 32, args.seq_len, 3e-3, coef, g)
    model.eval()
    x, y = make_windows(corpus.val, args.seq_len, args.n_windows)
    choices = expert_choices(model, x)
    load = np.bincount(choices.numpy().ravel(), minlength=args.n_experts).astype(np.float64)
    trans = transition_matrix(choices, args.n_experts)
    degree, strength = functional_degree(trans)
    out = {
        "train_ce": train_ce,
        "load": load.tolist(),
        "gini": gini(load), "max_over_mean": max_over_mean(load), "entropy": normalized_entropy(load),
        "n_used": int((load > 0).sum()),
        "degree": degree.tolist(), "strength": strength.tolist(),
        "spearman_load_degree": spearman(load, degree), "spearman_load_strength": spearman(load, strength),
        "lesion": {},
    }
    for mode in MODES:
        r = lesion_study(model, x, y, load, args.k, mode)
        r["ratio"] = r["d_target"] / max(r["d_random_mean"], args.eps)
        out["lesion"][mode] = r
    return out


def summarize(runs: list[dict], coefs: list[float], eps: float) -> dict:
    def ms(vals):
        # 定義できない値（nan。例: 結合相手の数が全専門家で同じときの順位相関）は除き、使った数を残す
        v = [float(a) for a in vals if not np.isnan(a)]
        if not v:
            return {"mean": float("nan"), "sd": float("nan"), "n": 0}
        return {"mean": statistics.mean(v), "sd": statistics.stdev(v) if len(v) > 1 else 0.0, "n": len(v)}

    s: dict = {}
    for c in coefs:
        key = f"coef{c:g}"
        rs = [r[key] for r in runs]
        d = {m: ms([r[m] for r in rs]) for m in ("gini", "max_over_mean", "entropy", "n_used", "train_ce",
                                                  "spearman_load_degree", "spearman_load_strength")}
        for mode in MODES:
            ls = [r["lesion"][mode] for r in rs]
            dt = [l["d_target"] for l in ls]
            dr = [l["d_random_mean"] for l in ls]
            d[mode] = {"base_ce": ms([l["base_ce"] for l in ls]), "d_target": ms(dt), "d_random": ms(dr),
                       "ratio_of_means": statistics.mean(dt) / max(statistics.mean(dr), eps),
                       "ratio_per_seed": [l["ratio"] for l in ls],
                       "target_gt_random": sum(a > b for a, b in zip(dt, dr)),
                       "spearman_load_single": ms([l["spearman_load_single"] for l in ls]),
                       "hub_load_frac": ms([l["hub_load_frac"] for l in ls]),
                       "d_target_per_load": ms([l["d_target_per_load"] for l in ls]),
                       "d_random_per_load": ms([l["d_random_per_load"] for l in ls])}
        s[key] = d
    return s


def judge(runs: list[dict], s: dict) -> dict:
    """12.6.123節の合格条件（主除去 = 出力の除去、係数 0 と 0.05 の比較）。"""
    n = len(runs)
    g0 = [r["coef0"]["gini"] for r in runs]
    g1 = [r["coef0.05"]["gini"] for r in runs]
    h1 = statistics.mean(g0) - statistics.mean(g1) >= 0.10 and all(a > b for a, b in zip(g0, g1))
    z0 = s["coef0"]["zero"]
    h2 = z0["ratio_of_means"] >= 2.0 and z0["target_gt_random"] == n
    r0 = z0["ratio_per_seed"]
    r1 = s["coef0.05"]["zero"]["ratio_per_seed"]
    h3 = sum(b < a for a, b in zip(r0, r1)) >= 4 and statistics.mean(r1) < statistics.mean(r0)
    return {"H1_補助損失なしで負荷が集中": h1, "H2_ハブ破壊が非ハブ破壊より悪化": h2,
            "H3_補助損失で非対称性が減る": h3}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    p.add_argument("--coefs", type=float, nargs="+", default=[0.0, 0.01, 0.05])
    p.add_argument("--n-sentences", type=int, default=3000)
    p.add_argument("--d", type=int, default=48)
    p.add_argument("--n-experts", type=int, default=8)
    p.add_argument("--train-steps", type=int, default=600)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--n-windows", type=int, default=64)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--out", type=str, default="results/hub_load/dev.json")
    args = p.parse_args()

    runs = []
    for seed in args.seeds:
        corpus = CharCorpus(synthetic_corpus(n_sentences=args.n_sentences, seed=seed))
        r = {"seed": seed}
        for c in args.coefs:
            r[f"coef{c:g}"] = run_one(seed, c, corpus, args)
            z = r[f"coef{c:g}"]
            print(f"seed={seed} coef={c:g} gini={z['gini']:.3f} used={z['n_used']} "
                  f"base={z['lesion']['zero']['base_ce']:.3f} "
                  f"dT={z['lesion']['zero']['d_target']:.3f} dR={z['lesion']['zero']['d_random_mean']:.3f} "
                  f"ratio={z['lesion']['zero']['ratio']:.2f} "
                  f"perload T={z['lesion']['zero']['d_target_per_load']:.3f} "
                  f"R={z['lesion']['zero']['d_random_per_load']:.3f} | reroute dT={z['lesion']['reroute']['d_target']:.3f} "
                  f"dR={z['lesion']['reroute']['d_random_mean']:.3f}", flush=True)
        runs.append(r)
    s = summarize(runs, args.coefs, args.eps)
    verdict = judge(runs, s) if {0.0, 0.05} <= set(args.coefs) else {}
    out = {"環境": env_info(), "設定": vars(args), "要約": s, "判定": verdict, "シード別": runs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({"要約": s, "判定": verdict}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
