"""ステップ6（海馬モジュール — パターン補完, 12.6.19節）の主実験。

hopfield（複数回反復）・single-step（対照群、統制5。注意機構と同型）・
knn（対照群、既存手法）の3条件で、手がかりの欠損率を振りながら想起精度を比較する。
主張: 反復ダイナミクスは、欠損率が高い（手がかりが曖昧な）条件でsingle-stepより
優れる。パラメータ・数式はhopfieldとsingle-stepで完全に共有し、反復回数だけが違う。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_hopfield --seeds 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..hopfield import (
    MemorySpec,
    hopfield_retrieve,
    knn_retrieve,
    make_cue,
    make_patterns,
    nearest_pattern_index,
)

CONDITIONS = ("hopfield", "single-step", "knn")


def run_trial(condition: str, K: torch.Tensor, idx: int, corruption_rate: float,
              beta: float, n_iters: int, generator: torch.Generator) -> bool:
    cue = make_cue(K, idx, corruption_rate, generator)
    if condition == "hopfield":
        out, _ = hopfield_retrieve(cue, K, beta, n_iters)
    elif condition == "single-step":
        out, _ = hopfield_retrieve(cue, K, beta, 1)
    elif condition == "knn":
        out = knn_retrieve(cue, K)
    else:
        raise ValueError(condition)
    return nearest_pattern_index(out, K) == idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--n-patterns", type=int, nargs="+", default=[16, 48])
    ap.add_argument("--beta", type=float, default=8.0)
    ap.add_argument("--n-iters", type=int, default=8)
    ap.add_argument("--corruption-rates", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.625, 0.75, 0.875])
    ap.add_argument("--trials-per-condition", type=int, default=200)
    ap.add_argument("--control1-threshold", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=Path("results/hopfield/completion_vs_single_step.json"))
    args = ap.parse_args()

    t0 = time.time()
    rows: list[dict] = []
    for n_patterns in args.n_patterns:
        spec = MemorySpec(dim=args.dim, n_patterns=n_patterns, beta=args.beta)
        for seed in range(args.seeds):
            K = make_patterns(spec, seed=seed)
            g = torch.Generator().manual_seed(seed + 90_000)
            for corruption_rate in args.corruption_rates:
                for condition in CONDITIONS:
                    correct = 0
                    for _ in range(args.trials_per_condition):
                        idx = int(torch.randint(n_patterns, (1,), generator=g))
                        if run_trial(condition, K, idx, corruption_rate, args.beta,
                                    args.n_iters, g):
                            correct += 1
                    rows.append({
                        "n_patterns": n_patterns,
                        "seed": seed,
                        "corruption_rate": corruption_rate,
                        "condition": condition,
                        "accuracy": correct / args.trials_per_condition,
                    })

    def key(n_patterns: int, corruption_rate: float, condition: str) -> str:
        return f"n{n_patterns}_c{corruption_rate}_{condition}"

    summary: dict[str, dict] = {}
    for n_patterns in args.n_patterns:
        for corruption_rate in args.corruption_rates:
            for condition in CONDITIONS:
                vals = torch.tensor([r["accuracy"] for r in rows
                                     if r["n_patterns"] == n_patterns
                                     and r["corruption_rate"] == corruption_rate
                                     and r["condition"] == condition])
                summary[key(n_patterns, corruption_rate, condition)] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std(unbiased=False)),
                }

    # 統制1: 欠損率0（易しい条件）での3条件の想起精度が同等であること。
    control1_diffs = []
    for n_patterns in args.n_patterns:
        accs0 = [summary[key(n_patterns, 0.0, c)]["mean"] for c in CONDITIONS]
        control1_diffs.append(max(accs0) - min(accs0))
    control1_ok = max(control1_diffs) < args.control1_threshold
    if not control1_ok:
        print(f"警告: 統制1（易しい条件での性能統制）が崩れています。"
              f" 最大条件間差: {max(control1_diffs):.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "ステップ6 海馬モジュール（パターン補完）: Hopfield反復更新 対 "
                "1ステップ照合（統制5）対 kNN（既存手法）の欠損率別想起精度比較"
                "（12.6.19節）",
        "条件": vars(args) | {"out": str(args.out)},
        "control1_ok": control1_ok,
        "summary": summary,
        "rows": rows,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n書き出し: {args.out}")
    for n_patterns in args.n_patterns:
        print(f"\n-- n_patterns={n_patterns} --")
        for corruption_rate in args.corruption_rates:
            line = f"  corruption={corruption_rate:.3f}  "
            for condition in CONDITIONS:
                s = summary[key(n_patterns, corruption_rate, condition)]
                line += f"{condition}={s['mean']:.3f}±{s['std']:.3f}  "
            print(line)


if __name__ == "__main__":
    main()
