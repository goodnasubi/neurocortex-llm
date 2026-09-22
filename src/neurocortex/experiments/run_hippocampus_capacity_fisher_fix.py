"""ステップ13（海馬モジュール — フィッシャー崩壊への対策の検証, 12.6.34節）の主実験。

ステップ12（12.6.32〜33節）と完全に同一のタスク設定（統制1）のもとで、6条件
（unbounded / fifo / uniform-compress / importance-weighted / importance-weighted-fisher-decay /
importance-weighted-loss-based）を比較する。動作点はステップ12で確認済みのため、
本ステップでは段階1の動作点探索を行わず、本実験のみを実行する。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_hippocampus_capacity_fisher_fix
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..hippocampus_capacity import CONDITIONS, make_task_specs, run_condition


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-tasks", type=int, default=5)
    ap.add_argument("--n-inputs", type=int, default=10)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--snapshot-size", type=int, default=60)
    ap.add_argument("--capacity", type=int, default=120,
                    help="ステップ12（12.6.32節）で確認済みの動作点をそのまま使う")
    ap.add_argument("--task-steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--fisher-batches", type=int, default=10)
    ap.add_argument("--fisher-batch-size", type=int, default=32)
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--fisher-decay-weight-decay", type=float, default=0.01,
                    help="ステップ10のEWC事前学習と同じ値（12.6.29節）")
    ap.add_argument("--out", type=Path,
                    default=Path("results/hippocampus_capacity/fisher_collapse_mitigations.json"))
    args = ap.parse_args()

    t0 = time.time()
    task_specs = make_task_specs(args.n_tasks, args.n_inputs)

    all_results: dict[str, dict] = {c: {"mean_old_task_acc": [], "oldest_task_acc": []}
                                    for c in CONDITIONS}
    for condition in CONDITIONS:
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            g = torch.Generator().manual_seed(seed + 70_000)  # ステップ12と同一のシード列
            r = run_condition(condition, task_specs, args.hidden_dim, args.capacity,
                              args.snapshot_size, args.task_steps, args.batch_size, args.lr,
                              args.fisher_batches, args.fisher_batch_size, args.n_eval, g,
                              fisher_decay_weight_decay=args.fisher_decay_weight_decay)
            all_results[condition]["mean_old_task_acc"].append(r.mean_old_task_acc)
            all_results[condition]["oldest_task_acc"].append(r.oldest_task_acc)

    summary: dict[str, dict] = {}
    for condition in CONDITIONS:
        h1 = torch.tensor(all_results[condition]["mean_old_task_acc"])
        h2 = torch.tensor(all_results[condition]["oldest_task_acc"])
        summary[condition] = {
            "h1_mean_old_task_acc": {"mean": float(h1.mean()), "std": float(h1.std(unbiased=False)),
                                     "per_seed": h1.tolist()},
            "h2_oldest_task_acc": {"mean": float(h2.mean()), "std": float(h2.std(unbiased=False)),
                                   "per_seed": h2.tolist()},
        }

    # 主張(強い形, L2): 対策版importance-weightedのいずれかがfifo・uniform-compressの両方を上回るか。
    fifo_h1 = summary["fifo"]["h1_mean_old_task_acc"]["mean"]
    uniform_h1 = summary["uniform-compress"]["h1_mean_old_task_acc"]["mean"]
    baseline_h1 = summary["importance-weighted"]["h1_mean_old_task_acc"]["mean"]
    claim_strong_ok = {}
    claim_weak_improvement_ok = {}
    for cond in ("importance-weighted-fisher-decay", "importance-weighted-loss-based"):
        h1 = summary[cond]["h1_mean_old_task_acc"]["mean"]
        claim_strong_ok[cond] = bool(h1 > fifo_h1 and h1 > uniform_h1)
        claim_weak_improvement_ok[cond] = bool(h1 > baseline_h1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "ステップ13 海馬モジュール（フィッシャー崩壊への対策の検証）: "
                "ステップ12（12.6.32〜33節）と同一のタスク設定で、"
                "importance-weighted-fisher-decay（本設計a）・importance-weighted-loss-based"
                "（本設計b）が、fifo・uniform-compress（統制群）およびステップ12の"
                "importance-weighted（対策なし、比較の基準点）と比べてどう変わるかを検証する"
                "（12.6.34節）",
        "条件": vars(args) | {"out": str(args.out)},
        "summary": summary,
        "claim_strong_ok（fifo・uniform-compressの両方を上回るか）": claim_strong_ok,
        "claim_weak_improvement_ok（対策なしimportance-weightedより改善したか）": claim_weak_improvement_ok,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n書き出し: {args.out}")
    print("-- 本実験 --")
    for condition in CONDITIONS:
        s = summary[condition]
        h1, h2 = s["h1_mean_old_task_acc"], s["h2_oldest_task_acc"]
        print(f"{condition:>34}  H1={h1['mean']:.3f}±{h1['std']:.3f}  H2={h2['mean']:.3f}±{h2['std']:.3f}")
    print("-- 主張（強い形: fifo・uniform-compressの両方を上回るか） --")
    for cond, ok in claim_strong_ok.items():
        print(f"  {cond}: {ok}")
    print("-- 副次的主張（弱い形: 対策なしimportance-weightedより改善したか） --")
    for cond, ok in claim_weak_improvement_ok.items():
        print(f"  {cond}: {ok}")


if __name__ == "__main__":
    main()
