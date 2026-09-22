"""ステップ12（海馬モジュール — リプレイバッファの容量制約下での刈り込み戦略, 12.6.32節）の主実験。

段階1: unbounded条件でタスク数を段階的に増やし、多タスクXORパリティ課題自体が
       解ける（統制1の前提）ことと、容量制約下（fifo）で実際に溢れが起きることを
       軽量に確認する（実施前メモで予告した動作点探索）。
段階2: 段階1で選んだ設定のもとで、4条件（unbounded / fifo / uniform-compress /
       importance-weighted）を比較する主実験を行う。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_hippocampus_capacity
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..hippocampus_capacity import CONDITIONS, make_task_specs, run_condition


def probe_capacity(n_tasks: int, n_inputs: int, hidden_dim: int, snapshot_size: int,
                   task_steps: int, batch_size: int, lr: float, fisher_batches: int,
                   fisher_batch_size: int, eval_n: int, capacities: list[int],
                   seeds: int) -> dict:
    """段階1: unbounded（参考上限）とfifo（溢れの発生確認）を、容量候補ごとに軽く比較する。"""
    task_specs = make_task_specs(n_tasks, n_inputs)
    results = {}
    for capacity in capacities:
        row = {}
        for condition in ("unbounded", "fifo"):
            accs = []
            for seed in range(seeds):
                torch.manual_seed(seed)
                g = torch.Generator().manual_seed(seed)
                r = run_condition(condition, task_specs, hidden_dim, capacity, snapshot_size,
                                  task_steps, batch_size, lr, fisher_batches, fisher_batch_size,
                                  eval_n, g)
                accs.append(r.mean_old_task_acc)
            row[condition] = sum(accs) / len(accs)
        results[capacity] = row
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-tasks", type=int, default=5)
    ap.add_argument("--n-inputs", type=int, default=10)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--snapshot-size", type=int, default=60,
                    help="タスク収束時に切り出す生サンプル数（各タスク共通）")
    ap.add_argument("--capacity", type=int, default=120,
                    help="バッファの総容量（2タスク分相当）。段階1のプローブで選定")
    ap.add_argument("--task-steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--fisher-batches", type=int, default=10)
    ap.add_argument("--fisher-batch-size", type=int, default=32)
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--control1-threshold", type=float, default=0.1,
                    help="unboundedがこの正解率を下回れば、課題設定自体が解けないとみなす")
    ap.add_argument("--out", type=Path,
                    default=Path("results/hippocampus_capacity/capacity_pruning_strategies.json"))
    args = ap.parse_args()

    t0 = time.time()
    task_specs = make_task_specs(args.n_tasks, args.n_inputs)

    probe = probe_capacity(args.n_tasks, args.n_inputs, args.hidden_dim, args.snapshot_size,
                           task_steps=200, batch_size=64, lr=0.05, fisher_batches=5,
                           fisher_batch_size=32, eval_n=500,
                           capacities=[60, 90, 120, 180, args.n_tasks * args.snapshot_size],
                           seeds=3)

    all_results: dict[str, dict] = {c: {"mean_old_task_acc": [], "oldest_task_acc": []}
                                    for c in CONDITIONS}
    for condition in CONDITIONS:
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            g = torch.Generator().manual_seed(seed + 70_000)
            r = run_condition(condition, task_specs, args.hidden_dim, args.capacity,
                              args.snapshot_size, args.task_steps, args.batch_size, args.lr,
                              args.fisher_batches, args.fisher_batch_size, args.n_eval, g)
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

    # 統制1: unbounded（参考上限）がチャンスを明確に上回ること（多タスク保持自体が可能なことの確認）。
    unbounded_h1 = summary["unbounded"]["h1_mean_old_task_acc"]["mean"]
    control1_ok = unbounded_h1 > (0.5 + args.control1_threshold)
    if not control1_ok:
        print(f"警告: 統制1（課題の可解性）が崩れています。unboundedのH1: {unbounded_h1:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "ステップ12 海馬モジュール（リプレイバッファの容量制約下での刈り込み戦略）: "
                "unbounded（参考上限・統制1）対 fifo（統制1、最弱ベースライン）対 "
                "uniform-compress（統制3）対 importance-weighted（本設計）の、"
                "多タスクXORパリティ継続学習での旧タスク保持率比較（12.6.32節）",
        "条件": vars(args) | {"out": str(args.out)},
        "capacity_probe": probe,
        "control1_ok": control1_ok,
        "summary": summary,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n書き出し: {args.out}")
    print("-- 容量プローブ（unbounded対fifo、H1） --")
    for capacity, row in probe.items():
        print(f"  capacity={capacity}: unbounded={row['unbounded']:.4f}  fifo={row['fifo']:.4f}")
    print("-- 本実験 --")
    for condition in CONDITIONS:
        s = summary[condition]
        h1, h2 = s["h1_mean_old_task_acc"], s["h2_oldest_task_acc"]
        print(f"{condition:>20}  H1={h1['mean']:.3f}±{h1['std']:.3f}  H2={h2['mean']:.3f}±{h2['std']:.3f}")


if __name__ == "__main__":
    main()
