"""ステップ15（マルチ学習則の学習安定性 — 目的関数が対立するシナリオ, 12.6.38節）の主実験。

タスクA（海馬が教師あり学習、リプレイバッファに保存）→タスクB（基底核がRLで学習）の
逐次シナリオで、3条件（shared-conflict / independent / rl-only-no-replay）を比較する。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_gradient_conflict
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..basal_ganglia_core import ParitySpec
from ..gradient_conflict import CONDITIONS, run_condition


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-inputs", type=int, default=8)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--buffer-size", type=int, default=50)
    ap.add_argument("--pretrain-steps", type=int, default=500)
    ap.add_argument("--pretrain-batch", type=int, default=64)
    ap.add_argument("--pretrain-lr", type=float, default=0.05)
    ap.add_argument("--b-steps", type=int, default=500)
    ap.add_argument("--b-batch", type=int, default=64)
    ap.add_argument("--b-lr", type=float, default=0.02)
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--threshold", type=float, default=0.1,
                    help="主張の判定に使う最小差")
    ap.add_argument("--out", type=Path,
                    default=Path("results/gradient_conflict/conflicting_objectives.json"))
    args = ap.parse_args()

    t0 = time.time()
    spec_a = ParitySpec(n_inputs=args.n_inputs, relevant=(0, 1))
    spec_b = ParitySpec(n_inputs=args.n_inputs, relevant=(2, 3))

    all_results: dict[str, dict] = {c: {"task_a_retention": [], "task_b_acc": []} for c in CONDITIONS}
    for condition in CONDITIONS:
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            g = torch.Generator().manual_seed(seed + 50_000)
            r = run_condition(condition, spec_a, spec_b, args.hidden_dim, args.buffer_size,
                              args.pretrain_steps, args.pretrain_batch, args.pretrain_lr,
                              args.b_steps, args.b_batch, args.b_lr, args.n_eval, g)
            all_results[condition]["task_a_retention"].append(r.task_a_retention)
            all_results[condition]["task_b_acc"].append(r.task_b_acc)

    def _stats(values: list[float]) -> dict:
        t = torch.tensor(values)
        return {"mean": float(t.mean()), "std": float(t.std(unbiased=False)), "per_seed": t.tolist()}

    summary: dict[str, dict] = {}
    for condition in CONDITIONS:
        r = all_results[condition]
        summary[condition] = {
            "task_a_retention": _stats(r["task_a_retention"]),
            "task_b_acc": _stats(r["task_b_acc"]),
        }

    shared_a = summary["shared-conflict"]["task_a_retention"]["mean"]
    shared_b = summary["shared-conflict"]["task_b_acc"]["mean"]
    indep_a = summary["independent"]["task_a_retention"]["mean"]
    rl_only_b = summary["rl-only-no-replay"]["task_b_acc"]["mean"]

    task_a_degraded = bool(shared_a < indep_a - args.threshold)
    task_b_degraded = bool(shared_b < rl_only_b - args.threshold)
    claim_ok = bool(task_a_degraded and task_b_degraded)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "ステップ15 マルチ学習則の学習安定性（目的関数が対立するシナリオ）: "
                "shared-conflict（本設計）対 independent（統制1、タスクA保持率の参照点）対 "
                "rl-only-no-replay（統制2、タスクB性能の参照点）の、タスクA(海馬・教師あり)→"
                "タスクB(基底核・RL)逐次シナリオでの対立の代償比較（12.6.38節）",
        "条件": vars(args) | {"out": str(args.out)},
        "summary": summary,
        "task_a_degraded（shared-conflictがindependentよりタスクA保持率で劣るか）": task_a_degraded,
        "task_b_degraded（shared-conflictがrl-only-no-replayよりタスクB性能で劣るか）": task_b_degraded,
        "claim_ok（両方向で対立の代償が出たか）": claim_ok,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n書き出し: {args.out}")
    print("-- 本実験 --")
    for condition in CONDITIONS:
        s = summary[condition]
        a, b = s["task_a_retention"], s["task_b_acc"]
        print(f"{condition:>20}  task_a_retention={a['mean']:.3f}±{a['std']:.3f}  "
              f"task_b_acc={b['mean']:.3f}±{b['std']:.3f}")
    print(f"\ntask_a_degraded: {task_a_degraded}  task_b_degraded: {task_b_degraded}  claim_ok: {claim_ok}")


if __name__ == "__main__":
    main()
