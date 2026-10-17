"""ステップ12・13（12.6.32〜35節）の6条件を、忘却指標 ACC・BWT・F・LA（12.6.101節）で評価し直す。

学習の設定・シードは run_hippocampus_capacity_fisher_fix.py と同一。評価専用の乱数で各タスク学習後に全タスクを
評価して正確度行列を作る（学習過程は変わらない）。最終時点の H1 が既存の結果と一致することも確認する。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_capacity_forgetting_metrics
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..continual_metrics import forgetting_metrics
from ..hippocampus_capacity import CONDITIONS, make_task_specs, run_condition

METRICS = ("ACC", "BWT", "F", "LA")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--reference", type=Path,
                    default=Path("results/hippocampus_capacity/fisher_collapse_mitigations.json"))
    ap.add_argument("--out", type=Path, default=Path("results/continual_forgetting/capacity_conditions.json"))
    args = ap.parse_args()
    ref = json.loads(args.reference.read_text())["summary"] if args.reference.exists() else None
    t0 = time.time()
    specs = make_task_specs(5, 10)
    out: dict = {"説明": "ステップ12・13の6条件を ACC・BWT・F・LA（chance 0.5 で正規化）で再評価（12.6.103節）",
                 "seeds": args.seeds, "conditions": {}}
    for cond in CONDITIONS:
        rows = []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            g = torch.Generator().manual_seed(seed + 70_000)
            ge = torch.Generator().manual_seed(seed + 95_000)
            r = run_condition(cond, specs, 32, 120, 60, 400, 64, 0.05, 10, 32, 1000, g,
                              fisher_decay_weight_decay=0.01, eval_generator=ge)
            rows.append({"R": r.acc_matrix, "h1": r.mean_old_task_acc,
                         **forgetting_metrics(r.acc_matrix, chance=0.5)})
        summ = {}
        for k in METRICS + ("h1",):
            t = torch.tensor([x[k] for x in rows])
            summ[k] = {"mean": float(t.mean()), "std": float(t.std(unbiased=False))}
        h1_match = None
        if ref is not None:
            ref_h1 = ref[cond]["h1_mean_old_task_acc"]["per_seed"][:args.seeds]
            h1_match = max(abs(a - b["h1"]) for a, b in zip(ref_h1, rows))
        out["conditions"][cond] = {"summary": summ, "h1_max_abs_diff_vs_reference": h1_match, "per_seed": rows}
        s = summ
        print(f"{cond:>34} ACC={s['ACC']['mean']:.3f}±{s['ACC']['std']:.3f} BWT={s['BWT']['mean']:+.3f}±{s['BWT']['std']:.3f}"
              f" F={s['F']['mean']:.3f} LA={s['LA']['mean']:.3f} H1={s['h1']['mean']:.3f} (ref差 {h1_match})", flush=True)
    out["sec"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"書き出し: {args.out}")


if __name__ == "__main__":
    main()
