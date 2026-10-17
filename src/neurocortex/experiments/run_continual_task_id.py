"""仮説の検証（12.6.104節）: ステップ12・13で安定性と可塑性のトレードオフが出なかったのは、入力にタスク ID が
あるためか。ステップ10の4タスク列（12.6.102節）にタスク ID の one-hot を加え、加えない場合と比べる。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_continual_task_id --seeds 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..basal_ganglia_core import ParitySpec
from ..continual_metrics import forgetting_metrics
from ..continual_sequence import run_sequence

CONDITIONS = ("naive-finetune", "replay", "ewc")
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("results/continual_forgetting/task_id_input.json"))
    args = ap.parse_args()
    specs = [ParitySpec(n_inputs=8, relevant=p) for p in PAIRS]
    out: dict = {"説明": "4タスク列でのタスク ID 入力の有無の比較（12.6.104節）", "seeds": args.seeds, "runs": {}}
    for task_id in (False, True):
        key = "task_id" if task_id else "no_task_id"
        out["runs"][key] = {}
        for cond in CONDITIONS:
            rows = []
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                R = run_sequence(cond, specs, 32, torch.Generator().manual_seed(seed + 50_000),
                                 torch.Generator().manual_seed(seed + 90_000), task_id_input=task_id)
                rows.append({"R": R, **forgetting_metrics(R, chance=0.5)})
            summ = {k: {"mean": float(torch.tensor([r[k] for r in rows]).mean()),
                        "std": float(torch.tensor([r[k] for r in rows]).std(unbiased=False))}
                    for k in ("ACC", "BWT", "F", "LA")}
            out["runs"][key][cond] = {"summary": summ, "per_seed": rows}
            s = summ
            print(f"{key:>10} {cond:>15} ACC={s['ACC']['mean']:.3f}±{s['ACC']['std']:.3f} "
                  f"BWT={s['BWT']['mean']:+.3f}±{s['BWT']['std']:.3f} LA={s['LA']['mean']:.3f}±{s['LA']['std']:.3f}",
                  flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
