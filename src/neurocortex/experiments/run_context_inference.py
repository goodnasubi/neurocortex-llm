"""文脈推定機構の検証（12.6.105〜106節）。4タスク列で oracle / none / inferred / random を比較する。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_inference --seeds 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..basal_ganglia_core import ParitySpec
from ..context_inference import CONDITIONS, assignment_purity, run_context_sequence
from ..continual_metrics import forgetting_metrics

PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]
KEYS = ("ACC", "BWT", "F", "LA", "purity", "distinct", "n_contexts")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.4)
    ap.add_argument("--out", type=Path, default=Path("results/context_inference/conditions.json"))
    args = ap.parse_args()
    specs = [ParitySpec(n_inputs=8, relevant=p) for p in PAIRS]
    out: dict = {"説明": "文脈推定機構の検証（12.6.105節）", "seeds": args.seeds, "tau": args.tau, "conditions": {}}
    for cond in CONDITIONS:
        rows = []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            r = run_context_sequence(cond, specs, 32, torch.Generator().manual_seed(seed + 50_000),
                                     torch.Generator().manual_seed(seed + 90_000), tau=args.tau)
            rows.append({"R": r["R"], "n_contexts": r["n_contexts"], **forgetting_metrics(r["R"], chance=0.5),
                         **assignment_purity(r["assign"], len(specs))})
        summ = {k: {"mean": float(torch.tensor([float(x[k]) for x in rows]).mean()),
                    "std": float(torch.tensor([float(x[k]) for x in rows]).std(unbiased=False))} for k in KEYS}
        out["conditions"][cond] = {"summary": summ, "per_seed": rows}
        s = summ
        print(f"{cond:>9} ACC={s['ACC']['mean']:.3f}±{s['ACC']['std']:.3f} BWT={s['BWT']['mean']:+.3f}±{s['BWT']['std']:.3f}"
              f" LA={s['LA']['mean']:.3f} purity={s['purity']['mean']:.3f} distinct={s['distinct']['mean']:.2f}"
              f" n_ctx={s['n_contexts']['mean']:.1f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
