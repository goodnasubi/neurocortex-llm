"""手がかりの数を減らしたときの文脈推定の精度（12.6.107節）。4タスク列・inferred で、テスト時の手がかりを
1〜32個に変える。学習過程は手がかりの数に依存しない（手がかりは評価用の乱数だけを使う）。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_cue_size --seeds 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..run_env import env_info
from ..basal_ganglia_core import ParitySpec
from ..context_inference import run_context_sequence
from ..continual_metrics import forgetting_metrics

PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]
CUE_SIZES = (1, 2, 4, 8, 16, 32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--out", type=Path, default=Path("results/context_inference/cue_size.json"))
    args = ap.parse_args()
    specs = [ParitySpec(n_inputs=8, relevant=p) for p in PAIRS]
    out: dict = {"環境": env_info(), "説明": "手がかりの数と文脈推定の精度（12.6.107節）", "seeds": args.seeds, "trials": args.trials,
                 "cue_sizes": {}}
    for n in CUE_SIZES:
        rows = []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            r = run_context_sequence("inferred", specs, 32, torch.Generator().manual_seed(seed + 50_000),
                                     torch.Generator().manual_seed(seed + 90_000), cue_n=n,
                                     selection_trials=args.trials)
            rows.append({"ACC": forgetting_metrics(r["R"], chance=0.5)["ACC"], "selection_acc": r["selection_acc"],
                         "R": r["R"]})
        summ = {k: {"mean": float(torch.tensor([x[k] for x in rows]).mean()),
                    "std": float(torch.tensor([x[k] for x in rows]).std(unbiased=False))} for k in ("ACC", "selection_acc")}
        out["cue_sizes"][str(n)] = {"summary": summ, "per_seed": rows}
        print(f"cue={n:>2} ACC={summ['ACC']['mean']:.3f}±{summ['ACC']['std']:.3f} "
              f"selection={summ['selection_acc']['mean']:.3f}±{summ['selection_acc']['std']:.3f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
