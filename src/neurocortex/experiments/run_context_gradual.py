"""少しずつ変わる文脈での文脈推定（12.6.109節）。4タスク列（ステップ10と同じ8入力）で、各タスクの最初の
L ステップを前のタスクとの混合期間にする（L = 0, 100, 250, 500）。oracle / none / inferred を比べる。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_gradual --seeds 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..run_env import env_info
from ..basal_ganglia_core import ParitySpec
from ..context_inference import assignment_purity, run_context_sequence
from ..continual_metrics import forgetting_metrics

PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("results/context_inference/gradual.json"))
    args = ap.parse_args()
    specs = [ParitySpec(n_inputs=8, relevant=p) for p in PAIRS]
    out: dict = {"環境": env_info(), "説明": "少しずつ変わる文脈での文脈推定（12.6.109節）", "seeds": args.seeds, "L": {}}
    for L in (0, 100, 250, 500):
        out["L"][str(L)] = {}
        for cond in ("oracle", "none", "inferred"):
            rows = []
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                r = run_context_sequence(cond, specs, 32, torch.Generator().manual_seed(seed + 50_000),
                                         torch.Generator().manual_seed(seed + 90_000), transition_steps=L,
                                         selection_trials=20)
                rows.append({"R": r["R"], "n_contexts": r["n_contexts"], "selection_acc": r["selection_acc"],
                             **forgetting_metrics(r["R"], chance=0.5), **assignment_purity(r["assign"], 4)})
            keys = ["ACC", "BWT", "LA", "n_contexts", "distinct"] + (["selection_acc"] if cond == "inferred" else [])
            summ = {k: {"mean": float(torch.tensor([float(x[k]) for x in rows]).mean()),
                        "std": float(torch.tensor([float(x[k]) for x in rows]).std(unbiased=False))} for k in keys}
            out["L"][str(L)][cond] = {"summary": summ, "per_seed": rows}
            s = summ
            extra = (f" n_ctx={s['n_contexts']['mean']:.1f} distinct={s['distinct']['mean']:.2f}"
                     f" sel={s['selection_acc']['mean']:.3f}") if cond == "inferred" else ""
            print(f"L={L:>3} {cond:>8} ACC={s['ACC']['mean']:.3f}±{s['ACC']['std']:.3f} "
                  f"BWT={s['BWT']['mean']:+.3f} LA={s['LA']['mean']:.3f}{extra}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
