"""似たタスクでの文脈推定（12.6.108節）。基本タスク b0 XOR b1 と、互いに素な m ビットがすべて +1 の領域でだけ
正解を反転させた3タスクの計4タスクを順に学習する。基本タスクと各反転タスクの不一致率は 2^-m。
m = 1〜4（不一致率 50%〜6.25%）で oracle / none / inferred を比べる。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_similar_tasks --seeds 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..run_env import env_info
from ..context_inference import FlipSpec, assignment_purity, run_context_sequence
from ..continual_metrics import forgetting_metrics

N_INPUTS = 14


def task_specs(m: int) -> list[FlipSpec]:
    flips = [tuple(range(2 + i * m, 2 + (i + 1) * m)) for i in range(3)]
    return [FlipSpec(N_INPUTS)] + [FlipSpec(N_INPUTS, flip=f) for f in flips]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.4)
    ap.add_argument("--out", type=Path, default=Path("results/context_inference/similar_tasks.json"))
    args = ap.parse_args()
    out: dict = {"環境": env_info(), "説明": "似たタスクでの文脈推定（12.6.108節）", "seeds": args.seeds, "tau": args.tau, "m": {}}
    for m in (1, 2, 3, 4):
        specs = task_specs(m)
        out["m"][str(m)] = {"disagreement": 2 ** -m}
        for cond in ("oracle", "none", "inferred"):
            rows = []
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                r = run_context_sequence(cond, specs, 32, torch.Generator().manual_seed(seed + 50_000),
                                         torch.Generator().manual_seed(seed + 90_000), tau=args.tau,
                                         selection_trials=20)
                rows.append({"R": r["R"], "n_contexts": r["n_contexts"], "selection_acc": r["selection_acc"],
                             **forgetting_metrics(r["R"], chance=0.5), **assignment_purity(r["assign"], 4)})
            keys = ["ACC", "BWT", "LA", "n_contexts", "purity", "distinct"] + (["selection_acc"] if cond == "inferred" else [])
            summ = {k: {"mean": float(torch.tensor([float(x[k]) for x in rows]).mean()),
                        "std": float(torch.tensor([float(x[k]) for x in rows]).std(unbiased=False))} for k in keys}
            out["m"][str(m)][cond] = {"summary": summ, "per_seed": rows}
            s = summ
            extra = (f" n_ctx={s['n_contexts']['mean']:.1f} purity={s['purity']['mean']:.3f} "
                     f"distinct={s['distinct']['mean']:.2f} sel={s['selection_acc']['mean']:.3f}") if cond == "inferred" else ""
            print(f"m={m} d={2**-m:.4f} {cond:>8} ACC={s['ACC']['mean']:.3f}±{s['ACC']['std']:.3f} "
                  f"BWT={s['BWT']['mean']:+.3f} LA={s['LA']['mean']:.3f}{extra}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
