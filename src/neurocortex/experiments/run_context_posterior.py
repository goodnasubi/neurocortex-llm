"""サンプルごとの事後推定による文脈推定の検証（12.6.110節）。12.6.109節と同じ移行期間の設定で比べる。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_posterior --seeds 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..basal_ganglia_core import ParitySpec
from ..context_inference import assignment_purity
from ..context_posterior import run_posterior_sequence
from ..continual_metrics import forgetting_metrics

PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("results/context_inference/posterior.json"))
    args = ap.parse_args()
    specs = [ParitySpec(n_inputs=8, relevant=p) for p in PAIRS]
    out: dict = {"説明": "サンプルごとの事後推定による文脈推定（12.6.110節）", "seeds": args.seeds, "L": {}}
    for L in (0, 100, 250, 500):
        rows = []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            r = run_posterior_sequence(specs, 32, torch.Generator().manual_seed(seed + 50_000),
                                       torch.Generator().manual_seed(seed + 90_000), transition_steps=L)
            rows.append({"R": r["R"], "n_contexts": r["n_contexts"], **forgetting_metrics(r["R"], chance=0.5),
                         **assignment_purity(r["assign"], 4)})
        summ = {k: {"mean": float(torch.tensor([float(x[k]) for x in rows]).mean()),
                    "std": float(torch.tensor([float(x[k]) for x in rows]).std(unbiased=False))}
                for k in ("ACC", "BWT", "LA", "n_contexts", "distinct")}
        out["L"][str(L)] = {"summary": summ, "per_seed": rows}
        s = summ
        print(f"L={L:>3} ACC={s['ACC']['mean']:.3f}±{s['ACC']['std']:.3f} BWT={s['BWT']['mean']:+.3f} "
              f"LA={s['LA']['mean']:.3f} n_ctx={s['n_contexts']['mean']:.1f} distinct={s['distinct']['mean']:.2f}",
              flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
