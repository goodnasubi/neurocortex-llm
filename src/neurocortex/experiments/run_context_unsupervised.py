"""入力分布からのラベルなし文脈推定の検証（12.6.111節）。手がかりの強さ q と移行期間 L を変えて、
oracle / none / inferred（ラベルあり・バッチ単位）/ unsupervised（ラベルなし・サンプル単位）を比べる。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_unsupervised --seeds 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..context_inference import CueSpec, run_context_sequence
from ..context_unsupervised import run_unsupervised_sequence
from ..continual_metrics import forgetting_metrics

PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]


def specs_for(q: float) -> list[CueSpec]:
    return [CueSpec(16, p, (8 + 2 * i, 9 + 2 * i), q) for i, p in enumerate(PAIRS)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("results/context_inference/unsupervised.json"))
    args = ap.parse_args()
    out: dict = {"説明": "入力分布からのラベルなし文脈推定（12.6.111節）", "seeds": args.seeds, "runs": {}}
    for q in (0.9, 0.75):
        for L in (0, 250):
            key = f"q{q}_L{L}"
            out["runs"][key] = {}
            for cond in ("oracle", "none", "inferred", "unsupervised"):
                rows = []
                for seed in range(args.seeds):
                    torch.manual_seed(seed)
                    g, ge = torch.Generator().manual_seed(seed + 50_000), torch.Generator().manual_seed(seed + 90_000)
                    if cond == "unsupervised":
                        r = run_unsupervised_sequence(specs_for(q), 32, g, ge, transition_steps=L)
                    else:
                        r = run_context_sequence(cond, specs_for(q), 32, g, ge, transition_steps=L)
                    rows.append({"R": r["R"], "n_contexts": r["n_contexts"], "sample_purity": r.get("sample_purity"),
                                 **forgetting_metrics(r["R"], chance=0.5)})
                acc = torch.tensor([x["ACC"] for x in rows])
                summ = {"ACC": {"mean": float(acc.mean()), "std": float(acc.std(unbiased=False))},
                        "n_contexts": float(torch.tensor([float(x["n_contexts"]) for x in rows]).mean())}
                if cond == "unsupervised":
                    summ["sample_purity"] = float(torch.tensor([x["sample_purity"] for x in rows]).mean())
                out["runs"][key][cond] = {"summary": summ, "per_seed": rows}
                print(f"{key:>10} {cond:>12} ACC={summ['ACC']['mean']:.3f}±{summ['ACC']['std']:.3f} "
                      f"n_ctx={summ['n_contexts']:.1f}" + (f" purity={summ['sample_purity']:.3f}" if cond == "unsupervised" else ""),
                      flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
