"""評価指標「継続学習後の破局的忘却の程度」の検証（12.6.101節）。

T = 2（ステップ10との整合性確認）と T = 4（(0,1)→(2,3)→(4,5)→(6,7)）で、naive-finetune / replay / ewc の
正確度行列とチャンス正規化した ACC・BWT・F・LA を計算する。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_continual_forgetting --seeds 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..basal_ganglia_core import ParitySpec
from ..continual_metrics import forgetting_metrics
from ..continual_sequence import run_sequence

CONDITIONS = ("naive-finetune", "replay", "ewc")
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]


def _mean_std(xs: list[float]) -> dict[str, float]:
    t = torch.tensor(xs)
    return {"mean": float(t.mean()), "std": float(t.std(unbiased=False))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path("results/continual_forgetting/metrics.json"))
    args = ap.parse_args()
    t0 = time.time()
    out: dict = {"説明": "破局的忘却の評価指標（12.6.101節）。chance=0.5 で正規化", "seeds": args.seeds, "runs": {}}
    for T in (2, 4):
        specs = [ParitySpec(n_inputs=8, relevant=p) for p in PAIRS[:T]]
        out["runs"][f"T{T}"] = {}
        for cond in CONDITIONS:
            per_seed = []
            for seed in range(args.seeds):
                g = torch.Generator().manual_seed(seed + 50_000)  # ステップ10と同じ
                ge = torch.Generator().manual_seed(seed + 90_000)
                torch.manual_seed(seed)
                R = run_sequence(cond, specs, args.hidden_dim, g, ge)
                per_seed.append({"R": R, **forgetting_metrics(R, chance=0.5)})
            agg = {k: _mean_std([r[k] for r in per_seed]) for k in ("ACC", "BWT", "F", "LA")}
            agg["R_last_first_task"] = _mean_std([r["R"][-1][0] for r in per_seed])
            out["runs"][f"T{T}"][cond] = {"summary": agg, "per_seed": per_seed}
            s = agg
            print(f"T={T} {cond:>15}  ACC={s['ACC']['mean']:+.3f}  BWT={s['BWT']['mean']:+.3f}±{s['BWT']['std']:.3f}"
                  f"  F={s['F']['mean']:.3f}  LA={s['LA']['mean']:.3f}  R[T-1][0]={s['R_last_first_task']['mean']:.3f}",
                  flush=True)
    out["sec"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"書き出し: {args.out}")


if __name__ == "__main__":
    main()
