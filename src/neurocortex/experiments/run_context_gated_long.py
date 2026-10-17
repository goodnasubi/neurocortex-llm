"""ゲート付き文脈推定の長い移行期間での限界（12.6.116節）の原因の検証。開発用シード 100〜104 のみ。

仮説: L = 500（移行期間 = 学習期間）では新タスクが多数派になるのが後半で、新しい文脈を学習する時間が足りない。
検証: L = 500 のまま学習ステップを 500 → 1000 に延ばす（移行後に 500 ステップの純粋な期間ができる）。
仮説が正しければ、ACC と「タスクごとに別の文脈」が L = 250 並みに戻る。あわせて新しい文脈の作成時点を記録する。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_gated_long
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..basal_ganglia_core import ParitySpec
from ..context_inference import assignment_purity, run_context_sequence
from ..continual_metrics import forgetting_metrics
from ..run_env import env_info

PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]


def creation_steps(assign: list[tuple[int, int]], pretrain_steps: int) -> list[int]:
    """新しい文脈が初めて使われた時点を、そのタスク内のステップ番号で返す（タスク0を除く）。"""
    seen, out, start = {assign[0][1]}, [], {}
    for i, (k, c) in enumerate(assign):
        start.setdefault(k, i)
        if c not in seen:
            seen.add(c)
            out.append(i - start[k])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    ap.add_argument("--out", type=Path, default=Path("results/context_inference/gated_long_dev.json"))
    args = ap.parse_args()
    specs = [ParitySpec(n_inputs=8, relevant=p) for p in PAIRS]
    out: dict = {"環境": env_info(), "説明": "長い移行期間での限界の原因の検証（12.6.116節）。開発用シード",
                 "seeds": args.seeds, "runs": {}}
    for L, steps in ((250, 500), (500, 500), (500, 1000)):
        rows = []
        for seed in args.seeds:
            torch.manual_seed(seed)
            r = run_context_sequence("inferred_gated", specs, 32, torch.Generator().manual_seed(seed + 50_000),
                                     torch.Generator().manual_seed(seed + 90_000), transition_steps=L, steps=steps)
            rows.append({"R": r["R"], "n_contexts": r["n_contexts"], **forgetting_metrics(r["R"], chance=0.5),
                         **assignment_purity(r["assign"], 4), "created_at": creation_steps(r["assign"], 500)})
        summ = {k: {"mean": float(torch.tensor([float(x[k]) for x in rows]).mean()),
                    "std": float(torch.tensor([float(x[k]) for x in rows]).std(unbiased=False))}
                for k in ("ACC", "BWT", "LA", "n_contexts", "distinct")}
        out["runs"][f"L{L}_steps{steps}"] = {"summary": summ, "per_seed": rows}
        print(f"L={L} steps={steps} ACC={summ['ACC']['mean']:.3f}±{summ['ACC']['std']:.3f} "
              f"LA={summ['LA']['mean']:.3f} n_ctx={summ['n_contexts']['mean']:.1f} "
              f"distinct={summ['distinct']['mean']:.2f} created={[x['created_at'] for x in rows]}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
