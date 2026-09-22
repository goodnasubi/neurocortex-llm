"""ステップ4（小脳モジュール, 12.6.14節）の主実験。

固定されたブラックボックスツールに対する複数エピソードの呼び出し探索で、
順モデルが持続するかどうか（cerebellum対from-scratch）が、目標到達までに
必要な実呼び出し回数にどれだけ差を生むかを比較する。random-search は
統制1の下限を確認するための素朴な基準。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_cerebellum --seeds 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..cerebellum import ForwardModel, make_tool, run_episode, sample_target

CONDITIONS = ("cerebellum", "from-scratch", "random-search")


def run_condition(condition: str, tool, args: argparse.Namespace,
                  generator: torch.Generator) -> list[dict]:
    dim = tool.weight.shape[0]
    # cerebellum条件だけ、この1個のインスタンスを全エピソードで使い回す（持続性）。
    # from-scratchはエピソードごとに新しいインスタンスを作る。random-searchはモデルなし。
    persistent_model = ForwardModel(dim) if condition == "cerebellum" else None

    rows = []
    for ep in range(args.n_episodes):
        target = sample_target(tool, generator, scale=args.target_scale)
        if condition == "cerebellum":
            model = persistent_model
        elif condition == "from-scratch":
            model = ForwardModel(dim)
        else:
            model = None
        result = run_episode(tool, target, args.eps, args.max_real_calls,
                             args.n_candidates, args.step_size, args.lr, generator, model)
        rows.append({"episode": ep, "real_calls_used": result.real_calls_used,
                    "success": result.success})
    return rows


def run_one_seed(spec_dim: int, args: argparse.Namespace, seed: int) -> list[dict]:
    tool = make_tool(spec_dim, seed=seed)
    rows = []
    for condition in args.conditions:
        g = torch.Generator().manual_seed(seed + 30_000)
        for row in run_condition(condition, tool, args, g):
            rows.append(dict(row, condition=condition, seed=seed))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--n-episodes", type=int, default=30)
    ap.add_argument("--early-episodes", type=int, default=3,
                    help="「立ち上がりの速さ」を見るための先頭エピソード数")
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--max-real-calls", type=int, default=30)
    ap.add_argument("--n-candidates", type=int, default=64)
    ap.add_argument("--step-size", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--target-scale", type=float, default=1.0)
    ap.add_argument("--out", type=Path,
                    default=Path("results/cerebellum/persistence_vs_from_scratch.json"))
    args = ap.parse_args()

    t0 = time.time()
    rows: list[dict] = []
    for seed in range(args.seeds):
        rows.extend(run_one_seed(args.dim, args, seed))

    for condition in args.conditions:
        cond_rows = [r for r in rows if r["condition"] == condition]
        early = [r["real_calls_used"] for r in cond_rows if r["episode"] < args.early_episodes]
        late = [r["real_calls_used"] for r in cond_rows if r["episode"] >= args.n_episodes // 2]
        print(f"{condition:>14}  早期(ep<{args.early_episodes})平均呼び出し数="
              f"{sum(early) / len(early):.2f}  後期平均={sum(late) / len(late):.2f}"
              f"  成功率={sum(r['success'] for r in cond_rows) / len(cond_rows):.3f}")

    summary: dict[str, dict] = {}
    for condition in args.conditions:
        cond_rows = [r for r in rows if r["condition"] == condition]
        by_episode: dict[int, list[int]] = {}
        for r in cond_rows:
            by_episode.setdefault(r["episode"], []).append(r["real_calls_used"])
        early = torch.tensor([r["real_calls_used"] for r in cond_rows if r["episode"] < args.early_episodes],
                             dtype=torch.float32)
        late = torch.tensor([r["real_calls_used"] for r in cond_rows if r["episode"] >= args.n_episodes // 2],
                            dtype=torch.float32)
        success = torch.tensor([float(r["success"]) for r in cond_rows])
        summary[condition] = {
            "early_calls_mean": float(early.mean()),
            "early_calls_std": float(early.std(unbiased=False)),
            "late_calls_mean": float(late.mean()),
            "late_calls_std": float(late.std(unbiased=False)),
            "success_rate": float(success.mean()),
            "calls_by_episode_mean": {str(ep): float(sum(v) / len(v)) for ep, v in sorted(by_episode.items())},
        }

    # 統制1（単体エピソードでの学習能力）は tests/test_cerebellum.py が機械的に保証する
    # （cerebellumとfrom-scratchが同一クラス・同一updateを使うため、後期エピソードの
    #  探索効率が揃うことをここでも確認する）。
    if "cerebellum" in summary and "from-scratch" in summary:
        late_gap = abs(summary["cerebellum"]["late_calls_mean"] - summary["from-scratch"]["late_calls_mean"])
        if late_gap > 2.0:
            print(f"警告: 統制1/2の想定と異なり、後期エピソードでも"
                  f" cerebellum と from-scratch の呼び出し数に {late_gap:.2f} の差が残っています。")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "ステップ4 小脳モジュール: 持続する順モデル(cerebellum)対"
                "エピソードごとに初期化する順モデル(from-scratch)の、目標到達に"
                "必要な実呼び出し回数の比較（12.6.14節）",
        "条件": vars(args) | {"out": str(args.out)},
        "summary": summary,
        "rows": rows,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n書き出し: {args.out}")


if __name__ == "__main__":
    main()
