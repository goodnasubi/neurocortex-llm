"""ステップ20（評価指標の設計 — 小脳モジュールのツール呼び出し試行回数削減率,
12.6.48節）の主実験。

`cerebellum.py`の実験基盤（`make_tool`・`run_episode`）を再利用し、`cerebellum`
条件の`random-search`基準の削減率（主指標）と`from-scratch`基準の削減率
（補足値）を、早期・後期エピソードに分けて測定する。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_tool_call_reduction
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..cerebellum import make_tool
from ..tool_call_reduction import reduction_rate, run_episodes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--n-episodes", type=int, default=30)
    ap.add_argument("--early-episodes", type=int, default=3)
    ap.add_argument("--late-start-episode", type=int, default=15)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--max-real-calls", type=int, default=30)
    ap.add_argument("--n-candidates", type=int, default=64)
    ap.add_argument("--step-size", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--target-scale", type=float, default=1.0)
    ap.add_argument("--out", type=Path,
                    default=Path("results/tool_call_reduction/reduction_rate.json"))
    args = ap.parse_args()

    t0 = time.time()

    # 条件ごとに各シード・各エピソードの呼び出し回数を集める（エピソード順を保持）。
    per_seed_calls: dict[str, list[list[int]]] = {"cerebellum": [], "from-scratch": [], "random-search": []}
    for seed in range(args.seeds):
        tool = make_tool(args.dim, seed=seed)
        for condition in per_seed_calls:
            torch.manual_seed(seed)
            g = torch.Generator().manual_seed(seed + 70_000)
            calls = run_episodes(condition, tool, args.n_episodes, args.eps, args.max_real_calls,
                                 args.n_candidates, args.step_size, args.lr, args.target_scale, g)
            per_seed_calls[condition].append(calls)

    def _phase_calls(condition: str, phase: str) -> list[int]:
        """全シードを跨いで、指定フェーズ（early/late/all）の呼び出し回数を1本のリストにする。"""
        out: list[int] = []
        for seed_calls in per_seed_calls[condition]:
            if phase == "early":
                out.extend(seed_calls[: args.early_episodes])
            elif phase == "late":
                out.extend(seed_calls[args.late_start_episode :])
            else:
                out.extend(seed_calls)
        return out

    def _mean(values: list[int]) -> float:
        return sum(values) / len(values)

    phases = ("early", "late", "all")
    summary: dict[str, dict] = {}
    for phase in phases:
        calls = {c: _phase_calls(c, phase) for c in per_seed_calls}
        summary[phase] = {
            "mean_calls": {c: _mean(v) for c, v in calls.items()},
            "reduction_vs_random_search": reduction_rate(calls["cerebellum"], calls["random-search"]),
            "reduction_vs_from_scratch": reduction_rate(calls["cerebellum"], calls["from-scratch"]),
        }

    # 合格条件(1): cerebellumのrandom-search基準削減率が明確に正であること。
    claim1_positive_reduction_ok = bool(summary["all"]["reduction_vs_random_search"] > 0.1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "ステップ20 評価指標の設計（小脳モジュールのツール呼び出し試行回数削減率）: "
                "cerebellum（本設計）のrandom-search基準削減率（主指標）と"
                "from-scratch基準削減率（補足値、持続性固有の効果）を"
                "早期/後期エピソードに分けて実測（12.6.48節）",
        "条件": vars(args) | {"out": str(args.out)},
        "claim1_positive_reduction_ok（random-search基準削減率が明確に正か）": claim1_positive_reduction_ok,
        "summary": summary,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n書き出し: {args.out}")
    for phase in phases:
        s = summary[phase]
        mc = s["mean_calls"]
        print(f"-- {phase} --  cerebellum={mc['cerebellum']:.2f}  "
              f"from-scratch={mc['from-scratch']:.2f}  random-search={mc['random-search']:.2f}")
        print(f"   削減率(vs random-search) = {s['reduction_vs_random_search']:.3f}"
              f"   削減率(vs from-scratch, 補足) = {s['reduction_vs_from_scratch']:.3f}")
    print(f"\n合格条件(1) claim1_positive_reduction_ok: {claim1_positive_reduction_ok}")


if __name__ == "__main__":
    main()
