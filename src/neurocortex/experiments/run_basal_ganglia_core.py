"""ステップ7（基底核モジュール — アクター・クリティック本体, 12.6.22節）の主実験。

pretrained（事前学習済み・凍結バックボーン） / random-frozen（未学習・凍結、
統制5） / rl-only（事前学習なし、RLだけでバックボーンごと学習、統制5b）の
3条件で、少数試行のRL収束の速さを比較する。

主張1: pretrainedがrandom-frozenを上回る（事前学習が固定表現の次元合わせ以上の
価値を持つ）。主張2: pretrainedがrl-onlyを上回る（皮質表現との結合に意味がある）。
統制1: 十分な試行数の極限では3条件とも同程度に到達すること。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_basal_ganglia_core --seeds 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..basal_ganglia_core import CONDITIONS, ParitySpec, run_condition


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-inputs", type=int, default=8)
    ap.add_argument("--hidden-dim", type=int, default=16)
    ap.add_argument("--pretrain-steps", type=int, default=500)
    ap.add_argument("--pretrain-batch", type=int, default=64)
    ap.add_argument("--pretrain-lr", type=float, default=0.05)
    ap.add_argument("--rl-steps", type=int, default=300)
    ap.add_argument("--rl-batch", type=int, default=16)
    ap.add_argument("--rl-lr", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-batch", type=int, default=256)
    ap.add_argument("--control1-threshold", type=float, default=0.05)
    ap.add_argument("--out", type=Path,
                    default=Path("results/basal_ganglia_core/pretrained_vs_controls.json"))
    args = ap.parse_args()

    spec = ParitySpec(n_inputs=args.n_inputs, relevant=(0, 1))
    t0 = time.time()

    all_curves: dict[str, list[list[float]]] = {c: [] for c in CONDITIONS}
    for condition in CONDITIONS:
        for seed in range(args.seeds):
            g = torch.Generator().manual_seed(seed + 70_000)
            accs = run_condition(condition, spec, args.hidden_dim, args.pretrain_steps,
                                 args.pretrain_batch, args.pretrain_lr, args.rl_steps,
                                 args.rl_batch, args.rl_lr, args.eval_every, args.eval_batch, g)
            all_curves[condition].append(accs)

    n_points = args.rl_steps // args.eval_every
    summary: dict[str, dict] = {}
    for condition in CONDITIONS:
        curves = torch.tensor(all_curves[condition])  # [seeds, n_points]
        summary[condition] = {
            "mean_curve": curves.mean(dim=0).tolist(),
            "std_curve": curves.std(dim=0, unbiased=False).tolist(),
            "final_mean": float(curves[:, -1].mean()),
            "final_std": float(curves[:, -1].std(unbiased=False)),
            "early_mean": float(curves[:, min(4, n_points - 1)].mean()),
            "early_std": float(curves[:, min(4, n_points - 1)].std(unbiased=False)),
        }

    # 統制1: 十分な試行数の極限（最終評価点）で3条件が同程度に到達すること。
    finals = [summary[c]["final_mean"] for c in CONDITIONS]
    control1_ok = (max(finals) - min(finals)) < args.control1_threshold
    if not control1_ok:
        print(f"警告: 統制1（極限での性能統制）が崩れています。条件間差: "
              f"{max(finals) - min(finals):.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "ステップ7 基底核モジュール（アクター・クリティック本体）: "
                "事前学習済み凍結バックボーン 対 固定ランダム凍結バックボーン（統制5）"
                "対 RL単独学習（統制5b）の少数試行RL収束比較（12.6.22節）",
        "条件": vars(args) | {"out": str(args.out)},
        "control1_ok": control1_ok,
        "summary": summary,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n書き出し: {args.out}")
    for condition in CONDITIONS:
        s = summary[condition]
        print(f"{condition:>14}  early={s['early_mean']:.3f}±{s['early_std']:.3f}"
              f"  final={s['final_mean']:.3f}±{s['final_std']:.3f}")


if __name__ == "__main__":
    main()
