"""ステップ9（基底核モジュール — アクター・クリティック本体の再検証, 12.6.26節）の主実験。

ステップ7（12.6.22〜23節）で統制1が確立できなかった問題を、2段階の事前チェックで
解消してから主張1・2を再検証する。

段階A: RLを使わず教師あり勾配降下で、random-frozenバックボーンが教師ありの極限で
高精度に到達する隠れ次元を特定する。
段階B: 段階Aの隠れ次元のもとで、pretrained条件のRLハイパーパラメータ
（学習率・バッチサイズ）を、確実に収束する設定になるまで調整し、その後
random-frozen・rl-onlyが同じ設定・十分なステップ予算で追いつくか（統制1）を確認する。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_basal_ganglia_core_v2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..basal_ganglia_core import CONDITIONS, ParitySpec, run_condition, supervised_capacity_check


def stage_a(spec: ParitySpec, hidden_dims: list[int], steps: int, batch_size: int, lr: float,
           seeds: int) -> dict:
    """段階A: random-frozenバックボーンの教師あり容量チェック。"""
    results = {}
    for hidden_dim in hidden_dims:
        accs = []
        for seed in range(seeds):
            torch.manual_seed(seed)
            g = torch.Generator().manual_seed(seed)
            accs.append(supervised_capacity_check("random-frozen", spec, hidden_dim, steps,
                                                   batch_size, lr, g))
        results[hidden_dim] = {
            "mean": sum(accs) / len(accs),
            "accs": accs,
        }
    return results


def stage_b(spec: ParitySpec, hidden_dim: int, rl_lrs: list[float], rl_steps: int,
           rl_batch: int, seeds: int) -> dict:
    """段階B: pretrained条件だけを使ったRL学習率のキャリブレーション。"""
    results = {}
    for rl_lr in rl_lrs:
        finals = []
        for seed in range(seeds):
            g = torch.Generator().manual_seed(seed)
            accs = run_condition("pretrained", spec, hidden_dim, 1500, 64, 0.05, rl_steps,
                                 rl_batch, rl_lr, rl_steps // 5, 256, g)
            finals.append(accs[-1])
        results[rl_lr] = finals
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-inputs", type=int, default=8)
    ap.add_argument("--hidden-dim", type=int, default=192,
                    help="段階Aで選定済みの隠れ次元（教師ありの極限で0.996に到達）")
    ap.add_argument("--rl-lr", type=float, default=0.02,
                    help="段階Bで選定済みのRL学習率（pretrained条件が安定して1.0に収束する設定）")
    ap.add_argument("--rl-batch", type=int, default=32)
    ap.add_argument("--rl-steps", type=int, default=6000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--pretrain-steps", type=int, default=1500)
    ap.add_argument("--pretrain-batch", type=int, default=64)
    ap.add_argument("--pretrain-lr", type=float, default=0.05)
    ap.add_argument("--eval-batch", type=int, default=256)
    ap.add_argument("--control1-threshold", type=float, default=0.05)
    ap.add_argument("--out", type=Path,
                    default=Path("results/basal_ganglia_core_v2/staged_recheck.json"))
    args = ap.parse_args()

    spec = ParitySpec(n_inputs=args.n_inputs, relevant=(0, 1))
    t0 = time.time()

    stage_a_results = stage_a(spec, [64, 96, 128, 192, 256], steps=1500, batch_size=64,
                              lr=0.05, seeds=args.seeds)
    stage_b_results = stage_b(spec, args.hidden_dim, [0.005, 0.01, 0.02, 0.05],
                              rl_steps=1000, rl_batch=32, seeds=3)

    all_curves: dict[str, list[list[float]]] = {c: [] for c in CONDITIONS}
    for condition in CONDITIONS:
        for seed in range(args.seeds):
            g = torch.Generator().manual_seed(seed + 80_000)
            accs = run_condition(condition, spec, args.hidden_dim, args.pretrain_steps,
                                 args.pretrain_batch, args.pretrain_lr, args.rl_steps,
                                 args.rl_batch, args.rl_lr, args.eval_every, args.eval_batch, g)
            all_curves[condition].append(accs)

    summary: dict[str, dict] = {}
    for condition in CONDITIONS:
        curves = torch.tensor(all_curves[condition])
        summary[condition] = {
            "mean_curve": curves.mean(dim=0).tolist(),
            "std_curve": curves.std(dim=0, unbiased=False).tolist(),
            "final_mean": float(curves[:, -1].mean()),
            "final_std": float(curves[:, -1].std(unbiased=False)),
            "final_per_seed": curves[:, -1].tolist(),
        }

    finals = [summary[c]["final_mean"] for c in CONDITIONS]
    control1_ok = (max(finals) - min(finals)) < args.control1_threshold
    if not control1_ok:
        print(f"警告: 統制1（極限での性能統制）が崩れています。条件間差: "
              f"{max(finals) - min(finals):.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "ステップ9 基底核モジュール（アクター・クリティック本体の再検証）: "
                "段階A（教師あり容量チェック）・段階B（RL学習率キャリブレーション）"
                "の記録と、統制1確立後の主張1・2の本実験（12.6.26節）",
        "条件": vars(args) | {"out": str(args.out)},
        "stage_a": stage_a_results,
        "stage_b": stage_b_results,
        "control1_ok": control1_ok,
        "summary": summary,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n書き出し: {args.out}")
    print("-- 段階A（教師あり容量チェック） --")
    for hidden_dim, r in stage_a_results.items():
        print(f"  hidden_dim={hidden_dim}: mean_acc={r['mean']:.4f}")
    print("-- 段階B（pretrained条件のRL学習率） --")
    for rl_lr, finals_b in stage_b_results.items():
        print(f"  rl_lr={rl_lr}: finals={finals_b}")
    print("-- 本実験 --")
    for condition in CONDITIONS:
        s = summary[condition]
        print(f"{condition:>14}  final={s['final_mean']:.3f}±{s['final_std']:.3f}"
              f"  per_seed={[round(v, 3) for v in s['final_per_seed']]}")


if __name__ == "__main__":
    main()
