"""ステップ8（小脳モジュール — 逆モデル本体, 12.6.24節）の主実験。

inverse-model（持続する逆モデル、本設計） / pseudo-inverse（持続する順モデルの
疑似逆行列、統制5） / inverse-from-scratch（統制5b）の3条件で、引数`x`の再現誤差
を比較する。真の生成分布`D`は原点から離れた固定平均を持つ（最小ノルム解との
乖離を確保するため）。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_cerebellum_inverse --seeds 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..cerebellum import make_tool
from ..cerebellum_inverse import run_condition

CONDITIONS = ("inverse-model", "pseudo-inverse", "inverse-from-scratch")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--dim", type=int, default=4)
    ap.add_argument("--d-mean", type=float, default=3.0)
    ap.add_argument("--d-std", type=float, default=0.5)
    ap.add_argument("--n-calls", type=int, default=30_000,
                    help="疑似逆行列条件の順モデル（相関の強いD由来の入力でLMS収束が"
                    "遅い）が真の重みに十分収束するまでに必要な呼び出し数")
    ap.add_argument("--episode-len", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.005,
                    help="逆モデルの学習率（yのスケールに対して安定な小さい値）")
    ap.add_argument("--lr-forward", type=float, default=0.03,
                    help="疑似逆行列条件が内部で使う順モデルの学習率（xのスケールに対して安定な値）")
    ap.add_argument("--eval-every", type=int, default=3_000)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--control1-threshold", type=float, default=0.1)
    ap.add_argument("--out", type=Path,
                    default=Path("results/cerebellum_inverse/inverse_vs_pseudo_inverse.json"))
    args = ap.parse_args()

    t0 = time.time()
    all_rows: dict[str, list[list[dict]]] = {c: [] for c in CONDITIONS}
    for condition in CONDITIONS:
        for seed in range(args.seeds):
            tool = make_tool(dim=args.dim, seed=seed)
            g = torch.Generator().manual_seed(seed + 60_000)
            rows = run_condition(condition, tool, args.n_calls, args.episode_len,
                                 args.d_mean, args.d_std, args.lr, args.eval_every,
                                 args.n_eval, g, lr_forward=args.lr_forward)
            all_rows[condition].append(rows)

    n_points = args.n_calls // args.eval_every
    summary: dict[str, dict] = {}
    for condition in CONDITIONS:
        x_errs = torch.tensor([[r["x_error"] for r in seed_rows] for seed_rows in all_rows[condition]])
        y_errs = torch.tensor([[r["y_error"] for r in seed_rows] for seed_rows in all_rows[condition]])
        summary[condition] = {
            "x_error_mean_curve": x_errs.mean(dim=0).tolist(),
            "x_error_std_curve": x_errs.std(dim=0, unbiased=False).tolist(),
            "y_error_final_mean": float(y_errs[:, -1].mean()),
            "y_error_final_std": float(y_errs[:, -1].std(unbiased=False)),
            "x_error_final_mean": float(x_errs[:, -1].mean()),
            "x_error_final_std": float(x_errs[:, -1].std(unbiased=False)),
        }

    # 統制1: 十分な履歴の極限で、inverse-model・pseudo-inverseのyの再現誤差が同程度に小さいこと。
    y_finals = [summary[c]["y_error_final_mean"] for c in ("inverse-model", "pseudo-inverse")]
    control1_ok = max(y_finals) < args.control1_threshold
    if not control1_ok:
        print(f"警告: 統制1（yの再現性の統制）が崩れています。y_error_final: {y_finals}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "ステップ8 小脳モジュール（逆モデル本体）: 持続する逆モデル 対 "
                "持続する順モデルの疑似逆行列（統制5）対 from-scratch逆モデル"
                "（統制5b）の引数x再現誤差比較（12.6.24節）",
        "条件": vars(args) | {"out": str(args.out)},
        "control1_ok": control1_ok,
        "summary": summary,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n書き出し: {args.out}")
    for condition in CONDITIONS:
        s = summary[condition]
        print(f"{condition:>22}  x_error_final={s['x_error_final_mean']:.4f}±{s['x_error_final_std']:.4f}"
              f"  y_error_final={s['y_error_final_mean']:.4f}±{s['y_error_final_std']:.4f}")


if __name__ == "__main__":
    main()
