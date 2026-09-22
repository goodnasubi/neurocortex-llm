"""ステップ10（海馬モジュール — リプレイによる固定化, 12.6.28節）の主実験。

naive-finetune（統制1の前提確認） / replay（本設計） / ewc（対照群、統制5）の
3条件で、タスクA→タスクB（排他的なXORパリティ課題）の順次学習後のタスクA
性能保持を比較する。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_hippocampus_replay --seeds 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..basal_ganglia_core import ParitySpec
from ..hippocampus_replay import CONDITIONS, run_condition


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-inputs", type=int, default=8)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--pretrain-steps", type=int, default=500)
    ap.add_argument("--pretrain-batch", type=int, default=64)
    ap.add_argument("--pretrain-lr", type=float, default=0.05)
    ap.add_argument("--pretrain-weight-decay", type=float, default=0.01,
                    help="タスクA事前学習の重み減衰。過度な自信によりフィッシャー情報量が"
                    "無意味に潰れるのを防ぐ（実測で確認、12.6.29節）")
    ap.add_argument("--buffer-size", type=int, default=50)
    ap.add_argument("--fisher-batches", type=int, default=20)
    ap.add_argument("--fisher-batch-size", type=int, default=64)
    ap.add_argument("--ewc-lambda", type=float, default=1000.0,
                    help="10〜100,000の範囲で事前調査したが効果に差がなかったため中央付近を採用")
    ap.add_argument("--b-steps", type=int, default=500)
    ap.add_argument("--b-batch", type=int, default=64)
    ap.add_argument("--b-lr", type=float, default=0.05)
    ap.add_argument("--eval-n", type=int, default=2000)
    ap.add_argument("--control1-threshold", type=float, default=0.15,
                    help="naive-finetuneのタスクA性能がこの値を下回れば忘却が起きたとみなす")
    ap.add_argument("--out", type=Path,
                    default=Path("results/hippocampus_replay/replay_vs_ewc.json"))
    args = ap.parse_args()

    spec_a = ParitySpec(n_inputs=args.n_inputs, relevant=(0, 1))
    spec_b = ParitySpec(n_inputs=args.n_inputs, relevant=(2, 3))
    t0 = time.time()

    all_results: dict[str, list[dict]] = {c: [] for c in CONDITIONS}
    for condition in CONDITIONS:
        for seed in range(args.seeds):
            g = torch.Generator().manual_seed(seed + 50_000)
            torch.manual_seed(seed)
            result = run_condition(condition, args.n_inputs, args.hidden_dim, spec_a, spec_b,
                                   args.pretrain_steps, args.pretrain_batch, args.pretrain_lr,
                                   args.buffer_size, args.fisher_batches, args.fisher_batch_size,
                                   args.ewc_lambda, args.b_steps, args.b_batch, args.b_lr,
                                   args.eval_n, g,
                                   pretrain_weight_decay=args.pretrain_weight_decay)
            all_results[condition].append(result)

    summary: dict[str, dict] = {}
    for condition in CONDITIONS:
        acc_a = torch.tensor([r["acc_a"] for r in all_results[condition]])
        acc_b = torch.tensor([r["acc_b"] for r in all_results[condition]])
        summary[condition] = {
            "acc_a_mean": float(acc_a.mean()),
            "acc_a_std": float(acc_a.std(unbiased=False)),
            "acc_b_mean": float(acc_b.mean()),
            "acc_b_std": float(acc_b.std(unbiased=False)),
        }

    # 統制1: naive-finetuneでタスクA性能が明確に低下すること（忘却が実際に起きる設定であることの確認）。
    naive_acc_a = summary["naive-finetune"]["acc_a_mean"]
    control1_ok = naive_acc_a < (1.0 - args.control1_threshold)
    if not control1_ok:
        print(f"警告: 統制1（忘却が実際に起きることの確認）が崩れています。"
              f" naive-finetuneのacc_a: {naive_acc_a:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "ステップ10 海馬モジュール（リプレイによる固定化）: "
                "naive-finetune（統制1）対 replay（本設計）対 ewc（統制5）の"
                "タスクA→B順次学習後のタスクA性能保持比較（12.6.28節）",
        "条件": vars(args) | {"out": str(args.out)},
        "control1_ok": control1_ok,
        "summary": summary,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n書き出し: {args.out}")
    for condition in CONDITIONS:
        s = summary[condition]
        print(f"{condition:>16}  acc_a={s['acc_a_mean']:.3f}±{s['acc_a_std']:.3f}"
              f"  acc_b={s['acc_b_mean']:.3f}±{s['acc_b_std']:.3f}")


if __name__ == "__main__":
    main()
