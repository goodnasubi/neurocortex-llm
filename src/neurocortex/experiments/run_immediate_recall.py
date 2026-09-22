"""ステップ19（評価指標の設計 — 海馬モジュールの即時想起率, 12.6.46節）の主実験。

`hippocampus.HippocampalMemory`の3モード（sdr・dense・identity）を並べ、
即時想起率・誤想起率を測定する。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_immediate_recall
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..immediate_recall import run_condition

MODES = ("sdr", "dense", "identity")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--value-dim", type=int, default=64)
    ap.add_argument("--n-units", type=int, default=2048)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--n-facts", type=int, default=50)
    ap.add_argument("--n-unknown", type=int, default=50)
    ap.add_argument("--gate-slope", type=float, default=20.0)
    ap.add_argument("--gate-bias", type=float, default=0.3)
    ap.add_argument("--out", type=Path,
                    default=Path("results/immediate_recall/hippocampus_recall_rate.json"))
    args = ap.parse_args()

    t0 = time.time()

    all_results: dict[str, dict] = {m: {"recall_rate": [], "false_positive_rate": []} for m in MODES}
    chance_recall_rate = 1.0 / args.n_facts
    for mode in MODES:
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            g = torch.Generator().manual_seed(seed + 60_000)
            r = run_condition(mode, args.d_model, args.value_dim, args.n_units, args.k,
                              args.n_facts, args.n_unknown, args.gate_slope, args.gate_bias,
                              g, seed)
            all_results[mode]["recall_rate"].append(r.recall_rate)
            all_results[mode]["false_positive_rate"].append(r.false_positive_rate)

    def _stats(values: list[float]) -> dict:
        t = torch.tensor(values)
        return {"mean": float(t.mean()), "std": float(t.std(unbiased=False)), "per_seed": t.tolist()}

    summary: dict[str, dict] = {}
    for mode in MODES:
        r = all_results[mode]
        summary[mode] = {
            "recall_rate": _stats(r["recall_rate"]),
            "false_positive_rate": _stats(r["false_positive_rate"]),
        }

    # 合格条件(1): sdrモードの即時想起率がチャンス水準を明確に上回ること。
    sdr_recall = summary["sdr"]["recall_rate"]["mean"]
    claim1_signal_ok = bool(sdr_recall > chance_recall_rate * 10)
    # 合格条件(2): sdrモードの誤想起率（誤棄却率）が低い水準に収まること。
    sdr_fpr = summary["sdr"]["false_positive_rate"]["mean"]
    claim2_low_false_positive_ok = bool(sdr_fpr < 0.2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "ステップ19 評価指標の設計（海馬モジュールの即時想起率）: "
                "HippocampalMemoryの3分離層モード（sdr・dense・identity）での"
                "即時想起率・誤想起率の実測（12.6.46節）",
        "条件": vars(args) | {"out": str(args.out)},
        "chance_recall_rate（1/n_facts）": chance_recall_rate,
        "claim1_signal_ok（sdrの想起率がチャンス水準を明確に上回るか）": claim1_signal_ok,
        "claim2_low_false_positive_ok（sdrの誤想起率が低いか）": claim2_low_false_positive_ok,
        "summary": summary,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n書き出し: {args.out}")
    print(f"チャンス水準（1/n_facts）: {chance_recall_rate:.4f}")
    print("-- 本実験 --")
    for mode in MODES:
        s = summary[mode]
        rr = s["recall_rate"]
        fp = s["false_positive_rate"]
        print(f"{mode:>10}  recall_rate={rr['mean']:.3f}±{rr['std']:.3f}  "
              f"false_positive_rate={fp['mean']:.3f}±{fp['std']:.3f}")
    print(f"\n合格条件(1) claim1_signal_ok: {claim1_signal_ok}")
    print(f"合格条件(2) claim2_low_false_positive_ok: {claim2_low_false_positive_ok}")


if __name__ == "__main__":
    main()
