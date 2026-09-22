"""ステップ24（`AssociativeStore`ANNモードのベクトル化再実装による速度優位の再検証,
12.6.56節）の主実験。

`AssociativeStore.read`（exact）・`read_approx`（ステップ23の旧Pythonループ実装）・
`read_approx_batched`（ステップ24のベクトル化再実装）の3方式を、書き込み件数Nに
対して同一シードで比較する。`run_associative_store_ann.py`（ステップ23）の
`measure_one`をそのまま再利用し、CLI引数・出力JSON構造を踏襲した上で、
旧実装対新実装の回帰比較（top-1一致率の一致確認）を追加する。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_associative_store_ann_vectorized
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .run_associative_store_ann import DEFAULT_N_LIST, loglog_slope, measure_one

# ステップ23実測値（12.6.54〜55節）。統制・主張(b)の比較基準として使う。
STEP23_TOP1_MATCH_RATE_MEAN = 0.530
STEP23_TOP1_MATCH_RATE_BY_N: dict[int, float] = {}  # N別の詳細比較は行わない（設計は全体平均のみ要求）


def _loglog_slope_mid_large(n_sorted: list[int], ys: list[float]) -> float:
    if len(n_sorted) <= 2:
        return loglog_slope(n_sorted, ys)
    return loglog_slope(n_sorted[1:], ys[1:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--value-dim", type=int, default=64)
    ap.add_argument("--n-list", type=int, nargs="+", default=list(DEFAULT_N_LIST))
    ap.add_argument("--out", type=Path,
                    default=Path("results/associative_store_scaling/ann_vectorized.json"))
    args = ap.parse_args()

    t0 = time.time()

    per_n: dict[int, dict] = {
        n: {"exact_read_time_median_sec": [], "ann_read_time_median_sec": [],
            "ann_batched_read_time_median_sec": [], "memory_bytes": [],
            "top1_match_rate": [], "top1_match_rate_batched": [],
            "top1_match_rate_old_vs_batched": []}
        for n in args.n_list
    }

    for n in args.n_list:
        for seed in range(args.seeds):
            r = measure_one(n, args.d_model, args.value_dim, seed)
            per_n[n]["exact_read_time_median_sec"].append(r["exact_read_time_median_sec"])
            per_n[n]["ann_read_time_median_sec"].append(r["ann_read_time_median_sec"])
            per_n[n]["ann_batched_read_time_median_sec"].append(r["ann_batched_read_time_median_sec"])
            per_n[n]["memory_bytes"].append(r["memory_bytes"])
            per_n[n]["top1_match_rate"].append(r["top1_match_rate"])
            per_n[n]["top1_match_rate_batched"].append(r["top1_match_rate_batched"])
            per_n[n]["top1_match_rate_old_vs_batched"].append(r["top1_match_rate_old_vs_batched"])

    def _stats(values: list[float]) -> dict:
        t = torch.tensor(values, dtype=torch.float64)
        return {"mean": float(t.mean()), "std": float(t.std(unbiased=False)), "per_seed": t.tolist()}

    summary: dict[str, dict] = {}
    for n in args.n_list:
        summary[str(n)] = {
            "exact_read_time_sec": _stats(per_n[n]["exact_read_time_median_sec"]),
            "ann_read_time_sec": _stats(per_n[n]["ann_read_time_median_sec"]),
            "ann_batched_read_time_sec": _stats(per_n[n]["ann_batched_read_time_median_sec"]),
            "memory_bytes": _stats(per_n[n]["memory_bytes"]),
            "top1_match_rate": _stats(per_n[n]["top1_match_rate"]),
            "top1_match_rate_batched": _stats(per_n[n]["top1_match_rate_batched"]),
            "top1_match_rate_old_vs_batched": _stats(per_n[n]["top1_match_rate_old_vs_batched"]),
        }

    n_sorted = sorted(args.n_list)
    mean_exact_times = [summary[str(n)]["exact_read_time_sec"]["mean"] for n in n_sorted]
    mean_ann_batched_times = [summary[str(n)]["ann_batched_read_time_sec"]["mean"] for n in n_sorted]

    slope_exact_all = loglog_slope(n_sorted, mean_exact_times)
    slope_exact_mid_large = _loglog_slope_mid_large(n_sorted, mean_exact_times)
    slope_ann_batched_all = loglog_slope(n_sorted, mean_ann_batched_times)
    slope_ann_batched_mid_large = _loglog_slope_mid_large(n_sorted, mean_ann_batched_times)

    max_n = n_sorted[-1]
    exact_time_at_max_n = summary[str(max_n)]["exact_read_time_sec"]["mean"]
    ann_batched_time_at_max_n = summary[str(max_n)]["ann_batched_read_time_sec"]["mean"]
    speedup_at_max_n = (exact_time_at_max_n / ann_batched_time_at_max_n
                        if ann_batched_time_at_max_n > 0 else float("inf"))

    all_match_rates_batched = [rate for n in n_sorted for rate in per_n[n]["top1_match_rate_batched"]]
    mean_match_rate_batched = float(np.mean(all_match_rates_batched))

    # N別の新実装top-1一致率平均（統制の内訳確認用。ステップ23はN別実測値をこの形式では
    # 保存していないため、全体平均との比較のみを合否判定に用いる）
    match_rate_by_n = {
        str(n): float(np.mean(per_n[n]["top1_match_rate_batched"])) for n in n_sorted
    }

    # 統制: バッチ版top-1一致率の全体平均が、ステップ23の実測値（0.530）から±0.02以内
    control_diff = abs(mean_match_rate_batched - STEP23_TOP1_MATCH_RATE_MEAN)
    control_ok = bool(control_diff <= 0.02)

    # 主張(a): N=30万でバッチ版annがexactと比べ2倍以上高速、かつ対数-対数回帰の傾きが0.5以下
    claim_a_speedup_ok = bool(speedup_at_max_n >= 2.0)
    claim_a_slope_ok = bool(slope_ann_batched_all <= 0.5)
    claim_a_ok = bool(claim_a_speedup_ok and claim_a_slope_ok)

    # 主張(b): バッチ版top-1一致率の全体平均が、ステップ23実測値0.530から±0.05以内
    claim_b_diff = abs(mean_match_rate_batched - STEP23_TOP1_MATCH_RATE_MEAN)
    claim_b_ok = bool(claim_b_diff <= 0.05)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "ステップ24（AssociativeStoreANNモードのベクトル化再実装による速度優位の"
                "再検証, 12.6.56節）: exact（厳密探索）・ann（ステップ23の旧Pythonループ版）・"
                "ann_batched（本ステップのベクトル化再実装）を同一N・同一シードで比較し、"
                "read壁時計時間とtop-1一致率を実測",
        "条件": vars(args) | {"out": str(args.out), "n_warmup": 1, "n_measure": 5},
        "n_list": n_sorted,
        "summary": summary,
        "control_top1_match_rate_batched_mean": mean_match_rate_batched,
        "control_top1_match_rate_batched_by_n": match_rate_by_n,
        "control_diff_from_step23（0.530）": control_diff,
        "control_ok（ステップ23実測値0.530から±0.02以内か）": control_ok,
        "control_exact_read_time_loglog_slope_mid_large": slope_exact_mid_large,
        "control_exact_read_time_loglog_slope_all": slope_exact_all,
        "claim_a_ann_batched_read_time_loglog_slope_all": slope_ann_batched_all,
        "claim_a_ann_batched_read_time_loglog_slope_mid_large": slope_ann_batched_mid_large,
        "claim_a_speedup_at_max_n": speedup_at_max_n,
        "claim_a_speedup_ok（2倍以上高速か）": claim_a_speedup_ok,
        "claim_a_slope_ok（傾きが0.5以下か）": claim_a_slope_ok,
        "claim_a_ok": claim_a_ok,
        "claim_b_mean_top1_match_rate_batched": mean_match_rate_batched,
        "claim_b_diff_from_step23（0.530）": claim_b_diff,
        "claim_b_ok（ステップ23実測値0.530から±0.05以内か）": claim_b_ok,
        "max_n": max_n,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n書き出し: {args.out}")
    print("-- 本実験（N x 指標） --")
    for n in n_sorted:
        s = summary[str(n)]
        et = s["exact_read_time_sec"]
        at = s["ann_read_time_sec"]
        bt = s["ann_batched_read_time_sec"]
        mr = s["top1_match_rate_batched"]
        ov = s["top1_match_rate_old_vs_batched"]
        print(f"N={n:>8}  exact={et['mean']*1000:.3f}ms±{et['std']*1000:.3f}ms  "
              f"ann(旧)={at['mean']*1000:.3f}ms±{at['std']*1000:.3f}ms  "
              f"ann_batched={bt['mean']*1000:.3f}ms±{bt['std']*1000:.3f}ms  "
              f"top1_match(vs exact)={mr['mean']:.3f}±{mr['std']:.3f}  "
              f"top1_match(旧vs新)={ov['mean']:.3f}±{ov['std']:.3f}")
    print(f"\n統制: バッチ版top1一致率平均={mean_match_rate_batched:.4f}  "
          f"ステップ23実測値との差={control_diff:.4f}  ok={control_ok}")
    print(f"主張(a): N={max_n}での高速化率={speedup_at_max_n:.2f}倍  "
          f"ok(speedup)={claim_a_speedup_ok}  ann_batched傾き={slope_ann_batched_all:.3f}  "
          f"ok(slope)={claim_a_slope_ok}  claim_a_ok={claim_a_ok}")
    print(f"主張(b): 全体top-1一致率平均={mean_match_rate_batched:.4f}  "
          f"ステップ23実測値との差={claim_b_diff:.4f}  claim_b_ok={claim_b_ok}")


if __name__ == "__main__":
    main()
