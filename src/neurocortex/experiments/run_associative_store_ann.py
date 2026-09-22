"""ステップ23（`AssociativeStore`へのANN候補絞り込みの実装と効果測定, 12.6.54節）の主実験。

`AssociativeStore.read`（exact=True相当の厳密探索）と、新設した
`AssociativeStore.read_approx`（k-means粗量子化による2段階近似探索）を、
書き込み件数Nに対して同一シードで比較する。ステップ22
（`run_associative_store_scaling.py`）のNループ構造を拡張し、`exact`・`ann`
の2モードでread壁時計時間とtop-1一致率（annの返す最近傍がexactのargmax結果
と一致する割合）を測定する。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_associative_store_ann
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from ..hippocampus import AssociativeStore

# ステップ22と同一のN範囲（100 〜 30万）。
DEFAULT_N_LIST = (100, 1000, 10_000, 100_000, 300_000)

QUERY_BATCH = 32
N_WARMUP = 1
N_MEASURE = 5


def _bytes_of(store: AssociativeStore) -> int:
    return (store._keys.element_size() * store._keys.nelement()
            + store._values.element_size() * store._values.nelement())


def measure_one(n: int, d_model: int, value_dim: int, seed: int) -> dict:
    """同一N・同一シードで`exact`（厳密）・`ann`（近似, 旧Pythonループ実装）・
    `ann_batched`（近似, ステップ24のベクトル化再実装）の3方式のreadを測定する。
    """
    g = torch.Generator().manual_seed(seed)
    keys = torch.randn(n, d_model, generator=g)
    keys = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    values = torch.randn(n, value_dim, generator=g)

    store_exact = AssociativeStore(d_model, value_dim, exact=True)
    store_exact.write(keys, values)
    store_ann = AssociativeStore(d_model, value_dim, exact=True)
    store_ann.write(keys, values)
    store_ann_batched = AssociativeStore(d_model, value_dim, exact=True)
    store_ann_batched.write(keys, values)

    query_g = torch.Generator().manual_seed(seed + 90_000)
    queries = torch.randn(QUERY_BATCH, d_model, generator=query_g)
    queries = queries / queries.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    # --- exact ---
    for _ in range(N_WARMUP):
        store_exact.read(queries)
    exact_times = []
    for _ in range(N_MEASURE):
        t0 = time.perf_counter()
        _, exact_stats = store_exact.read(queries)
        exact_times.append(time.perf_counter() - t0)

    # --- ann（旧実装, ウォームアップでk-means索引を構築・キャッシュしてから計測） ---
    for _ in range(N_WARMUP):
        store_ann.read_approx(queries, seed=seed)
    ann_times = []
    ann_stats = None
    for _ in range(N_MEASURE):
        t0 = time.perf_counter()
        _, ann_stats = store_ann.read_approx(queries, seed=seed)
        ann_times.append(time.perf_counter() - t0)

    # --- ann_batched（ステップ24のベクトル化再実装） ---
    for _ in range(N_WARMUP):
        store_ann_batched.read_approx_batched(queries, seed=seed)
    ann_batched_times = []
    ann_batched_stats = None
    for _ in range(N_MEASURE):
        t0 = time.perf_counter()
        _, ann_batched_stats = store_ann_batched.read_approx_batched(queries, seed=seed)
        ann_batched_times.append(time.perf_counter() - t0)

    top1_match = (exact_stats.top1_index == ann_stats.top1_index).float().mean().item()
    top1_match_batched = (exact_stats.top1_index == ann_batched_stats.top1_index).float().mean().item()
    # 統制: 旧実装と新実装の候補選択がアルゴリズム的に同じであることの確認
    top1_match_old_vs_batched = (ann_stats.top1_index == ann_batched_stats.top1_index).float().mean().item()

    return {
        "n": n,
        "exact_read_time_median_sec": statistics.median(exact_times),
        "ann_read_time_median_sec": statistics.median(ann_times),
        "ann_batched_read_time_median_sec": statistics.median(ann_batched_times),
        "memory_bytes": _bytes_of(store_exact),
        "top1_match_rate": top1_match,
        "top1_match_rate_batched": top1_match_batched,
        "top1_match_rate_old_vs_batched": top1_match_old_vs_batched,
    }


def loglog_slope(xs: list[float], ys: list[float]) -> float:
    """対数-対数回帰の傾き（numpy.polyfitの1次係数）。"""
    log_x = np.log10(np.asarray(xs, dtype=float))
    log_y = np.log10(np.asarray(ys, dtype=float))
    slope, _intercept = np.polyfit(log_x, log_y, 1)
    return float(slope)


def _loglog_slope_mid_large(n_sorted: list[int], ys: list[float]) -> float:
    """ステップ22に倣い、中〜大規模区間（下位1点を除く）での傾きも算出する。"""
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
                    default=Path("results/associative_store_scaling/ann_comparison.json"))
    args = ap.parse_args()

    t0 = time.time()

    # per_n[n] = {"exact_read_time_median_sec": [...per seed...], ...}
    per_n: dict[int, dict] = {
        n: {"exact_read_time_median_sec": [], "ann_read_time_median_sec": [],
            "memory_bytes": [], "top1_match_rate": []}
        for n in args.n_list
    }

    for n in args.n_list:
        for seed in range(args.seeds):
            r = measure_one(n, args.d_model, args.value_dim, seed)
            per_n[n]["exact_read_time_median_sec"].append(r["exact_read_time_median_sec"])
            per_n[n]["ann_read_time_median_sec"].append(r["ann_read_time_median_sec"])
            per_n[n]["memory_bytes"].append(r["memory_bytes"])
            per_n[n]["top1_match_rate"].append(r["top1_match_rate"])

    def _stats(values: list[float]) -> dict:
        t = torch.tensor(values, dtype=torch.float64)
        return {"mean": float(t.mean()), "std": float(t.std(unbiased=False)), "per_seed": t.tolist()}

    summary: dict[str, dict] = {}
    for n in args.n_list:
        summary[str(n)] = {
            "exact_read_time_sec": _stats(per_n[n]["exact_read_time_median_sec"]),
            "ann_read_time_sec": _stats(per_n[n]["ann_read_time_median_sec"]),
            "memory_bytes": _stats(per_n[n]["memory_bytes"]),
            "top1_match_rate": _stats(per_n[n]["top1_match_rate"]),
        }

    n_sorted = sorted(args.n_list)
    mean_exact_times = [summary[str(n)]["exact_read_time_sec"]["mean"] for n in n_sorted]
    mean_ann_times = [summary[str(n)]["ann_read_time_sec"]["mean"] for n in n_sorted]

    slope_exact_all = loglog_slope(n_sorted, mean_exact_times)
    slope_exact_mid_large = _loglog_slope_mid_large(n_sorted, mean_exact_times)
    slope_ann_all = loglog_slope(n_sorted, mean_ann_times)
    slope_ann_mid_large = _loglog_slope_mid_large(n_sorted, mean_ann_times)

    max_n = n_sorted[-1]
    exact_time_at_max_n = summary[str(max_n)]["exact_read_time_sec"]["mean"]
    ann_time_at_max_n = summary[str(max_n)]["ann_read_time_sec"]["mean"]
    speedup_at_max_n = exact_time_at_max_n / ann_time_at_max_n if ann_time_at_max_n > 0 else float("inf")

    all_match_rates = [rate for n in n_sorted for rate in per_n[n]["top1_match_rate"]]
    mean_match_rate = float(np.mean(all_match_rates))

    # 統制: exactモードの傾き（中〜大規模区間）がステップ22の実測値（0.986）を再現するか
    control_ok = bool(0.7 <= slope_exact_mid_large <= 1.3)
    # 主張(a): N=30万でannが2倍以上高速、かつannの対数-対数回帰の傾きが0.5以下
    claim_a_speedup_ok = bool(speedup_at_max_n >= 2.0)
    claim_a_slope_ok = bool(slope_ann_all <= 0.5)
    claim_a_ok = bool(claim_a_speedup_ok and claim_a_slope_ok)
    # 主張(b): 全N・全シードでのtop-1一致率の平均が0.95以上
    claim_b_ok = bool(mean_match_rate >= 0.95)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "ステップ23（AssociativeStoreへのANN候補絞り込みの実装と効果測定, "
                "12.6.54節）: exact（厳密探索）・ann（k-means粗量子化による近似探索）を"
                "同一N・同一シードで比較し、read壁時計時間とtop-1一致率を実測",
        "条件": vars(args) | {"out": str(args.out), "query_batch": QUERY_BATCH,
                              "n_warmup": N_WARMUP, "n_measure": N_MEASURE},
        "n_list": n_sorted,
        "summary": summary,
        "control_exact_read_time_loglog_slope_mid_large": slope_exact_mid_large,
        "control_exact_read_time_loglog_slope_all": slope_exact_all,
        "control_ok（0.7-1.3に収まるか、ステップ22の0.986を再現）": control_ok,
        "claim_a_ann_read_time_loglog_slope_all": slope_ann_all,
        "claim_a_ann_read_time_loglog_slope_mid_large": slope_ann_mid_large,
        "claim_a_speedup_at_max_n": speedup_at_max_n,
        "claim_a_speedup_ok（2倍以上高速か）": claim_a_speedup_ok,
        "claim_a_slope_ok（傾きが0.5以下か）": claim_a_slope_ok,
        "claim_a_ok": claim_a_ok,
        "claim_b_mean_top1_match_rate": mean_match_rate,
        "claim_b_ok（平均一致率が0.95以上か）": claim_b_ok,
        "max_n": max_n,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n書き出し: {args.out}")
    print("-- 本実験（N x 指標） --")
    for n in n_sorted:
        s = summary[str(n)]
        et = s["exact_read_time_sec"]
        at = s["ann_read_time_sec"]
        mr = s["top1_match_rate"]
        print(f"N={n:>8}  exact={et['mean']*1000:.3f}ms±{et['std']*1000:.3f}ms  "
              f"ann={at['mean']*1000:.3f}ms±{at['std']*1000:.3f}ms  "
              f"top1_match={mr['mean']:.3f}±{mr['std']:.3f}")
    print(f"\n統制: exact傾き(中〜大規模)={slope_exact_mid_large:.3f}  ok={control_ok}")
    print(f"主張(a): N={max_n}での高速化率={speedup_at_max_n:.2f}倍  "
          f"ok(speedup)={claim_a_speedup_ok}  ann傾き={slope_ann_all:.3f}  ok(slope)={claim_a_slope_ok}"
          f"  claim_a_ok={claim_a_ok}")
    print(f"主張(b): 全体top-1一致率平均={mean_match_rate:.4f}  claim_b_ok={claim_b_ok}")


if __name__ == "__main__":
    main()
