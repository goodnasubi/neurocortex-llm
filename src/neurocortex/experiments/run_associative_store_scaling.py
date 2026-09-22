"""ステップ22（海馬モジュール`AssociativeStore`の検索コスト, 12.6.52節）の主実験。

`AssociativeStore`（`hippocampus.py`）単体を対象に、書き込み件数Nに対する
`read`呼び出しの壁時計時間とテンソルのメモリ占有量のスケーリングを実測する。
`run_immediate_recall_capacity.py`（ステップ21）の書き込み件数ループ構造を流用する。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_associative_store_scaling
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

# 対数的に広い範囲（100 〜 30万）。100万は開発環境の実行時間・メモリの都合で
# 30万に調整した（30万件×d_model=64でも十分に対数-対数回帰・外挿の材料になる）。
DEFAULT_N_LIST = (100, 1000, 10_000, 100_000, 300_000)

QUERY_BATCH = 32
N_WARMUP = 1
N_MEASURE = 5


def _bytes_of(store: AssociativeStore) -> int:
    return (store._keys.element_size() * store._keys.nelement()
            + store._values.element_size() * store._values.nelement())


def measure_one(n: int, d_model: int, value_dim: int, seed: int,
                 device: torch.device) -> dict:
    g = torch.Generator().manual_seed(seed)
    store = AssociativeStore(d_model, value_dim, device=device)
    keys = torch.randn(n, d_model, generator=g)
    keys = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    values = torch.randn(n, value_dim, generator=g)
    store.write(keys, values)

    query_g = torch.Generator().manual_seed(seed + 90_000)
    queries = torch.randn(QUERY_BATCH, d_model, generator=query_g)
    queries = queries / queries.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    for _ in range(N_WARMUP):
        store.read(queries)
        if device.type == "cuda":
            torch.cuda.synchronize()

    times = []
    for _ in range(N_MEASURE):
        t0 = time.perf_counter()
        store.read(queries)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    result = {
        "n": n,
        "read_time_median_sec": statistics.median(times),
        "read_time_all_sec": times,
        "memory_bytes": _bytes_of(store),
    }
    if device.type == "cuda":
        result["cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    return result


def loglog_slope(xs: list[float], ys: list[float]) -> float:
    """対数-対数回帰の傾き（numpy.polyfitの1次係数）。"""
    log_x = np.log10(np.asarray(xs, dtype=float))
    log_y = np.log10(np.asarray(ys, dtype=float))
    slope, _intercept = np.polyfit(log_x, log_y, 1)
    return float(slope)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--value-dim", type=int, default=64)
    ap.add_argument("--n-list", type=int, nargs="+", default=list(DEFAULT_N_LIST))
    ap.add_argument("--out", type=Path,
                    default=Path("results/associative_store_scaling/scaling.json"))
    args = ap.parse_args()

    t0 = time.time()
    cuda_available = torch.cuda.is_available()
    device_cpu = torch.device("cpu")

    # per_n[n] = {"read_time_median_sec": [...per seed...], "memory_bytes": [...]}
    per_n: dict[int, dict] = {n: {"read_time_median_sec": [], "memory_bytes": []}
                              for n in args.n_list}
    cuda_per_n: dict[int, dict] = {n: {"read_time_median_sec": [], "cuda_max_memory_allocated_bytes": []}
                                   for n in args.n_list} if cuda_available else {}

    for n in args.n_list:
        for seed in range(args.seeds):
            r = measure_one(n, args.d_model, args.value_dim, seed, device_cpu)
            per_n[n]["read_time_median_sec"].append(r["read_time_median_sec"])
            per_n[n]["memory_bytes"].append(r["memory_bytes"])
            if cuda_available:
                device_cuda = torch.device("cuda")
                rc = measure_one(n, args.d_model, args.value_dim, seed, device_cuda)
                cuda_per_n[n]["read_time_median_sec"].append(rc["read_time_median_sec"])
                cuda_per_n[n]["cuda_max_memory_allocated_bytes"].append(
                    rc["cuda_max_memory_allocated_bytes"])

    def _stats(values: list[float]) -> dict:
        t = torch.tensor(values, dtype=torch.float64)
        return {"mean": float(t.mean()), "std": float(t.std(unbiased=False)), "per_seed": t.tolist()}

    summary: dict[str, dict] = {}
    for n in args.n_list:
        summary[str(n)] = {
            "read_time_sec": _stats(per_n[n]["read_time_median_sec"]),
            "memory_bytes": _stats(per_n[n]["memory_bytes"]),
        }
        if cuda_available:
            summary[str(n)]["cuda_read_time_sec"] = _stats(cuda_per_n[n]["read_time_median_sec"])
            summary[str(n)]["cuda_max_memory_allocated_bytes"] = _stats(
                cuda_per_n[n]["cuda_max_memory_allocated_bytes"])

    n_sorted = sorted(args.n_list)
    mean_read_times = [summary[str(n)]["read_time_sec"]["mean"] for n in n_sorted]
    mean_memory = [summary[str(n)]["memory_bytes"]["mean"] for n in n_sorted]

    slope_a_read_time = loglog_slope(n_sorted, mean_read_times)
    slope_b_memory = loglog_slope(n_sorted, mean_memory)

    # メモリは理論的に厳密な線形（bytes = n * (d_model + value_dim) * 4）なので、
    # 実測の傾きに加えて外挿値も算出する。
    bytes_per_n = 4 * (args.d_model + args.value_dim)  # float32
    extrapolate_n = 1_000_000
    memory_extrapolated_bytes_theoretical = bytes_per_n * extrapolate_n
    # 実測の対数-対数回帰から外挿（切片も使う）
    log_n_sorted = np.log10(np.asarray(n_sorted, dtype=float))
    log_mem = np.log10(np.asarray(mean_memory, dtype=float))
    slope_fit, intercept_fit = np.polyfit(log_n_sorted, log_mem, 1)
    memory_extrapolated_bytes_fit = float(10 ** (slope_fit * np.log10(extrapolate_n) + intercept_fit))

    max_n = n_sorted[-1]
    memory_at_max_n_bytes = summary[str(max_n)]["memory_bytes"]["mean"]

    claim_a_control_ok = bool(0.7 <= slope_a_read_time <= 1.3)
    claim_b_slope_ok = bool(0.9 <= slope_b_memory <= 1.1)
    claim_b_extrapolation_ok = bool(memory_extrapolated_bytes_fit > 100 * 1024 * 1024)
    claim_b_ok = bool(claim_b_slope_ok and claim_b_extrapolation_ok)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "ステップ22（海馬モジュールAssociativeStoreの検索コスト — 書き込み件数に対する"
                "探索時間・メモリのスケーリング, 12.6.52節）: AssociativeStore単体を対象に、"
                "read呼び出しの壁時計時間中央値とテンソルのメモリ占有量を書き込み件数Nに対して実測",
        "条件": vars(args) | {"out": str(args.out), "query_batch": QUERY_BATCH,
                              "n_warmup": N_WARMUP, "n_measure": N_MEASURE,
                              "cuda_available": cuda_available},
        "n_list": n_sorted,
        "summary": summary,
        "claim_a_read_time_loglog_slope": slope_a_read_time,
        "claim_a_control_ok（0.7-1.3に収まるか）": claim_a_control_ok,
        "claim_b_memory_loglog_slope": slope_b_memory,
        "claim_b_slope_ok（0.9-1.1に収まるか）": claim_b_slope_ok,
        "memory_at_max_n_bytes": memory_at_max_n_bytes,
        "max_n": max_n,
        "memory_extrapolated_bytes_at_n=1e6_theoretical": memory_extrapolated_bytes_theoretical,
        "memory_extrapolated_bytes_at_n=1e6_fit": memory_extrapolated_bytes_fit,
        "claim_b_extrapolation_ok（外挿値が100MBを超えるか）": claim_b_extrapolation_ok,
        "claim_b_ok": claim_b_ok,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n書き出し: {args.out}")
    print("-- 本実験（N x 指標） --")
    for n in n_sorted:
        s = summary[str(n)]
        rt = s["read_time_sec"]
        mem = s["memory_bytes"]
        print(f"N={n:>8}  read_time={rt['mean']*1000:.3f}ms±{rt['std']*1000:.3f}ms  "
              f"memory={mem['mean']/1024/1024:.2f}MB±{mem['std']/1024/1024:.4f}MB")
    print(f"\n主張(a) read_time loglog slope: {slope_a_read_time:.3f}  ok={claim_a_control_ok}")
    print(f"主張(b) memory loglog slope: {slope_b_memory:.3f}  ok(slope)={claim_b_slope_ok}")
    print(f"  N=1e6 外挿(理論値): {memory_extrapolated_bytes_theoretical/1024/1024:.1f}MB, "
          f"外挿(回帰): {memory_extrapolated_bytes_fit/1024/1024:.1f}MB  ok(extrap)={claim_b_extrapolation_ok}")
    print(f"主張(b) claim_b_ok: {claim_b_ok}")


if __name__ == "__main__":
    main()
