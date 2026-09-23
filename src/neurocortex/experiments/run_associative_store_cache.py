"""ステップ33（海馬モジュール`DiskBackedAssociativeStore`のreadレイテンシ緩和：
頻出キーのRAMキャッシュ層, 12.6.74〜75節）の主実験。

ステップ22・32の計測パイプライン（N=100〜30万・64次元・壁時計時間中央値・
ピークワーキングセット〈Windows: ctypes経由PeakWorkingSetSize〉の実測、
`torch.equal`による正確性検証）を踏襲する。

一様分布クエリ（ステップ22・32と同条件）に加え、Zipf分布で偏りを持たせた
合成クエリ集合（上位10%チャンクが全クエリの50〜90%を占める）を新設し、
キャッシュ容量0（ベースライン）・4・16・64チャンクで比較する。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_associative_store_cache
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wintypes
import json
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from ..hippocampus import DiskBackedAssociativeStore

# ステップ22・32と揃えたレンジ。N=100万は実行時間の都合で除外（ステップ22・32と同じ理由）。
DEFAULT_N_LIST = (100, 1000, 10_000, 100_000, 300_000)
DEFAULT_CACHE_SIZES = (0, 4, 16, 64)

QUERY_BATCH = 32
N_WARMUP = 1
N_MEASURE = 5
DISK_KEY_CHUNK = 4096

STEP22_RAM_MEMORY_AT_N300K_BYTES = 146.48 * 1024 * 1024  # 12.7節記載のステップ22実測値


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def peak_working_set_bytes() -> int:
    """プロセス全体のピークワーキングセットサイズ（Windows版`resource.getrusage`代替）。"""
    if sys.platform != "win32":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCounters),
                                            wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    if not ok:
        raise OSError(f"GetProcessMemoryInfoに失敗した (GetLastError={ctypes.get_last_error()})")
    return int(counters.PeakWorkingSetSize)


def make_uniform_queries(n_queries: int, key_dim: int, g: torch.Generator) -> torch.Tensor:
    return torch.nn.functional.normalize(torch.randn(n_queries, key_dim, generator=g), dim=-1)


def make_skewed_queries(n_queries: int, key_dim: int, n_chunks: int, key_chunk: int,
                         keys: torch.Tensor, g: torch.Generator,
                         zipf_s: float = 1.5) -> torch.Tensor:
    """Zipf分布で「どのチャンクの最良一致になるか」に偏りを持たせた合成クエリ集合。

    各クエリについて、Zipf分布でチャンク番号を選び、そのチャンク内のキーの1つを
    小さなノイズを加えて再利用することで、そのキーが（高確率で）最良一致になる
    ようにする。上位10%のチャンクへの集中度はzipf_sで制御する（大きいほど偏る）。
    """
    ranks = np.arange(1, n_chunks + 1, dtype=np.float64)
    weights = 1.0 / (ranks ** zipf_s)
    weights /= weights.sum()
    rng = np.random.default_rng(int(g.initial_seed()) % (2**32))
    chosen_chunks = rng.choice(n_chunks, size=n_queries, p=weights)

    queries = torch.empty(n_queries, key_dim)
    for i, c in enumerate(chosen_chunks):
        start = c * key_chunk
        end = min(start + key_chunk, keys.shape[0])
        row = start + int(rng.integers(0, max(end - start, 1)))
        row = min(row, keys.shape[0] - 1)
        noise = torch.from_numpy(rng.normal(scale=0.01, size=key_dim)).to(torch.float32)
        q = keys[row] + noise
        queries[i] = q
    return torch.nn.functional.normalize(queries, dim=-1)


def top10pct_share(chosen_chunks: np.ndarray, n_chunks: int) -> float:
    counts = np.bincount(chosen_chunks, minlength=n_chunks)
    top_k = max(1, n_chunks // 10)
    sorted_counts = np.sort(counts)[::-1]
    return float(sorted_counts[:top_k].sum() / max(counts.sum(), 1))


def measure(n: int, d_model: int, value_dim: int, seed: int, cache_size: int,
            persist_root: Path, query_kind: str) -> dict:
    persist_dir = persist_root / f"n{n}_seed{seed}_c{cache_size}_{query_kind}"
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    peak_baseline = peak_working_set_bytes()
    g = torch.Generator().manual_seed(seed)
    store = DiskBackedAssociativeStore(d_model, value_dim, persist_dir=persist_dir,
                                        exact=True, key_chunk=DISK_KEY_CHUNK,
                                        cache_size=cache_size)
    keys = torch.nn.functional.normalize(torch.randn(n, d_model, generator=g), dim=-1)
    values = torch.nn.functional.normalize(torch.randn(n, value_dim, generator=g), dim=-1)
    store.write(keys, values)

    query_g = torch.Generator().manual_seed(seed + 90_000)
    n_chunks = max(1, (n + DISK_KEY_CHUNK - 1) // DISK_KEY_CHUNK)
    if query_kind == "uniform":
        queries = make_uniform_queries(QUERY_BATCH, d_model, query_g)
        skew = None
    else:
        rng = np.random.default_rng(int(query_g.initial_seed()) % (2**32))
        ranks = np.arange(1, n_chunks + 1, dtype=np.float64)
        weights = 1.0 / (ranks ** 1.5)
        weights /= weights.sum()
        chosen_chunks = rng.choice(n_chunks, size=QUERY_BATCH, p=weights)
        skew = top10pct_share(chosen_chunks, n_chunks)
        queries = make_skewed_queries(QUERY_BATCH, d_model, n_chunks, DISK_KEY_CHUNK,
                                       keys, query_g)

    for _ in range(N_WARMUP):
        store.read(queries)
    times = []
    for _ in range(N_MEASURE):
        # キャッシュ状態をウォームアップ後の定常状態に保つため、測定間でクリアしない
        # （実運用のホットキャッシュを想定）。ベースライン（cache_size=0）は毎回無効。
        t0 = time.perf_counter()
        out, stats = store.read(queries)
        times.append(time.perf_counter() - t0)
    peak_after = peak_working_set_bytes()

    result = {
        "read_time_median_sec": statistics.median(times),
        "peak_working_set_delta_bytes": max(0, peak_after - peak_baseline),
        "out": out,
        "stats": stats,
        "queries": queries,
        "top10pct_share": skew,
        "cache_hits": store.cache_hits,
        "cache_misses": store.cache_misses,
    }
    shutil.rmtree(persist_dir, ignore_errors=True)
    return result


def loglog_slope(xs: list[float], ys: list[float]) -> float:
    log_x = np.log10(np.asarray(xs, dtype=float))
    log_y = np.log10(np.clip(np.asarray(ys, dtype=float), 1.0, None))
    slope, _intercept = np.polyfit(log_x, log_y, 1)
    return float(slope)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--value-dim", type=int, default=64)
    ap.add_argument("--n-list", type=int, nargs="+", default=list(DEFAULT_N_LIST))
    ap.add_argument("--cache-sizes", type=int, nargs="+", default=list(DEFAULT_CACHE_SIZES))
    ap.add_argument("--out", type=Path,
                    default=Path("results/associative_store_scaling/cache_latency.json"))
    args = ap.parse_args()

    t0 = time.time()
    persist_root = Path(tempfile.mkdtemp(prefix="assoc_store_cache_"))

    # per_n[query_kind][cache_size][n] = {read_time:[], peak_ws:[]}
    per: dict[str, dict[int, dict[int, dict]]] = {
        qk: {c: {n: {"read_time": [], "peak_ws": [], "top10pct_share": []}
                 for n in args.n_list}
             for c in args.cache_sizes}
        for qk in ("uniform", "skewed")
    }
    accuracy_mismatches: list[str] = []

    try:
        for query_kind in ("uniform", "skewed"):
            for n in args.n_list:
                for seed in range(args.seeds):
                    baseline = measure(n, args.d_model, args.value_dim, seed, 0,
                                        persist_root, query_kind)
                    per[query_kind][0][n]["read_time"].append(baseline["read_time_median_sec"])
                    per[query_kind][0][n]["peak_ws"].append(baseline["peak_working_set_delta_bytes"])
                    if baseline["top10pct_share"] is not None:
                        per[query_kind][0][n]["top10pct_share"].append(baseline["top10pct_share"])

                    for cache_size in args.cache_sizes:
                        if cache_size == 0:
                            continue
                        cached = measure(n, args.d_model, args.value_dim, seed, cache_size,
                                          persist_root, query_kind)
                        per[query_kind][cache_size][n]["read_time"].append(
                            cached["read_time_median_sec"])
                        per[query_kind][cache_size][n]["peak_ws"].append(
                            cached["peak_working_set_delta_bytes"])
                        if cached["top10pct_share"] is not None:
                            per[query_kind][cache_size][n]["top10pct_share"].append(
                                cached["top10pct_share"])

                        # 主張(b): 同一キー・バリュー・クエリでキャッシュあり/なしのread出力を比較。
                        if not torch.equal(baseline["stats"].top1_index, cached["stats"].top1_index):
                            accuracy_mismatches.append(
                                f"{query_kind} n={n} seed={seed} cache={cache_size}: top1_index不一致")
                        if not torch.equal(baseline["out"], cached["out"]):
                            accuracy_mismatches.append(
                                f"{query_kind} n={n} seed={seed} cache={cache_size}: value(out)不一致")
                        if not torch.equal(baseline["stats"].max_score, cached["stats"].max_score):
                            accuracy_mismatches.append(
                                f"{query_kind} n={n} seed={seed} cache={cache_size}: max_score不一致")
    finally:
        shutil.rmtree(persist_root, ignore_errors=True)

    def _stats(values: list[float]) -> dict:
        if not values:
            return {"mean": None, "std": None, "median": None, "per_seed": []}
        t = torch.tensor(values, dtype=torch.float64)
        return {"mean": float(t.mean()), "std": float(t.std(unbiased=False)),
                "median": float(statistics.median(values)), "per_seed": t.tolist()}

    summary: dict[str, dict] = {}
    for query_kind in ("uniform", "skewed"):
        summary[query_kind] = {}
        for n in args.n_list:
            summary[query_kind][str(n)] = {}
            baseline_stats = _stats(per[query_kind][0][n]["read_time"])
            summary[query_kind][str(n)]["cache0_read_time_sec"] = baseline_stats
            summary[query_kind][str(n)]["cache0_peak_ws_bytes"] = _stats(per[query_kind][0][n]["peak_ws"])
            if per[query_kind][0][n]["top10pct_share"]:
                summary[query_kind][str(n)]["top10pct_share_mean"] = float(
                    np.mean(per[query_kind][0][n]["top10pct_share"]))
            for cache_size in args.cache_sizes:
                if cache_size == 0:
                    continue
                rt_stats = _stats(per[query_kind][cache_size][n]["read_time"])
                ws_stats = _stats(per[query_kind][cache_size][n]["peak_ws"])
                summary[query_kind][str(n)][f"cache{cache_size}_read_time_sec"] = rt_stats
                summary[query_kind][str(n)][f"cache{cache_size}_peak_ws_bytes"] = ws_stats
                if baseline_stats["median"]:
                    reduction = 1.0 - (rt_stats["median"] / baseline_stats["median"])
                else:
                    reduction = None
                summary[query_kind][str(n)][f"cache{cache_size}_read_time_reduction_vs_cache0"] = reduction

    # 主張(a): 偏りのあるクエリ分布・N=30万・キャッシュ容量64チャンク
    max_n = max(args.n_list)
    nonzero_cache_sizes = [c for c in args.cache_sizes if c > 0]
    cache64 = 64 if 64 in nonzero_cache_sizes else max(nonzero_cache_sizes)
    claim_a_reduction = None
    claim_a_ok = False
    if max_n in args.n_list:
        claim_a_reduction = summary["skewed"][str(max_n)].get(
            f"cache{cache64}_read_time_reduction_vs_cache0")
        claim_a_ok = bool(claim_a_reduction is not None and claim_a_reduction >= 0.30)

    # 主張(b)
    claim_b_ok = bool(len(accuracy_mismatches) == 0)

    # 主張(c): ピークRAM使用量増分がキャッシュ容量にほぼ比例（対数-対数回帰）、
    # かつキャッシュ容量64・N=30万時点での増分がステップ22実測値の20%以下。
    # ピークWSの単調非減少性（ステップ32の「実装中に発覚した問題」3.）を踏まえ、
    # N=30万・全シードのうち各キャッシュ容量での最大値（最も情報量の大きい測定）を使う。
    cache_sizes_for_c = [c for c in args.cache_sizes if c > 0]
    mean_peak_ws_by_cache = []
    for c in cache_sizes_for_c:
        vals = per["uniform"][c][max_n]["peak_ws"] if max_n in args.n_list else []
        mean_peak_ws_by_cache.append(max(vals) if vals else 0.0)
    slope_cache_ws = None
    claim_c_slope_ok = False
    if len(cache_sizes_for_c) >= 2 and all(v > 0 for v in mean_peak_ws_by_cache):
        slope_cache_ws = loglog_slope(cache_sizes_for_c, mean_peak_ws_by_cache)
        claim_c_slope_ok = bool(0.8 <= slope_cache_ws <= 1.2)
    cache64_peak_ws_at_maxn = (max(per["uniform"][cache64][max_n]["peak_ws"])
                                if max_n in args.n_list and per["uniform"][cache64][max_n]["peak_ws"]
                                else None)
    claim_c_ratio = (cache64_peak_ws_at_maxn / STEP22_RAM_MEMORY_AT_N300K_BYTES
                      if cache64_peak_ws_at_maxn is not None else None)
    claim_c_ratio_ok = bool(claim_c_ratio is not None and claim_c_ratio <= 0.20)
    claim_c_ok = bool(claim_c_slope_ok and claim_c_ratio_ok)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "ステップ33（DiskBackedAssociativeStoreのreadレイテンシ緩和：頻出キーのRAM"
                "キャッシュ層, 12.6.74〜75節）: 一様分布・Zipf偏り分布クエリで、"
                "キャッシュ容量0（ベースライン）・4・16・64チャンクを比較",
        "条件": vars(args) | {"out": str(args.out), "query_batch": QUERY_BATCH,
                              "n_warmup": N_WARMUP, "n_measure": N_MEASURE,
                              "disk_key_chunk": DISK_KEY_CHUNK,
                              "zipf_s": 1.5,
                              "platform": sys.platform,
                              "peak_memory_metric": ("Windows: GetProcessMemoryInfo."
                                                      "PeakWorkingSetSize (resource.getrusageの代替)"
                                                      if sys.platform == "win32"
                                                      else "resource.getrusage().ru_maxrss")},
        "n_list": list(args.n_list),
        "cache_sizes": list(args.cache_sizes),
        "summary": summary,
        "claim_a_query_kind": "skewed",
        "claim_a_n": max_n,
        "claim_a_cache_size": cache64,
        "claim_a_read_time_reduction_vs_cache0（30%以上が合格）": claim_a_reduction,
        "claim_a_ok": claim_a_ok,
        "claim_b_accuracy_mismatches": accuracy_mismatches,
        "claim_b_ok（value・best・argがtorch.equalで完全一致）": claim_b_ok,
        "claim_c_cache_sizes_used": cache_sizes_for_c,
        "claim_c_mean_peak_ws_by_cache_bytes": mean_peak_ws_by_cache,
        "claim_c_loglog_slope（0.8〜1.2が合格）": slope_cache_ws,
        "claim_c_slope_ok": claim_c_slope_ok,
        "claim_c_cache64_peak_ws_at_maxn_bytes": cache64_peak_ws_at_maxn,
        "claim_c_step22_ram_memory_at_n300k_bytes": STEP22_RAM_MEMORY_AT_N300K_BYTES,
        "claim_c_ratio_to_step22（20%以下が合格）": claim_c_ratio,
        "claim_c_ratio_ok": claim_c_ratio_ok,
        "claim_c_ok": claim_c_ok,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n書き出し: {args.out}")
    print(f"主張(a) skewed N={max_n} cache={cache64}: 短縮率={claim_a_reduction}  ok={claim_a_ok}")
    print(f"主張(b) 不一致件数: {len(accuracy_mismatches)}  ok={claim_b_ok}")
    print(f"主張(c) slope={slope_cache_ws}  ratio={claim_c_ratio}  ok={claim_c_ok}")
    print(f"所要時間: {time.time() - t0:.1f}秒")


if __name__ == "__main__":
    main()
