"""ステップ32（海馬モジュール`AssociativeStore`のディスクストリーミング実装, 12.6.72節）の主実験。

`AssociativeStore`（全件RAM常駐, (i)）と`DiskBackedAssociativeStore`（ディスク常駐, (ii)）を
同一のキー・バリュー・クエリ条件で比較する。ステップ22
（`run_associative_store_scaling.py`）の計測パイプライン（N=100〜30万・64次元・
壁時計時間中央値・メモリ占有量の実測）を流用する。

Windows環境には`resource`モジュールがない（Unix専用）ため、ピークRSSの計測は
`ctypes`経由のWindows API（`GetProcessMemoryInfo`の`PeakWorkingSetSize`）で代替する。
プロセス全体のピークワーキングセットサイズであり、`resource.getrusage().ru_maxrss`
（Unixのピーク常駐セットサイズ）の直接の代替指標として扱う。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_associative_store_disk_streaming
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

from ..hippocampus import AssociativeStore, DiskBackedAssociativeStore

# ステップ22と揃えたレンジ。実行時間の都合で100万は含めない（ステップ22と同じ理由）。
DEFAULT_N_LIST = (100, 1000, 10_000, 100_000, 300_000)

QUERY_BATCH = 32
N_WARMUP = 1
N_MEASURE = 5
DISK_KEY_CHUNK = 4096

STEP22_RAM_MEMORY_AT_N300K_BYTES = 146.48 * 1024 * 1024  # ステップ22実測値（12.7節記載）


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


def measure_ram(n: int, d_model: int, value_dim: int, seed: int) -> dict:
    """(i) 全件RAM常駐方式（`AssociativeStore`, exact=True）の計測。

    `PeakWorkingSetSize`はプロセス起動時からの単調非減少カウンタで、Python/PyTorchの
    実行時オーバーヘッド（約200MB）がベースラインに常時乗る。ステップ22実測値
    （テンソルのみのバイト数146.48MB）と条件を揃えて比較するため、ストア構築前を
    ベースラインとした**増分**（delta）を主指標として記録する。
    """
    peak_baseline = peak_working_set_bytes()
    g = torch.Generator().manual_seed(seed)
    store = AssociativeStore(d_model, value_dim, exact=True)
    keys = torch.nn.functional.normalize(torch.randn(n, d_model, generator=g), dim=-1)
    values = torch.nn.functional.normalize(torch.randn(n, value_dim, generator=g), dim=-1)
    store.write(keys, values)

    query_g = torch.Generator().manual_seed(seed + 90_000)
    queries = torch.nn.functional.normalize(torch.randn(QUERY_BATCH, d_model, generator=query_g), dim=-1)

    for _ in range(N_WARMUP):
        store.read(queries)
    times = []
    for _ in range(N_MEASURE):
        t0 = time.perf_counter()
        out, stats = store.read(queries)
        times.append(time.perf_counter() - t0)
    peak_after = peak_working_set_bytes()

    tensor_bytes = (store._keys.element_size() * store._keys.nelement()
                     + store._values.element_size() * store._values.nelement())
    return {
        "read_time_median_sec": statistics.median(times),
        "tensor_memory_bytes": tensor_bytes,
        "peak_working_set_delta_bytes": max(0, peak_after - peak_baseline),
        "out": out,
        "stats": stats,
        "queries": queries,
    }


def measure_disk(n: int, d_model: int, value_dim: int, seed: int, persist_root: Path) -> dict:
    """(ii) ディスク常駐方式（`DiskBackedAssociativeStore`, exact=True）の計測。増分方式は`measure_ram`と同じ。"""
    persist_dir = persist_root / f"n{n}_seed{seed}"
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    peak_baseline = peak_working_set_bytes()
    g = torch.Generator().manual_seed(seed)
    store = DiskBackedAssociativeStore(d_model, value_dim, persist_dir=persist_dir,
                                        exact=True, key_chunk=DISK_KEY_CHUNK)
    keys = torch.nn.functional.normalize(torch.randn(n, d_model, generator=g), dim=-1)
    values = torch.nn.functional.normalize(torch.randn(n, value_dim, generator=g), dim=-1)
    store.write(keys, values)

    query_g = torch.Generator().manual_seed(seed + 90_000)
    queries = torch.nn.functional.normalize(torch.randn(QUERY_BATCH, d_model, generator=query_g), dim=-1)

    for _ in range(N_WARMUP):
        store.read(queries)
    times = []
    for _ in range(N_MEASURE):
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
    }
    shutil.rmtree(persist_dir, ignore_errors=True)
    return result


def loglog_slope(xs: list[float], ys: list[float]) -> float:
    log_x = np.log10(np.asarray(xs, dtype=float))
    log_y = np.log10(np.clip(np.asarray(ys, dtype=float), 1.0, None))  # 0除け（測定上のデルタ0対策）
    slope, _intercept = np.polyfit(log_x, log_y, 1)
    return float(slope)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--value-dim", type=int, default=64)
    ap.add_argument("--n-list", type=int, nargs="+", default=list(DEFAULT_N_LIST))
    ap.add_argument("--out", type=Path,
                    default=Path("results/associative_store_scaling/disk_streaming.json"))
    args = ap.parse_args()

    t0 = time.time()
    persist_root = Path(tempfile.mkdtemp(prefix="disk_backed_store_"))

    per_n: dict[int, dict] = {n: {"ram_read_time": [], "disk_read_time": [],
                                   "ram_tensor_bytes": [], "ram_peak_ws_bytes": [],
                                   "disk_peak_ws_bytes": []} for n in args.n_list}
    accuracy_mismatches: list[str] = []
    max_score_max_abs_diff = 0.0

    try:
        for n in args.n_list:
            for seed in range(args.seeds):
                ram = measure_ram(n, args.d_model, args.value_dim, seed)
                disk = measure_disk(n, args.d_model, args.value_dim, seed, persist_root)

                per_n[n]["ram_read_time"].append(ram["read_time_median_sec"])
                per_n[n]["disk_read_time"].append(disk["read_time_median_sec"])
                per_n[n]["ram_tensor_bytes"].append(ram["tensor_memory_bytes"])
                per_n[n]["ram_peak_ws_bytes"].append(ram["peak_working_set_delta_bytes"])
                per_n[n]["disk_peak_ws_bytes"].append(disk["peak_working_set_delta_bytes"])

                # 主張(b): value・arg（top1_index）はtorch.equalで完全一致を要求する。
                # max_scoreはBLAS内部のリダクション順序の違いにより厳密なビット一致が
                # 保証されないため（12.6.73節参照）、allcloseで別途記録する。
                if not torch.equal(ram["stats"].top1_index, disk["stats"].top1_index):
                    accuracy_mismatches.append(f"n={n} seed={seed}: top1_index不一致")
                if not torch.equal(ram["out"], disk["out"]):
                    accuracy_mismatches.append(f"n={n} seed={seed}: value(out)不一致")
                diff = float((ram["stats"].max_score - disk["stats"].max_score).abs().max())
                max_score_max_abs_diff = max(max_score_max_abs_diff, diff)
    finally:
        shutil.rmtree(persist_root, ignore_errors=True)

    def _stats(values: list[float]) -> dict:
        t = torch.tensor(values, dtype=torch.float64)
        return {"mean": float(t.mean()), "std": float(t.std(unbiased=False)), "per_seed": t.tolist()}

    summary: dict[str, dict] = {}
    for n in args.n_list:
        summary[str(n)] = {
            "ram_read_time_sec": _stats(per_n[n]["ram_read_time"]),
            "disk_read_time_sec": _stats(per_n[n]["disk_read_time"]),
            "ram_tensor_memory_bytes": _stats(per_n[n]["ram_tensor_bytes"]),
            "ram_peak_working_set_bytes": _stats(per_n[n]["ram_peak_ws_bytes"]),
            "disk_peak_working_set_bytes": _stats(per_n[n]["disk_peak_ws_bytes"]),
        }
        rt_ratio = (summary[str(n)]["disk_read_time_sec"]["mean"]
                    / max(summary[str(n)]["ram_read_time_sec"]["mean"], 1e-12))
        summary[str(n)]["read_time_ratio_disk_over_ram"] = rt_ratio

    n_sorted = sorted(args.n_list)
    mean_disk_peak_ws = [summary[str(n)]["disk_peak_working_set_bytes"]["mean"] for n in n_sorted]
    slope_disk_peak_ws = loglog_slope(n_sorted, mean_disk_peak_ws)

    max_n = n_sorted[-1]
    disk_peak_ws_at_max_n = summary[str(max_n)]["disk_peak_working_set_bytes"]["mean"]

    # 主張(a)
    claim_a_ratio = disk_peak_ws_at_max_n / STEP22_RAM_MEMORY_AT_N300K_BYTES
    claim_a_ratio_ok = bool(max_n == 300_000 and claim_a_ratio <= 0.20)
    claim_a_slope_ok = bool(slope_disk_peak_ws <= 0.3)
    claim_a_ok = bool(claim_a_ratio_ok and claim_a_slope_ok)

    # 主張(b)
    claim_b_ok = bool(len(accuracy_mismatches) == 0)

    # 主張(c)（合否対象外、定量記録のみ）
    read_time_ratios = {str(n): summary[str(n)]["read_time_ratio_disk_over_ram"] for n in n_sorted}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "ステップ32（海馬モジュールAssociativeStoreのディスクストリーミング実装, "
                "12.6.72〜73節）: 全件RAM常駐方式(AssociativeStore, exact=True)とディスク常駐方式"
                "(DiskBackedAssociativeStore, exact=True)を、read壁時計時間・プロセスピーク"
                "ワーキングセットサイズ・read出力の一致で比較",
        "条件": vars(args) | {"out": str(args.out), "query_batch": QUERY_BATCH,
                              "n_warmup": N_WARMUP, "n_measure": N_MEASURE,
                              "disk_key_chunk": DISK_KEY_CHUNK,
                              "platform": sys.platform,
                              "peak_memory_metric": ("Windows: GetProcessMemoryInfo."
                                                      "PeakWorkingSetSize (resource.getrusageの代替)"
                                                      if sys.platform == "win32"
                                                      else "resource.getrusage().ru_maxrss")},
        "n_list": n_sorted,
        "summary": summary,
        "claim_a_disk_peak_working_set_loglog_slope": slope_disk_peak_ws,
        "claim_a_disk_peak_ws_at_max_n_bytes": disk_peak_ws_at_max_n,
        "claim_a_step22_ram_memory_at_n300k_bytes": STEP22_RAM_MEMORY_AT_N300K_BYTES,
        "claim_a_ratio_to_step22（20%以下が合格）": claim_a_ratio,
        "claim_a_ratio_ok": claim_a_ratio_ok,
        "claim_a_slope_ok（0.3以下が合格）": claim_a_slope_ok,
        "claim_a_ok": claim_a_ok,
        "claim_b_accuracy_mismatches": accuracy_mismatches,
        "claim_b_max_score_max_abs_diff（BLAS丸め誤差の実測上限）": max_score_max_abs_diff,
        "claim_b_ok（value・argがtorch.equalで完全一致）": claim_b_ok,
        "claim_c_read_time_ratio_disk_over_ram（定量記録のみ）": read_time_ratios,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n書き出し: {args.out}")
    print("-- 本実験（N x 指標） --")
    for n in n_sorted:
        s = summary[str(n)]
        print(f"N={n:>8}  ram_read={s['ram_read_time_sec']['mean']*1000:.3f}ms  "
              f"disk_read={s['disk_read_time_sec']['mean']*1000:.3f}ms  "
              f"ratio={s['read_time_ratio_disk_over_ram']:.2f}x  "
              f"ram_peak_ws={s['ram_peak_working_set_bytes']['mean']/1024/1024:.2f}MB  "
              f"disk_peak_ws={s['disk_peak_working_set_bytes']['mean']/1024/1024:.2f}MB")
    print(f"\n主張(a) disk peak working set loglog slope: {slope_disk_peak_ws:.3f}  "
          f"ok(slope)={claim_a_slope_ok}")
    print(f"  N={max_n} disk_peak_ws={disk_peak_ws_at_max_n/1024/1024:.2f}MB / "
          f"step22実測{STEP22_RAM_MEMORY_AT_N300K_BYTES/1024/1024:.2f}MB = {claim_a_ratio*100:.1f}%  "
          f"ok(ratio)={claim_a_ratio_ok}")
    print(f"主張(a) claim_a_ok: {claim_a_ok}")
    print(f"主張(b) 不一致件数: {len(accuracy_mismatches)}  max_score最大絶対誤差: "
          f"{max_score_max_abs_diff:.2e}  claim_b_ok: {claim_b_ok}")


if __name__ == "__main__":
    main()
