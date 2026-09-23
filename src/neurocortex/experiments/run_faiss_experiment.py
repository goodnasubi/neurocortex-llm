"""ステップ36（faiss IVF統合による高速キー検索, 12.6.80節）の主実験。

4条件比較：
  1. brute-force（ステップ32）
  2. faiss(nprobe=5)
  3. faiss(nprobe=10)
  4. faiss(nprobe=20)

パラメータ:
  N ∈ {10000, 100000, 300000}
  k_candidate ∈ {10, 20, sqrt(N)}
  seed = 3個（0, 1, 2）

計測項目:
  - read壁時計時間（中央値, 秒）
  - argmax一致率（%）
  - 索引構築時間（秒）

結果保存: results/faiss_integration/experiment_results.json

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_faiss_experiment
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from ..hippocampus import DiskBackedAssociativeStore

DEFAULT_N_LIST = (10_000, 100_000, 300_000)
DEFAULT_K_CANDIDATES = (10, 20, None)  # None means sqrt(N)
DEFAULT_NPROBES = (5, 10, 20)
N_SEEDS = 3
N_WARMUP = 1
N_MEASURE = 5
QUERY_BATCH = 32


def run_benchmark(
    n: int,
    key_dim: int = 64,
    value_dim: int = 32,
    seed: int = 0,
) -> dict:
    """1つの N値に対するベンチマークを実行。

    Returns:
        {
            "brute_force": {k_cand: {...}},
            "faiss_nprobe_5": {k_cand: {...}},
            ...
        }
    """
    torch.manual_seed(seed)
    keys = torch.nn.functional.normalize(torch.randn(n, key_dim), dim=-1)
    values = torch.nn.functional.normalize(torch.randn(n, value_dim), dim=-1)

    # クエリセット（全キーを対象に検索）
    query_keys = keys.clone()

    results = {}

    # === ブルートフォース版（参照実装） ===
    with tempfile.TemporaryDirectory() as tmpdir:
        store_bf = DiskBackedAssociativeStore(key_dim, value_dim, tmpdir, exact=True)
        store_bf.write(keys, values)

        results["brute_force"] = {}
        for k_cand in DEFAULT_K_CANDIDATES:
            k_display = k_cand if k_cand is not None else f"sqrt({n})"
            times = []

            # ウォームアップ
            for _ in range(N_WARMUP):
                store_bf.read(query_keys, chunk=QUERY_BATCH)

            # 計測
            for _ in range(N_MEASURE):
                t0 = time.perf_counter()
                out_bf, stats_bf = store_bf.read(query_keys, chunk=QUERY_BATCH)
                t1 = time.perf_counter()
                times.append(t1 - t0)

            results["brute_force"][k_display] = {
                "median_time_sec": statistics.median(times),
                "argmax_match_pct": 100.0,  # 参照実装
            }

    # === faiss 版（複数 nprobe） ===
    with tempfile.TemporaryDirectory() as tmpdir:
        store_faiss = DiskBackedAssociativeStore(key_dim, value_dim, tmpdir, exact=True)
        store_faiss.write(keys, values)

        # faiss インデックス構築
        keys_np = keys.detach().to("cpu", dtype=torch.float32).numpy()
        t_index_start = time.perf_counter()
        try:
            store_faiss._train_faiss_index(keys_np)
        except ImportError:
            print("WARNING: faiss がインストールされていません。スキップします。")
            return results
        t_index_end = time.perf_counter()
        index_time = t_index_end - t_index_start

        for nprobe in DEFAULT_NPROBES:
            store_faiss._faiss_nprobe = nprobe
            store_faiss._faiss_index.nprobe = nprobe

            results[f"faiss_nprobe_{nprobe}"] = {}

            for k_cand in DEFAULT_K_CANDIDATES:
                k_display = k_cand if k_cand is not None else f"sqrt({n})"
                k_actual = k_cand if k_cand is not None else max(10, int(n**0.5))
                times = []
                match_count = 0
                match_total = 0

                # ウォームアップ
                for _ in range(N_WARMUP):
                    store_faiss.read_with_faiss_search(
                        query_keys, chunk=QUERY_BATCH, k_candidates=k_actual
                    )

                # 計測
                for _ in range(N_MEASURE):
                    t0 = time.perf_counter()
                    out_faiss, stats_faiss = store_faiss.read_with_faiss_search(
                        query_keys, chunk=QUERY_BATCH, k_candidates=k_actual
                    )
                    t1 = time.perf_counter()
                    times.append(t1 - t0)

                    # argmax 一致率を計測（最後の計測時）
                    if match_total == 0:
                        match_count = (stats_faiss.top1_index == stats_bf.top1_index).sum().item()
                        match_total = query_keys.shape[0]

                results[f"faiss_nprobe_{nprobe}"][k_display] = {
                    "median_time_sec": statistics.median(times),
                    "argmax_match_pct": 100.0 * match_count / match_total,
                    "index_time_sec": index_time,
                }

    return results


def main() -> None:
    """メイン実験ループ。"""
    print("ステップ36: faiss IVF統合実験")
    print("=" * 60)

    all_results = {}

    for n in DEFAULT_N_LIST:
        print(f"\n[N={n:,}]")
        all_results[n] = {}

        for seed in range(N_SEEDS):
            print(f"  seed={seed}...", end=" ", flush=True)
            result = run_benchmark(n, seed=seed)
            all_results[n][seed] = result
            print("done")

    # 結果を JSON で保存
    output_dir = Path(__file__).parent.parent.parent / "results" / "faiss_integration"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "experiment_results.json"

    output_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\n結果を保存しました: {output_path}")

    # === 結果サマリー ===
    print("\n" + "=" * 60)
    print("結果サマリー")
    print("=" * 60)

    for n in DEFAULT_N_LIST:
        print(f"\n[N={n:,}]")

        # 各 seed の結果を平均
        agg = {}
        for seed in range(N_SEEDS):
            for method_key, method_results in all_results[n][seed].items():
                if method_key not in agg:
                    agg[method_key] = {}
                for k_display, metrics in method_results.items():
                    if k_display not in agg[method_key]:
                        agg[method_key][k_display] = []
                    agg[method_key][k_display].append(metrics)

        # 平均値を計算して表示
        for method_key in ["brute_force", "faiss_nprobe_5", "faiss_nprobe_10", "faiss_nprobe_20"]:
            if method_key not in agg:
                continue
            print(f"  {method_key}:")
            for k_display in agg[method_key]:
                times = [m["median_time_sec"] for m in agg[method_key][k_display]]
                matches = [m["argmax_match_pct"] for m in agg[method_key][k_display]]
                avg_time = statistics.mean(times)
                avg_match = statistics.mean(matches)
                print(f"    k_candidate={k_display:>8} | time={avg_time*1000:7.2f}ms | match={avg_match:6.2f}%")


if __name__ == "__main__":
    main()
