"""ステップ39: GPU vs CPU PQ 性能ベンチマーク。

N=100K, 500K, 1M でのインデックス構築・検索速度を測定。
GPU 環境で実行時は自動的に GPU 性能を測定。
CPU 環境では CPU フォールバック性能を記録。
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.hippocampus import DiskBackedAssociativeStore


def benchmark_pq_scaling():
    """大規模 N での PQ 性能ベンチマーク。"""
    print("=" * 70)
    print("GPU vs CPU PQ ベンチマーク（ステップ39）")
    print("=" * 70)

    results = []

    for N in [100000, 500000, 1000000]:
        print(f"\n【N={N:,} でのベンチマーク】")

        key_dim = 128
        value_dim = 64
        M = 16

        g = torch.Generator().manual_seed(200 + N)

        with tempfile.TemporaryDirectory() as tmpdir:
            store = DiskBackedAssociativeStore(
                key_dim=key_dim,
                value_dim=value_dim,
                persist_dir=tmpdir,
                exact=True,
            )

            # 大規模データ書き込み（段階的）
            chunk_size = 10000
            print(f"  データ書き込み中... ({N // chunk_size} チャンク)")
            write_start = time.perf_counter()
            for chunk_idx in range(N // chunk_size):
                keys_chunk = torch.randn(chunk_size, key_dim, generator=g)
                values_chunk = torch.randn(chunk_size, value_dim, generator=g)
                store.write(keys_chunk, values_chunk)
            write_time = time.perf_counter() - write_start
            print(f"  ✓ 書き込み完了: {write_time:.2f}s")

            # CPU PQ インデックス構築
            keys_mm = store._keys_memmap()
            print(f"  CPU PQ インデックス構築中...")
            cpu_build_start = time.perf_counter()
            store._build_pq_index(keys_mm, M=M)
            cpu_build_time = time.perf_counter() - cpu_build_start
            print(f"  ✓ CPU PQ 構築: {cpu_build_time:.2f}s")

            # GPU PQ インデックス構築（GPU 不可時は CPU フォールバック）
            print(f"  GPU PQ インデックス構築中...")
            gpu_build_start = time.perf_counter()
            store._build_pq_index_gpu(keys_mm, M=M)
            gpu_build_time = time.perf_counter() - gpu_build_start
            print(f"  ✓ GPU/CPU PQ 構築: {gpu_build_time:.2f}s")

            # 検索ベンチマーク
            query_keys = torch.randn(8, key_dim, generator=g)
            k_cand = max(int(N**0.5), 50)

            # CPU PQ 検索
            print(f"  CPU PQ 検索中 (k_cand={k_cand})...")
            cpu_search_start = time.perf_counter()
            values_cpu, stats_cpu = store.read_with_pq_search(
                query_keys, k_candidates=k_cand, M=M
            )
            cpu_search_time = time.perf_counter() - cpu_search_start
            print(f"  ✓ CPU PQ 検索: {cpu_search_time:.3f}s")

            # GPU PQ 検索
            print(f"  GPU PQ 検索中 (k_cand={k_cand})...")
            gpu_search_start = time.perf_counter()
            values_gpu, stats_gpu = store.read_with_pq_search_gpu(
                query_keys, k_candidates=k_cand, M=M
            )
            gpu_search_time = time.perf_counter() - gpu_search_start
            print(f"  ✓ GPU/CPU PQ 検索: {gpu_search_time:.3f}s")

            # 精度確認
            match_count = (stats_cpu.top1_index == stats_gpu.top1_index).sum().item()
            match_rate = match_count / len(stats_cpu.top1_index)
            print(f"  ✓ 精度（top-1 一致率）: {match_rate:.1%}")

            results.append({
                "N": N,
                "write_time": write_time,
                "cpu_build_time": cpu_build_time,
                "gpu_build_time": gpu_build_time,
                "cpu_search_time": cpu_search_time,
                "gpu_search_time": gpu_search_time,
                "speedup": cpu_build_time / gpu_build_time if gpu_build_time > 0 else 0,
                "match_rate": match_rate,
                "k_cand": k_cand,
            })

    # 結果集計
    print("\n" + "=" * 70)
    print("【ベンチマーク結果サマリー】")
    print("=" * 70)

    print("\nインデックス構築時間:")
    print(f"{'N':>10s} {'CPU(s)':>12s} {'GPU/CPU(s)':>12s} {'速度比':>10s}")
    for r in results:
        speedup = r["cpu_build_time"] / r["gpu_build_time"] if r["gpu_build_time"] > 0 else 0
        print(
            f"{r['N']:>10,d} {r['cpu_build_time']:>12.2f} {r['gpu_build_time']:>12.2f} "
            f"{speedup:>10.1f}x"
        )

    print("\n検索時間（8クエリ）:")
    print(f"{'N':>10s} {'k_cand':>8s} {'CPU(ms)':>12s} {'GPU/CPU(ms)':>12s}")
    for r in results:
        print(
            f"{r['N']:>10,d} {r['k_cand']:>8d} {r['cpu_search_time']*1000:>12.2f} "
            f"{r['gpu_search_time']*1000:>12.2f}"
        )

    print("\n精度維持:")
    print(f"{'N':>10s} {'top-1 一致率':>15s}")
    for r in results:
        print(f"{r['N']:>10,d} {r['match_rate']:>14.1%}")

    # 結論
    print("\n" + "=" * 70)
    print("【結論】")
    print("=" * 70)
    print(f"""
インデックス構築:
  - CPU での構築時間: {results[-1]['cpu_build_time']:.2f}s (N={results[-1]['N']:,})
  - GPU/CPU フォールバック: {results[-1]['gpu_build_time']:.2f}s
  - 速度比（GPU/CPU）: {results[-1]['speedup']:.1f}x

検索性能（N={results[-1]['N']:,}, k_cand={results[-1]['k_cand']}）:
  - CPU PQ: {results[-1]['cpu_search_time']*1000:.2f}ms
  - GPU/CPU: {results[-1]['gpu_search_time']*1000:.2f}ms

精度維持:
  - GPU/CPU 間の top-1 一致率: {results[-1]['match_rate']:.1%}

【注記】
現在の環境: CPU フォールバック実装テスト完了
GPU 環境での実行: faiss-gpu インストール後、同じスクリプトで GPU 性能測定可能
期待値: GPU により 5-10倍高速化（大規模 N>100K）
""")


if __name__ == "__main__":
    benchmark_pq_scaling()
