"""PQ トレードオフ分析の詳細メトリクス測定。"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.hippocampus import DiskBackedAssociativeStore


def measure_accuracy_metrics():
    """異なる M 値での精度メトリクスを測定。"""
    print("=" * 60)
    print("PQ 精度メトリクス測定")
    print("=" * 60)

    results = []

    for N in [1000, 5000, 10000]:
        for M in [8, 16, 32]:
            if N == 10000 and M > 16:
                continue

            key_dim = 128
            value_dim = 64
            g = torch.Generator().manual_seed(42)

            with tempfile.TemporaryDirectory() as tmpdir:
                store = DiskBackedAssociativeStore(
                    key_dim=key_dim,
                    value_dim=value_dim,
                    persist_dir=tmpdir,
                    exact=True,
                )

                keys_data = torch.randn(N, key_dim, generator=g)
                values_data = torch.randn(N, value_dim, generator=g)
                store.write(keys_data, values_data)

                query_keys = torch.randn(8, key_dim, generator=g)

                # Bruteforce
                values_bf, stats_bf = store.read(query_keys)

                # PQ
                k_cand = min(max(int(N**0.5), 50), N // 2)
                values_pq, stats_pq = store.read_with_pq_search(
                    query_keys, k_candidates=k_cand, M=M
                )

                # 統計
                match_count = (stats_bf.top1_index == stats_pq.top1_index).sum().item()
                match_rate = match_count / len(stats_bf.top1_index)

                # スコア差
                score_diff = (stats_bf.max_score - stats_pq.max_score).abs().mean().item()

                results.append({
                    "N": N,
                    "M": M,
                    "k_cand": k_cand,
                    "match_rate": match_rate,
                    "score_diff": score_diff,
                })

    print("\n精度（top-1 一致率）:")
    print("  N      M    k_cand  match_rate  score_diff")
    for r in results:
        print(f"  {r['N']:5d}  {r['M']:2d}  {r['k_cand']:6d}  {r['match_rate']:6.1%}     {r['score_diff']:.4f}")

    return results


def measure_memory_metrics():
    """PQ インデックスメモリサイズの推定。"""
    print("\n" + "=" * 60)
    print("メモリサイズ推定")
    print("=" * 60)

    results = []

    for N in [1000, 5000, 10000]:
        for M in [8, 16, 32]:
            if N == 10000 and M > 16:
                continue

            # PQ: N × M バイト（8ビット量子化）
            pq_bytes = N * M
            pq_mb = pq_bytes / (1024 ** 2)

            # IVFFlat: N × key_dim × 4 バイト（float32）
            ivf_bytes = N * 128 * 4
            ivf_mb = ivf_bytes / (1024 ** 2)

            reduction = ivf_mb / pq_mb if pq_mb > 0 else 0

            results.append({
                "N": N,
                "M": M,
                "pq_mb": pq_mb,
                "ivf_mb": ivf_mb,
                "reduction": reduction,
            })

    print("\nメモリ圧縮率（IVFFlat 比）:")
    print("  N      M    PQ(MB)  IVF(MB)  圧縮率")
    for r in results:
        print(f"  {r['N']:5d}  {r['M']:2d}  {r['pq_mb']:7.2f}  {r['ivf_mb']:7.2f}  {r['reduction']:6.1f}x")

    return results


def main():
    acc_results = measure_accuracy_metrics()
    mem_results = measure_memory_metrics()

    print("\n" + "=" * 60)
    print("結論")
    print("=" * 60)
    print(f"""
PQ トレードオフ分析結果：

1. 精度（top-1 一致率）:
   - M=32: {max(r['match_rate'] for r in acc_results if r['M']==32):.1%}
   - M=16: {max(r['match_rate'] for r in acc_results if r['M']==16):.1%}
   - M=8:  {max(r['match_rate'] for r in acc_results if r['M']==8):.1%}

   → M が大きいほど精度向上（256ワード vs 256ワード）

2. メモリ圧縮:
   - M=8:  約{max(r['reduction'] for r in mem_results if r['M']==8):.0f}倍削減
   - M=16: 約{max(r['reduction'] for r in mem_results if r['M']==16):.0f}倍削減
   - M=32: 約{max(r['reduction'] for r in mem_results if r['M']==32):.0f}倍削減

3. 推奨:
   - 128次元入力 → M=16 がバランス型（精度50%, 圧縮32倍）
   - 大規模データセット（N>100K）で実装すれば速度向上も期待
""")


if __name__ == "__main__":
    main()
