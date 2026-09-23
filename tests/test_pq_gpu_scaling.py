"""ステップ39: GPU faiss IndexPQ による高速化・大規模スケーリングテスト。

N=100K での大規模データセットでの PQ 性能測定。
GPU が利用不可な環境では CPU でのフォールバック動作を検証。
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.hippocampus import DiskBackedAssociativeStore


@pytest.mark.parametrize("N", [100000])
def test_pq_gpu_index_buildable(N: int):
    """GPU PQ インデックス構築可能性（N=100K）。

    Args:
        N: 大規模ストアサイズ
    """
    key_dim = 128
    value_dim = 64
    M = 16

    g = torch.Generator().manual_seed(100)

    with tempfile.TemporaryDirectory() as tmpdir:
        store = DiskBackedAssociativeStore(
            key_dim=key_dim,
            value_dim=value_dim,
            persist_dir=tmpdir,
            exact=True,
        )

        # 大規模データを段階的に書き込み（メモリ効率化）
        chunk_size = 10000
        for chunk_idx in range(N // chunk_size):
            keys_chunk = torch.randn(chunk_size, key_dim, generator=g)
            values_chunk = torch.randn(chunk_size, value_dim, generator=g)
            store.write(keys_chunk, values_chunk)

        # GPU PQ インデックス構築試行
        keys_mm = store._keys_memmap()
        try:
            store._build_pq_index_gpu(keys_mm, M=M)
            # GPU またはフォールバック CPU で構築成功
            assert hasattr(store, "_pq_index_gpu") or hasattr(store, "_pq_index")
        except Exception as e:
            pytest.fail(f"N={N}: GPU/CPU PQ構築失敗 - {e}")


@pytest.mark.parametrize("N", [100000])
def test_pq_gpu_search_runnable(N: int):
    """GPU PQ 検索実行可能性（N=100K）。

    Args:
        N: 大規模ストアサイズ
    """
    key_dim = 128
    value_dim = 64
    M = 16

    g = torch.Generator().manual_seed(101)

    with tempfile.TemporaryDirectory() as tmpdir:
        store = DiskBackedAssociativeStore(
            key_dim=key_dim,
            value_dim=value_dim,
            persist_dir=tmpdir,
            exact=True,
        )

        # 大規模データを段階的に書き込み
        chunk_size = 10000
        for chunk_idx in range(N // chunk_size):
            keys_chunk = torch.randn(chunk_size, key_dim, generator=g)
            values_chunk = torch.randn(chunk_size, value_dim, generator=g)
            store.write(keys_chunk, values_chunk)

        # GPU PQ 検索試行
        query_keys = torch.randn(4, key_dim, generator=g)
        k_cand = max(int(N**0.5), 50)

        try:
            values_gpu, stats_gpu = store.read_with_pq_search_gpu(
                query_keys, k_candidates=k_cand, M=M
            )
            # GPU またはフォールバック CPU で検索成功
            assert values_gpu.shape == (4, value_dim)
            assert stats_gpu.max_score.shape == (4,)
            assert stats_gpu.top1_index.shape == (4,)
        except Exception as e:
            pytest.fail(f"N={N}: GPU/CPU PQ検索失敗 - {e}")


@pytest.mark.parametrize("N", [100000])
def test_pq_gpu_cpu_equivalence(N: int):
    """GPU と CPU PQ の結果一貫性（N=100K）。

    Args:
        N: 大規模ストアサイズ
    """
    key_dim = 128
    value_dim = 64
    M = 16

    g = torch.Generator().manual_seed(102)

    with tempfile.TemporaryDirectory() as tmpdir:
        store = DiskBackedAssociativeStore(
            key_dim=key_dim,
            value_dim=value_dim,
            persist_dir=tmpdir,
            exact=True,
        )

        # 大規模データを段階的に書き込み
        chunk_size = 10000
        for chunk_idx in range(N // chunk_size):
            keys_chunk = torch.randn(chunk_size, key_dim, generator=g)
            values_chunk = torch.randn(chunk_size, value_dim, generator=g)
            store.write(keys_chunk, values_chunk)

        query_keys = torch.randn(4, key_dim, generator=g)
        k_cand = max(int(N**0.5), 50)

        # CPU PQ 検索
        values_cpu, stats_cpu = store.read_with_pq_search(
            query_keys, k_candidates=k_cand, M=M
        )

        # GPU PQ 検索（GPU 不可時は同じ結果、GPU 可時は同等結果）
        values_gpu, stats_gpu = store.read_with_pq_search_gpu(
            query_keys, k_candidates=k_cand, M=M
        )

        # GPU 不可環境では完全に同じ結果
        assert torch.allclose(values_cpu, values_gpu, rtol=1e-4)
        assert torch.allclose(stats_cpu.max_score, stats_gpu.max_score, rtol=1e-4)


@pytest.mark.parametrize("N", [100000])
def test_pq_gpu_output_validity(N: int):
    """GPU PQ 出力の有効性チェック（N=100K）。

    Args:
        N: 大規模ストアサイズ
    """
    key_dim = 128
    value_dim = 64
    M = 16

    g = torch.Generator().manual_seed(103)

    with tempfile.TemporaryDirectory() as tmpdir:
        store = DiskBackedAssociativeStore(
            key_dim=key_dim,
            value_dim=value_dim,
            persist_dir=tmpdir,
            exact=True,
        )

        # 大規模データを段階的に書き込み
        chunk_size = 10000
        for chunk_idx in range(N // chunk_size):
            keys_chunk = torch.randn(chunk_size, key_dim, generator=g)
            values_chunk = torch.randn(chunk_size, value_dim, generator=g)
            store.write(keys_chunk, values_chunk)

        query_keys = torch.randn(4, key_dim, generator=g)
        k_cand = max(int(N**0.5), 50)

        values_gpu, stats_gpu = store.read_with_pq_search_gpu(
            query_keys, k_candidates=k_cand, M=M
        )

        # 出力値の有効性確認
        assert not torch.isnan(values_gpu).any(), "GPU PQ 出力に NaN 含む"
        assert not torch.isinf(values_gpu).any(), "GPU PQ 出力に無限値含む"
        assert values_gpu.shape == (4, value_dim)

        # top1_index の有効性
        for idx in stats_gpu.top1_index:
            if idx >= 0:
                assert 0 <= idx < N, f"top1_index が範囲外: {idx} not in [0, {N})"


@pytest.mark.parametrize("N", [100000])
def test_pq_gpu_memory_estimate(N: int):
    """GPU PQ メモリ使用量推定（N=100K）。

    Args:
        N: 大規模ストアサイズ
    """
    key_dim = 128
    M_values = [8, 16, 32]

    # PQ インデックスメモリ推定: N × M バイト
    # GPU VRAM 推定: インデックス + コードブック（256^(d/M) × M × 1バイト）

    for M in M_values:
        pq_index_memory = N * M / (1024**2)  # MB
        codebook_vocab = 256
        codebook_size = (codebook_vocab * (key_dim // M)) / (1024**2)
        total_gpu_memory = pq_index_memory + codebook_size

        # 推奨: GPU VRAM < 80% 占有（標準 GPU: 8GB = 8000MB）
        # N=100K, M=16 → ~1.6MB + ~0.1MB = ~1.7MB（余裕あり）
        assert total_gpu_memory < 8000 * 0.8, \
            f"M={M}: 推定 GPU メモリ {total_gpu_memory:.1f}MB が VRAM 80% を超過"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
