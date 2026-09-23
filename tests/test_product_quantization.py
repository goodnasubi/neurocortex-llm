"""ステップ38: Product Quantization（PQ）によるインデックスサイズ削減とトレードオフ分析。

インデックスサイズ・検索精度・検索速度の関係を測定。
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


@pytest.mark.parametrize("N", [1000, 5000])
def test_pq_index_buildable(N: int):
    """PQインデックスが構築可能であることを確認。

    Args:
        N: ストアサイズ
    """
    key_dim = 128
    value_dim = 64

    g = torch.Generator().manual_seed(0)

    with tempfile.TemporaryDirectory() as tmpdir:
        store = DiskBackedAssociativeStore(
            key_dim=key_dim,
            value_dim=value_dim,
            persist_dir=tmpdir,
            exact=True,
        )

        # ストアにデータ書き込み
        keys_data = torch.randn(N, key_dim, generator=g)
        values_data = torch.randn(N, value_dim, generator=g)
        store.write(keys_data, values_data)

        # PQインデックス構築可能
        keys_mm = store._keys_memmap()
        for M in [8, 16]:
            try:
                store._build_pq_index(keys_mm, M=M)
                assert hasattr(store, "_pq_index") and store._pq_index is not None
            except Exception as e:
                pytest.fail(f"N={N}, M={M}: PQ構築失敗 - {e}")
        # Windows ではmemmapがファイルハンドルを保持するため、明示的に解放する
        del keys_mm
        del store


@pytest.mark.parametrize("N", [1000, 5000])
def test_pq_search_runnable(N: int):
    """PQ検索が実行可能であることを確認。

    Args:
        N: ストアサイズ
    """
    key_dim = 128
    value_dim = 64

    g = torch.Generator().manual_seed(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        store = DiskBackedAssociativeStore(
            key_dim=key_dim,
            value_dim=value_dim,
            persist_dir=tmpdir,
            exact=True,
        )

        # ストアにデータ書き込み
        keys_data = torch.randn(N, key_dim, generator=g)
        values_data = torch.randn(N, value_dim, generator=g)
        store.write(keys_data, values_data)

        # クエリ生成
        B = 8
        query_keys = torch.randn(B, key_dim, generator=g)

        # PQ検索実行
        try:
            values_pq, stats_pq = store.read_with_pq_search(query_keys, k_candidates=10, M=16)
            assert values_pq.shape == (B, value_dim)
            assert stats_pq.max_score.shape == (B,)
            assert stats_pq.top1_index.shape == (B,)
        except Exception as e:
            pytest.fail(f"N={N}: PQ検索失敗 - {e}")


@pytest.mark.parametrize("N,M", [(1000, 8), (1000, 16), (5000, 8), (5000, 16)])
def test_pq_output_validity(N: int, M: int):
    """PQ検索の出力が有効な値であることを確認。

    Args:
        N: ストアサイズ
        M: PQのサブクォンタイザ数
    """
    key_dim = 128
    value_dim = 64

    g = torch.Generator().manual_seed(2)

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

        query_keys = torch.randn(4, key_dim, generator=g)

        # PQ検索
        values_pq, stats_pq = store.read_with_pq_search(query_keys, k_candidates=20, M=M)

        # 出力値の有効性確認
        assert not torch.isnan(values_pq).any(), "PQ出力にNaN含む"
        assert not torch.isinf(values_pq).any(), "PQ出力に無限値含む"
        assert values_pq.shape == (4, value_dim)

        # top1_index の有効性
        for idx in stats_pq.top1_index:
            if idx >= 0:
                assert 0 <= idx < N, f"top1_index が範囲外: {idx} not in [0, {N})"


@pytest.mark.parametrize("N", [1000, 5000])
def test_pq_vs_bruteforce_similarity(N: int):
    """PQ検索がbrute force と統計的に近い結果を返すことを確認。

    Args:
        N: ストアサイズ
    """
    key_dim = 128
    value_dim = 64

    g = torch.Generator().manual_seed(3)

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

        query_keys = torch.randn(4, key_dim, generator=g)

        # Bruteforce
        values_bf, stats_bf = store.read(query_keys)

        # PQ（k_candidates を十分大きく）
        k_cand = min(int(N**0.5), 100)
        values_pq, stats_pq = store.read_with_pq_search(
            query_keys, k_candidates=k_cand, M=16
        )

        # top1 インデックスの一致率を確認（完全一致は期待しない）
        match_count = (stats_bf.top1_index == stats_pq.top1_index).sum().item()
        match_rate = match_count / len(stats_bf.top1_index)
        # 少なくとも50%は一致（PQは完全ではないが、方向性は一致）
        assert match_rate >= 0.5, f"N={N}: top1一致率 {match_rate} < 0.5"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
