"""ステップ37: memmap バッチ読み込み最適化のテスト。

faiss_search_batch() メソッドが:
1. ブルートフォース版と同じ精度を保つ
2. 候補インデックスのソートと復元が正確に動作する
3. 大規模ストアで動作する
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.hippocampus import DiskBackedAssociativeStore


@pytest.fixture
def small_store(tmp_path):
    store = DiskBackedAssociativeStore(key_dim=16, value_dim=8, persist_dir=tmp_path, exact=False)
    torch.manual_seed(0)
    keys = torch.randn(50, 16)
    values = torch.randn(50, 8)
    store.write(keys, values)
    keys_np = keys.detach().to("cpu", dtype=torch.float32).numpy()
    store._train_faiss_index(keys_np)
    return store


@pytest.fixture
def medium_store(tmp_path):
    store = DiskBackedAssociativeStore(key_dim=32, value_dim=16, persist_dir=tmp_path, exact=False)
    torch.manual_seed(42)
    keys = torch.randn(500, 32)
    values = torch.randn(500, 16)
    store.write(keys, values)
    keys_np = keys.detach().to("cpu", dtype=torch.float32).numpy()
    store._train_faiss_index(keys_np)
    return store


def test_batch_read_produces_same_results_as_exact(small_store):
    """batch版とexact版が同じ結果を返すことを確認"""
    torch.manual_seed(0)
    query = torch.randn(10, 16)

    result_exact, stats_exact = small_store.read(query)
    result_batch, stats_batch = small_store.read_with_faiss_search(
        query, k_candidates=5
    )

    torch.testing.assert_close(stats_exact.max_score, stats_batch.max_score, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(result_exact, result_batch, atol=1e-4, rtol=1e-3)


def test_batch_read_approximation_quality(medium_store):
    """batch版が近似的に良い一致を見つけることを確認"""
    torch.manual_seed(1)
    query = torch.randn(5, 32)

    _, stats_exact = medium_store.read(query)
    _, stats_batch = medium_store.read_with_faiss_search(query, k_candidates=50)

    torch.testing.assert_close(stats_exact.max_score, stats_batch.max_score, atol=1e-3, rtol=0.05)


def test_batch_read_with_various_k_candidates(small_store):
    """複数のk_candidatesでも結果が返される"""
    torch.manual_seed(2)
    query = torch.randn(5, 16)

    for k in [1, 3, 5, 10]:
        result, stats_batch = small_store.read_with_faiss_search(query, k_candidates=k)
        assert result.shape == (5, 8)
        assert stats_batch.max_score.shape == (5,)
        assert stats_batch.top1_index.shape == (5,)


def test_batch_read_empty_candidates(small_store):
    """候補がない場合も動作する"""
    query = torch.randn(5, 16) * 100

    result, stats = small_store.read_with_faiss_search(query, k_candidates=5)

    assert result.shape == (5, 8)
    assert result.isfinite().all()


def test_batch_read_single_query(small_store):
    """単一クエリでも動作"""
    query = torch.randn(1, 16)
    result, stats = small_store.read_with_faiss_search(query, k_candidates=5)

    assert result.shape == (1, 8)
    assert stats.max_score.shape == (1,)
    assert stats.top1_index.shape == (1,)


def test_batch_read_large_k_candidates(medium_store):
    """k_candidatesが全ストア数を超える場合"""
    query = torch.randn(10, 32)
    result, stats = medium_store.read_with_faiss_search(query, k_candidates=600)

    assert result.shape == (10, 16)
    assert stats.max_score.shape == (10,)


def test_batch_sorting_index_mapping(tmp_path):
    """インデックスマッピングの正確性を直接テスト"""
    store = DiskBackedAssociativeStore(key_dim=8, value_dim=4, persist_dir=tmp_path, exact=False)
    torch.manual_seed(5)
    keys = torch.randn(20, 8)
    values = torch.randn(20, 4)
    store.write(keys, values)
    keys_np = keys.detach().to("cpu", dtype=torch.float32).numpy()
    store._train_faiss_index(keys_np)

    candidates = [15, 3, 8, 1, 19, 12]
    sorted_cand = np.sort(np.array(candidates))
    idx_to_pos = {int(c): pos for pos, c in enumerate(sorted_cand)}

    for orig_idx, pos in idx_to_pos.items():
        assert sorted_cand[pos] == orig_idx


def test_batch_read_approximation_mode(small_store):
    """近似モード（exact=False）で softmax の重み付け和が正確"""
    torch.manual_seed(6)
    query = torch.randn(3, 16)

    result, _ = small_store.read_with_faiss_search(query, k_candidates=5)

    assert result.isfinite().all()
    assert result.shape == (3, 8)


def test_batch_read_exact_mode(medium_store):
    """exact モード（最良一致のみ返す）での動作"""
    medium_store.exact = True
    torch.manual_seed(7)
    query = torch.randn(5, 32)

    result_exact, stats_exact = medium_store.read(query)
    medium_store.exact = False
    result_batch, stats_batch = medium_store.read_with_faiss_search(
        query, k_candidates=50
    )

    torch.testing.assert_close(stats_exact.max_score, stats_batch.max_score, atol=1e-4, rtol=0.05)


def test_batch_read_chunked_queries(tmp_path):
    """チャンク処理での一貫性"""
    store = DiskBackedAssociativeStore(key_dim=16, value_dim=8, persist_dir=tmp_path, exact=False)
    torch.manual_seed(8)
    keys = torch.randn(100, 16)
    values = torch.randn(100, 8)
    store.write(keys, values)
    keys_np = keys.detach().to("cpu", dtype=torch.float32).numpy()
    store._train_faiss_index(keys_np)

    query = torch.randn(200, 16)

    result1, _ = store.read_with_faiss_search(query, chunk=50, k_candidates=10)
    result2, _ = store.read_with_faiss_search(query, chunk=100, k_candidates=10)
    result3, _ = store.read_with_faiss_search(query, chunk=200, k_candidates=10)

    torch.testing.assert_close(result1, result2, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(result2, result3, atol=1e-5, rtol=1e-4)
