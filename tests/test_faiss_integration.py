"""ステップ36（faiss IVF統合による高速キー検索, 12.6.80節）の単体テスト。

faiss インデックスを用いた候補絞り込みが、従来の read() と同一の結果を
生成することを検証する。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.hippocampus import DiskBackedAssociativeStore


@pytest.fixture
def store_1000(tmp_path):
    """N=1000 のテスト用ストア（seed=0）。"""
    torch.manual_seed(0)
    key_dim, value_dim = 64, 32
    n = 1000

    keys = torch.nn.functional.normalize(torch.randn(n, key_dim), dim=-1)
    values = torch.nn.functional.normalize(torch.randn(n, value_dim), dim=-1)

    store = DiskBackedAssociativeStore(key_dim, value_dim, tmp_path, exact=True)
    store.write(keys, values)

    # faiss インデックス構築
    keys_np = keys.detach().to("cpu", dtype=torch.float32).numpy()
    try:
        store._train_faiss_index(keys_np)
    except ImportError:
        pytest.skip("faiss がインストールされていません")

    return store, keys, values


@pytest.fixture
def store_5000(tmp_path):
    """N=5000 のテスト用ストア（seed=0）。"""
    torch.manual_seed(0)
    key_dim, value_dim = 64, 32
    n = 5000

    keys = torch.nn.functional.normalize(torch.randn(n, key_dim), dim=-1)
    values = torch.nn.functional.normalize(torch.randn(n, value_dim), dim=-1)

    store = DiskBackedAssociativeStore(key_dim, value_dim, tmp_path, exact=True)
    store.write(keys, values)

    keys_np = keys.detach().to("cpu", dtype=torch.float32).numpy()
    try:
        store._train_faiss_index(keys_np)
    except ImportError:
        pytest.skip("faiss がインストールされていません")

    return store, keys, values


@pytest.fixture
def store_10000(tmp_path):
    """N=10000 のテスト用ストア（seed=0）。"""
    torch.manual_seed(0)
    key_dim, value_dim = 64, 32
    n = 10000

    keys = torch.nn.functional.normalize(torch.randn(n, key_dim), dim=-1)
    values = torch.nn.functional.normalize(torch.randn(n, value_dim), dim=-1)

    store = DiskBackedAssociativeStore(key_dim, value_dim, tmp_path, exact=True)
    store.write(keys, values)

    keys_np = keys.detach().to("cpu", dtype=torch.float32).numpy()
    try:
        store._train_faiss_index(keys_np)
    except ImportError:
        pytest.skip("faiss がインストールされていません")

    return store, keys, values


def test_faiss_exact_roundtrip_1000(store_1000):
    """N=1000: faiss 候補絞り込み + read が brute-force と同じ結果。"""
    store, keys, values = store_1000

    out_brute, stats_brute = store.read(keys, chunk=32)

    k_cand = max(10, int(len(keys)**0.5))
    out_faiss, stats_faiss = store.read_with_faiss_search(
        keys, chunk=32, k_candidates=k_cand
    )

    assert torch.equal(out_faiss, out_brute), \
        "N=1000: faiss版と brute-force版の値が一致しない"
    assert torch.equal(stats_faiss.top1_index, stats_brute.top1_index), \
        "N=1000: argmax が一致しない"


def test_faiss_exact_roundtrip_5000(store_5000):
    """N=5000: faiss 候補絞り込み + read が brute-force と同じ結果。"""
    store, keys, values = store_5000

    out_brute, stats_brute = store.read(keys, chunk=32)

    k_cand = max(10, int(len(keys)**0.5))
    out_faiss, stats_faiss = store.read_with_faiss_search(
        keys, chunk=32, k_candidates=k_cand
    )

    assert torch.equal(out_faiss, out_brute), \
        "N=5000: faiss版と brute-force版の値が一致しない"
    assert torch.equal(stats_faiss.top1_index, stats_brute.top1_index), \
        "N=5000: argmax が一致しない"


def test_faiss_exact_roundtrip_10000(store_10000):
    """N=10000: faiss 候補絞り込み + read が brute-force と同じ結果。"""
    store, keys, values = store_10000

    out_brute, stats_brute = store.read(keys, chunk=32)

    k_cand = max(10, int(len(keys)**0.5))
    out_faiss, stats_faiss = store.read_with_faiss_search(
        keys, chunk=32, k_candidates=k_cand
    )

    assert torch.equal(out_faiss, out_brute), \
        "N=10000: faiss版と brute-force版の値が一致しない"
    assert torch.equal(stats_faiss.top1_index, stats_brute.top1_index), \
        "N=10000: argmax が一致しない"


@pytest.mark.parametrize("k_cand", [10, 20])
def test_faiss_candidate_precision(k_cand: int):
    """faiss 候補に最適キーが含まれる率が 99.5% 以上であること。"""
    n = 10000
    key_dim, value_dim = 64, 32

    torch.manual_seed(0)
    keys = torch.nn.functional.normalize(torch.randn(n, key_dim), dim=-1)
    values = torch.nn.functional.normalize(torch.randn(n, value_dim), dim=-1)

    with tempfile.TemporaryDirectory() as tmpdir:
        store = DiskBackedAssociativeStore(key_dim, value_dim, tmpdir, exact=True)
        store.write(keys, values)

        keys_np = keys.detach().to("cpu", dtype=torch.float32).numpy()
        try:
            store._train_faiss_index(keys_np)
        except ImportError:
            pytest.skip("faiss がインストールされていません")

        # brute-force で真の最近傍インデックスを取得
        _, stats_brute = store.read(keys, chunk=32)
        true_nn = stats_brute.top1_index.long()  # [B]

        # faiss 候補を取得
        distances, indices = store._faiss_index.search(keys_np, k_cand)
        # indices: [n, k_cand]

        # 候補集合に最適キーが含まれているか判定
        contained = 0
        for i in range(n):
            if true_nn[i].item() in indices[i]:
                contained += 1

        precision = contained / n
        assert precision >= 0.995, \
            f"候補精度が低い: {precision:.3%} < 99.5% (k_cand={k_cand})"


def test_faiss_small_store(tmp_path):
    """小規模ストア（N < faiss_nlist）でも動作すること。"""
    n = 100  # nlist=100 より小さい
    key_dim, value_dim = 64, 32

    torch.manual_seed(0)
    keys = torch.nn.functional.normalize(torch.randn(n, key_dim), dim=-1)
    values = torch.nn.functional.normalize(torch.randn(n, value_dim), dim=-1)

    store = DiskBackedAssociativeStore(key_dim, value_dim, tmp_path, exact=True)
    store.write(keys, values)

    keys_np = keys.detach().to("cpu", dtype=torch.float32).numpy()
    try:
        store._train_faiss_index(keys_np)
    except ImportError:
        pytest.skip("faiss がインストールされていません")

    k_cand = 10
    out_faiss, _ = store.read_with_faiss_search(keys, chunk=32, k_candidates=k_cand)
    out_brute, _ = store.read(keys, chunk=32)

    assert torch.allclose(out_faiss, out_brute, atol=1e-5), \
        "小規模ストア: faiss版と brute-force版が一致しない"


def test_faiss_requires_index_init(tmp_path):
    """インデックス初期化なしで read_with_faiss_search を呼ぶと失敗すること。"""
    store = DiskBackedAssociativeStore(64, 32, tmp_path, exact=True)
    keys = torch.randn(10, 64)

    with pytest.raises(RuntimeError, match="未初期化"):
        store.read_with_faiss_search(keys)


def test_faiss_with_different_k_candidates(tmp_path):
    """異なる k_candidates 値で結果が正確であること。"""
    n = 5000
    key_dim, value_dim = 64, 32

    torch.manual_seed(0)
    keys = torch.nn.functional.normalize(torch.randn(n, key_dim), dim=-1)
    values = torch.nn.functional.normalize(torch.randn(n, value_dim), dim=-1)

    store = DiskBackedAssociativeStore(key_dim, value_dim, tmp_path, exact=True)
    store.write(keys, values)

    keys_np = keys.detach().to("cpu", dtype=torch.float32).numpy()
    try:
        store._train_faiss_index(keys_np)
    except ImportError:
        pytest.skip("faiss がインストールされていません")

    out_brute, stats_brute = store.read(keys, chunk=32)

    for k_cand in [10, 20, int(n**0.5)]:
        out_faiss, stats_faiss = store.read_with_faiss_search(
            keys, chunk=32, k_candidates=k_cand
        )
        assert torch.equal(out_faiss, out_brute), \
            f"k_candidates={k_cand}: faiss版が brute-force版と一致しない"
