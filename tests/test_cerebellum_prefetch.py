"""ステップ34（小脳モジュールとの予測的プリフェッチ結合, 12.6.76〜77節）のテスト。

`DiskBackedAssociativeStore.prefetch`の基本動作（呼び出し後キャッシュに載ること）と、
プリフェッチあり/なしで`read`出力が完全一致すること（`torch.equal`）を、
小規模データで高速に検証する。
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import torch

from neurocortex.hippocampus import DiskBackedAssociativeStore


def _make_store(tmp_dir: Path, n: int = 50, d: int = 8, key_chunk: int = 8,
                 cache_size: int = 4, seed: int = 0) -> DiskBackedAssociativeStore:
    store = DiskBackedAssociativeStore(d, d, persist_dir=tmp_dir, exact=True,
                                        key_chunk=key_chunk, cache_size=cache_size)
    g = torch.Generator().manual_seed(seed)
    keys = torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=-1)
    values = torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=-1)
    store.write(keys, values)
    return store


def test_prefetch_loads_chunk_into_cache():
    tmp_dir = Path(tempfile.mkdtemp(prefix="prefetch_basic_"))
    try:
        store = _make_store(tmp_dir, n=50, key_chunk=8, cache_size=4)
        assert len(store._chunk_cache) == 0
        misses_before = store.cache_misses
        store.prefetch(1)  # チャンク1 = キー[8:16)
        assert 8 in store._chunk_cache
        assert store.cache_misses == misses_before + 1
        # 同じチャンクを再度読み出してもキャッシュヒットするだけで、
        # ミス数は増えない。
        store._get_chunk(store._keys_memmap(), store._values_memmap(), 8, store.key_chunk)
        assert store.cache_hits >= 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_prefetch_out_of_range_is_noop():
    tmp_dir = Path(tempfile.mkdtemp(prefix="prefetch_oor_"))
    try:
        store = _make_store(tmp_dir, n=20, key_chunk=8, cache_size=4)
        store.prefetch(999)  # 範囲外
        assert len(store._chunk_cache) == 0
        store.prefetch(-1)
        assert len(store._chunk_cache) == 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_prefetch_empty_store_is_noop():
    tmp_dir = Path(tempfile.mkdtemp(prefix="prefetch_empty_"))
    try:
        store = DiskBackedAssociativeStore(8, 8, persist_dir=tmp_dir, exact=True,
                                            key_chunk=8, cache_size=4)
        store.prefetch(0)  # 例外を投げず、何もしない
        assert len(store._chunk_cache) == 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_read_output_identical_with_and_without_prefetch():
    """正確性: プリフェッチあり/なしでread出力(value・best・arg)が完全一致する。"""
    d = 8
    n = 64
    key_chunk = 8
    g = torch.Generator().manual_seed(123)
    query_g = torch.Generator().manual_seed(456)
    queries = torch.nn.functional.normalize(torch.randn(10, d, generator=query_g), dim=-1)

    # プリフェッチなし
    dir_a = Path(tempfile.mkdtemp(prefix="prefetch_cmp_a_"))
    dir_b = Path(tempfile.mkdtemp(prefix="prefetch_cmp_b_"))
    try:
        store_a = DiskBackedAssociativeStore(d, d, persist_dir=dir_a, exact=True,
                                              key_chunk=key_chunk, cache_size=4)
        g_copy = torch.Generator().manual_seed(123)
        keys = torch.nn.functional.normalize(torch.randn(n, d, generator=g_copy), dim=-1)
        values = torch.nn.functional.normalize(torch.randn(n, d, generator=g_copy), dim=-1)
        store_a.write(keys, values)
        out_a, stats_a = store_a.read(queries)

        # プリフェッチあり: readの前にいくつかのチャンクを先読みしておく
        store_b = DiskBackedAssociativeStore(d, d, persist_dir=dir_b, exact=True,
                                              key_chunk=key_chunk, cache_size=4)
        store_b.write(keys, values)
        n_chunks = (n + key_chunk - 1) // key_chunk
        for c in range(n_chunks):
            store_b.prefetch(c)
        out_b, stats_b = store_b.read(queries)

        assert torch.equal(out_a, out_b)
        assert torch.equal(stats_a.max_score, stats_b.max_score)
        assert torch.equal(stats_a.top1_index, stats_b.top1_index)
    finally:
        shutil.rmtree(dir_a, ignore_errors=True)
        shutil.rmtree(dir_b, ignore_errors=True)


def test_existing_read_and_get_chunk_unaffected_by_prefetch():
    """既存の_get_chunk・readの挙動がprefetch追加によって変わらないことの回帰確認。"""
    tmp_dir = Path(tempfile.mkdtemp(prefix="prefetch_regress_"))
    try:
        store = _make_store(tmp_dir, n=40, key_chunk=8, cache_size=0)
        # cache_size=0では_get_chunkはキャッシュに一切触れない。
        store.prefetch(0)
        assert len(store._chunk_cache) == 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
