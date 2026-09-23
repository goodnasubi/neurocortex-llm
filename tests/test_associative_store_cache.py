"""ステップ33（12.6.74〜75節）: `DiskBackedAssociativeStore`のRAMキャッシュ層の単体テスト。

キャッシュ層追加が既存の`read`出力・コードパスを変えないこと（cache_size=0）、
キャッシュあり/なしで出力が完全一致すること（正確性、主張(b)の単体版）、
キャッシュヒット時にディスク読み出し回数が減ること、を確認する最小テスト。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.hippocampus import DiskBackedAssociativeStore  # noqa: E402


def _make_store(tmp_path: Path, n: int, key_dim: int, value_dim: int, key_chunk: int,
                 cache_size: int, seed: int) -> tuple[DiskBackedAssociativeStore, torch.Tensor]:
    torch.manual_seed(seed)
    keys = torch.nn.functional.normalize(torch.randn(n, key_dim), dim=-1)
    values = torch.nn.functional.normalize(torch.randn(n, value_dim), dim=-1)
    store = DiskBackedAssociativeStore(key_dim, value_dim, persist_dir=tmp_path, exact=True,
                                        key_chunk=key_chunk, cache_size=cache_size)
    store.write(keys, values)
    queries = torch.nn.functional.normalize(torch.randn(40, key_dim), dim=-1)
    return store, queries


def test_cache_size_zero_matches_existing_behaviour(tmp_path: Path) -> None:
    """cache_size=0（既定）ではステップ32の`read`と完全に同一の出力になること。"""
    store0, queries = _make_store(tmp_path / "a", n=300, key_dim=16, value_dim=8,
                                   key_chunk=37, cache_size=0, seed=0)
    out0, stats0 = store0.read(queries, chunk=13)

    torch.manual_seed(0)
    from neurocortex.hippocampus import AssociativeStore
    keys = torch.nn.functional.normalize(torch.randn(300, 16), dim=-1)
    values = torch.nn.functional.normalize(torch.randn(300, 8), dim=-1)
    ram_store = AssociativeStore(16, 8, exact=True)
    ram_store.write(keys, values)
    ram_out, ram_stats = ram_store.read(queries)

    assert torch.equal(ram_stats.top1_index, stats0.top1_index)
    assert torch.equal(ram_out, out0)
    # cache_size=0時はキャッシュが一切使われないこと。
    assert store0.cache_hits == 0
    assert store0.cache_misses == 0
    assert len(store0._chunk_cache) == 0


def test_cache_matches_no_cache_output_exact_mode(tmp_path: Path) -> None:
    """主張(b)の単体版（exact=True）: キャッシュあり/なしのread出力が完全一致すること。"""
    n, key_dim, value_dim, key_chunk = 500, 16, 8, 37
    for cache_size in (4, 16, 64):
        store_nc, queries = _make_store(tmp_path / f"nc{cache_size}", n, key_dim, value_dim,
                                         key_chunk, cache_size=0, seed=1)
        out_nc, stats_nc = store_nc.read(queries, chunk=13)

        store_c, _ = _make_store(tmp_path / f"c{cache_size}", n, key_dim, value_dim,
                                  key_chunk, cache_size=cache_size, seed=1)
        out_c, stats_c = store_c.read(queries, chunk=13)

        assert torch.equal(out_nc, out_c), f"cache_size={cache_size}: value不一致"
        assert torch.equal(stats_nc.top1_index, stats_c.top1_index), f"cache_size={cache_size}: arg不一致"
        assert torch.equal(stats_nc.max_score, stats_c.max_score), f"cache_size={cache_size}: best不一致"


def test_cache_matches_no_cache_output_non_exact_mode(tmp_path: Path) -> None:
    """主張(b)の単体版（exact=False、softmax加重和）でもvalueが完全一致すること。"""
    torch.manual_seed(2)
    n, key_dim, value_dim, key_chunk = 300, 12, 6, 29
    keys = torch.nn.functional.normalize(torch.randn(n, key_dim), dim=-1)
    values = torch.nn.functional.normalize(torch.randn(n, value_dim), dim=-1)
    queries = torch.nn.functional.normalize(torch.randn(20, key_dim), dim=-1)

    store_nc = DiskBackedAssociativeStore(key_dim, value_dim, persist_dir=tmp_path / "nc",
                                           beta=50.0, exact=False, key_chunk=key_chunk, cache_size=0)
    store_nc.write(keys, values)
    out_nc, stats_nc = store_nc.read(queries)

    store_c = DiskBackedAssociativeStore(key_dim, value_dim, persist_dir=tmp_path / "c",
                                          beta=50.0, exact=False, key_chunk=key_chunk, cache_size=8)
    store_c.write(keys, values)
    out_c, stats_c = store_c.read(queries)

    assert torch.equal(out_nc, out_c)
    assert torch.equal(stats_nc.top1_index, stats_c.top1_index)


def test_cache_hit_avoids_disk_read(tmp_path: Path, monkeypatch) -> None:
    """キャッシュヒット時にディスク読み出し（np.array(memmap[...])）回数が減ること。"""
    import numpy as np

    n, key_dim, value_dim, key_chunk = 400, 8, 4, 50  # n_total//key_chunk = 8チャンク
    store, _ = _make_store(tmp_path, n, key_dim, value_dim, key_chunk, cache_size=8, seed=3)

    call_count = {"n": 0}
    orig_array = np.array

    def counting_array(*args, **kwargs):
        call_count["n"] += 1
        return orig_array(*args, **kwargs)

    monkeypatch.setattr(np, "array", counting_array)

    query = torch.nn.functional.normalize(torch.randn(1, key_dim), dim=-1)
    store.read(query)
    first_calls = call_count["n"]
    assert store.cache_misses == 8  # 全8チャンクが初回ミス

    call_count["n"] = 0
    store.read(query)  # 同じキャッシュ容量（8）で全チャンク保持されているため全ヒット
    second_calls = call_count["n"]

    assert second_calls < first_calls
    assert store.cache_hits >= 8
