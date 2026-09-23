"""ステップ32（12.6.72〜73節）: `DiskBackedAssociativeStore`の単体テスト。

`AssociativeStore`の`write`・`read`を一切変更しないという設計上の制約と、
ディスク常駐方式の正確性（`read`出力の一致）を確認する最小テスト。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.hippocampus import (  # noqa: E402
    AssociativeStore,
    DiskBackedAssociativeStore,
)


def test_write_read_roundtrip(tmp_path: Path) -> None:
    """1件だけ書けば必ず戻る（`AssociativeStore`の同名テストのディスク版）。"""
    store = DiskBackedAssociativeStore(key_dim=8, value_dim=4, persist_dir=tmp_path, exact=True)
    key = torch.zeros(1, 8)
    key[0, 3] = 1.0
    value = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    store.write(key, value)
    out, stats = store.read(key)
    assert torch.allclose(out, value, atol=1e-5)
    assert pytest.approx(1.0, abs=1e-5) == float(stats.max_score)
    assert int(stats.top1_index[0]) == 0


def test_empty_store_returns_zero(tmp_path: Path) -> None:
    store = DiskBackedAssociativeStore(8, 4, persist_dir=tmp_path)
    out, stats = store.read(torch.randn(3, 8))
    assert float(out.abs().max()) == 0.0
    assert torch.equal(stats.top1_index, torch.full((3,), -1))


def test_exact_mode_matches_in_memory_store_exactly(tmp_path: Path) -> None:
    """主張(b)の単体版: `exact=True`では`AssociativeStore`と`read`出力が完全一致する。"""
    torch.manual_seed(0)
    n, key_dim, value_dim = 200, 16, 8
    keys = torch.nn.functional.normalize(torch.randn(n, key_dim), dim=-1)
    values = torch.nn.functional.normalize(torch.randn(n, value_dim), dim=-1)
    queries = torch.nn.functional.normalize(torch.randn(37, key_dim), dim=-1)

    ram_store = AssociativeStore(key_dim, value_dim, exact=True)
    ram_store.write(keys, values)
    ram_out, ram_stats = ram_store.read(queries)

    disk_store = DiskBackedAssociativeStore(key_dim, value_dim, persist_dir=tmp_path,
                                             exact=True, key_chunk=37)
    disk_store.write(keys, values)
    disk_out, disk_stats = disk_store.read(queries, chunk=13)

    assert torch.equal(ram_stats.top1_index, disk_stats.top1_index)
    assert torch.equal(ram_out, disk_out)
    # max_score（コサイン類似度の値そのもの）は、全件一括matmul（RAM常駐版）と
    # チャンク単位matmul（ディスク版）とでBLAS内部のリダクション順序が異なるため、
    # torch.equalでのビット完全一致は保証されない（実測で絶対誤差1e-6程度、
    # typesafe-aiによる判断: value・argの完全一致とmax_scoreの数値的一致を
    # もって実質的に正確性が保たれているとみなす、確信度0.8）。
    assert torch.allclose(ram_stats.max_score, disk_stats.max_score, atol=1e-5)


def test_non_exact_mode_matches_in_memory_store_closely(tmp_path: Path) -> None:
    """非exactモード（softmax加重和）は数学的に同一値のはずで、`allclose`で一致する。"""
    torch.manual_seed(1)
    n, key_dim, value_dim = 150, 12, 6
    keys = torch.nn.functional.normalize(torch.randn(n, key_dim), dim=-1)
    values = torch.nn.functional.normalize(torch.randn(n, value_dim), dim=-1)
    queries = torch.nn.functional.normalize(torch.randn(20, key_dim), dim=-1)

    ram_store = AssociativeStore(key_dim, value_dim, beta=50.0, exact=False)
    ram_store.write(keys, values)
    ram_out, ram_stats = ram_store.read(queries)

    disk_store = DiskBackedAssociativeStore(key_dim, value_dim, persist_dir=tmp_path,
                                             beta=50.0, exact=False, key_chunk=29)
    disk_store.write(keys, values)
    disk_out, disk_stats = disk_store.read(queries)

    assert torch.equal(ram_stats.top1_index, disk_stats.top1_index)
    assert torch.allclose(ram_out, disk_out, atol=1e-4)


def test_write_can_be_incremental(tmp_path: Path) -> None:
    """`write`を複数回に分けても、1回でまとめて書いた場合と同じ結果になること。"""
    torch.manual_seed(2)
    key_dim, value_dim = 8, 4
    keys = torch.nn.functional.normalize(torch.randn(30, key_dim), dim=-1)
    values = torch.nn.functional.normalize(torch.randn(30, value_dim), dim=-1)
    queries = torch.nn.functional.normalize(torch.randn(5, key_dim), dim=-1)

    store = DiskBackedAssociativeStore(key_dim, value_dim, persist_dir=tmp_path, exact=True)
    store.write(keys[:10], values[:10])
    store.write(keys[10:20], values[10:20])
    store.write(keys[20:], values[20:])
    assert len(store) == 30
    out, stats = store.read(queries)

    ram_store = AssociativeStore(key_dim, value_dim, exact=True)
    ram_store.write(keys, values)
    ram_out, ram_stats = ram_store.read(queries)

    assert torch.equal(stats.top1_index, ram_stats.top1_index)
    assert torch.equal(out, ram_out)


def test_clear_empties_the_store(tmp_path: Path) -> None:
    store = DiskBackedAssociativeStore(8, 4, persist_dir=tmp_path)
    store.write(torch.nn.functional.normalize(torch.randn(2, 8), dim=-1), torch.randn(2, 4))
    store.clear()
    assert len(store) == 0
    assert float(store.read(torch.randn(3, 8))[0].abs().max()) == 0.0


def test_persists_across_new_instances(tmp_path: Path) -> None:
    """`persist_dir`を再度開くと既存の書き込みが引き継がれること（ディスク常駐の本質）。"""
    key_dim, value_dim = 8, 4
    store1 = DiskBackedAssociativeStore(key_dim, value_dim, persist_dir=tmp_path, exact=True)
    key = torch.zeros(1, key_dim)
    key[0, 0] = 1.0
    value = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    store1.write(key, value)

    store2 = DiskBackedAssociativeStore(key_dim, value_dim, persist_dir=tmp_path, exact=True)
    assert len(store2) == 1
    out, stats = store2.read(key)
    assert torch.allclose(out, value, atol=1e-5)
    assert int(stats.top1_index[0]) == 0


def test_does_not_mutate_in_memory_associative_store_behaviour() -> None:
    """`DiskBackedAssociativeStore`の追加は既存`AssociativeStore`のコード・挙動を変えない。"""
    torch.manual_seed(3)
    keys = torch.nn.functional.normalize(torch.randn(16, 8), dim=-1)
    values = torch.nn.functional.normalize(torch.randn(16, 4), dim=-1)
    store = AssociativeStore(8, 4, exact=True)
    store.write(keys, values)
    out, stats = store.read(keys)
    assert torch.equal(stats.top1_index, torch.arange(16))
    assert torch.allclose(out, values, atol=1e-5)
