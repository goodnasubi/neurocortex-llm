"""海馬モジュール（12.6.4節・ステップ1）の単体テスト。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.hippocampus import (  # noqa: E402
    AssociativeStore,
    HippocampalMemory,
    PatternSeparator,
)
from neurocortex.tasks import FactSpec, make_fact_batch  # noqa: E402


# --- パターン分離層（DG） ---------------------------------------------------

def test_sdr_sparsity_is_exactly_k() -> None:
    """k-WTA は疎度をkに固定する（固定閾値方式との差。12.6.4節）。"""
    sep = PatternSeparator(16, n_units=256, k=10, mode="sdr")
    for scale in (1e-3, 1.0, 1e3):  # 入力ノルムを変えても疎度は変わらない
        out = sep(torch.randn(8, 16) * scale)
        assert torch.all((out != 0).sum(dim=-1) == 10)


@pytest.mark.parametrize("mode", ["sdr", "dense", "identity"])
def test_keys_are_unit_norm(mode: str) -> None:
    """キーがL2正規化されていること（内積＝コサイン類似度の前提）。"""
    sep = PatternSeparator(16, n_units=256, k=10, mode=mode)
    out = sep(torch.randn(8, 16))
    assert torch.allclose(out.norm(dim=-1), torch.ones(8), atol=1e-5)


def test_separator_has_no_trainable_parameters() -> None:
    """射影は学習しない（ステップ1）。勾配を持つテンソルが存在しないこと。"""
    sep = PatternSeparator(16, n_units=64, k=4, mode="sdr")
    assert list(sep.parameters()) == []


def test_pattern_separation_reduces_similarity() -> None:
    """似た入力の類似度が、分離層を通すと下がること（6.1節の定義そのもの）。

    これは主指標の前提であり、成り立たなければ分離層を名乗れない。
    """
    torch.manual_seed(0)
    base = torch.randn(64, 32)
    near = base + 0.25 * torch.randn(64, 32)  # わずかに異なる入力
    ident = PatternSeparator(32, mode="identity")
    sdr = PatternSeparator(32, n_units=1024, k=20, mode="sdr")
    cos_in = (ident(base) * ident(near)).sum(-1).mean()
    cos_out = (sdr(base) * sdr(near)).sum(-1).mean()
    assert cos_out < cos_in, f"分離できていない（入力 {cos_in:.3f} → 出力 {cos_out:.3f}）"


# --- 連想ストア（CA3） -------------------------------------------------------

def test_write_read_roundtrip() -> None:
    """1件だけ書けば必ず戻る（統制1「容量の統制」の単体版）。"""
    store = AssociativeStore(key_dim=8, value_dim=4, beta=1e3)
    key = torch.zeros(1, 8)
    key[0, 3] = 1.0
    value = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    store.write(key, value)
    out, stats = store.read(key)
    assert torch.allclose(out, value, atol=1e-5)
    assert pytest.approx(1.0, abs=1e-5) == float(stats.max_score)


def test_exact_mode_is_nearest_neighbour() -> None:
    """exact=True は β→∞ の極限、すなわち厳密最近傍（dict 相当）であること。"""
    torch.manual_seed(0)
    keys = torch.nn.functional.normalize(torch.randn(16, 8), dim=-1)
    values = torch.nn.functional.normalize(torch.randn(16, 4), dim=-1)
    store = AssociativeStore(8, 4, exact=True)
    store.write(keys, values)
    out, stats = store.read(keys)
    assert torch.equal(stats.top1_index, torch.arange(16))
    assert torch.allclose(out, values, atol=1e-5)


def test_empty_store_returns_zero() -> None:
    """統制2「因果的除去」: ストアを空にすると注入がゼロになる。"""
    store = AssociativeStore(8, 4)
    out, stats = store.read(torch.randn(3, 8))
    assert float(out.abs().max()) == 0.0
    assert torch.equal(stats.top1_index, torch.full((3,), -1))
    store.write(torch.nn.functional.normalize(torch.randn(2, 8), dim=-1), torch.randn(2, 4))
    store.clear()
    assert len(store) == 0
    assert float(store.read(torch.randn(3, 8))[0].abs().max()) == 0.0


def test_write_does_not_build_a_graph() -> None:
    """書き込みは勾配を通さない局所更新である（12.3.2節）。

    勾配が有効な文脈で書き込んでも、ストア内のテンソルが計算グラフに繋がらないこと。
    """
    mem = HippocampalMemory(8, value_dim=4, n_units=64, k=4)
    x = torch.randn(5, 8, requires_grad=True)
    v = torch.randn(5, 4, requires_grad=True)
    with torch.enable_grad():
        mem.write(x, v)
        inject, gate = mem.read(x)
    assert not mem.store._keys.requires_grad
    assert not mem.store._values.requires_grad
    assert not inject.requires_grad and not gate.requires_grad


def test_gate_closes_for_unknown_cues() -> None:
    """知らない手がかりではゲートが閉じる（CA1の新規性検出, 6.1節）。"""
    torch.manual_seed(0)
    mem = HippocampalMemory(32, value_dim=8, n_units=1024, k=20, gate_bias=0.3)
    known = torch.randn(32, 32)
    mem.write(known, torch.randn(32, 8))
    _, g_known = mem.read(known)
    _, g_unknown = mem.read(torch.randn(32, 32))
    assert float(g_known.mean()) > float(g_unknown.mean())


# --- 三つ組課題 --------------------------------------------------------------

def test_fact_batch_subjects_are_unique() -> None:
    """主語が重複しないこと（重複すると別種の失敗が混ざる）。"""
    spec = FactSpec(n_prefix=4, n_suffix=64, n_object=16)
    prompts, objects = make_fact_batch(spec, 200, torch.Generator().manual_seed(0))
    subjects = {tuple(r.tolist()) for r in prompts[:, :2]}
    assert len(subjects) == 200
    assert torch.all(prompts[:, 2] == spec.relation_id)
    assert torch.all(prompts[:, 0] < spec.n_prefix)
    assert torch.all(objects >= spec.object_offset)
    assert torch.all(objects < spec.vocab_size)


def test_fact_batch_rejects_too_many() -> None:
    spec = FactSpec(n_prefix=2, n_suffix=4, n_object=4)
    with pytest.raises(ValueError):
        make_fact_batch(spec, 9, torch.Generator().manual_seed(0))
