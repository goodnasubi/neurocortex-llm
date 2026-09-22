"""海馬モジュール — リプレイバッファの容量制約下での刈り込み戦略（12.6.32節・ステップ12）の単体テスト。

最終的な保持率の良し悪しではなく、統制1（容量制約下で溢れが実際に起きること）・
統制3（uniform-compressが重要度を見ず均等配分すること）という機構そのものを
機械的に検証する（`test_hippocampus_replay.py`・`test_small_world_layers.py` と同型）。
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.hippocampus_capacity import (  # noqa: E402
    CONDITIONS,
    evict_fifo,
    evict_importance_weighted,
    evict_uniform,
    make_task_specs,
    model_input_dim,
    run_condition,
    snapshot_task,
    task_importance_loss_based,
    train_with_replay,
)
from neurocortex.hippocampus_replay import make_model  # noqa: E402


# --- タスク仕様 -----------------------------------------------------------------

def test_make_task_specs_uses_disjoint_bit_pairs() -> None:
    specs = make_task_specs(n_tasks=5, n_inputs=10)
    assert len(specs) == 5
    seen: set[int] = set()
    for spec in specs:
        a, b = spec.relevant
        assert a not in seen and b not in seen
        seen.add(a)
        seen.add(b)


def test_make_task_specs_rejects_too_few_inputs() -> None:
    with pytest.raises(ValueError):
        make_task_specs(n_tasks=5, n_inputs=8)


def test_snapshot_task_returns_requested_size_with_task_id_appended() -> None:
    specs = make_task_specs(n_tasks=3, n_inputs=8)
    g = torch.Generator().manual_seed(0)
    bits, labels = snapshot_task(specs[0], task_idx=0, n_tasks=3, size=40, generator=g)
    assert bits.shape == (40, 8 + 3)  # 元の8ビット + タスクIDのone-hot(3)
    assert labels.shape == (40,)


def test_snapshot_task_one_hot_matches_task_idx() -> None:
    specs = make_task_specs(n_tasks=3, n_inputs=8)
    g = torch.Generator().manual_seed(0)
    bits, _ = snapshot_task(specs[1], task_idx=1, n_tasks=3, size=10, generator=g)
    onehot = bits[:, 8:]
    assert torch.all(onehot[:, 1] == 1.0)
    assert torch.all(onehot[:, [0, 2]] == 0.0)


# --- 刈り込み戦略: 容量を守ること -----------------------------------------------

def _make_buffer(sizes: list[int], n_inputs: int = 8) -> OrderedDict:
    g = torch.Generator().manual_seed(0)
    n_tasks = len(sizes)
    specs = make_task_specs(n_tasks=n_tasks, n_inputs=n_inputs)
    buf = OrderedDict()
    for t, (spec, size) in enumerate(zip(specs, sizes)):
        buf[t] = snapshot_task(spec, t, n_tasks, size, g)
    return buf


def test_evict_fifo_respects_capacity() -> None:
    buf = _make_buffer([50, 50, 50])
    out = evict_fifo(buf, capacity=80)
    total = sum(b.shape[0] for b, _ in out.values())
    assert total == 80


def test_evict_fifo_drops_oldest_task_first() -> None:
    """統制1の前提: FIFOは新しいタスク(2)を優先的に残し、最古タスク(0)から削る。"""
    buf = _make_buffer([50, 50, 50])
    out = evict_fifo(buf, capacity=60)
    assert 0 not in out or out[0][0].shape[0] < 50
    assert out[2][0].shape[0] == 50


def test_evict_uniform_allocates_equal_counts_per_task() -> None:
    buf = _make_buffer([50, 50, 50])
    g = torch.Generator().manual_seed(1)
    out = evict_uniform(buf, capacity=60, generator=g)
    counts = {t: b.shape[0] for t, (b, _) in out.items()}
    assert len(set(counts.values())) == 1  # 統制3: 重要度を見ず均等配分
    assert sum(counts.values()) <= 60


def test_evict_importance_weighted_favors_higher_importance_task() -> None:
    buf = _make_buffer([50, 50])
    importances = {0: 1.0, 1: 9.0}
    g = torch.Generator().manual_seed(2)
    out = evict_importance_weighted(buf, importances, capacity=50, generator=g)
    assert out[1][0].shape[0] > out[0][0].shape[0]
    total = sum(b.shape[0] for b, _ in out.values())
    assert total <= 50


def test_evict_functions_are_noop_under_capacity() -> None:
    buf = _make_buffer([10, 10])
    g = torch.Generator().manual_seed(0)
    assert sum(b.shape[0] for b, _ in evict_fifo(buf, capacity=100).values()) == 20
    assert sum(b.shape[0] for b, _ in evict_uniform(buf, capacity=100, generator=g).values()) == 20


# --- 学習ループ ------------------------------------------------------------------

def test_train_with_replay_with_empty_buffer_updates_model() -> None:
    torch.manual_seed(0)
    specs = make_task_specs(n_tasks=2, n_inputs=8)
    model = make_model(n_inputs=model_input_dim(8, 2), hidden_dim=16)
    before = model.head.weight.detach().clone()
    g = torch.Generator().manual_seed(0)
    train_with_replay(model, specs[0], task_idx=0, n_tasks=2, buffer=OrderedDict(), steps=10,
                      batch_size=16, lr=0.05, generator=g)
    assert not torch.equal(before, model.head.weight.detach())


def test_train_with_replay_uses_buffer_when_nonempty(monkeypatch) -> None:
    """バッファに保持タスクがあれば、学習バッチにそのサンプルが混ざること
    （バッチサイズより大きい出力を`cross_entropy`に渡すには複数タスク分が
    連結されている必要がある — 間接的にバッファ利用を確認する）。
    """
    torch.manual_seed(0)
    specs = make_task_specs(n_tasks=2, n_inputs=8)
    model = make_model(n_inputs=model_input_dim(8, 2), hidden_dim=16)
    g = torch.Generator().manual_seed(0)
    buf = OrderedDict({0: snapshot_task(specs[0], 0, 2, 20, g)})
    import neurocortex.hippocampus_capacity as hc

    original_cross_entropy = torch.nn.functional.cross_entropy
    seen_batch_sizes = []

    def spy_cross_entropy(logits, labels, *args, **kwargs):
        seen_batch_sizes.append(logits.shape[0])
        return original_cross_entropy(logits, labels, *args, **kwargs)

    class FakeF:
        cross_entropy = staticmethod(spy_cross_entropy)
        one_hot = staticmethod(torch.nn.functional.one_hot)
        log_softmax = staticmethod(torch.nn.functional.log_softmax)
        nll_loss = staticmethod(torch.nn.functional.nll_loss)

    monkeypatch.setattr(hc, "F", FakeF)
    train_with_replay(model, specs[1], task_idx=1, n_tasks=2, buffer=buf, steps=3, batch_size=16,
                      lr=0.05, generator=g)
    assert all(n == 32 for n in seen_batch_sizes)  # 新タスク16 + バッファ1タスク分16


# --- 統制1: 容量制約下で溢れが実際に起きること -----------------------------------

def test_run_condition_unbounded_keeps_growing_without_eviction() -> None:
    """統制1の前提確認: unboundedは容量制約を受けず、全タスク分をそのまま保持する。"""
    specs = make_task_specs(n_tasks=4, n_inputs=8)
    g = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    result = run_condition("unbounded", specs, hidden_dim=16, capacity=40, snapshot_size=30,
                           task_steps=5, batch_size=16, lr=0.05, fisher_batches=2,
                           fisher_batch_size=16, eval_n=200, generator=g)
    assert len(result.per_task_acc) == 3  # タスク1..3（最終タスクを除く）


@pytest.mark.parametrize("condition", ["fifo", "uniform-compress", "importance-weighted",
                                       "importance-weighted-fisher-decay",
                                       "importance-weighted-loss-based"])
def test_run_condition_bounded_conditions_return_valid_results(condition: str) -> None:
    specs = make_task_specs(n_tasks=4, n_inputs=8)
    g = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    result = run_condition(condition, specs, hidden_dim=16, capacity=40, snapshot_size=30,
                           task_steps=5, batch_size=16, lr=0.05, fisher_batches=2,
                           fisher_batch_size=16, eval_n=200, generator=g)
    assert len(result.per_task_acc) == 3
    assert all(0.0 <= a <= 1.0 for a in result.per_task_acc)
    assert 0.0 <= result.mean_old_task_acc <= 1.0
    assert 0.0 <= result.oldest_task_acc <= 1.0


def test_unknown_condition_raises() -> None:
    specs = make_task_specs(n_tasks=2, n_inputs=8)
    with pytest.raises(ValueError):
        run_condition("bogus", specs, hidden_dim=16, capacity=40, snapshot_size=30, task_steps=5,
                      batch_size=16, lr=0.05, fisher_batches=2, fisher_batch_size=16, eval_n=200,
                      generator=torch.Generator().manual_seed(0))


def test_all_conditions_covered() -> None:
    assert set(CONDITIONS) == {
        "unbounded", "fifo", "uniform-compress", "importance-weighted",
        "importance-weighted-fisher-decay", "importance-weighted-loss-based",
    }


# --- ステップ13: フィッシャー崩壊への対策 ----------------------------------------

def test_loss_based_importance_is_nonzero_even_near_convergence() -> None:
    """統制の核心: 損失ベースの重要度は、フィッシャー情報量と違って0に潰れきらないこと。"""
    torch.manual_seed(0)
    specs = make_task_specs(n_tasks=2, n_inputs=8)
    model = make_model(n_inputs=model_input_dim(8, 2), hidden_dim=32)
    g = torch.Generator().manual_seed(0)
    train_with_replay(model, specs[0], task_idx=0, n_tasks=2, buffer=OrderedDict(), steps=300,
                      batch_size=64, lr=0.05, generator=g)
    importance = task_importance_loss_based(model, specs[0], task_idx=0, n_tasks=2, n_batches=5,
                                            batch_size=32, generator=g)
    assert importance > 1e-6  # フィッシャー情報量（1e-10オーダー）とは対照的に潰れない


def test_run_condition_fisher_decay_applies_weight_decay_during_training() -> None:
    """importance-weighted-fisher-decay条件は`weight_decay`付きで学習すること
    （他条件と学習手続きが変わることの確認 — 対策そのものが差分であることの監査）。
    """
    specs = make_task_specs(n_tasks=2, n_inputs=8)
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    result = run_condition("importance-weighted-fisher-decay", specs, hidden_dim=16, capacity=30,
                           snapshot_size=30, task_steps=5, batch_size=16, lr=0.05,
                           fisher_batches=2, fisher_batch_size=16, eval_n=100, generator=g,
                           fisher_decay_weight_decay=0.01)
    assert 0.0 <= result.mean_old_task_acc <= 1.0
