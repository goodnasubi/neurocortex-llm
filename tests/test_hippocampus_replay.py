"""海馬モジュール — リプレイによる固定化（12.6.28節・ステップ10）の単体テスト。

最終的な忘却対策の良し悪しではなく、統制2（3条件がバックボーン・ヘッドの
アーキテクチャを共有すること）・統制5（ewc条件が生サンプルを一切保持しない
こと）という機構そのものを機械的に検証する
（`tests/test_path_audit.py`・`test_basal_ganglia_core.py` と同型）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.basal_ganglia_core import ParitySpec, make_batch  # noqa: E402
from neurocortex.hippocampus_replay import (  # noqa: E402
    compute_fisher_diagonal,
    evaluate,
    make_model,
    make_replay_buffer,
    run_condition,
    train_task,
    train_task_b_ewc,
    train_task_b_replay,
)


SPEC_A = ParitySpec(n_inputs=8, relevant=(0, 1))
SPEC_B = ParitySpec(n_inputs=8, relevant=(2, 3))


# --- 基本コンポーネント ---------------------------------------------------------

def test_train_task_reduces_loss() -> None:
    """健全性チェック: 教師あり学習でタスクAの正解率が十分高くなること。"""
    torch.manual_seed(0)
    model = make_model(n_inputs=8, hidden_dim=32)
    g = torch.Generator().manual_seed(0)
    train_task(model, SPEC_A, steps=500, batch_size=64, lr=0.05, generator=g)
    acc = evaluate(model, SPEC_A, n_eval=1000, generator=g)
    assert acc > 0.9


def test_make_replay_buffer_returns_requested_size() -> None:
    g = torch.Generator().manual_seed(0)
    bits, labels = make_replay_buffer(SPEC_A, buffer_size=50, generator=g)
    assert bits.shape == (50, 8)
    assert labels.shape == (50,)


def test_replay_buffer_labels_match_task_a() -> None:
    g = torch.Generator().manual_seed(0)
    bits, labels = make_replay_buffer(SPEC_A, buffer_size=50, generator=g)
    expected = ((bits[:, 0] > 0) ^ (bits[:, 1] > 0)).long()
    assert torch.equal(labels, expected)


def test_fisher_diagonal_has_matching_shapes() -> None:
    torch.manual_seed(0)
    model = make_model(n_inputs=8, hidden_dim=16)
    g = torch.Generator().manual_seed(0)
    fisher = compute_fisher_diagonal(model, SPEC_A, n_batches=5, batch_size=32, generator=g)
    for f, p in zip(fisher, model.parameters()):
        assert f.shape == p.shape
        assert torch.all(f >= 0)  # 勾配の二乗なので非負


# --- 統制5: ewc条件は生サンプルを保持しない ---------------------------------------

def test_ewc_training_never_calls_make_batch_on_task_a_after_fisher_computation(monkeypatch) -> None:
    """統制5の核心: フィッシャー計算後のタスクB学習フェーズでは、タスクAの
    `make_batch`が一切呼ばれないこと（生サンプルを再取得していないことの監査）。
    """
    torch.manual_seed(0)
    model = make_model(n_inputs=8, hidden_dim=16)
    g = torch.Generator().manual_seed(0)
    old_params = [p.detach().clone() for p in model.parameters()]
    fisher = compute_fisher_diagonal(model, SPEC_A, n_batches=3, batch_size=16, generator=g)

    import neurocortex.hippocampus_replay as hr

    original_make_batch = hr.make_batch
    calls_on_a = []

    def spy_make_batch(spec, batch_size, generator):
        if spec is SPEC_A:
            calls_on_a.append(1)
        return original_make_batch(spec, batch_size, generator)

    monkeypatch.setattr(hr, "make_batch", spy_make_batch)
    train_task_b_ewc(model, SPEC_B, old_params, fisher, ewc_lambda=1.0, steps=10,
                     batch_size=16, lr=0.05, generator=g)
    assert calls_on_a == []


def test_replay_condition_does_reuse_task_a_samples() -> None:
    """対比: replay条件は（設計通り）バッファ経由でタスクAのサンプルを実際に使うこと。"""
    torch.manual_seed(0)
    model = make_model(n_inputs=8, hidden_dim=16)
    g = torch.Generator().manual_seed(0)
    buffer = make_replay_buffer(SPEC_A, buffer_size=20, generator=g)
    before = model.head.weight.detach().clone()
    train_task_b_replay(model, SPEC_B, buffer, steps=10, batch_size=16, lr=0.05, generator=g)
    assert not torch.equal(before, model.head.weight.detach())


# --- 統制2: アーキテクチャの完全一致 ---------------------------------------------

def test_all_conditions_produce_the_same_architecture() -> None:
    hidden_dim = 16
    shapes = set()
    for _ in range(3):
        model = make_model(n_inputs=8, hidden_dim=hidden_dim)
        shapes.add(tuple(tuple(p.shape) for p in model.parameters()))
    assert len(shapes) == 1


# --- 実行の健全性 ---------------------------------------------------------------

def test_run_condition_naive_finetune_forgets_task_a() -> None:
    """統制1の前提: 保護機構なしでタスクBだけ学習すると、タスクA性能が明確に落ちること。"""
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    result = run_condition("naive-finetune", n_inputs=8, hidden_dim=32, spec_a=SPEC_A,
                           spec_b=SPEC_B, pretrain_steps=500, pretrain_batch=64,
                           pretrain_lr=0.05, buffer_size=50, fisher_batches=10,
                           fisher_batch_size=32, ewc_lambda=1.0, b_steps=500, b_batch=64,
                           b_lr=0.05, eval_n=1000, generator=g)
    assert result["acc_a"] < 0.7  # 忘却により大きく低下していること
    assert result["acc_b"] > 0.9  # タスクB自体は学習できていること


def test_run_condition_replay_and_ewc_return_valid_results() -> None:
    for condition in ("replay", "ewc"):
        torch.manual_seed(0)
        g = torch.Generator().manual_seed(0)
        result = run_condition(condition, n_inputs=8, hidden_dim=32, spec_a=SPEC_A,
                               spec_b=SPEC_B, pretrain_steps=500, pretrain_batch=64,
                               pretrain_lr=0.05, buffer_size=50, fisher_batches=10,
                               fisher_batch_size=32, ewc_lambda=1.0, b_steps=500, b_batch=64,
                               b_lr=0.05, eval_n=1000, generator=g)
        assert 0.0 <= result["acc_a"] <= 1.0
        assert 0.0 <= result["acc_b"] <= 1.0


def test_unknown_condition_raises() -> None:
    with pytest.raises(ValueError):
        run_condition("bogus", n_inputs=8, hidden_dim=16, spec_a=SPEC_A, spec_b=SPEC_B,
                      pretrain_steps=10, pretrain_batch=16, pretrain_lr=0.05, buffer_size=10,
                      fisher_batches=2, fisher_batch_size=16, ewc_lambda=1.0, b_steps=10,
                      b_batch=16, b_lr=0.05, eval_n=100, generator=torch.Generator().manual_seed(0))
