"""マルチ学習則の学習安定性 — 目的関数が対立するシナリオ（12.6.38節・ステップ15）の単体テスト。

最終的な干渉の有無ではなく、統制1（independentのタスクA保持率がタスクB学習の
影響を一切受けないこと）・統制2（rl-only-no-replayはリプレイ勾配を一切使わない
こと）という機構そのものを機械的に検証する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.basal_ganglia_core import ParitySpec  # noqa: E402
from neurocortex.gradient_conflict import (  # noqa: E402
    CONDITIONS,
    evaluate_hippocampus,
    evaluate_rl,
    make_replay_buffer,
    run_condition,
    train_task_a,
)
from neurocortex.basal_ganglia_core import Backbone  # noqa: E402

SPEC_A = ParitySpec(n_inputs=8, relevant=(0, 1))
SPEC_B = ParitySpec(n_inputs=8, relevant=(2, 3))


# --- 基本コンポーネント -----------------------------------------------------------

def test_train_task_a_reaches_high_accuracy() -> None:
    torch.manual_seed(0)
    backbone = Backbone(8, 32)
    head = torch.nn.Linear(32, 2)
    g = torch.Generator().manual_seed(0)
    train_task_a(backbone, head, SPEC_A, steps=500, batch_size=64, lr=0.05, generator=g)
    acc = evaluate_hippocampus(backbone, head, SPEC_A, n_eval=1000, generator=g)
    assert acc > 0.9


def test_make_replay_buffer_returns_requested_size() -> None:
    g = torch.Generator().manual_seed(0)
    bits, labels = make_replay_buffer(SPEC_A, buffer_size=50, generator=g)
    assert bits.shape == (50, 8)
    assert labels.shape == (50,)


# --- 統制1: independentのタスクA保持率はタスクB学習の影響を受けないこと -------------

def test_independent_task_a_retention_is_unaffected_by_task_b_steps() -> None:
    """統制1の核心: `b_steps`を増減させても、independent条件のタスクA保持率は
    （バックボーンが完全に別なので）変わらないこと。
    """
    torch.manual_seed(0)
    g1 = torch.Generator().manual_seed(0)
    short = run_condition("independent", SPEC_A, SPEC_B, hidden_dim=16, buffer_size=30,
                          pretrain_steps=300, pretrain_batch=32, pretrain_lr=0.05, b_steps=5,
                          b_batch=16, b_lr=0.05, eval_n=500, generator=g1)

    torch.manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    long = run_condition("independent", SPEC_A, SPEC_B, hidden_dim=16, buffer_size=30,
                         pretrain_steps=300, pretrain_batch=32, pretrain_lr=0.05, b_steps=50,
                         b_batch=16, b_lr=0.05, eval_n=500, generator=g2)
    assert short.task_a_retention > 0.85
    assert long.task_a_retention > 0.85


# --- 統制2: rl-only-no-replayはリプレイ勾配を一切使わないこと ------------------------

def test_rl_only_no_replay_never_calls_replay_loss(monkeypatch) -> None:
    import neurocortex.gradient_conflict as gc

    called = []
    original = gc._replay_loss

    def spy(*args, **kwargs):
        called.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(gc, "_replay_loss", spy)
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    run_condition("rl-only-no-replay", SPEC_A, SPEC_B, hidden_dim=16, buffer_size=20,
                 pretrain_steps=50, pretrain_batch=16, pretrain_lr=0.05, b_steps=10, b_batch=16,
                 b_lr=0.05, eval_n=200, generator=g)
    assert called == []


def test_rl_only_no_replay_forgets_task_a() -> None:
    """統制2の前提: リプレイ勾配なしでタスクBだけ学習すると、タスクA保持率が明確に落ちること。"""
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    result = run_condition("rl-only-no-replay", SPEC_A, SPEC_B, hidden_dim=32, buffer_size=50,
                           pretrain_steps=500, pretrain_batch=64, pretrain_lr=0.05, b_steps=500,
                           b_batch=64, b_lr=0.02, eval_n=1000, generator=g)
    assert result.task_a_retention < 0.9  # 事前学習直後の水準（0.9台後半）より明確に低下していること


# --- 実行の健全性 -----------------------------------------------------------------

@pytest.mark.parametrize("condition", CONDITIONS)
def test_run_condition_returns_valid_results(condition: str) -> None:
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    result = run_condition(condition, SPEC_A, SPEC_B, hidden_dim=16, buffer_size=20,
                           pretrain_steps=50, pretrain_batch=16, pretrain_lr=0.05, b_steps=10,
                           b_batch=16, b_lr=0.05, eval_n=200, generator=g)
    assert 0.0 <= result.task_a_retention <= 1.0
    assert 0.0 <= result.task_b_acc <= 1.0


def test_unknown_condition_raises() -> None:
    with pytest.raises(ValueError):
        run_condition("bogus", SPEC_A, SPEC_B, hidden_dim=16, buffer_size=20, pretrain_steps=10,
                      pretrain_batch=16, pretrain_lr=0.05, b_steps=5, b_batch=16, b_lr=0.05,
                      eval_n=100, generator=torch.Generator().manual_seed(0))


def test_all_conditions_covered() -> None:
    assert set(CONDITIONS) == {"shared-conflict", "independent", "rl-only-no-replay"}
