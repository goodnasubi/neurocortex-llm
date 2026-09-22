"""マルチ学習則の学習安定性 — 3モジュール構成（12.6.44節・ステップ18）の単体テスト。

最終的な干渉の有無ではなく、統制1（independent-allの3バックボーンは互いに影響
しないこと）・統制2（単独/ペア共有条件は該当ヘッドのみ学習し、他はNoneの
ままであること）・コサイン類似度がペア数ぶん記録されることという機構そのものを
機械的に検証する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.basal_ganglia_core import ActorCritic, Backbone, ParitySpec  # noqa: E402
from neurocortex.multi_learning_rule_3way import (  # noqa: E402
    CONDITIONS,
    evaluate_cerebellum,
    evaluate_hippocampus,
    evaluate_rl,
    run_condition,
)

SPEC = ParitySpec(n_inputs=8, relevant=(0, 1))


# --- 基本コンポーネント -----------------------------------------------------------

def test_evaluate_hippocampus_returns_valid_accuracy() -> None:
    torch.manual_seed(0)
    backbone = Backbone(8, 16)
    head = torch.nn.Linear(16, 2)
    g = torch.Generator().manual_seed(0)
    acc = evaluate_hippocampus(backbone, head, SPEC, n_eval=200, generator=g)
    assert 0.0 <= acc <= 1.0


def test_evaluate_rl_returns_valid_accuracy() -> None:
    torch.manual_seed(0)
    backbone = Backbone(8, 16)
    head = ActorCritic(16)
    g = torch.Generator().manual_seed(0)
    acc = evaluate_rl(backbone, head, SPEC, n_eval=200, generator=g)
    assert 0.0 <= acc <= 1.0


def test_evaluate_cerebellum_returns_nonnegative_mse() -> None:
    torch.manual_seed(0)
    backbone = Backbone(8, 16)
    head = torch.nn.Linear(16, 1)
    g = torch.Generator().manual_seed(0)
    mse = evaluate_cerebellum(backbone, head, SPEC, n_eval=200, generator=g)
    assert mse >= 0.0


# --- 統制2: 単独共有条件は該当ヘッドのみ学習し、コサイン類似度は空であること --------

@pytest.mark.parametrize("condition,active,inactive", [
    ("shared-hippo-only", "hippo_acc", ("rl_acc", "cereb_mse")),
    ("shared-rl-only", "rl_acc", ("hippo_acc", "cereb_mse")),
    ("shared-cereb-only", "cereb_mse", ("hippo_acc", "rl_acc")),
])
def test_solo_conditions_learn_only_the_active_head(condition, active, inactive) -> None:
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    result = run_condition(condition, SPEC, hidden_dim=16, steps=10, batch_size=16, lr=0.05,
                           eval_n=200, generator=g)
    assert getattr(result, active) is not None
    for field in inactive:
        assert getattr(result, field) is None
    assert result.cosine_similarities == {}


# --- ペア共有条件: 2ヘッドが学習され、コサイン類似度が1ペアぶん記録されること -------

@pytest.mark.parametrize("condition,pair_key", [
    ("shared-hippo-rl", "hippoxrl"),
    ("shared-hippo-cereb", "hippoxcereb"),
    ("shared-rl-cereb", "rlxcereb"),
])
def test_pair_conditions_record_one_cosine_similarity_series(condition, pair_key) -> None:
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    steps = 12
    result = run_condition(condition, SPEC, hidden_dim=16, steps=steps, batch_size=16, lr=0.05,
                           eval_n=200, generator=g)
    assert len(result.cosine_similarities) == 1
    series = next(iter(result.cosine_similarities.values()))
    assert len(series) == steps
    assert all(-1.0 <= c <= 1.0 for c in series)


# --- shared-all: 3ヘッド全てが学習され、コサイン類似度が3ペアぶん記録されること -----

def test_shared_all_returns_three_metrics_and_three_cosine_series() -> None:
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    steps = 12
    result = run_condition("shared-all", SPEC, hidden_dim=16, steps=steps, batch_size=16, lr=0.05,
                           eval_n=200, generator=g)
    assert result.hippo_acc is not None
    assert result.rl_acc is not None
    assert result.cereb_mse is not None
    assert len(result.cosine_similarities) == 3
    for series in result.cosine_similarities.values():
        assert len(series) == steps
        assert all(-1.0 <= c <= 1.0 for c in series)


def test_independent_all_returns_three_metrics_and_no_cosine_similarities() -> None:
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    result = run_condition("independent-all", SPEC, hidden_dim=16, steps=10, batch_size=16,
                           lr=0.05, eval_n=200, generator=g)
    assert result.hippo_acc is not None
    assert result.rl_acc is not None
    assert result.cereb_mse is not None
    assert result.cosine_similarities == {}


# --- 統制1: independent-allの3バックボーンは互いに独立であること -------------------

def test_independent_all_backbones_do_not_share_parameters(monkeypatch) -> None:
    """統制1の核心: 基底核側の学習を無効化しても、海馬側・小脳側の最終指標が
    変わらないことを確認する（独立バックボーンが互いに影響しないことの監査）。
    """
    import neurocortex.multi_learning_rule_3way as mlr3

    torch.manual_seed(0)
    g1 = torch.Generator().manual_seed(0)
    baseline = run_condition("independent-all", SPEC, hidden_dim=16, steps=10, batch_size=16,
                             lr=0.05, eval_n=500, generator=g1)

    original_rl_step_loss = mlr3._rl_step_loss

    def zeroed_rl_step_loss(*args, **kwargs):
        loss = original_rl_step_loss(*args, **kwargs)
        return loss * 0.0

    monkeypatch.setattr(mlr3, "_rl_step_loss", zeroed_rl_step_loss)
    torch.manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    with_zeroed_rl = run_condition("independent-all", SPEC, hidden_dim=16, steps=10,
                                   batch_size=16, lr=0.05, eval_n=500, generator=g2)
    assert with_zeroed_rl.hippo_acc == pytest.approx(baseline.hippo_acc, abs=1e-6)
    assert with_zeroed_rl.cereb_mse == pytest.approx(baseline.cereb_mse, abs=1e-6)


def test_unknown_condition_raises() -> None:
    with pytest.raises(ValueError):
        run_condition("bogus", SPEC, hidden_dim=16, steps=5, batch_size=16, lr=0.05, eval_n=100,
                      generator=torch.Generator().manual_seed(0))


def test_all_conditions_covered() -> None:
    assert set(CONDITIONS) == {
        "independent-all", "shared-all", "shared-hippo-only", "shared-rl-only",
        "shared-cereb-only", "shared-hippo-rl", "shared-hippo-cereb", "shared-rl-cereb",
    }
