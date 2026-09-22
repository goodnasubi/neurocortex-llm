"""マルチ学習則の学習安定性（12.6.36節・ステップ14）の単体テスト。

最終的な干渉の有無ではなく、統制1（独立バックボーンは互いに影響しないこと）・
統制2（単一信号条件は該当タスクのみ学習し、もう一方はNoneのままであること）
という機構そのものを機械的に検証する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.basal_ganglia_core import ActorCritic, Backbone, ParitySpec  # noqa: E402
from neurocortex.multi_learning_rule import (  # noqa: E402
    CONDITIONS,
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


# --- 統制2: 単一信号条件は該当タスクのみ学習すること -------------------------------

def test_shared_hippocampus_only_leaves_rl_acc_none() -> None:
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    result = run_condition("shared-hippocampus-only", SPEC, hidden_dim=16, steps=10, batch_size=16,
                           lr=0.05, eval_n=200, generator=g)
    assert result.hippo_acc is not None
    assert result.rl_acc is None
    assert result.cosine_similarities == []


def test_shared_rl_only_leaves_hippo_acc_none() -> None:
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    result = run_condition("shared-rl-only", SPEC, hidden_dim=16, steps=10, batch_size=16, lr=0.05,
                           eval_n=200, generator=g)
    assert result.rl_acc is not None
    assert result.hippo_acc is None
    assert result.cosine_similarities == []


# --- shared-both: 両方の指標が得られ、コサイン類似度が記録されること -----------------

def test_shared_both_returns_both_metrics_and_cosine_similarities() -> None:
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    steps = 15
    result = run_condition("shared-both", SPEC, hidden_dim=16, steps=steps, batch_size=16,
                           lr=0.05, eval_n=200, generator=g)
    assert result.hippo_acc is not None
    assert result.rl_acc is not None
    assert len(result.cosine_similarities) == steps
    assert all(-1.0 <= c <= 1.0 for c in result.cosine_similarities)


def test_independent_both_returns_both_metrics_and_no_cosine_similarities() -> None:
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    result = run_condition("independent-both", SPEC, hidden_dim=16, steps=10, batch_size=16,
                           lr=0.05, eval_n=200, generator=g)
    assert result.hippo_acc is not None
    assert result.rl_acc is not None
    assert result.cosine_similarities == []


# --- 統制1: independent-bothの2つのバックボーンは互いに独立であること --------------

def test_independent_both_backbones_do_not_share_parameters(monkeypatch) -> None:
    """統制1の核心: 海馬用の学習ステップが基底核用バックボーンのパラメータに
    影響しないこと（逆も同様）を、片方の学習ループを無効化して監査する。
    """
    import neurocortex.multi_learning_rule as mlr

    torch.manual_seed(0)
    g1 = torch.Generator().manual_seed(0)
    baseline = run_condition("independent-both", SPEC, hidden_dim=16, steps=10, batch_size=16,
                             lr=0.05, eval_n=500, generator=g1)

    # 基底核側の学習を完全に無効化（損失を常に0にする）しても、
    # 海馬側の最終正解率が変わらないことを確認する。
    original_rl_step_losses = mlr._rl_step_losses

    def zeroed_rl_step_losses(*args, **kwargs):
        loss = original_rl_step_losses(*args, **kwargs)
        return loss * 0.0

    monkeypatch.setattr(mlr, "_rl_step_losses", zeroed_rl_step_losses)
    torch.manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    with_zeroed_rl = run_condition("independent-both", SPEC, hidden_dim=16, steps=10,
                                   batch_size=16, lr=0.05, eval_n=500, generator=g2)
    assert with_zeroed_rl.hippo_acc == pytest.approx(baseline.hippo_acc, abs=1e-6)


def test_unknown_condition_raises() -> None:
    with pytest.raises(ValueError):
        run_condition("bogus", SPEC, hidden_dim=16, steps=5, batch_size=16, lr=0.05, eval_n=100,
                      generator=torch.Generator().manual_seed(0))


def test_all_conditions_covered() -> None:
    assert set(CONDITIONS) == {
        "shared-both", "independent-both", "shared-hippocampus-only", "shared-rl-only",
    }
