"""評価指標 — 小脳モジュールのツール呼び出し試行回数削減率（12.6.48節・ステップ20）の単体テスト。

指標そのものの機構（削減率の算出式・条件の妥当性検査）を検証する。3条件の
実測値そのものは実験スクリプト（`run_tool_call_reduction.py`）の役割であり、
ここでは扱わない。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.cerebellum import make_tool  # noqa: E402
from neurocortex.tool_call_reduction import (  # noqa: E402
    evaluate_reduction,
    reduction_rate,
    run_episodes,
)


def test_reduction_rate_basic_arithmetic() -> None:
    assert reduction_rate([5, 5], [10, 10]) == pytest.approx(0.5)
    assert reduction_rate([10, 10], [10, 10]) == pytest.approx(0.0)


def test_reduction_rate_negative_when_condition_worse_than_baseline() -> None:
    assert reduction_rate([20, 20], [10, 10]) == pytest.approx(-1.0)


def test_run_episodes_unknown_condition_raises() -> None:
    tool = make_tool(dim=3, seed=0)
    g = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError):
        run_episodes("bogus", tool, n_episodes=3, eps=0.1, max_real_calls=10, n_candidates=16,
                     step_size=1.0, lr=0.3, target_scale=1.0, generator=g)


def test_run_episodes_returns_one_call_count_per_episode() -> None:
    tool = make_tool(dim=3, seed=0)
    g = torch.Generator().manual_seed(0)
    calls = run_episodes("random-search", tool, n_episodes=5, eps=0.1, max_real_calls=10,
                         n_candidates=16, step_size=1.0, lr=0.3, target_scale=1.0, generator=g)
    assert len(calls) == 5
    assert all(1 <= c <= 10 for c in calls)


# --- random-search（内部モデルなし）はcerebellumより一貫して呼び出し回数が多いこと ---
# （統制1の下限として機能していることの健全性確認、ステップ4の既知の結果と整合）

def test_cerebellum_uses_fewer_calls_than_random_search_baseline() -> None:
    torch.manual_seed(0)
    tool = make_tool(dim=3, seed=0)
    g = torch.Generator().manual_seed(0)
    result = evaluate_reduction(tool, n_episodes=30, eps=0.1, max_real_calls=30, n_candidates=64,
                                step_size=1.0, lr=0.3, target_scale=1.0, generator=g)
    assert result.reduction_vs_random_search > 0.0
    assert result.mean_calls["cerebellum"] < result.mean_calls["random-search"]


def test_evaluate_reduction_reports_all_three_mean_calls() -> None:
    torch.manual_seed(0)
    tool = make_tool(dim=3, seed=0)
    g = torch.Generator().manual_seed(0)
    result = evaluate_reduction(tool, n_episodes=10, eps=0.1, max_real_calls=20, n_candidates=32,
                                step_size=1.0, lr=0.3, target_scale=1.0, generator=g)
    assert set(result.mean_calls.keys()) == {"cerebellum", "from-scratch", "random-search"}
