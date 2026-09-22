"""評価指標 — 小脳モジュールのツール呼び出し試行回数削減率（12.6.48節・ステップ20）。

`cerebellum.py`の`run_episode`が返す`real_calls_used`（目標到達までの実呼び出し
回数）を土台に、削減率という比率指標を定義する。

  削減率 = 1 - mean(real_calls_used[条件]) / mean(real_calls_used[基準条件])

基準条件は`random-search`（内部モデルを一切持たず、候補を実際に呼び出して
探索する下限）を使う。`from-scratch`基準の削減率（持続性固有の効果）も
補足として計算できるが、ステップ4（12.6.14〜15節）の結論により主張の
成立/不成立の判定には使わない。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .cerebellum import ForwardModel, Tool, run_episode, sample_target


def run_episodes(condition: str, tool: Tool, n_episodes: int, eps: float, max_real_calls: int,
                 n_candidates: int, step_size: float, lr: float, target_scale: float,
                 generator: torch.Generator) -> list[int]:
    """1条件ぶんの複数エピソードを実行し、各エピソードの`real_calls_used`を返す。"""
    if condition not in ("cerebellum", "from-scratch", "random-search"):
        raise ValueError(condition)

    dim = tool.weight.shape[0]
    persistent_model = ForwardModel(dim) if condition == "cerebellum" else None

    calls: list[int] = []
    for _ in range(n_episodes):
        target = sample_target(tool, generator, scale=target_scale)
        if condition == "cerebellum":
            model = persistent_model
        elif condition == "from-scratch":
            model = ForwardModel(dim)
        else:
            model = None
        result = run_episode(tool, target, eps, max_real_calls, n_candidates, step_size, lr,
                             generator, model)
        calls.append(result.real_calls_used)
    return calls


def reduction_rate(condition_calls: list[int], baseline_calls: list[int]) -> float:
    """削減率 = 1 - mean(condition) / mean(baseline)。"""
    condition_mean = sum(condition_calls) / len(condition_calls)
    baseline_mean = sum(baseline_calls) / len(baseline_calls)
    return 1.0 - condition_mean / baseline_mean


@dataclass
class ReductionResult:
    reduction_vs_random_search: float   # 主要指標: random-search基準の削減率
    reduction_vs_from_scratch: float    # 補足値: from-scratch基準の削減率（持続性固有）
    mean_calls: dict[str, float]        # 条件ごとの平均呼び出し回数


def evaluate_reduction(tool: Tool, n_episodes: int, eps: float, max_real_calls: int,
                       n_candidates: int, step_size: float, lr: float, target_scale: float,
                       generator: torch.Generator) -> ReductionResult:
    """1シードぶんの3条件を実行し、削減率を計算する。"""
    calls: dict[str, list[int]] = {}
    for condition in ("cerebellum", "from-scratch", "random-search"):
        calls[condition] = run_episodes(condition, tool, n_episodes, eps, max_real_calls,
                                        n_candidates, step_size, lr, target_scale, generator)

    return ReductionResult(
        reduction_vs_random_search=reduction_rate(calls["cerebellum"], calls["random-search"]),
        reduction_vs_from_scratch=reduction_rate(calls["cerebellum"], calls["from-scratch"]),
        mean_calls={c: sum(v) / len(v) for c, v in calls.items()},
    )
