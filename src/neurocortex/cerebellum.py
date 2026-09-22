"""小脳モジュール（12.6.14節・ステップ4）: 持続する順モデルによるツール呼び出し探索。

12.3.4節の当初案（順モデル・逆モデルをWidrow-Hoff則で更新する）は、素朴には
「デルタ則という学習則そのもの」に主張を置きがちだが、デルタ則は線形モデルに
対する通常の勾配降下法と数学的に同一であり、これ自体に固有の主張は持たせられ
ない（12.6.6節のSTDP対Hebb学習、12.6.11節のGo/NoGo対非対称学習率と同型の罠）。

本モジュールの主張は学習則ではなく、**順モデルがツール呼び出し（エピソード）を
またいで持続すること**に置く（12.6.14節）。したがって `ForwardModel` は
cerebellum条件・from-scratch条件の両方が**同一のクラス・同一の `update()`**を
使う。差は「インスタンスをエピソードをまたいで使い回すか、エピソードごとに
作り直すか」だけであり、`experiments/run_cerebellum.py` の呼び出し側にしか
現れない（統制5「学習則の監査」に対応する設計）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Tool:
    """未知のブラックボックスツール `y = weight・args + bias`（エージェントからは不可視）。"""

    weight: torch.Tensor  # [dim]
    bias: float


def make_tool(dim: int, seed: int) -> Tool:
    g = torch.Generator().manual_seed(seed)
    weight = torch.randn(dim, generator=g)
    bias = float(torch.randn(1, generator=g))
    return Tool(weight=weight, bias=bias)


def call_tool(tool: Tool, args: torch.Tensor) -> torch.Tensor:
    """1回の実呼び出し。args: [..., dim] → y: [...]"""
    return args @ tool.weight + tool.bias


class ForwardModel:
    """順モデル `f̂(args) = weight・args + bias`。デルタ則（Widrow-Hoff）でのみ更新する。

    `weight`・`bias` はゼロ初期化（何も呼び出していない状態を明示的に表す）。
    """

    def __init__(self, dim: int) -> None:
        self.weight = torch.zeros(dim)
        self.bias = 0.0

    def predict(self, args: torch.Tensor) -> torch.Tensor:
        return args @ self.weight + self.bias

    def update(self, args: torch.Tensor, y_true: float, lr: float) -> float:
        """1件の (args, y_true) でデルタ則更新する。戻り値は更新前の予測誤差。"""
        error = y_true - float(self.predict(args))
        self.weight += lr * error * args
        self.bias += lr * error
        return error


@dataclass
class EpisodeResult:
    real_calls_used: int
    success: bool


def run_episode(tool: Tool, target: float, eps: float, max_real_calls: int,
                n_candidates: int, step_size: float, lr: float,
                generator: torch.Generator, model: ForwardModel | None) -> EpisodeResult:
    """1エピソード。目標 `target` に `|y - target| < eps` で到達するまで実呼び出しを繰り返す。

    `model is None` が random-search条件（12.6.14節「構成」）: 候補を `f̂` で
    screeningせず、探索範囲から直接ランダムに選んで実際に呼び出す。
    `model` が渡された場合（cerebellum・from-scratchの両条件）は、多数の候補を
    `f̂` だけで（実呼び出しなしに）評価し、最良の1つだけを実際に呼び出す。
    探索アルゴリズム自体は両条件で完全に同一（統制2）。
    """
    dim = tool.weight.shape[0]
    current = torch.zeros(dim)
    for call_idx in range(1, max_real_calls + 1):
        if model is not None:
            candidates = current + step_size * torch.randn(n_candidates, dim, generator=generator)
            preds = candidates @ model.weight + model.bias
            chosen = candidates[int((preds - target).abs().argmin())]
        else:
            chosen = step_size * 3.0 * torch.randn(dim, generator=generator)

        y = float(call_tool(tool, chosen))
        if model is not None:
            model.update(chosen, y, lr)
        current = chosen
        if abs(y - target) < eps:
            return EpisodeResult(call_idx, True)
    return EpisodeResult(max_real_calls, False)


def sample_target(tool: Tool, generator: torch.Generator, scale: float = 1.0) -> float:
    """到達可能性を保証するため、ランダムな args におけるツールの実際の出力を目標にする。"""
    args = scale * torch.randn(tool.weight.shape[0], generator=generator)
    return float(call_tool(tool, args))
