"""サロゲート勾配（12.3.1節: arctan型）。

順伝播は厳密なHeaviside階段関数、逆伝播のみ arctan の導関数（= 1/(1+x^2) 型）で
置き換える。straight-through estimator（逆伝播で恒等 or hardtanh を使う方式）とは
別物である点が本プロジェクトの本質的な設計判断（12.3.1節）。
"""

from __future__ import annotations

import math

import torch


class ArcTanSpike(torch.autograd.Function):
    """Heaviside順伝播 + arctanサロゲート逆伝播。

    代理導関数:
        d/dx surrogate(x) = alpha / (2 * (1 + (pi/2 * alpha * x)^2))
    これは (1/pi) * arctan(pi/2 * alpha * x) + 1/2 の厳密な導関数であり、
    alpha -> inf でHeavisideの導関数（デルタ関数）に収束する。
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        # 厳密な二値出力 {0.0, 1.0}
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        shared = (math.pi / 2.0) * alpha * x
        grad = alpha / (2.0 * (1.0 + shared * shared))
        return grad_output * grad, None


def spike(x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """x > 0 で 1、それ以外で 0 を返す（逆伝播はarctanサロゲート）。"""
    return ArcTanSpike.apply(x, alpha)
