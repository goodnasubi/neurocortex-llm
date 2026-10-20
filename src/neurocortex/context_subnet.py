"""文脈でゲートした文脈固有の部分ネットワーク（12.6.219節）。

共有の隠れ層（`Backbone`、tanh）に加えて、文脈の枠ごとに K 個の隠れユニットを持ち、入力の末尾の文脈の one-hot が
指す枠のユニットだけを働かせる（他の枠の出力は 0）。働かないユニットの入力重みと読み出しの重みには勾配が流れない
ので、新しい文脈の写像を、古い文脈の部分ネットワークを動かさずに学べる（文脈依存のゲーティング: Masse, Grant &
Freedman 2018, PNAS 115:E10467。相補的学習系の、文脈ごとに重なりの少ない表現: McClelland, McNaughton & O'Reilly
1995）。共有の重みは従来どおり全ての文脈の勾配を受ける。

`Model`（hippocampus_replay）と同じ呼び出し方（`model(x)`・`model.parameters()`・`model.backbone.fc1`・
`model.head`）を持つので、`run_context_sequence` の他の処理はそのまま使える。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .basal_ganglia_core import Backbone


class GatedModel:
    """共有の隠れ層 H 個 + 文脈ごとに K 個の部分ネットワーク（C 枠）+ 線形の読み出し。

    入力 x は [ビット n_bits 個, 文脈の one-hot C 個]。部分ネットワークの入力は x 全体（共有の層と同じ）。
    """

    def __init__(self, n_bits: int, n_ctx: int, hidden_dim: int, ctx_units: int) -> None:
        if n_ctx <= 0 or ctx_units <= 0:
            raise ValueError("n_ctx・ctx_units は正")
        self.n_bits, self.n_ctx, self.hidden_dim, self.ctx_units = n_bits, n_ctx, hidden_dim, ctx_units
        self.backbone = Backbone(n_bits + n_ctx, hidden_dim)
        self.sub = nn.Linear(n_bits + n_ctx, n_ctx * ctx_units)
        self.head = nn.Linear(hidden_dim + n_ctx * ctx_units, 2)

    def gate(self, x: torch.Tensor) -> torch.Tensor:
        """文脈の one-hot を各枠の K 個のユニットに広げたマスク。"""
        return x[:, self.n_bits:].repeat_interleave(self.ctx_units, dim=1)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.backbone(x), torch.tanh(self.sub(x)) * self.gate(x)], dim=1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(x))

    def parameters(self) -> list[nn.Parameter]:
        return list(self.backbone.parameters()) + list(self.sub.parameters()) + list(self.head.parameters())

    def ctx_slices(self, ctx: int) -> list[tuple[nn.Parameter, tuple]]:
        """文脈 ctx の部分ネットワークのパラメータ（入力重みの行・バイアス・読み出しの列）の (パラメータ, 添字)。"""
        K = self.ctx_units
        rows = slice(ctx * K, (ctx + 1) * K)
        cols = slice(self.hidden_dim + ctx * K, self.hidden_dim + (ctx + 1) * K)
        return [(self.sub.weight, (rows, slice(None))), (self.sub.bias, (rows,)),
                (self.head.weight, (slice(None), cols))]


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
