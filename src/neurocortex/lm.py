"""小規模言語モデル（12.3.1節の設計に沿う副指標用）。

`SpikingLM`     : 因果的線形アテンション + 適応閾値LIF（膜電位はトークン位置方向）
`DenseBaseline` : 同規模の密なTransformer（softmaxアテンション + GELU）

12.3.1節のハブスパースMoE・スモールワールド層間結合はステップ3以降の課題であり、
ステップ0ではスパイキング基盤の健全性確認に必要な部分だけを実装する。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalLinearAttention
from .neurons import ALIFNeuron, NeuronConfig, make_activation


class SpikingBlock(nn.Module):
    """線形アテンション経路とFFN経路の双方にスパイキングニューロンを置くブロック。"""

    def __init__(self, d_model: int, n_heads: int, cfg: NeuronConfig, activation: str) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalLinearAttention(d_model, n_heads)
        self.act_attn = make_activation(activation, cfg)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.act_ffn = make_activation(activation, cfg)
        self.fc2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.act_attn(self.attn(self.ln1(x)))
        x = x + self.fc2(self.act_ffn(self.fc1(self.ln2(x))))
        return x


class SpikingLM(nn.Module):
    """スパイキング言語モデル。"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        max_len: int = 128,
        cfg: NeuronConfig | None = None,
        activation: str = "spiking",
    ) -> None:
        super().__init__()
        cfg = cfg or NeuronConfig()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            SpikingBlock(d_model, n_heads, cfg, activation) for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def firing_rates(self) -> list[float]:
        return [
            r
            for blk in self.blocks
            for r in (blk.act_attn.last_firing_rate, blk.act_ffn.last_firing_rate)
        ]

    def set_carry_membrane(self, enabled: bool) -> None:
        for m in self.modules():
            if isinstance(m, ALIFNeuron):
                m.carry_membrane = enabled

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, t = tokens.shape
        pos = torch.arange(t, device=tokens.device)
        x = self.embed(tokens) + self.pos(pos)[None]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


class DenseBlock(nn.Module):
    """密なTransformerブロック（因果softmaxアテンション + GELU FFN）。"""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        shape = (b, t, self.n_heads, self.d_head)
        q, k, v = (z.view(shape).transpose(1, 2) for z in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        att = att.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(b, t, d)
        x = x + self.proj(y)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class DenseBaseline(nn.Module):
    """同規模の密なTransformerベースライン。"""

    def __init__(
        self, vocab_size: int, d_model: int = 128, n_layers: int = 3,
        n_heads: int = 4, max_len: int = 128,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(DenseBlock(d_model, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, t = tokens.shape
        pos = torch.arange(t, device=tokens.device)
        x = self.embed(tokens) + self.pos(pos)[None]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
