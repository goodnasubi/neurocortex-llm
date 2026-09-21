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
    """スパイクを射影の「前」に置くブロック（12.3.1節 2026-09-22 の設計訂正）。

    配置は `x → LN → ALIF → Linear → …` であり、qkv・fc1・fc2 のすべてが
    スパイクを入力に取る。これにより「発火しないニューロンは計算コストを消費しない」
    というイベント駆動性が、線形演算の大部分（qkv 25% + fc1 33% + fc2 33% ≒ 91%）に
    対して成立しうる配置になる。残差ストリーム自体は連続値のまま（加算のみ）。

    旧配置（`x + Linear(ALIF(Attention(LN(x))))`, `x + fc2(ALIF(fc1(LN(x))))`）では
    スパイクを入力に取る行列積は fc2 だけで、削減余地は約30%にとどまっていた。

    アテンション内部の出力射影 `attn.out` は線形アテンションの連続値出力を受けるため、
    ここだけは密なままである（線形演算全体の約8%）。
    """

    def __init__(self, d_model: int, n_heads: int, cfg: NeuronConfig, activation: str,
                 placement: str = "pre") -> None:
        super().__init__()
        if placement not in ("pre", "post"):
            raise ValueError(f"未知の配置: {placement}")
        # "pre"  … 新配置（スパイクを射影の前に置く。本設計）
        # "post" … 旧配置（2026-09-22の訂正以前。比較のためだけに残す）
        self.placement = placement
        self.ln1 = nn.LayerNorm(d_model)
        # アテンション経路: LN → スパイク → qkv（qkvがスパイクを入力に取る）
        self.act_attn = make_activation(activation, cfg)
        self.attn = CausalLinearAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        # FFN経路: LN → スパイク → fc1 → スパイク → fc2（両方がスパイクを入力に取る）
        self.act_fc1 = make_activation(activation, cfg)
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.act_fc2 = make_activation(activation, cfg)
        self.fc2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.placement == "post":
            # 旧配置。スパイクを入力に取る行列積は fc2 だけ（act_fc1 は未使用）。
            x = x + self.attn.out_only(self.act_attn(self.attn.attend(self.ln1(x))))
            return x + self.fc2(self.act_fc2(self.fc1(self.ln2(x))))
        x = x + self.attn(self.act_attn(self.ln1(x)))
        x = x + self.fc2(self.act_fc2(self.fc1(self.act_fc1(self.ln2(x)))))
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
        placement: str = "pre",
    ) -> None:
        super().__init__()
        cfg = cfg or NeuronConfig()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            SpikingBlock(d_model, n_heads, cfg, activation, placement)
            for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def firing_rates(self) -> list[float]:
        # ブロックあたり3つ（qkv前・fc1前・fc2前）。旧配置では2つだった。
        return [
            r
            for blk in self.blocks
            for r in (
                blk.act_attn.last_firing_rate,
                blk.act_fc1.last_firing_rate,
                blk.act_fc2.last_firing_rate,
            )
            if r is not None  # 旧配置では act_fc1 が使われないので None になる
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
