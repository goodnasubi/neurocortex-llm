"""テストB（12.6.1節・確認用）用の単層モデル。

因果的線形アテンションの状態 S_t = Σ k_i v_i^T は前文脈の単なる和なので順序に不変。
一方、漏れのある膜電位は指数的な直近重み付けを与えるので順序を判別できる。
**この順序不変性は単層の場合にのみ成立する**ため、本モデルは必ず単層で使う。

位置を跨ぐ経路を持つため、統制5の経路監査（tests/test_path_audit.py）の対象である
`models.py` とは意図的にファイルを分けている。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import CausalLinearAttention
from .neurons import ALIFNeuron, NeuronConfig, make_activation


class OrderModel(nn.Module):
    """単層の因果的線形アテンション + 活性化。位置エンコーディングは持たない。"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        cfg: NeuronConfig | None = None,
        activation: str = "spiking",
    ) -> None:
        super().__init__()
        cfg = cfg or NeuronConfig()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.attn = CausalLinearAttention(d_model, n_heads)
        self.act = make_activation(activation, cfg)
        self.head = nn.Linear(d_model, 2)

    def set_carry_membrane(self, enabled: bool) -> None:
        for m in self.modules():
            if isinstance(m, ALIFNeuron):
                m.carry_membrane = enabled

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # スパイク化してから射影する（12.3.1節 2026-09-22 の設計訂正）。
        # アテンション内部の qkv がスパイクを入力に取る配置になる。
        h = self.attn(self.act(self.embed(tokens)))
        return self.head(h[:, -1, :])
