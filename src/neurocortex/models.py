"""識別テスト（12.6.1節 テストA / テストB）用のモデル。

テストAの構成は「埋め込み → スパイキングニューロン層のスタック → 出力」であり、
アテンション・畳み込み・cumsum など**位置を跨ぐ演算を一切含まない**。位置間を結ぶ
経路は ALIFNeuron の膜電位・適応閾値の持ち越しだけである（統制5で機械的に検査する）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .neurons import ALIFNeuron, NeuronConfig, StepActivation, make_activation


class RecallModel(nn.Module):
    """遅延想起課題を解くモデル（テストA構成）。

    Args:
        vocab_size: 入力語彙数。
        n_classes: 出力クラス数（cue記号数）。
        d_model: 隠れ次元。
        n_layers: [活性化 → Linear] ブロックの段数。両群で必ずそろえる。
            （12.3.1節の設計訂正により、スパイクを射影の「前」に置く配置にした。
            旧配置は [Linear → 活性化] であった。）
        cfg: ニューロン設定。
        activation: "spiking"（ALIF）または "step"（12.9節の階段関数＝対照群）。
    """

    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        d_model: int = 64,
        n_layers: int = 2,
        cfg: NeuronConfig | None = None,
        activation: str = "spiking",
    ) -> None:
        super().__init__()
        cfg = cfg or NeuronConfig()
        self.cfg = cfg
        self.activation_kind = activation
        self.embed = nn.Embedding(vocab_size, d_model)
        self.projections = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(n_layers))
        self.activations = nn.ModuleList(
            make_activation(activation, cfg) for _ in range(n_layers)
        )
        self.head = nn.Linear(d_model, n_classes)

    def set_carry_membrane(self, enabled: bool) -> None:
        """膜電位の位置方向の持ち越しを切り替える（12.6.1節 統制2の因果的除去）。

        重みには一切触れないので、学習済みモデルに対する純粋な介入になる。
        """
        for act in self.activations:
            if isinstance(act, ALIFNeuron):
                act.carry_membrane = enabled

    def firing_rates(self) -> list[float]:
        """直近フォワードでの層ごとの発火率。"""
        return [a.last_firing_rate for a in self.activations]  # type: ignore[misc]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Args: tokens [B, T]。Returns: 最終位置のロジット [B, n_classes]。"""
        h = self.embed(tokens)
        for proj, act in zip(self.projections, self.activations):
            # スパイク化してから射影する（12.3.1節 2026-09-22 の設計訂正）
            h = proj(act(h))
        return self.head(h[:, -1, :])

    def forward_all(self, tokens: torch.Tensor) -> torch.Tensor:
        """全位置のロジット [B, T, n_classes]（経路監査で使う）。"""
        h = self.embed(tokens)
        for proj, act in zip(self.projections, self.activations):
            h = proj(act(h))
        return self.head(h)


class EmbeddingFreeRecallModel(RecallModel):
    """経路監査のために埋め込みを恒等入力に差し替えた版（連続値入力を直接受ける）。"""

    def forward_from_dense(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x [B, T, d_model]。Returns: [B, T, n_classes]。"""
        h = x
        for proj, act in zip(self.projections, self.activations):
            h = proj(act(h))
        return self.head(h)


__all__ = ["RecallModel", "EmbeddingFreeRecallModel", "NeuronConfig", "ALIFNeuron", "StepActivation"]
