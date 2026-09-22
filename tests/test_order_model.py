"""`OrderModel`（`order_model.py`, テストB）の常設識別テスト（12.6.40節・ステップ16）。

`run_order.py`が既に手動でアブレーション評価を実装しているが、常設のpytestとしては
存在しなかった。ここでは共有ユーティリティ経由で標準化する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neurocortex.neurons import NeuronConfig  # noqa: E402
from neurocortex.order_model import OrderModel  # noqa: E402
from neurocortex.tasks import OrderSpec, make_order_batch  # noqa: E402

from spiking_audit import assert_membrane_ablation_degrades, assert_not_permutation_invariant  # noqa: E402

SPEC = OrderSpec(seq_len=16)


def _make_model(seed: int = 0) -> OrderModel:
    ncfg = NeuronConfig(beta=0.95)
    return OrderModel(SPEC.vocab_size, d_model=32, n_heads=2, cfg=ncfg)


def test_not_permutation_invariant_on_untrained_model() -> None:
    torch.manual_seed(0)
    model = _make_model()
    model.eval()
    g = torch.Generator().manual_seed(0)
    tokens, _ = make_order_batch(SPEC, 8, g)
    assert_not_permutation_invariant(lambda t: model(t), tokens, output_has_seq_dim=False)


def test_membrane_ablation_degrades_order_accuracy() -> None:
    """V2: 順序判別課題は、アテンション状態だけでは原理的に解けない（順序不変）ため、
    膜電位ゼロ化で正解率がチャンス近辺まで明確に落ちるはず。
    """
    torch.manual_seed(0)
    model = _make_model()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    for _ in range(400):
        model.train()
        x, y = make_order_batch(SPEC, 64, g)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        eval_g = torch.Generator().manual_seed(999)
        correct = total = 0
        for _ in range(8):
            x, y = make_order_batch(SPEC, 256, eval_g)
            correct += int((model(x).argmax(-1) == y).sum())
            total += y.numel()
        return correct / total

    baseline, ablated = assert_membrane_ablation_degrades(evaluate, model.set_carry_membrane,
                                                           min_drop=0.15)
    assert baseline > 0.8  # 事前学習が十分であることの健全性確認
    assert ablated < baseline
