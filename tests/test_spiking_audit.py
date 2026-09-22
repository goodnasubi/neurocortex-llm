"""共有監査ユーティリティ（`spiking_audit.py`、12.6.40節・ステップ16）自体の単体テスト。

既知の「合格するはずの」ケース（`models.py`のスパイキング版RecallModel、
12.6.1節Test Aで合格確認済み）と「不合格になるはずの」ケース（12.9節の
階段関数方式そのもの、`activation="step"`）の両方で、ユーティリティが
正しく判定できることを先に確認する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.models import RecallModel  # noqa: E402
from neurocortex.neurons import NeuronConfig  # noqa: E402
from neurocortex.tasks import RecallSpec, make_recall_batch  # noqa: E402

from spiking_audit import assert_membrane_ablation_degrades, assert_not_permutation_invariant  # noqa: E402

SPEC = RecallSpec(n_cue=8, n_filler=8, max_delay=16)


def _tokens(n: int, delay: int, generator: torch.Generator) -> torch.Tensor:
    tokens, _ = make_recall_batch(SPEC, n, generator, delay=delay)
    return tokens


def test_assert_not_permutation_invariant_passes_for_spiking_model() -> None:
    torch.manual_seed(0)
    model = RecallModel(vocab_size=SPEC.pad_id + 1, n_classes=SPEC.n_cue, activation="spiking")
    model.eval()
    g = torch.Generator().manual_seed(0)
    tokens = _tokens(8, delay=15, generator=g)
    assert_not_permutation_invariant(lambda t: model.forward_all(t), tokens)


def test_assert_not_permutation_invariant_fails_for_step_model() -> None:
    """既知の不合格ケース: 12.9節の階段関数方式は順列不変なので、監査は必ず検出できること。"""
    torch.manual_seed(0)
    model = RecallModel(vocab_size=SPEC.pad_id + 1, n_classes=SPEC.n_cue, activation="step")
    model.eval()
    g = torch.Generator().manual_seed(0)
    tokens = _tokens(8, delay=15, generator=g)
    with pytest.raises(AssertionError):
        assert_not_permutation_invariant(lambda t: model.forward_all(t), tokens)


def test_assert_membrane_ablation_degrades_detects_real_drop() -> None:
    torch.manual_seed(0)
    model = RecallModel(vocab_size=SPEC.pad_id + 1, n_classes=SPEC.n_cue, activation="spiking")
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    for _ in range(300):
        tokens, labels = make_recall_batch(SPEC, 32, g, delay=15)
        loss = torch.nn.functional.cross_entropy(model(tokens), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        tokens, labels = make_recall_batch(SPEC, 500, torch.Generator().manual_seed(999), delay=15)
        pred = model(tokens).argmax(-1)
        return float((pred == labels).float().mean())

    baseline, ablated = assert_membrane_ablation_degrades(evaluate, model.set_carry_membrane,
                                                           min_drop=0.2)
    assert baseline > ablated


def test_assert_membrane_ablation_raises_when_no_real_drop() -> None:
    """既知の不合格ケース: アブレーションしても性能が落ちない（＝時間ダイナミクスを
    使っていない）架空のモデルに対しては、監査が必ず検出できること。
    """
    def evaluate_constant() -> float:
        return 0.9  # 常に同じ値 = アブレーションの影響を一切受けないふり

    def noop_set_carry_membrane(enabled: bool) -> None:
        pass

    with pytest.raises(AssertionError):
        assert_membrane_ablation_degrades(evaluate_constant, noop_set_carry_membrane, min_drop=0.2)
