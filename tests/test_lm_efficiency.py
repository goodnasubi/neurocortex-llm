"""ステップ17（スパイクベース設計の学習効率実測, 12.6.42節）の単体テスト。

最終的な効率差の有無ではなく、集計ロジック（目標PPL到達ステップの抽出、
モデル構築の分岐）を機械的に検証する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.experiments.run_lm_efficiency import (  # noqa: E402
    build_model,
    steps_to_reach_ppl,
)
from neurocortex.lm import DenseBaseline, SpikingLM  # noqa: E402
from neurocortex.neurons import NeuronConfig  # noqa: E402


def test_steps_to_reach_ppl_returns_first_crossing_step() -> None:
    curve = [
        {"step": 100, "val_ppl": 20.0},
        {"step": 200, "val_ppl": 12.0},
        {"step": 300, "val_ppl": 7.5},
        {"step": 400, "val_ppl": 6.0},
    ]
    assert steps_to_reach_ppl(curve, target_ppl=8.0) == 300


def test_steps_to_reach_ppl_returns_none_when_never_reached() -> None:
    curve = [{"step": 100, "val_ppl": 20.0}, {"step": 200, "val_ppl": 15.0}]
    assert steps_to_reach_ppl(curve, target_ppl=8.0) is None


def test_steps_to_reach_ppl_handles_empty_curve() -> None:
    assert steps_to_reach_ppl([], target_ppl=8.0) is None


def test_build_model_spiking_returns_spiking_lm() -> None:
    model = build_model("spiking", vocab_size=20, d_model=16, n_layers=1, n_heads=2, seq_len=8,
                        ncfg=NeuronConfig())
    assert isinstance(model, SpikingLM)


def test_build_model_dense_returns_dense_baseline() -> None:
    model = build_model("dense", vocab_size=20, d_model=16, n_layers=1, n_heads=2, seq_len=8,
                        ncfg=NeuronConfig())
    assert isinstance(model, DenseBaseline)


def test_build_model_rejects_unknown_tag() -> None:
    with pytest.raises(ValueError):
        build_model("bogus", vocab_size=20, d_model=16, n_layers=1, n_heads=2, seq_len=8,
                   ncfg=NeuronConfig())


def test_build_model_produces_comparable_architectures() -> None:
    """統制2の健全性確認: 同じd_model・n_layers・n_headsを渡せば、両モデルの入出力形状が一致すること。"""
    torch.manual_seed(0)
    spiking = build_model("spiking", vocab_size=20, d_model=16, n_layers=1, n_heads=2, seq_len=8,
                          ncfg=NeuronConfig())
    dense = build_model("dense", vocab_size=20, d_model=16, n_layers=1, n_heads=2, seq_len=8,
                        ncfg=NeuronConfig())
    tokens = torch.randint(0, 20, (2, 8))
    assert spiking(tokens).shape == dense(tokens).shape == (2, 8, 20)
