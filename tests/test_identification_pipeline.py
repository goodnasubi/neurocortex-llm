"""識別テストのパイプライン自体が正しく働くかの軽量テスト。

本番実験（run_identification）は数分〜十数分かかるため、ここでは同じ関数を
極小の設定で呼び、統制1・統制2の仕掛けが期待通り動くことだけを確認する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.experiments.identification import (  # noqa: E402
    TrainConfig,
    evaluate,
    train_recall,
)
from neurocortex.neurons import NeuronConfig  # noqa: E402
from neurocortex.tasks import RecallSpec  # noqa: E402


def _cfg() -> tuple[RecallSpec, NeuronConfig, TrainConfig]:
    spec = RecallSpec(n_cue=4, n_filler=4, max_delay=4)
    ncfg = NeuronConfig(beta=0.95)
    tcfg = TrainConfig(steps=250, batch_size=64, lr=3e-3, d_model=32, n_layers=2,
                       eval_batches=2, eval_batch_size=128)
    return spec, ncfg, tcfg


def test_control1_step_model_can_solve_zero_delay() -> None:
    """統制1: 対照群（階段関数）でも遅延ゼロなら解けること＝容量・学習は足りている。"""
    spec, ncfg, tcfg = _cfg()
    model, _ = train_recall(spec, ncfg, tcfg, "step", seed=0, train_delay=0)
    assert evaluate(model, spec, tcfg, 0, delay=0) >= 0.95


def test_control2_zeroing_membrane_carry_destroys_accuracy() -> None:
    """統制2: 学習済みスパイキングモデルの膜電位の持ち越しだけを切るとチャンスに落ちる。"""
    spec, ncfg, tcfg = _cfg()
    model, _ = train_recall(spec, ncfg, tcfg, "spiking", seed=0)
    acc_on = evaluate(model, spec, tcfg, 0, delay=spec.max_delay)
    model.set_carry_membrane(False)
    acc_off = evaluate(model, spec, tcfg, 0, delay=spec.max_delay)
    assert acc_on >= 0.90
    assert acc_off <= spec.chance + 0.06


def test_ablation_does_not_change_weights() -> None:
    """統制2の介入が重みに一切触れないこと（純粋な推論時の操作であること）。"""
    spec, ncfg, tcfg = _cfg()
    model, _ = train_recall(spec, ncfg, tcfg, "spiking", seed=0)
    before = [p.detach().clone() for p in model.parameters()]
    model.set_carry_membrane(False)
    evaluate(model, spec, tcfg, 0, delay=spec.max_delay)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_both_groups_share_the_training_budget() -> None:
    """統制3: 同じ TrainConfig から両群が同じ形の学習ログを得ること。"""
    spec, ncfg, tcfg = _cfg()
    _, log_sp = train_recall(spec, ncfg, tcfg, "spiking", seed=1)
    _, log_st = train_recall(spec, ncfg, tcfg, "step", seed=1)
    assert log_sp["train_config"] == log_st["train_config"]
    assert not log_sp["diverged"] and not log_st["diverged"]


def test_firing_rate_is_not_saturated() -> None:
    """副指標: 発火率が 0% や 100% に張り付いていないこと。"""
    spec, ncfg, tcfg = _cfg()
    _, log = train_recall(spec, ncfg, tcfg, "spiking", seed=0)
    for rate in log["firing_rates"]:
        assert 0.005 < rate < 0.95, f"発火率が飽和している: {rate}"
