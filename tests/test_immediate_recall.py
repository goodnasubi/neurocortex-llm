"""評価指標 — 海馬モジュールの即時想起率（12.6.46節・ステップ19）の単体テスト。

指標そのものの機構（正しいスロットとゲート開放の両方を要求すること・未知キーの
誤想起率を別に測ること）を検証する。3モード比較の実測値そのものは実験スクリプト
（`run_immediate_recall.py`）の役割であり、ここでは扱わない。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.hippocampus import HippocampalMemory  # noqa: E402
from neurocortex.immediate_recall import (  # noqa: E402
    evaluate_immediate_recall,
    run_condition,
    sample_facts,
    sample_unknown_keys,
)


def test_sample_facts_shapes() -> None:
    g = torch.Generator().manual_seed(0)
    keys, values = sample_facts(10, d_model=16, value_dim=8, generator=g)
    assert keys.shape == (10, 16)
    assert values.shape == (10, 8)


def test_sample_unknown_keys_shape() -> None:
    g = torch.Generator().manual_seed(0)
    keys = sample_unknown_keys(5, d_model=16, generator=g)
    assert keys.shape == (5, 16)


# --- 統制: 空ストアに対する未知キーは全て正しく棄却されること -----------------------

def test_empty_store_never_opens_gate_for_unknown_keys() -> None:
    memory = HippocampalMemory(d_model=16, value_dim=8, n_units=256, k=8, mode="sdr", seed=0)
    g = torch.Generator().manual_seed(0)
    unknown = sample_unknown_keys(20, d_model=16, generator=g)
    # 既知キーが0件でも評価関数自体は例外を出さない設計にする（n_facts=0のchance_recall_rateは特別扱い）。
    known_keys = torch.zeros(0, 16)
    _, stats = memory.store.read(memory.separator(unknown))
    assert (stats.top1_index == -1).all()


# --- 書き込んだ事実は高い確率で正しく想起できること（sdrモード、健全性確認） --------

def test_sdr_mode_recalls_written_facts_reliably() -> None:
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    result = run_condition("sdr", d_model=64, value_dim=64, n_units=2048, k=40, n_facts=50,
                           n_unknown=50, gate_slope=20.0, gate_bias=0.3, generator=g, seed=0)
    assert result.recall_rate > 0.9
    assert result.recall_rate > result.chance_recall_rate * 10  # チャンス水準を明確に上回る
    assert result.false_positive_rate < 0.2


def test_chance_recall_rate_is_inverse_of_n_facts() -> None:
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    result = run_condition("sdr", d_model=32, value_dim=32, n_units=512, k=16, n_facts=20,
                           n_unknown=20, gate_slope=20.0, gate_bias=0.3, generator=g, seed=0)
    assert result.chance_recall_rate == pytest.approx(1.0 / 20)


# --- 成功の判定は「正しいスロット」と「ゲート開放」の両方を要求すること -------------

def test_success_requires_both_correct_slot_and_open_gate(monkeypatch) -> None:
    """ゲートを常に閉じるよう介入すると、正しいスロットを指していても想起率が0になること。"""
    memory = HippocampalMemory(d_model=32, value_dim=32, n_units=512, k=16, mode="sdr",
                               gate_slope=20.0, gate_bias=0.3, seed=0)
    g = torch.Generator().manual_seed(0)
    known_keys, known_values = sample_facts(10, d_model=32, value_dim=32, generator=g)
    memory.write(known_keys, known_values)
    unknown_keys = sample_unknown_keys(10, d_model=32, generator=g)

    baseline = evaluate_immediate_recall(memory, known_keys, unknown_keys)
    assert baseline.recall_rate > 0.0  # 通常は想起できる

    # gate_biasを極端に引き上げ、どんな類似度でもゲートが開かないようにする。
    memory.gate_bias = 100.0
    forced_closed = evaluate_immediate_recall(memory, known_keys, unknown_keys)
    assert forced_closed.recall_rate == 0.0
    assert forced_closed.false_positive_rate == 0.0


def test_unknown_condition_mode_raises() -> None:
    with pytest.raises(ValueError):
        run_condition("bogus", d_model=16, value_dim=16, n_units=64, k=8, n_facts=5, n_unknown=5,
                      gate_slope=20.0, gate_bias=0.3, generator=torch.Generator().manual_seed(0),
                      seed=0)
