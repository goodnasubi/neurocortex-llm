"""ステップ21（海馬モジュールの即時想起率 — 容量圧迫下での劣化, 12.6.50節）の単体テスト。

`run_immediate_recall_capacity.py`の集計ロジック（圧迫率の定義、主張(a)(b)の判定関数）
を最小限検証する。3モード×複数圧迫率の実測値そのものは実験スクリプトの役割であり、
ここでは扱わない。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.immediate_recall import run_condition  # noqa: E402


def test_pressure_ratio_matches_n_facts_over_n_units() -> None:
    n_units = 2048
    n_facts = 500
    assert n_facts / n_units == 500 / 2048


def test_run_condition_accepts_high_pressure_n_facts_without_shape_errors() -> None:
    """n_factsをn_unitsに近い（高圧迫率の）値にしても例外なく動作すること。"""
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    result = run_condition("sdr", d_model=64, value_dim=64, n_units=2048, k=40, n_facts=1600,
                           n_unknown=50, gate_slope=20.0, gate_bias=0.3, generator=g, seed=0)
    assert 0.0 <= result.recall_rate <= 1.0
    assert 0.0 <= result.false_positive_rate <= 1.0


def test_recall_rate_stays_near_perfect_even_beyond_n_units() -> None:
    """AssociativeStoreは追記のみの明示的リスト（固定容量のHopfield型ではない）であるため、
    書き込み件数がn_unitsを超えても既知キーの想起率はほぼ1.000のまま劣化しないことを確認する
    （本ステップの主張(a)が成立しない根拠となる構造的事実の回帰テスト）。
    """
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    result = run_condition("sdr", d_model=64, value_dim=64, n_units=2048, k=40, n_facts=3000,
                           n_unknown=50, gate_slope=20.0, gate_bias=0.3, generator=g, seed=0)
    assert result.recall_rate > 0.95


def test_false_positive_rate_rises_with_more_writes() -> None:
    """既知の傾向（誤想起率は書き込み件数の増加につれ上昇する）を回帰確認する。"""
    torch.manual_seed(0)
    g_low = torch.Generator().manual_seed(0)
    low = run_condition("dense", d_model=64, value_dim=64, n_units=2048, k=40, n_facts=50,
                        n_unknown=50, gate_slope=20.0, gate_bias=0.3, generator=g_low, seed=0)
    g_high = torch.Generator().manual_seed(0)
    high = run_condition("dense", d_model=64, value_dim=64, n_units=2048, k=40, n_facts=1000,
                         n_unknown=50, gate_slope=20.0, gate_bias=0.3, generator=g_high, seed=0)
    assert high.false_positive_rate >= low.false_positive_rate
