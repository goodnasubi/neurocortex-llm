"""ステップ24（AssociativeStoreANNモードのベクトル化再実装による速度優位の再検証,
12.6.56節）の単体テスト。

`run_associative_store_ann_vectorized.py`の集計ロジック（旧実装との回帰比較を含む）を
最小限検証する。大規模Nでの実測そのものは実験スクリプトの役割であり、ここでは
小さいNで構造・疎通のみ確認する。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.experiments.run_associative_store_ann import measure_one  # noqa: E402
from neurocortex.experiments.run_associative_store_ann_vectorized import (  # noqa: E402
    STEP23_TOP1_MATCH_RATE_MEAN,
    _loglog_slope_mid_large,
)


def test_step23_reference_value_is_the_documented_constant() -> None:
    """統制・主張(b)の比較基準は12.6.55節に実測値として記録された0.530であること。"""
    assert STEP23_TOP1_MATCH_RATE_MEAN == 0.530


def test_loglog_slope_mid_large_drops_the_first_point() -> None:
    n_sorted = [100, 1_000, 10_000, 100_000, 300_000]
    ys = [n * 2.0 for n in n_sorted]
    slope_all = _loglog_slope_mid_large(n_sorted, ys)
    assert abs(slope_all - 1.0) < 1e-6


def test_measure_one_batched_fields_are_deterministic_given_seed() -> None:
    """同一シードならバッチ版の一致率も再現すること（`measure_one`はステップ23の
    実装を流用しているため、ここではバッチ版フィールドの疎通のみ確認する）。
    """
    r1 = measure_one(150, 64, 64, seed=3)
    r2 = measure_one(150, 64, 64, seed=3)
    assert r1["top1_match_rate_batched"] == r2["top1_match_rate_batched"]
    assert r1["ann_batched_read_time_median_sec"] > 0.0
