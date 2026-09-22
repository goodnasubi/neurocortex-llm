"""ステップ23（AssociativeStoreへのANN候補絞り込みの実装と効果測定, 12.6.54節）の単体テスト。

`run_associative_store_ann.py`の集計ロジック（exact・ann比較、top-1一致率算出）を
最小限検証する。大規模Nでの実測そのものは実験スクリプトの役割であり、ここでは
小さいNで構造・疎通のみ確認する。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.experiments.run_associative_store_ann import (  # noqa: E402
    loglog_slope,
    measure_one,
)


def test_loglog_slope_of_perfectly_linear_data_is_one() -> None:
    xs = [100, 1_000, 10_000, 100_000]
    ys = [x * 3.0 for x in xs]  # y = 3x は対数-対数で傾き1
    slope = loglog_slope(xs, ys)
    assert abs(slope - 1.0) < 1e-6


def test_measure_one_returns_both_modes_and_valid_match_rate() -> None:
    n = 200
    d_model = 64
    value_dim = 64
    result = measure_one(n, d_model, value_dim, seed=0)
    assert result["n"] == n
    assert result["exact_read_time_median_sec"] > 0.0
    assert result["ann_read_time_median_sec"] > 0.0
    assert 0.0 <= result["top1_match_rate"] <= 1.0
    # float32は4バイト。 keys: n*d_model, values: n*value_dim
    expected_bytes = 4 * n * d_model + 4 * n * value_dim
    assert result["memory_bytes"] == expected_bytes


def test_measure_one_is_deterministic_given_seed() -> None:
    """同一シードなら書き込み内容・クエリが同一になり、結果が再現すること。"""
    r1 = measure_one(150, 64, 64, seed=3)
    r2 = measure_one(150, 64, 64, seed=3)
    assert r1["top1_match_rate"] == r2["top1_match_rate"]
    assert r1["memory_bytes"] == r2["memory_bytes"]


def test_measure_one_includes_batched_ann_fields() -> None:
    """ステップ24（12.6.56節）: `measure_one`はバッチ版ann（`ann_batched`）の
    計測結果・旧実装との回帰比較用一致率も返す。
    """
    result = measure_one(200, 64, 64, seed=0)
    assert result["ann_batched_read_time_median_sec"] > 0.0
    assert 0.0 <= result["top1_match_rate_batched"] <= 1.0
    assert 0.0 <= result["top1_match_rate_old_vs_batched"] <= 1.0


def test_full_candidate_coverage_gives_perfect_match_rate() -> None:
    """`read_approx`が全件を候補にできる小規模ストアでは、厳密探索と一致するはず
    （候補絞り込みロジック自体のバグがないことの健全性確認、hippocampus.py側の
    `test_read_approx_matches_exact_when_min_candidates_covers_all`と対をなす）。
    """
    from neurocortex.hippocampus import AssociativeStore
    import torch

    torch.manual_seed(0)
    keys = torch.nn.functional.normalize(torch.randn(30, 8), dim=-1)
    values = torch.nn.functional.normalize(torch.randn(30, 4), dim=-1)

    store_exact = AssociativeStore(8, 4, exact=True)
    store_exact.write(keys, values)
    _, stats_exact = store_exact.read(keys)

    store_ann = AssociativeStore(8, 4)
    store_ann.write(keys, values)
    _, stats_ann = store_ann.read_approx(keys, min_candidates=30)

    match = (stats_exact.top1_index == stats_ann.top1_index).float().mean().item()
    assert match == 1.0
