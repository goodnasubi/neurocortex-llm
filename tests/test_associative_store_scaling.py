"""ステップ22（海馬モジュールAssociativeStoreの検索コスト, 12.6.52節）の単体テスト。

`run_associative_store_scaling.py`の集計ロジック（対数-対数回帰の傾き算出、
メモリ占有量の理論値との整合性）を最小限検証する。大規模Nでの実測そのものは
実験スクリプトの役割であり、ここでは小さいNで構造を確認する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.experiments.run_associative_store_scaling import (  # noqa: E402
    loglog_slope,
    measure_one,
)
from neurocortex.hippocampus import AssociativeStore  # noqa: E402


def test_loglog_slope_of_perfectly_linear_data_is_one() -> None:
    xs = [100, 1_000, 10_000, 100_000]
    ys = [x * 3.0 for x in xs]  # y = 3x は対数-対数で傾き1
    slope = loglog_slope(xs, ys)
    assert abs(slope - 1.0) < 1e-6


def test_loglog_slope_of_constant_data_is_zero() -> None:
    xs = [100, 1_000, 10_000]
    ys = [5.0, 5.0, 5.0]
    slope = loglog_slope(xs, ys)
    assert abs(slope - 0.0) < 1e-6


def test_measure_one_returns_positive_read_time_and_correct_memory_bytes() -> None:
    device = torch.device("cpu")
    n = 200
    d_model = 64
    value_dim = 64
    result = measure_one(n, d_model, value_dim, seed=0, device=device)
    assert result["n"] == n
    assert result["read_time_median_sec"] > 0.0
    assert len(result["read_time_all_sec"]) == 5
    # float32は4バイト。 keys: n*d_model, values: n*value_dim
    expected_bytes = 4 * n * d_model + 4 * n * value_dim
    assert result["memory_bytes"] == expected_bytes


def test_memory_bytes_matches_associative_store_tensor_sizes_directly() -> None:
    """AssociativeStore自体のテンソルサイズからも同じ計算式が導けることを確認する
    （実験スクリプトが依拠する計測式の回帰テスト）。
    """
    store = AssociativeStore(key_dim=64, value_dim=64)
    keys = torch.randn(500, 64)
    keys = keys / keys.norm(dim=-1, keepdim=True)
    values = torch.randn(500, 64)
    store.write(keys, values)
    computed = store._keys.element_size() * store._keys.nelement() + \
        store._values.element_size() * store._values.nelement()
    assert computed == 4 * 500 * 64 * 2


def test_read_time_grows_with_n_on_small_scale() -> None:
    """N=100とN=20000で、20000の方が概ね遅い（同一条件で複数回試行しての緩い確認）。
    壁時計時間はノイズが大きいため、厳密な比較ではなく大きな差での確認に留める。
    """
    device = torch.device("cpu")
    small = measure_one(100, 64, 64, seed=0, device=device)
    large = measure_one(50_000, 64, 64, seed=0, device=device)
    assert large["read_time_median_sec"] >= small["read_time_median_sec"]
