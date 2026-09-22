"""STDPパターン分離層の学習安定性計測基盤（ステップ26・12.6.60節）の単体テスト。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.hippocampus import HippocampalMemory, fit_stdp  # noqa: E402
from neurocortex.experiments.run_stdp_stability import (  # noqa: E402
    degeneracy_stats,
    run_one,
)


def test_fit_stdp_callback_fires_once_per_epoch() -> None:
    """新規追加した`callback`引数が、各epoch末にちょうど1回ずつ呼ばれること。"""
    mem = HippocampalMemory(8, value_dim=8, n_units=32, k=4, mode="sdr")
    spikes = (torch.rand(20, 3, 8) > 0.5).float()
    calls: list[int] = []
    fit_stdp(mem.separator, spikes, epochs=4, batch_size=8,
             generator=torch.Generator().manual_seed(0),
             callback=lambda ep, sep: calls.append(ep))
    assert calls == [0, 1, 2, 3]


def test_fit_stdp_without_callback_is_unchanged() -> None:
    """`callback=None`（既定）のときは従来どおり動作し、戻り値の形も変わらないこと。"""
    mem = HippocampalMemory(8, value_dim=8, n_units=32, k=4, mode="sdr")
    spikes = (torch.rand(20, 3, 8) > 0.5).float()
    info = fit_stdp(mem.separator, spikes, epochs=2, batch_size=8,
                     generator=torch.Generator().manual_seed(0))
    assert info["rule"] == "stdp"
    assert info["epochs"] == 2


def test_degeneracy_stats_detects_identical_rows() -> None:
    """全ユニットが同一方向に縮退していれば、ペアワイズコサイン類似度が1に近いこと。"""
    w = torch.ones(16, 8) + 1e-6 * torch.randn(16, 8)
    stats = degeneracy_stats(w)
    assert stats["mean_pairwise_cos"] > 0.99
    assert stats["max_pairwise_cos"] > 0.99
    assert stats["n_unique_rows"] <= 2


def test_degeneracy_stats_detects_diverse_rows() -> None:
    """直交に近いランダムな重みでは類似度が低く、一意な行数がほぼ全数であること。"""
    torch.manual_seed(0)
    w = torch.randn(64, 32)
    stats = degeneracy_stats(w)
    assert stats["mean_pairwise_cos"] < 0.3
    assert stats["n_unique_rows"] == 64


def test_run_one_reports_nan_inf_and_norm_per_epoch() -> None:
    """`run_one`が各epochのフロベニウスノルム・NaN/Inf有無を記録すること（健全性）。"""
    torch.manual_seed(0)
    spikes = (torch.rand(200, 4, 16) > 0.7).float()
    result = run_one(spikes, d_model=16, n_units=64, k=8, backbone_seed=0,
                      stdp_seed=0, a_plus=0.01, a_minus=0.008, tau=0.9, epochs=3)
    assert result["diverged"] is False
    assert len(result["epoch_records"]) == 3
    for rec in result["epoch_records"]:
        assert rec["has_nan"] is False
        assert rec["has_inf"] is False
        assert rec["frobenius_norm"] == rec["frobenius_norm"]  # NaNでない


def test_run_one_is_deterministic_given_same_seed() -> None:
    """同一シードなら結果が再現すること（実験の再現性の前提）。"""
    torch.manual_seed(0)
    spikes = (torch.rand(200, 4, 16) > 0.7).float()
    r1 = run_one(spikes, d_model=16, n_units=64, k=8, backbone_seed=1,
                 stdp_seed=1, a_plus=0.01, a_minus=0.008, tau=0.9, epochs=3)
    r2 = run_one(spikes, d_model=16, n_units=64, k=8, backbone_seed=1,
                 stdp_seed=1, a_plus=0.01, a_minus=0.008, tau=0.9, epochs=3)
    n1 = [r["frobenius_norm"] for r in r1["epoch_records"]]
    n2 = [r["frobenius_norm"] for r in r2["epoch_records"]]
    assert n1 == n2
