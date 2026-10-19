"""ハブを許す目標域つきの乗法的な恒常性スケーリング（12.6.195節）のテスト。"""

from __future__ import annotations

import math

import numpy as np
import torch

from neurocortex import hub_load
from neurocortex.moe_homeo_band import BandHomeostaticLM, band_violation, single_lesions, topk_share
from neurocortex.moe_scale import ScaledRoutedCharLM, eval_ce, load_stats, train_scaled

V = 12


def _choice_with_usage(counts: list[int]) -> torch.Tensor:
    """選択の数が counts になる choice [1,1,T]。"""
    return torch.tensor([e for e, c in enumerate(counts) for _ in range(c)]).view(1, 1, -1)


def test_point_target_update_is_log_ratio_and_centered():
    m = BandHomeostaticLM(V, d=8, n_experts=4, d_ff=8, eta=0.5, alpha=1.0)
    m.homeostatic_step(_choice_with_usage([4, 2, 1, 1]))  # r = 2, 1, 0.5, 0.5
    raw = 0.5 * torch.log(torch.tensor([2.0, 1.0, 0.5, 0.5]))
    assert torch.allclose(m.threshold[0], raw - raw.mean(), atol=1e-6)
    assert abs(float(m.threshold[0].sum())) < 1e-6
    # 使われすぎの専門家の閾値が上がり、使われなさすぎの専門家の閾値が下がる
    assert m.threshold[0, 0] > m.threshold[0, 1] > m.threshold[0, 2]


def test_band_leaves_in_range_experts_and_pushes_to_nearest_edge():
    m = BandHomeostaticLM(V, d=8, n_experts=4, d_ff=8, eta=1.0, alpha=1.0, r_lo=0.5, r_hi=2.0)
    m.homeostatic_step(_choice_with_usage([4, 2, 1, 1]))  # r = 2, 1, 0.5, 0.5: すべて域内
    assert torch.allclose(m.threshold, torch.zeros_like(m.threshold))
    m.homeostatic_step(_choice_with_usage([6, 2, 0, 0]))  # r = 3, 1, 0, 0
    err = m.drive_error()[0]
    assert math.isclose(float(err[0]), math.log(3 / 2), rel_tol=1e-5)
    assert float(err[1]) == 0.0
    assert math.isclose(float(err[2]), math.log(0.01 / 0.5), rel_tol=1e-5)  # r_floor で抑える


def test_eta_zero_matches_select_scaled_model():
    torch.manual_seed(0)
    a = ScaledRoutedCharLM(V, d=8, n_experts=8, d_ff=8, kind="select", eta=0.0)
    torch.manual_seed(0)
    b = BandHomeostaticLM(V, d=8, n_experts=8, d_ff=8, eta=0.0, r_lo=0.5, r_hi=2.5)
    data = torch.tensor([i % V for i in range(500)])
    ca = train_scaled(a, data, 5, 4, 16, 3e-3, torch.Generator().manual_seed(1))
    cb = train_scaled(b, data, 5, 4, 16, 3e-3, torch.Generator().manual_seed(1))
    assert ca == cb
    x = torch.randint(0, V, (2, 7))
    assert torch.allclose(a(x)[0], b(x)[0])


def test_threshold_changes_selection_not_gate_probability():
    torch.manual_seed(0)
    m = BandHomeostaticLM(V, d=8, n_experts=4, d_ff=8)
    x = torch.randint(0, V, (2, 7))
    r0 = m.run(x)
    m.threshold[0] = torch.tensor([50.0, 50.0, 50.0, -150.0])
    r1 = m.run(x)
    assert (r1["choice"] == 3).all()
    assert torch.allclose(r1["router"], r0["router"])  # 閾値は確率（ルータの対数）に入らない


def test_metrics():
    assert math.isclose(topk_share([5, 3, 1, 1], 2), 0.8)
    assert math.isclose(topk_share([1, 1, 1, 1], 2), 0.5)
    assert band_violation([1, 1, 1, 1], 0.5, 2.0) == 0.0
    # r = 3.2, 0.8, 0, 0 → (3.2−2) + 0.5 + 0.5 = 2.2、÷E
    assert math.isclose(band_violation([4, 1, 0, 0], 0.5, 2.0), 2.2 / 4)


def test_single_lesions_matches_eval_ce():
    torch.manual_seed(0)
    m = BandHomeostaticLM(V, d=8, n_experts=4, d_ff=8)
    x, y = torch.randint(0, V, (2, 9)), torch.randint(0, V, (2, 9))
    d = single_lesions(m, x, y)
    mask = torch.zeros(1, 4, dtype=torch.bool)
    mask[0, 2] = True
    assert len(d) == 4 and math.isclose(d[2], eval_ce(m, x, y, disabled=mask) - eval_ce(m, x, y), abs_tol=1e-7)


def test_band_keeps_more_concentration_than_point_target():
    torch.manual_seed(0)
    data = torch.tensor([i % V for i in range(2000)])
    x, _ = hub_load.make_windows(data, 32, 8)
    res = {}
    for name, lo, hi, eta in (("plain", 1.0, 1.0, 0.0), ("point", 1.0, 1.0, 0.3), ("band", 0.5, 2.5, 0.3)):
        torch.manual_seed(0)
        m = BandHomeostaticLM(V, d=8, n_experts=8, d_ff=8, kind="plain" if eta == 0 else "select", eta=eta,
                              r_lo=lo, r_hi=hi)
        train_scaled(m, data, 60, 8, 32, 3e-3, torch.Generator().manual_seed(0))
        res[name] = load_stats(m, x)[1]["gini"]
    assert res["point"] < res["plain"]
    assert res["point"] <= res["band"] + 1e-9
    assert np.isfinite(list(res.values())).all()
