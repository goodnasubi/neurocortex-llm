"""恒常性 MoE の規模の変化（12.6.137節）のテスト。"""

from __future__ import annotations

import numpy as np
import torch

from neurocortex import hub_load
from neurocortex.expert_prefetch import RoutedCharLM, train_routed_lm
from neurocortex.homeostatic_moe import HomeostaticRoutedCharLM, train_homeostatic_lm, use_select_forward
from neurocortex.moe_scale import ScaledRoutedCharLM, eval_ce, hub_masks, lesion_study, load_stats, train_scaled

V = 12


def _pair(kind: str, n_layers: int = 1, seed: int = 0, n_experts: int = 8):
    torch.manual_seed(seed)
    if kind == "plain":
        ref = RoutedCharLM(V, d=8, n_experts=n_experts, d_ff=8)
    else:
        ref = HomeostaticRoutedCharLM(V, d=8, n_experts=n_experts, d_ff=8, eta=0.1, select_only=kind == "select")
    torch.manual_seed(seed)
    m = ScaledRoutedCharLM(V, d=8, n_experts=n_experts, d_ff=8, n_layers=n_layers, kind=kind, eta=0.1)
    return ref, m


def test_one_layer_matches_routed_char_lm():
    ref, m = _pair("plain")
    x = torch.randint(0, V, (3, 10))
    with torch.no_grad():
        la, ba, ca, _ = ref(x)
        lb, bb, cb = m(x)
    assert torch.allclose(la, lb, atol=1e-6) and torch.allclose(ba, bb) and torch.equal(ca, cb[0])


def test_one_layer_matches_homeostatic_variants_with_threshold():
    x = torch.randint(0, V, (3, 10))
    th = torch.tensor([2.0, -1.0, 0.5, 0.0, -3.0, 1.0, 0.2, 0.3])
    for kind in ("gate", "select"):
        ref, m = _pair(kind)
        ref.router.threshold.copy_(th)
        m.threshold[0].copy_(th)
        with torch.no_grad():
            la, _, ca, _ = ref(x)
            lb, _, cb = m(x)
        assert torch.allclose(la, lb, atol=1e-6) and torch.equal(ca, cb[0])


def test_one_layer_training_matches_existing_trainers():
    data = torch.randint(0, V, (400,))
    ref, m = _pair("plain")
    a = train_routed_lm(ref, data, 5, 4, 16, 3e-3, 0.05, torch.Generator().manual_seed(1))
    b = train_scaled(m, data, 5, 4, 16, 3e-3, torch.Generator().manual_seed(1), 0.05)
    assert abs(a - b) < 1e-5
    for kind in ("gate", "select"):
        ref, m = _pair(kind)
        a = train_homeostatic_lm(ref, data, 5, 4, 16, 3e-3, torch.Generator().manual_seed(1))
        b = train_scaled(m, data, 5, 4, 16, 3e-3, torch.Generator().manual_seed(1))
        assert abs(a - b) < 1e-5
        assert torch.allclose(ref.router.threshold, m.threshold[0], atol=1e-6)


def test_one_layer_lesion_matches_hub_load():
    data = torch.randint(0, V, (400,))
    x, y = hub_load.make_windows(data, 16, 8)
    for kind in ("plain", "select"):
        ref, m = _pair(kind)
        if kind == "select":
            th = torch.linspace(-1, 1, 8)
            ref.router.threshold.copy_(th)
            m.threshold[0].copy_(th)
        load, _ = load_stats(m, x)
        with use_select_forward(ref):
            ref_load = np.bincount(hub_load.expert_choices(ref, x).numpy().ravel(), minlength=8)
            assert np.array_equal(load[0], ref_load)
            for mode in ("zero", "reroute"):
                a = hub_load.lesion_study(ref, x, y, ref_load.astype(float), 2, mode)
                b = lesion_study(m, x, y, load, 2, mode)
                assert b["n_random"] == 15
                for key in ("base_ce", "d_target", "d_random_mean", "d_target_per_load"):
                    assert abs(a[key] - b[key]) < 1e-5, (kind, mode, key)


def test_two_layers_and_threshold_per_layer():
    torch.manual_seed(0)
    m = ScaledRoutedCharLM(V, d=8, n_experts=4, d_ff=8, n_layers=2, kind="select", eta=0.5)
    x = torch.randint(0, V, (3, 10))
    _, _, c = m(x)
    assert c.shape == (2, 3, 10)
    m.threshold[1] = torch.tensor([100.0, 100.0, -300.0, 100.0])
    _, _, c = m(x)
    assert (c[1] == 2).all() and not (c[0] == 2).all()
    m.threshold.zero_()
    m.homeostatic_step(torch.stack([torch.zeros(3, 10, dtype=torch.long), torch.ones(3, 10, dtype=torch.long)]))
    assert m.threshold[0, 0] > 0 and m.threshold[1, 1] > 0
    assert torch.allclose(m.threshold.sum(1), torch.zeros(2), atol=1e-6)


def test_gate_one_makes_gate_mass_equal_token_share():
    torch.manual_seed(0)
    m = ScaledRoutedCharLM(V, d=8, n_experts=8, d_ff=8, n_layers=2)
    data = torch.randint(0, V, (400,))
    x, y = hub_load.make_windows(data, 16, 8)
    load, st = load_stats(m, x)
    r = lesion_study(m, x, y, load, 2, gate_one=True)
    assert abs(r["hub_gate_mass"] - r["hub_load_frac"]) < 1e-9
    assert abs(r["d_target_per_gate"] - r["d_target_per_load"]) < 1e-9
    r = lesion_study(m, x, y, load, 2)
    # ゲート < 1 なので質量はトークン割合より小さく、質量あたりの悪化は大きい
    assert r["hub_gate_mass"] < r["hub_load_frac"] and abs(r["d_target_per_gate"]) >= abs(r["d_target_per_load"])
    assert 0 < st["gate_mean"] < 1
    assert eval_ce(m, x, y, gate_one=True) != eval_ce(m, x, y)


def test_hub_masks_samples_when_too_many_combinations():
    load = np.arange(64, dtype=float).reshape(2, 32)
    hub, masks = hub_masks(load, 8, 30, np.random.default_rng(0))
    assert hub.sum() == 16 and hub[0, 24:].all() and hub[1, 24:].all()
    assert len(masks) == 30
    for mm in masks:
        assert mm.sum() == 16 and not (mm & hub).any()


def test_select_homeostasis_reduces_imbalance_with_more_experts():
    torch.manual_seed(0)
    data = torch.tensor([i % V for i in range(2000)])
    x, _ = hub_load.make_windows(data, 32, 8)
    ginis = {}
    for kind, eta in (("plain", 0.0), ("select", 0.3)):
        torch.manual_seed(0)
        m = ScaledRoutedCharLM(V, d=8, n_experts=16, d_ff=8, n_layers=1, kind=kind, eta=eta)
        train_scaled(m, data, 60, 8, 32, 3e-3, torch.Generator().manual_seed(0))
        ginis[kind] = load_stats(m, x)[1]["gini"]
    assert ginis["select"] < ginis["plain"]
