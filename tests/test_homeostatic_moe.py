"""恒常性可塑性による MoE の負荷分散（12.6.127節）のテスト。"""

from __future__ import annotations

import numpy as np
import torch

from neurocortex.expert_prefetch import RoutedCharLM, train_routed_lm
from neurocortex.homeostatic_moe import HomeostaticRoutedCharLM, train_homeostatic_lm
from neurocortex.hub_load import gini, lesioned_forward, make_windows


def _pair(seed: int = 0):
    torch.manual_seed(seed)
    base = RoutedCharLM(vocab_size=12, d=8, n_experts=4, d_ff=8)
    torch.manual_seed(seed)
    homeo = HomeostaticRoutedCharLM(vocab_size=12, d=8, n_experts=4, d_ff=8, eta=0.1)
    return base, homeo


def test_same_init_and_forward_with_zero_threshold():
    base, homeo = _pair()
    x = torch.randint(0, 12, (3, 10))
    for name, p in base.named_parameters():
        q = dict(homeo.named_parameters())[name.replace("router.", "router.linear.")]
        assert torch.equal(p, q)
    with torch.no_grad():
        a, _, ca, _ = base(x)
        b, _, cb, _ = homeo(x)
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.equal(ca, cb)


def test_threshold_moves_against_usage_and_is_centered():
    _, homeo = _pair()
    homeo.alpha = 1.0
    homeo.homeostatic_step(torch.zeros(2, 5, dtype=torch.long))  # 専門家 0 だけが使われた
    th = homeo.router.threshold
    assert th[0] > 0 and (th[1:] < 0).all()  # 使われすぎは閾値が上がる（選ばれにくくなる）
    assert abs(float(th.mean())) < 1e-6
    assert torch.allclose(homeo.usage_ema, torch.tensor([1.0, 0, 0, 0]))


def test_threshold_shifts_routing_and_lesion_path():
    _, homeo = _pair()
    x = torch.randint(0, 12, (3, 10))
    homeo.router.threshold[:] = torch.tensor([50.0, 50.0, -100.0, 0.0])
    with torch.no_grad():
        logits, _, choice, _ = homeo(x)
        l2, c2 = lesioned_forward(homeo, x)  # hub_load の関数が閾値込みで動く
    assert (choice == 2).all()
    assert torch.equal(choice, c2) and torch.allclose(logits, l2, atol=1e-6)
    with homeo.without_threshold():
        assert (homeo.router.threshold == 0).all()
    assert float(homeo.router.threshold[2]) == -100.0  # 文脈を出ると戻る


def test_zero_eta_training_equals_baseline():
    torch.manual_seed(1)
    data = torch.randint(0, 12, (500,))
    base, homeo = _pair(1)
    homeo.eta = 0.0
    ce_a = train_routed_lm(base, data, 5, 4, 8, 3e-3, 0.0, torch.Generator().manual_seed(1))
    ce_b = train_homeostatic_lm(homeo, data, 5, 4, 8, 3e-3, torch.Generator().manual_seed(1))
    assert abs(ce_a - ce_b) < 1e-5
    assert (homeo.router.threshold == 0).all()


def test_homeostasis_reduces_load_imbalance():
    # ルータの初期値を専門家 0 に強く偏らせ、恒常性だけで偏りが減ることを確かめる
    torch.manual_seed(2)
    data = torch.randint(0, 12, (2000,))
    x, _ = make_windows(data, 16, 20)
    loads = {}
    for eta in (0.0, 0.3):
        torch.manual_seed(2)
        m = HomeostaticRoutedCharLM(vocab_size=12, d=8, n_experts=4, d_ff=8, eta=eta, alpha=0.2)
        with torch.no_grad():
            m.router.linear.bias[:] = torch.tensor([5.0, 0, 0, 0])
        train_homeostatic_lm(m, data, 60, 8, 16, 1e-3, torch.Generator().manual_seed(2))
        with torch.no_grad():
            c = lesioned_forward(m, x)[1]
        loads[eta] = np.bincount(c.numpy().ravel(), minlength=4)
    assert gini(loads[0.3]) < gini(loads[0.0]) - 0.2
