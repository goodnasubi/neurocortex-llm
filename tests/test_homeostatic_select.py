"""閾値を選択だけに効かせる恒常性 MoE（12.6.133節）のテスト。"""

from __future__ import annotations

import numpy as np
import torch

from neurocortex import hub_load
from neurocortex.homeostatic_moe import HomeostaticRoutedCharLM, train_homeostatic_lm, use_select_forward
from neurocortex.hub_load import eval_ce, expert_choices, gini, lesion_study, lesioned_forward, make_windows


def _model(seed: int = 0, **kw) -> HomeostaticRoutedCharLM:
    torch.manual_seed(seed)
    return HomeostaticRoutedCharLM(vocab_size=12, d=8, n_experts=4, d_ff=8, eta=0.1, **kw)


def test_select_only_equals_default_with_zero_threshold():
    a, b = _model(), _model(select_only=True)
    x = torch.randint(0, 12, (3, 10))
    with torch.no_grad():
        la, ba, ca, _ = a(x)
        lb, bb, cb, _ = b(x)
    assert torch.allclose(la, lb, atol=1e-6) and torch.equal(ca, cb) and torch.allclose(ba, bb)


def test_threshold_changes_choice_but_not_gate_probability():
    m = _model(select_only=True)
    x = torch.randint(0, 12, (3, 10))
    m.router.threshold[:] = torch.tensor([50.0, 50.0, -100.0, 0.0])
    with torch.no_grad():
        logits, _, choice, h = m(x)
        probs = torch.softmax(m.router.linear(h), -1)  # 閾値なしの確率
        gate = probs.gather(-1, choice.unsqueeze(-1))
        y = torch.einsum("btd,btdf->btf", h, m.w1[choice]).relu()
        y = torch.einsum("btf,btfd->btd", y, m.w2[choice])
        ref = m.head(h + gate * y)
    assert (choice == 2).all()
    assert torch.allclose(logits, ref, atol=1e-6)


def test_threshold_gets_no_gradient_and_router_grad_ignores_threshold():
    grads = []
    for th in (0.0, 1e-4):  # 選択を変えない小さな閾値なら勾配は同じ（ゲート込みなら変わる）
        m = _model(1, select_only=True)
        x = torch.randint(0, 12, (2, 6), generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            _, _, c0, _ = m(x)
        m.router.threshold[:] = th * torch.tensor([1.0, -1.0, 1.0, -1.0])
        logits, _, c1, _ = m(x)
        assert torch.equal(c0, c1)
        logits.sum().backward()
        grads.append(m.router.linear.weight.grad.clone())
        assert not m.router.threshold.requires_grad
    assert len(grads) == 2 and torch.allclose(grads[0], grads[1])


def test_use_select_forward_routes_hub_load_functions():
    m = _model(select_only=True)
    x = torch.randint(0, 12, (3, 10))
    y = torch.randint(0, 12, (3, 10))
    m.router.threshold[:] = torch.tensor([50.0, 50.0, -100.0, 0.0])
    orig = hub_load.lesioned_forward
    with use_select_forward(m):
        assert (expert_choices(m, x) == 2).all()
        base = eval_ce(m, x, y)
        with torch.no_grad():
            logits, _, _, _ = m(x)
        ref = float(torch.nn.functional.cross_entropy(logits.reshape(-1, 12), y.reshape(-1)))
        assert abs(base - ref) < 1e-5
        dis = torch.tensor([False, False, True, False])
        _, c = hub_load.lesioned_forward(m, x, dis, "reroute")
        assert not (c == 2).any()
        r = lesion_study(m, x, y, np.array([0, 0, 30, 0.0]), 2, "zero")
        assert r["hubs"][0] == 2
    assert hub_load.lesioned_forward is orig
    # 選択専用でないモデルでは何もしない
    d = _model()
    with use_select_forward(d):
        assert hub_load.lesioned_forward is orig
    assert torch.equal(lesioned_forward(d, x)[1], d(x)[2])


def test_leak_and_clip_bound_threshold():
    m = _model(leak=0.5, theta_max=0.2)
    m.alpha = 1.0
    for _ in range(20):
        m.homeostatic_step(torch.zeros(2, 5, dtype=torch.long))
    assert float(m.router.threshold.abs().max()) <= 0.2 + 1e-6
    m2 = _model(leak=0.5)
    m2.alpha = 1.0
    for _ in range(50):
        m2.homeostatic_step(torch.zeros(2, 5, dtype=torch.long))
    # 漏れのある積分は定常値 η (E u − 1) / λ に収束する（増分は既に平均 0 なので 0.1 * 3 / 0.5 = 0.6）
    assert abs(float(m2.router.threshold[0]) - 0.6) < 1e-3


def test_select_only_homeostasis_reduces_imbalance_with_small_threshold():
    torch.manual_seed(2)
    data = torch.randint(0, 12, (2000,))
    x, _ = make_windows(data, 16, 20)
    loads = {}
    for eta in (0.0, 0.3):
        torch.manual_seed(2)
        m = HomeostaticRoutedCharLM(vocab_size=12, d=8, n_experts=4, d_ff=8, eta=eta, alpha=0.2, select_only=True)
        with torch.no_grad():
            m.router.linear.bias[:] = torch.tensor([5.0, 0, 0, 0])
        train_homeostatic_lm(m, data, 60, 8, 16, 1e-3, torch.Generator().manual_seed(2))
        with use_select_forward(m), torch.no_grad():
            loads[eta] = np.bincount(expert_choices(m, x).numpy().ravel(), minlength=4)
    assert gini(loads[0.3]) < gini(loads[0.0]) - 0.2


def test_gate_stats_reports_threshold_shift():
    from neurocortex.experiments.run_homeostatic_select import gate_stats
    m = _model(select_only=True)
    x = torch.randint(0, 12, (3, 10))
    assert gate_stats(m, x)["shifted_frac"] == 0.0
    m.router.threshold[:] = torch.tensor([50.0, 50.0, -100.0, 0.0])
    g = gate_stats(m, x)
    with torch.no_grad():
        _, _, _, h = m(x)
        p2 = torch.softmax(m.router.linear(h), -1)[..., 2]
    assert abs(g["gate_mean"] - float(p2.mean())) < 1e-6 and g["shifted_frac"] > 0
