"""登上線維の誤差による先読みの閾値の適応（12.6.193節）のテスト。"""

from __future__ import annotations

import numpy as np

from neurocortex.expert_prefetch import AdaptiveThresholdGate, MultiPathExpertStore, simulate_prefetch_v2
from neurocortex.experiments.run_expert_prefetch_adaptive import h_pass, select_on_dev, warm


def _prob(c: int, p: float, n: int = 4) -> np.ndarray:
    pr = np.zeros(n)
    pr[c] = p
    return pr


def test_miss_raises_hit_lowers_threshold():
    g = AdaptiveThresholdGate(4, eta_up=0.1, eta_down=0.05, theta0=0.5)
    assert g.decide(_prob(2, 0.6), 0, lambda c: False) == 2
    g.feedback(3)  # 先読みして外れ → 上がる
    assert np.isclose(g.theta[2], 0.6)
    assert g.decide(_prob(2, 0.7), 0, lambda c: False) == 2
    g.feedback(2)  # 当たり → 下がる
    assert np.isclose(g.theta[2], 0.55)
    assert g.theta[[0, 1, 3]].tolist() == [0.5, 0.5, 0.5]  # 専門家ごと


def test_skipped_hit_lowers_and_skipped_miss_keeps():
    g = AdaptiveThresholdGate(4, eta_up=0.1, eta_down=0.05, theta0=0.5)
    assert g.decide(_prob(1, 0.3), 0, lambda c: False) is None
    g.feedback(1)  # 先読みしなかったが当たっていた → 下がる
    assert np.isclose(g.theta[1], 0.45)
    assert g.decide(_prob(1, 0.3), 0, lambda c: False) is None
    g.feedback(2)  # 先読みせず外れ → 変えない
    assert np.isclose(g.theta[1], 0.45)


def test_no_update_when_no_io_would_happen():
    g = AdaptiveThresholdGate(4, eta_up=0.1, eta_down=0.05, theta0=0.5)
    assert g.decide(_prob(0, 0.9), 0, lambda c: False) is None  # 直前と同じ
    g.feedback(1)
    assert g.decide(_prob(2, 0.9), 0, lambda c: True) is None  # 常駐・読み込み中
    g.feedback(1)
    assert np.allclose(g.theta, 0.5) and g.n_issued == 0


def test_global_threshold_shared_and_clipped():
    g = AdaptiveThresholdGate(4, eta_up=1.0, eta_down=0.05, theta0=0.5, per_expert=False, hi=1.2)
    g.decide(_prob(2, 0.9), 0, lambda c: False)
    g.feedback(3)
    assert g.theta.shape == (1,) and np.isclose(g.theta[0], 1.2)


def test_equilibrium_miss_to_hit_ratio():
    """釣り合いでは 外れた先読み ≈ 当たり × eta_down/eta_up。"""
    rng = np.random.default_rng(0)
    g = AdaptiveThresholdGate(2, eta_up=0.02, eta_down=0.01, theta0=0.5, per_expert=False)
    hits = 0
    for _ in range(20000):
        p = rng.uniform()
        hit = rng.uniform() < p  # 確信度 p が較正された予測
        g.decide(np.array([0.0, p]), 0, lambda c: False)
        g.feedback(1 if hit else 0)
        hits += hit
    ratio = g.n_issued_miss / hits
    assert abs(ratio - 0.5) < 0.05


class _ConstPred:
    """常に専門家 c を確信度 p で予測する予測器。"""

    def __init__(self, c: int, p: float, n: int = 4) -> None:
        self.pr = _prob(c, p, n)

    def prob(self, e, h):
        return self.pr

    def observe(self, e, h, e_next):
        pass


def test_simulation_threshold_rises_on_wrong_predictions(tmp_path):
    choices = np.array([0, 1] * 30, dtype=np.int64)  # 専門家 3 は決して使われない
    hidden = np.zeros((len(choices), 2), dtype=np.float32)
    weights = [np.full(8, e, dtype=np.float32) for e in range(4)]
    store = MultiPathExpertStore(weights, tmp_path, capacity=2, io_ms=0.5, n_paths=2)
    g = AdaptiveThresholdGate(4, eta_up=0.1, eta_down=0.01, theta0=0.5)
    res = simulate_prefetch_v2(choices, hidden, store, _ConstPred(3, 0.75), 1, 0.2, adaptive_gate=g)
    # 外れるたびに θ_3 が 0.1 ずつ上がり（0.5→0.6→0.7→0.8）、確信度 0.75 を超えたら先読みをやめる
    assert g.n_issued == 3 and g.n_issued_miss == 3
    assert g.theta[3] > 0.75
    assert res.prefetch_loads == 3


def test_warm_moves_threshold():
    choices = np.array([0, 1] * 50, dtype=np.int64)
    hidden = np.zeros((len(choices), 2), dtype=np.float32)
    g = AdaptiveThresholdGate(4, eta_up=0.05, eta_down=0.01, theta0=0.5)
    warm(_ConstPred(3, 0.7), choices, hidden, g, capacity=2)
    assert g.theta[3] > 0.7  # 外れ続けて閾値が確信度を超える


def _summ(walls, wasted):
    return {c: {"wall_s": {"mean": w}, "wasted_loads": {"mean": x},
                "reduction_vs_none": {"mean": 1 - w / walls["none"]},
                "reduction_per_seed": [1 - w / walls["none"]] * 5}
            for (c, w), x in zip(walls.items(), wasted)}


def test_selection_rule_and_pass_criteria():
    s = _summ({"none": 1.0, "last": 1.0, "freq": 0.97, "cerebellum": 0.88, "cerebellum_fixed0.3": 0.89,
               "cerebellum_fixed0.5": 0.95, "cerebellum_adapt1": 0.86, "cerebellum_adapt2": 0.88,
               "cerebellum_adaptglobal1": 0.80},
              [0, 0, 100, 120, 110, 40, 101, 90, 10])
    assert select_on_dev(s, "cerebellum_fixed") == "cerebellum_fixed0.5"  # 0.3 は無駄 I/O が freq 超え
    assert select_on_dev(s, "cerebellum_adapt") == "cerebellum_adapt2"  # global は候補に入れない
    r = h_pass(s, "cerebellum_adapt2")
    assert r["pass"] and r["c4_wasted"]
    assert not h_pass(s, "cerebellum")["pass"]
