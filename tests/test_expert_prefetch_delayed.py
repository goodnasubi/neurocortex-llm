"""遅れて届く登上線維の誤差と適格性の窓による先読みの閾値の適応（12.6.209節）のテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from neurocortex.expert_prefetch import MultiPathExpertStore, simulate_prefetch_v2
from neurocortex.expert_prefetch_delayed import (CFTaggedStore, DelayedCFGate, curve_residual, pareto_front,
                                                 warm_delayed)
from neurocortex.experiments.run_expert_prefetch_delayed import _resid_test, residuals, select_rho


def _prob(c: int, p: float, n: int = 4) -> np.ndarray:
    pr = np.zeros(n)
    pr[c] = p
    return pr


def test_outcome_updates_arrive_at_use_or_eviction():
    g = DelayedCFGate(4, eta_up=0.1, eta_down=0.05, theta0=0.5)
    assert g.decide(_prob(2, 0.6), 0, lambda c: False) == 2
    g.feedback(3)  # 次の時刻に使われなくても、まだ誤差は届かない（次の時刻の誤差とは違う）
    assert np.isclose(g.theta[2], 0.5)
    g.on_prefetch_used(2)  # 2ステップ後に使われた → LTP
    assert np.isclose(g.theta[2], 0.45) and g.delay_used == {1: 1}
    assert g.decide(_prob(1, 0.6), 0, lambda c: False) == 1
    g.on_prefetch_wasted(1)  # 使われずに追い出された → LTD
    assert np.isclose(g.theta[1], 0.6) and g.n_wasted == 1
    g.on_prefetch_wasted(1)  # 発行の印がなければ何もしない（二重に数えない）
    assert np.isclose(g.theta[1], 0.6)


def test_skipped_eligibility_window():
    for window, expect in ((2, 0.45), (1, 0.5)):
        g = DelayedCFGate(4, eta_up=0.1, eta_down=0.05, theta0=0.5, window=window)
        assert g.decide(_prob(1, 0.3), 0, lambda c: False) is None
        g.feedback(2)  # 次の時刻は別の専門家
        g.feedback(1)  # 2ステップ後に使われた: W=2 なら窓の中で LTP、W=1 なら窓の外
        assert np.isclose(g.theta[1], expect)
    g = DelayedCFGate(4, eta_up=0.1, eta_down=0.05, theta0=0.5, window=2)
    g.decide(_prob(1, 0.3), 0, lambda c: False)
    for e in (2, 3, 1):  # 3ステップ後は窓の外
        g.feedback(e)
    assert np.isclose(g.theta[1], 0.5) and g.n_skipped_hit == 0
    with pytest.raises(ValueError):
        DelayedCFGate(4, 0.1, 0.05, window=0)


def test_issue_clears_eligibility_and_busy_is_not_target():
    g = DelayedCFGate(4, eta_up=0.1, eta_down=0.05, theta0=0.5)
    g.decide(_prob(1, 0.3), 0, lambda c: False)  # 見送り → 適格性
    g.feedback(2)
    assert g.decide(_prob(1, 0.9), 2, lambda c: False) == 1  # 先読みしたら適格性は消える
    g.feedback(1)
    assert g.n_skipped_hit == 0
    assert g.decide(_prob(3, 0.9), 0, lambda c: True) is None  # 常駐・読み込み中は対象外
    assert g.decide(_prob(0, 0.9), 0, lambda c: False) is None  # 直前と同じ
    assert g.n_issued == 1


def test_store_without_gate_matches_multipath(tmp_path):
    """ゲートなしの CFTaggedStore は、同じ操作の列で MultiPathExpertStore と同じ常駐・数を持つ。"""
    weights = [np.full(8, e, dtype=np.float32) for e in range(4)]
    a = MultiPathExpertStore(weights, tmp_path / "a", 2, 0.0, 1)
    b = CFTaggedStore(weights, tmp_path / "b", 2, 0.0, 1)
    ops = [("u", 0), ("p", 1), ("u", 2), ("p", 3), ("u", 3), ("p", 0), ("p", 1), ("u", 2), ("u", 1), ("p", 2)]
    for st in (a, b):
        for op, e in ops:
            st.use_tracked(e) if op == "u" else st.try_prefetch(e)
        st.finish()
    assert list(a.resident) == list(b.resident)
    for k in ("demand_loads", "prefetch_loads", "wasted_loads"):
        assert getattr(a, k) == getattr(b, k)


class _ConstPred:
    def __init__(self, c: int, p: float, n: int = 4) -> None:
        self.pr = _prob(c, p, n)

    def prob(self, e, h):
        return self.pr

    def observe(self, e, h, e_next):
        pass


def test_simulation_wasted_matches_gate_errors(tmp_path):
    """ゲートの誤差の数が、ストアの無駄 I/O（列の終わりの分を除く）と一致する。"""
    choices = np.array([0, 1] * 30, dtype=np.int64)  # 専門家 3 は決して使われない
    hidden = np.zeros((len(choices), 2), dtype=np.float32)
    weights = [np.full(8, e, dtype=np.float32) for e in range(4)]
    g = DelayedCFGate(4, eta_up=0.1, eta_down=0.01, theta0=0.5)
    store = CFTaggedStore(weights, tmp_path, capacity=2, io_ms=0.5, n_paths=2, gate=g)
    res = simulate_prefetch_v2(choices, hidden, store, _ConstPred(3, 0.75), 1, 0.2, adaptive_gate=g)
    assert g.n_used == 0 and g.n_wasted >= 2
    assert res.wasted_loads == g.n_wasted + len(g.issued_at)  # 終わりまで残った先読みはゲートを更新しない
    assert g.theta[3] > 0.75  # 外れが届くたびに上がり、確信度を超えて先読みをやめる
    assert res.prefetch_loads == g.n_issued


def test_two_step_useful_prefetch_is_not_an_error(tmp_path):
    """次の時刻ではなく2ステップ後に使われる先読みは、遅れた誤差では当たりになる。"""
    choices = np.array([0, 1, 2, 3] * 10, dtype=np.int64)
    hidden = np.zeros((len(choices), 2), dtype=np.float32)

    class TwoAhead:
        def prob(self, e, h):
            return _prob((e + 2) % 4, 0.6)

        def observe(self, *a):
            pass

    g = DelayedCFGate(4, eta_up=0.1, eta_down=0.01, theta0=0.5)
    warm_delayed(TwoAhead(), choices, hidden, g, capacity=3)
    assert g.n_used > 0 and g.n_wasted == 0
    assert np.all(g.theta <= 0.5)
    assert not g.elig and not g.issued_at  # 暖機の終わりに印を捨てる


def test_pareto_residual():
    pts = [(0, 0.0), (40, 0.06), (80, 0.09), (60, 0.05), (120, 0.12)]
    assert pareto_front(pts) == [(0, 0.0), (40, 0.06), (80, 0.09), (120, 0.12)]
    assert np.isclose(curve_residual(pts, 60, 0.085), 0.085 - 0.075)
    assert np.isclose(curve_residual(pts, 200, 0.10), -0.02)  # 右端の先は端の値
    t = _resid_test([0.02, 0.01, 0.015, -0.001, 0.02])
    assert t["pass"] and t["n_pos"] == 4
    assert not _resid_test([0.005] * 5)["pass"]


def test_residuals_and_selection():
    conds = ["none", "freq", "cerebellum", "cerebellum_fixed0.3", "cerebellum_delayed1", "cerebellum_delayed2",
             "cerebellum_delayedw12"]
    sp = {"none": (1.0, 0), "freq": (0.97, 100), "cerebellum": (0.88, 120), "cerebellum_fixed0.3": (0.92, 60),
          "cerebellum_delayed1": (0.85, 110), "cerebellum_delayed2": (0.90, 60), "cerebellum_delayedw12": (0.91, 60)}
    run = {"speed": {"small": {c: {"wall_s": w, "wasted_loads": x, "stalls": 10} for c, (w, x) in sp.items()}}}
    r = residuals([run], conds, "small", "cerebellum_delayed2")
    assert np.isclose(r[0], 0.10 - 0.08)
    s = {c: {"wall_s": {"mean": w}, "wasted_loads": {"mean": x}} for c, (w, x) in sp.items()}
    assert select_rho(s) == "cerebellum_delayed2"  # delayed1 は無駄 I/O が freq を超える、w1 は候補外
