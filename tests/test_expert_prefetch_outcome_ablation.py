"""先読みの帰結で学ぶ小脳の順モデルの利得の源の切り分け（12.6.217節）のテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from neurocortex.expert_prefetch import simulate_prefetch_v2
from neurocortex.expert_prefetch_outcome import OutcomeCerebellum, OutcomeTaggedStore, warm_outcome
from neurocortex.expert_prefetch_outcome_ablation import OneSidedOutcomeCerebellum, SupervisedEligibleCerebellum
from neurocortex.experiments import run_expert_prefetch_outcome as base
from neurocortex.experiments.run_expert_prefetch_outcome_ablation import (FAMS, conditions, curve_conds, judge,
                                                                          make_outcome_model, parse, residuals,
                                                                          select_all)


def _h() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0])


def _trace(n=300, seed=0):
    rng = np.random.default_rng(seed)
    ch = np.arange(n) % 4
    flip = rng.random(n) < 0.1
    ch[flip] = rng.integers(0, 4, flip.sum())
    return ch, np.eye(4)[ch].astype(np.float32)


def test_supervised_labels_within_window_immediately():
    m = SupervisedEligibleCerebellum(4, 3, lr=0.5, theta=0.0, window=2)
    with pytest.raises(RuntimeError):
        m.decide(m.prob(0, _h()), 0, lambda c: False)
    m.set_trace(np.array([0, 2, 0, 1, 3]))
    v0 = m.prob(0, _h())
    assert m.decide(v0, 0, lambda c: c == 3) == 1  # 決定は更新の前（同点は番号の小さい順）
    # 対象は 1・2（0 は直前、3 は常駐）。次の2ステップは {2, 0} → 2 が正、1 が負（0 のままなので変わらない）
    assert m.counts["sup_pos"] == 1 and m.counts["sup_neg"] == 1
    v = m.prob(0, _h())
    assert v[2] > 0 and np.isclose(v[1], 0.0) and np.isclose(v[3], 0.0)
    w = m.weight.copy()
    m.feedback(2)
    m.on_prefetch_start(1)
    m.on_prefetch_used(1, late=True)
    m.on_prefetch_wasted(1)
    assert np.array_equal(m.weight, w)  # 帰結は数えるだけで学ばない
    assert m.counts["obs_used_late"] == 1 and m.counts["obs_wasted"] == 1
    with pytest.raises(RuntimeError):
        m.decide(m.prob(0, _h()), 0, lambda c: False)  # 列では時刻 1 は 2
    with pytest.raises(ValueError):
        SupervisedEligibleCerebellum(4, 3, lr=0.5, theta=0.0, shuffle=True)


def test_supervised_window1_matches_next_label_on_targets():
    m = SupervisedEligibleCerebellum(4, 3, lr=0.5, theta=10.0, window=1)
    m.set_trace(np.array([0, 2, 1]))
    m.decide(m.prob(0, _h()), 0, lambda c: False)
    assert m.counts["sup_pos"] == 1 and m.counts["sup_neg"] == 2  # 2 だけが正（1 は 2 ステップ後）


def test_supervised_runs_in_warm_and_simulation(tmp_path):
    ch, hid = _trace(400, seed=1)
    m = SupervisedEligibleCerebellum(4, 4, lr=0.2, theta=0.3, window=2)
    m.set_trace(ch[:201])
    warm_outcome(m, ch[:201], hid[:201], capacity=2)
    assert m.k == 200
    v = m.prob(0, np.eye(4)[0].astype(np.float32))
    assert v[1] > 0.5 and v[2] > 0.5  # 0 の後の 2 ステップには 1 も 2 も来る（W=2 の教師）
    sc, sh = ch[200:300], hid[200:300]
    m.set_trace(sc)
    started0 = m.counts["started"]
    ws = [np.full(8, float(e), dtype=np.float32) for e in range(4)]
    st = OutcomeTaggedStore(ws, tmp_path, 2, 0.2, n_paths=2, gate=m)
    r = simulate_prefetch_v2(sc, sh, st, m, 1, 0.2, parallel_predictor=True, adaptive_gate=m)
    assert m.k == len(sc) - 1
    assert 0 < m.counts["started"] - started0 <= r.prefetch_loads


def test_one_sided_plasticity():
    ltd = OneSidedOutcomeCerebellum(4, 3, lr=0.5, theta=0.0, plasticity="ltd")
    assert np.allclose(ltd.prob(0, _h()), 1.0)  # 増強された状態から始まる
    x = ltd._features(0, _h())
    ltd._learn(1, x, 1.0, "used")  # δ = 0
    ltd._learn(1, x, 0.0, "wasted")  # δ < 0 → 抑圧
    v = ltd.prob(0, _h())[1]
    assert v < 1.0
    ltd._learn(1, x, 1.0, "used")  # δ > 0 は通さない
    assert np.isclose(ltd.prob(0, _h())[1], v) and ltd.counts["blocked"] == 1
    ltp = OneSidedOutcomeCerebellum(4, 3, lr=0.5, theta=0.0, plasticity="ltp")
    assert np.allclose(ltp.prob(0, _h()), 0.0)
    ltp.bias[2] = 0.5
    ltp._learn(2, x, 0.0, "wasted")  # δ < 0 は通さない
    assert np.allclose(ltp.weight, 0.0) and ltp.counts["blocked"] == 1
    ltp._learn(2, x, 1.0, "used")
    assert ltp.prob(0, _h())[2] > 0
    assert np.isclose(OneSidedOutcomeCerebellum(4, 3, lr=0.5, theta=0.0, v0=0.3).bias[0], 0.3)
    with pytest.raises(ValueError):
        OneSidedOutcomeCerebellum(4, 3, lr=0.5, theta=0.0, plasticity="both")


def test_binary_outcome_counts_late_as_used():
    m = make_outcome_model("outbin", 0.0, 4, 3, 0.5, 0)
    assert isinstance(m, OutcomeCerebellum) and m.y_late == 1.0 and m.window == 2
    m.decide(m.prob(0, _h()), 0, lambda c: False)
    m.on_prefetch_start(1)
    m.on_prefetch_used(1, late=True)
    assert np.isclose(m.prob(0, _h())[1], 1.0)  # lr=0.5、|x|=1 と bias で 1.0（段階づけなし）


def test_factory_and_families():
    assert type(make_outcome_model("outcome", 0.5, 4, 3, 0.2, 0)) is OutcomeCerebellum
    assert make_outcome_model("outw4", 0.5, 4, 3, 0.2, 0).window == 4
    assert isinstance(make_outcome_model("supw1", 0.5, 4, 3, 0.2, 0), SupervisedEligibleCerebellum)
    assert make_outcome_model("outltp", 0.5, 4, 3, 0.2, 0).plasticity == "ltp"
    conds = conditions([0.2, 0.4], [0.5])
    assert parse("outw1@0.5") == ("outw1", 0.5) and parse("cerebellum_fixed0.4") == ("cerebellum_fixed", 0.4)
    assert all(f"{f}@0.5" in conds for f in FAMS)
    assert not any("@" in c for c in curve_conds(conds))


def _speed(conds, over):
    def sp(wall, wasted):
        return {"wall_s": wall, "stalls": wall, "stall_s": 0.0, "demand_loads": 0, "prefetch_loads": 0,
                "wasted_loads": wasted, "wasted_bytes": 0, "predictor_s": 0.0}

    return {c: sp(0.9, 50) for c in conds} | {"none": sp(1.0, 0), "cerebellum": sp(0.9, 100)} | {
        k: sp(*v) for k, v in over.items()}


def test_residuals_match_base_module_for_label_points():
    conds = conditions([0.2, 0.4], [0.5])
    bconds = base.conditions([0.2, 0.4])
    over = {"cerebellum_efixed0.2": (0.8, 150), "outcome@0.5": (0.85, 50), "outcome0.2": (0.85, 50)}
    runs = [{"speed": {"small": _speed(conds, over)}}]
    bruns = [{"speed": {"small": _speed(bconds, over)}}]
    r = residuals(runs, conds, "small", "outcome@0.5")
    assert np.isclose(r[0], base.residuals(bruns, bconds, "small", "outcome0.2")[0])


def test_select_and_judge_on_synthetic_runs():
    conds = conditions([0.4], [0.5, 0.7])
    runs = []
    for k in range(5):
        over = {"freq": (0.95, 100), "cerebellum_fixed0.4": (0.9, 80), "cerebellum_efixed0.4": (0.9, 90),
                "outcome@0.5": (0.80, 120), "outcome@0.7": (0.85 - 0.001 * k, 60), "supw2@0.7": (0.88, 60),
                "outbin@0.7": (0.855, 60), "supw1@0.7": (0.9, 60)}
        runs.append({"seed": k, "speed": {"small": _speed(conds, over), "large": _speed(conds, over)}})
    from neurocortex.experiments.run_expert_prefetch_adaptive import summarize
    s = summarize(runs, conds)
    sel = select_all(s, conds)
    assert sel["small"]["outcome"] == "outcome@0.7"  # outcome@0.5 は freq より無駄 I/O が多い
    assert sel["small"]["label_best"] == "cerebellum_fixed0.4"
    v = judge(runs, s, conds, sel)
    assert v["R0_small"]["pass"] and np.isclose(v["H2_small"]["mean"], 0.03 + 0.002, atol=1e-9)
    assert v["H2_small"]["pass"] and not v["H3_small"]["pass"]  # outbin との差は +0.7 ポイント
    assert "H5_small" in v and "ref_a_actionable_small" in v
