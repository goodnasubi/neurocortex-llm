"""小脳の順モデルを先読みの帰結で学習させる（12.6.211節）のテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from neurocortex.expert_prefetch import MultiPathExpertStore, simulate_prefetch_v2
from neurocortex.expert_prefetch_outcome import (EligibleTopGate, OutcomeCerebellum, OutcomeTaggedStore,
                                                 warm_outcome)
from neurocortex.experiments.run_expert_prefetch_outcome import _parse, conditions, curve_conds, residuals, select_all


def _model(**kw) -> OutcomeCerebellum:
    kw = {"lr": 0.5, "theta": 0.0} | kw
    return OutcomeCerebellum(4, 3, **kw)


def _h() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0])


def test_observe_ignores_labels_and_features_are_unit_norm():
    m = _model()
    w0 = m.weight.copy()
    m.observe(0, _h(), 2)
    assert np.array_equal(m.weight, w0)
    assert np.isclose(np.linalg.norm(m._features(1, _h())), 1.0)


def test_prefetch_outcome_used_late_wasted():
    m = _model()
    v = m.prob(0, _h())
    assert m.decide(v, 0, lambda c: False) == 1  # 同点は番号の小さい順、直前と同じ 0 は除く
    m.on_prefetch_used(1)  # 読み込みが始まっていなければ印は付いていない
    assert m.counts["used"] == 0
    m.on_prefetch_start(1)
    m.feedback(3)  # 次の時刻に使われなくても、先読みの帰結はまだ届かない
    assert m.counts["used"] == 0 and m.counts["wasted"] == 0
    m.on_prefetch_used(1, late=False)  # 間に合った → y = 1
    x = m._features(0, _h())
    assert np.isclose(m.weight[1] @ x + m.bias[1], 0.5 * 1.0 * 2)  # lr=0.5、|x|=1 と bias の2つで 1.0
    m.on_prefetch_used(1)  # 発行の印がなければ何もしない
    assert m.counts["used"] == 1
    m2 = _model()
    m2.decide(m2.prob(0, _h()), 0, lambda c: False)
    m2.on_prefetch_start(1)
    m2.on_prefetch_used(1, late=True)  # 待った → y = 0.5
    assert m2.counts["used_late"] == 1 and np.isclose(m2.prob(0, _h())[1], 0.5)
    m3 = _model()
    m3.weight[1] = 0.4 * m3._features(0, _h())
    m3.decide(m3.prob(0, _h()), 0, lambda c: False)
    m3.on_prefetch_start(1)
    m3.on_prefetch_wasted(1)  # 使われずに追い出された → y = 0（LTD 側、価値が下がる）
    assert m3.counts["wasted"] == 1 and m3.prob(0, _h())[1] < 0.4


def test_skip_tags_window_and_busy_candidates():
    m = _model(theta=10.0)  # 先読みしない → 対象の候補すべてにタグ
    assert m.decide(m.prob(0, _h()), 0, lambda c: c == 3) is None  # 3 は常駐なので対象外
    assert set(m.elig) == {1, 2}
    m.feedback(2)  # 2 は窓の中で使われた → y = 1
    assert m.counts["skip_hit"] == 1 and m.prob(0, _h())[2] > 0
    m.feedback(0)  # 1 は窓（W=2）の終わり → y = 0
    assert m.counts["skip_miss"] == 1 and not m.elig
    m1 = _model(theta=10.0, window=1)
    m1.decide(m1.prob(0, _h()), 0, lambda c: False)
    m1.feedback(0)  # W=1 では次の時刻で窓が閉じる
    assert m1.counts["skip_miss"] == 3 and not m1.elig
    with pytest.raises(ValueError):
        _model(window=0)


def test_shuffle_keeps_counts_but_breaks_pairing():
    m = _model(shuffle=True, shuffle_buf=4, seed=1)
    for _ in range(4):
        if _ == 3:
            assert np.allclose(m.weight, 0.0)  # バッファが埋まるまで更新しない
        m.decide(m.prob(0, _h()), 0, lambda c: False)
        m.on_prefetch_start(1)
        m.on_prefetch_used(1)
    assert m.counts["used"] == 4 and len(m.buf) == 3 and not np.allclose(m.weight, 0.0)


def test_eligible_top_gate_skips_resident():
    g = EligibleTopGate(0.3)
    pr = np.array([0.1, 0.9, 0.5, 0.2])
    assert g.decide(pr, 0, lambda c: False) == 1
    assert g.decide(pr, 0, lambda c: c == 1) == 2  # 最上位が常駐なら次の候補
    assert g.decide(pr, 0, lambda c: c in (1, 2)) is None  # 次の候補が閾値未満なら出さない
    assert g.decide(pr, 1, lambda c: c == 2) is None


def _trace(n=300, seed=0):
    rng = np.random.default_rng(seed)
    # 0 → 1 → 2 → 3 → 0 … の周期に少し乱れを入れた列。文脈は現在の専門家の onehot。
    ch = np.arange(n) % 4
    flip = rng.random(n) < 0.1
    ch[flip] = rng.integers(0, 4, flip.sum())
    hid = np.eye(4)[ch].astype(np.float32)
    return ch, hid


def test_warm_outcome_learns_useful_candidate():
    ch, hid = _trace(2000)
    m = OutcomeCerebellum(4, 4, lr=0.2, theta=0.3, window=2)
    warm_outcome(m, ch, hid, capacity=2)
    assert not m.elig and not m.issued
    v = m.prob(0, np.eye(4)[0].astype(np.float32))
    assert int(np.argmax(v)) == 1  # 0 の次は 1 → 先読みの価値が最も高い
    assert m.counts["used"] > m.counts["wasted"]


def test_store_without_gate_matches_multipath(tmp_path):
    ws = [np.full(8, float(e), dtype=np.float32) for e in range(4)]
    ops = [("p", 1), ("u", 0), ("u", 1), ("p", 2), ("p", 3), ("u", 0), ("u", 3), ("p", 1), ("u", 2), ("p", 0)]
    res = []
    for cls, sub in ((MultiPathExpertStore, "a"), (OutcomeTaggedStore, "b")):
        st = cls(ws, tmp_path / sub, 2, 0.0, n_paths=1)
        for op, e in ops:
            st.try_prefetch(e) if op == "p" else st.use_tracked(e)
        st.finish()
        res.append((st.demand_loads, st.prefetch_loads, st.wasted_loads, list(st.resident)))
    assert res[0] == res[1]


def test_store_reports_outcomes_and_late(tmp_path):
    ws = [np.full(8, float(e), dtype=np.float32) for e in range(4)]
    m = _model()
    st = OutcomeTaggedStore(ws, tmp_path, 2, 0.0, n_paths=1, gate=m)
    st.use_tracked(0)
    m.decide(m.prob(0, _h()), 0, st.is_resident_or_inflight)  # 1 を先読みと決める
    assert st.try_prefetch(1)
    st.use_tracked(1)  # 使われた（間に合った）
    assert m.counts["used"] == 1
    m.decide(m.prob(1, _h()), 1, st.is_resident_or_inflight)
    picked = next(iter(m.pending))
    st.try_prefetch(picked)
    st.use_tracked(3 if picked != 3 else 2)
    st.use_tracked(0)  # 2つ入れ替わると、使われなかった先読みは追い出される
    assert m.counts["wasted"] == 1 and st.wasted_loads == 1
    m.decide(m.prob(0, _h()), 0, st.is_resident_or_inflight)
    c = next(iter(m.pending))
    ev = st._claim(c)  # 読み込み中に使われると待つ（間に合わなかった）
    m.on_prefetch_start(c)
    import threading
    th = threading.Thread(target=st.use_tracked, args=(c,))
    th.start()
    st._load_claimed(c, ev, prefetched=True)
    th.join()
    assert m.counts["used_late"] == 1


def test_simulation_outcomes_match_store_waste(tmp_path):
    ch, hid = _trace(300, seed=3)
    ws = [np.full(8, float(e), dtype=np.float32) for e in range(4)]
    for k in range(3):
        m = OutcomeCerebellum(4, 4, lr=0.2, theta=0.0, window=2)
        st = OutcomeTaggedStore(ws, tmp_path / str(k), 2, 0.5, n_paths=2, gate=m)
        r = simulate_prefetch_v2(ch, hid, st, m, 1, 0.5, parallel_predictor=True, adaptive_gate=m)
        done = m.counts["used"] + m.counts["used_late"] + m.counts["wasted"]
        # 読み込みの始まった先読みの帰結は高々1回ずつ届き、届かないのは列の終わりに常駐していた分だけ
        assert done <= m.counts["started"] <= done + 2
        assert m.counts["started"] <= r.prefetch_loads
        assert m.counts["wasted"] <= r.wasted_loads <= m.counts["wasted"] + 2 + (r.prefetch_loads - m.counts["started"])


def test_conditions_parse_and_curve():
    conds = conditions([0.2, 0.4])
    assert _parse("outcomeshuf0.4") == ("outcomeshuf", 0.4)
    assert _parse("outcome0.2") == ("outcome", 0.2)
    assert _parse("cerebellum_efixed0.2") == ("cerebellum_efixed", 0.2)
    cc = curve_conds(conds)
    assert "cerebellum_efixed0.2" in cc and "cerebellum_fixed0.4" in cc
    assert not any(c.startswith("outcome") or c.startswith("cerebellum_delayed") for c in cc)

    def sp(wall, wasted):
        return {"wall_s": wall, "stalls": wall, "wasted_loads": wasted}

    speed = {c: sp(0.9, 50) for c in conds} | {"none": sp(1.0, 0), "cerebellum": sp(0.9, 100),
                                               "outcome0.2": sp(0.85, 50)}
    runs = [{"speed": {"small": speed}}]
    r = residuals(runs, conds, "small", "outcome0.2")
    assert np.isclose(r[0], 0.05)  # 曲線は 50 回で 10%、outcome は 15%


def test_select_all_rule():
    conds = conditions([0.2, 0.4])

    def d(wall, wasted):
        return {"wall_s": {"mean": wall}, "wasted_loads": {"mean": wasted}}

    s_size = {c: d(1.0, 200) for c in conds} | {"freq": d(1.0, 100), "outcome0.2": d(0.80, 150),
                                                "outcome0.4": d(0.85, 90), "cerebellum_efixed0.4": d(0.86, 80),
                                                "cerebellum_fixed0.4": d(0.88, 70)}
    sel = select_all({"small": s_size, "large": s_size})["small"]
    assert sel["outcome"] == "outcome0.4"  # freq 以下で最も速い
    assert sel["label_best"] == "cerebellum_efixed0.4"
    assert sel["outcomeshuf"] in ("outcomeshuf0.2", "outcomeshuf0.4")  # どれも freq 超 → 無駄 I/O 最小
