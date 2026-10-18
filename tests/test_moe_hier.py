"""可塑性のブレーキの範囲と時期・臨界期の階層的な順序・CE を下げる割り当て（12.6.153節）のテスト。"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from neurocortex.data import CharCorpus, synthetic_corpus
from neurocortex.experiments import run_moe_hier as R
from neurocortex.moe_brake import BrakedCriticalLM, train_braked
from neurocortex.moe_hier import HierBrakeLM, oracle_assignment, train_hier

V = 12
PROBE = (torch.arange(64) % V).reshape(4, 16)
KW = {"kind": "select", "eta": 0.3, "total_steps": 8, "open_frac": 0.25, "ramp_frac": 0.25, "freeze": True}


def _data():
    return torch.arange(500) % V


def _train(model, fn, steps=8):
    return fn(model, _data(), steps, 4, 16, 3e-3, torch.Generator().manual_seed(1),
              0.05 if model.kind == "plain" else 0.0)


def _make(cls, **kw):
    torch.manual_seed(0)
    return cls(V, d=8, n_experts=8, d_ff=8, **kw)


def _same(a, b):
    assert torch.equal(a.threshold, b.threshold) and torch.equal(a.threshold_state, b.threshold_state)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_default_matches_braked_model():
    for kind in ("plain", "gate", "select"):
        for layers in (1, 2):
            kw = {**KW, "kind": kind, "n_layers": layers, "probe": PROBE, "log_every": 2}
            a, b = _make(BrakedCriticalLM, **kw), _make(HierBrakeLM, **kw)
            assert _train(a, train_braked) == _train(b, train_hier)
            _same(a, b)
            for p, q in zip(a.series, b.series):
                assert all(q[k] == v for k, v in p.items())


def test_router_scope_matches_braked_router():
    """ルータだけを閾値と同じ閉じ方で止めると、12.6.149節の brake=β と同じ学習経過。入力にも掛けると brake_front と同じ。"""
    a = _make(BrakedCriticalLM, **KW, brake=0.1)
    b = _make(HierBrakeLM, **KW, brake_sched={"router": (0.25, 0.25)}, beta=0.1)
    assert _train(a, train_braked) == _train(b, train_hier)
    _same(a, b)
    s = (0.25, 0.25)
    c = _make(BrakedCriticalLM, **KW, brake=0.1, brake_front=True)
    d = _make(HierBrakeLM, **KW, brake_sched={"router": s, "rnn": s, "embed": s}, beta=0.1)
    assert _train(c, train_braked) == _train(d, train_hier)
    _same(c, d)


def test_group_brake_freezes_only_that_group():
    sched = {"rnn": (0.0, 0.25)}  # GRU だけ、ステップ 2 で閉じ終わる
    a = _make(HierBrakeLM, **KW, brake_sched=sched, beta=0.0)
    b = _make(HierBrakeLM, **KW, brake_sched=sched, beta=0.0)
    _train(a, train_hier, 8)
    _train(b, train_hier, 2)
    assert a.group_lr_scale["rnn"] == 0.0
    for pa, pb in zip(a.rnn.parameters(), b.rnn.parameters()):
        assert torch.equal(pa, pb)
    for m in ("embed", "head"):
        assert not torch.equal(getattr(a, m).weight, getattr(b, m).weight)
    assert not torch.equal(a.w1[0], b.w1[0]) and not torch.equal(a.routers[0].weight, b.routers[0].weight)
    c = _make(HierBrakeLM, **KW, brake_sched={"experts": (0.25, 0.5)}, beta=0.5)
    _train(c, train_hier, 4)  # s(4) = 1 − (4 − 2)/4 = 0.5 → 0.5 + 0.5·0.5
    assert abs(c.group_lr_scale["experts"] - 0.75) < 1e-12


def test_unknown_group_rejected():
    try:
        _make(HierBrakeLM, **KW, brake_sched={"head": (0.0, 0.1)})
    except ValueError:
        return
    raise AssertionError


def test_order_schedules():
    h, r, s = R.order_sched("hier"), R.order_sched("rev"), R.order_sched("sim")
    assert h["embed"] == (0.125, 0.125) and h["experts"] == (0.5, 0.125) and h["router"] == (0.375, 0.125)
    assert r["experts"] == h["embed"] and r["embed"] == h["experts"] and r["router"] == h["rnn"]
    assert all(v == (0.3125, 0.125) for v in s.values()) and len(s) == 4
    assert R.scope_sched("RGM") == {"router": (0.25, 0.25), "rnn": (0.25, 0.25), "embed": (0.25, 0.25)}
    assert R.scope_sched("none") == {}
    c8, c32 = R.conditions(8, 1), R.conditions(32, 1)
    assert list(c8) == ["sel0.01"]
    assert {"aux0", "aux0.05", "gate0.1", "sel1", "sc_RGMX", "ord_hier"} <= set(c32)
    assert c32["sc0_RGM"]["beta"] == 0.0 and c32["sc_RGM"]["beta"] == 0.1 and "ord0_hier" in c32
    assert c32["sc_R"]["freeze"] and c32["sc_R"]["open"] == 0.25 and R.conditions(16, 1)["ord_rev"]["eta"] == 0.3


def test_oracle_assignment_consistent():
    for kind in ("plain", "gate", "select"):
        m = _make(HierBrakeLM, kind=kind, eta=1.0)
        _train(m, train_hier, 6)
        x, y = PROBE[:, :-1], PROBE[:, 1:]
        o = oracle_assignment(m, x, y)
        with torch.no_grad():
            r = m.run(x)
        ce = float(F.cross_entropy(r["logits"].reshape(-1, V), y.reshape(-1)))
        assert abs(o["ce_on"] - ce) < 1e-5  # 選ばれた専門家に強制した CE は通常の順伝播と一致
        assert o["ce_oracle"] <= o["ce_on"] + 1e-7 and o["regret_on"] >= -1e-7
        assert 0 <= o["oracle_gini"] <= 1 and 1 <= o["oracle_n_used"] <= 8
        if kind == "plain":  # ゲート込みは閾値を外すと選択も変わる
            assert o["shift_frac"] == 0 and o["shift_cost"] is None and o["ce_on"] == o["ce_off"]


def test_end_to_end_small():
    torch.set_num_threads(1)
    args = argparse.Namespace(beta=0.1, alpha=0.05, d=8, train_steps=16, seq_len=16, n_windows=4, n_probe=2,
                              log_every=2, lag=4, k_num=1, k_den=4, n_random=5, eps=1e-4)
    corpus = CharCorpus(synthetic_corpus(n_sentences=60, seed=0))
    subs = {}
    for name, (e, l) in (("E8_L1", (8, 1)), ("E32_L1", (32, 1))):
        subs[name] = [{k: R.run_cond(s, e, l, k, c, corpus, args) for k, c in R.conditions(e, l).items()}
                      for s in (0, 1)]
    summ, v = R.analyze(subs, args)
    assert set(v["E32_L1"]) == {"β=0.1", "β=0"}
    assert subs["E32_L1"][0]["sc0_R"] != subs["E32_L1"][0]["sc_R"]
    j = v["E32_L1"]["β=0.1"]
    for k in ("H16a_入力の安定化が要る", "H16b_入力だけでルータ以上に戻りを防ぐ", "H16c_専門家のブレーキは戻りに効かない",
              "H17a_階層的な順序で学習の終わりの偏りが小さい", "H17b_階層的な順序で戻りが小さい_≦0.02",
              "H17c_階層的な順序で取り込み"):
        assert isinstance(j[k], bool)
    assert set(j["戻り_平均"]) == {f"sc_{s}" for s in R.SCOPES} | {f"ord_{o}" for o in R.ORDERS}
    assert "H15a_CEを下げる割り当ての偏りがE32で大きい" in v["H15"]
    assert "return" in summ["E32_L1"]["ord_hier"] and "oracle" in summ["E32_L1"]["aux0"]
