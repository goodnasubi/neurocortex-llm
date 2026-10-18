"""臨界期を閉じた後の可塑性のブレーキとトークンごとの取り込みの指標（12.6.149節）のテスト。"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from neurocortex.data import CharCorpus, synthetic_corpus
from neurocortex.experiments import run_moe_brake as R
from neurocortex.moe_brake import BrakedCriticalLM, adoption_stats, train_braked
from neurocortex.moe_critical import CriticalPeriodLM
from neurocortex.moe_scale import train_scaled

V = 12
PROBE = (torch.arange(64) % V).reshape(4, 16)


def _data():
    return torch.arange(500) % V


def _train(model, fn, steps=8):
    return fn(model, _data(), steps, 4, 16, 3e-3, torch.Generator().manual_seed(1),
              0.05 if model.kind == "plain" else 0.0)


def _pair(**kw):
    torch.manual_seed(0)
    a = CriticalPeriodLM(V, d=8, n_experts=8, d_ff=8, **kw)
    torch.manual_seed(0)
    b = BrakedCriticalLM(V, d=8, n_experts=8, d_ff=8, **kw)
    return a, b


def _braked(**kw):
    torch.manual_seed(0)
    return BrakedCriticalLM(V, d=8, n_experts=8, d_ff=8, **kw)


def test_default_matches_critical_model():
    for kind in ("plain", "gate", "select"):
        for layers in (1, 2):
            for sched in ({}, {"open_frac": 0.25, "ramp_frac": 0.25}):
                a, b = _pair(n_layers=layers, kind=kind, eta=0.1, total_steps=8, probe=PROBE, log_every=2, **sched)
                assert _train(a, train_scaled) == _train(b, train_braked)
                assert torch.equal(a.threshold, b.threshold) and torch.equal(a.threshold_state, b.threshold_state)
                for pa, pb in zip(a.parameters(), b.parameters()):
                    assert torch.equal(pa, pb)
                for p, q in zip(a.series, b.series):
                    assert all(q[k] == v for k, v in p.items())


def test_brake_one_is_no_brake():
    a = _braked(kind="select", eta=0.3, total_steps=8, open_frac=0.25, ramp_frac=0.25)
    b = _braked(kind="select", eta=0.3, total_steps=8, open_frac=0.25, ramp_frac=0.25, brake=1.0)
    assert _train(a, train_braked) == _train(b, train_braked)


def test_full_brake_freezes_router_after_close():
    kw = {"kind": "select", "eta": 0.3, "total_steps": 8, "open_frac": 0.25, "ramp_frac": 0.25, "brake": 0.0}
    a = _braked(**kw)
    b = _braked(**kw)
    _train(a, train_braked, 8)
    _train(b, train_braked, 4)  # 閉じ終わり（ステップ 4）まで
    assert a.router_lr_scale == 0.0
    for pa, pb in zip(a.routers.parameters(), b.routers.parameters()):
        assert torch.equal(pa, pb)  # 閉じた後はルータが動かない
    assert not torch.equal(a.w1[0], b.w1[0])  # 専門家は学習を続ける
    c = _braked(**{**kw, "brake": 0.5})
    _train(c, train_braked, 3)
    assert abs(c.router_lr_scale - 0.75) < 1e-12  # s(3) = 0.5 → 0.5 + 0.5·0.5


def test_freeze_stops_integration_when_closing_starts():
    kw = {"kind": "select", "eta": 0.3, "total_steps": 8, "open_frac": 0.5, "ramp_frac": 0.5, "freeze": True}
    a = _braked(**kw)
    b = _braked(**kw)
    _train(a, train_braked, 8)
    _train(b, train_braked, 4)
    assert torch.equal(a.threshold_state, b.threshold_state) and a.threshold_state.abs().sum() > 0
    assert torch.count_nonzero(a.threshold) == 0
    c = _braked(**kw)
    _train(c, train_braked, 6)
    assert torch.allclose(c.threshold, 0.5 * b.threshold_state)  # 閉じる間は s·θ（θ は開いている期間の終わりの値）


def test_mlp_router():
    torch.manual_seed(0)
    lin = BrakedCriticalLM(V, d=8, n_experts=8, d_ff=8, kind="select", eta=0.1)
    torch.manual_seed(0)
    mlp = BrakedCriticalLM(V, d=8, n_experts=8, d_ff=8, kind="select", eta=0.1, router_hidden=5, probe=PROBE)
    assert torch.equal(lin.w1[0], mlp.w1[0]) and torch.equal(lin.head.weight, mlp.head.weight)
    assert isinstance(mlp.routers[0], torch.nn.Sequential) and len(mlp.router_parameters()) == 4
    assert mlp.logit0.shape == (1, 8)
    w = mlp.routers[0][0].weight.clone()
    _train(mlp, train_braked, 3)
    assert not torch.equal(w, mlp.routers[0][0].weight)


def test_adoption_stats():
    on = np.array([1, 2, 3, 0, 0])
    off = np.array([0, 0, 0, 0, 1])
    later = np.array([1, 0, 4, 0, 0])
    a = adoption_stats(on, off, later, 6, "T")  # 変わったのは 0,1,2,4 番。採用 2/4、保持 1/4、他 1/4
    assert a["adopt_T"] == 0.5 and a["keep_T"] == 0.25 and a["chance_T"] == 0.25 / 4 and a["R_T"] == 8.0
    assert adoption_stats(on, on, later, 6, "lag")["adopt_lag"] is None
    assert adoption_stats(on, off, None, 6, "lag")["R_lag"] is None


def test_adoption_series():
    torch.manual_seed(0)
    m = BrakedCriticalLM(V, d=8, n_experts=8, d_ff=8, kind="select", eta=1.0, total_steps=8, probe=PROBE,
                         log_every=2, lag=4)
    _train(m, train_braked)
    m.adoption()
    steps = [p["step"] for p in m.series]
    assert steps == [1, 2, 4, 6, 8]
    for p in m.series:
        assert "adopt_T" in p and "shift" in p
        assert (p["adopt_lag"] is None) == (p["step"] + 4 not in steps or p["shift"] == 0)


def test_parse_and_conditions():
    c = R.parse_pb("pb1_o0.25_r0.25_frz_b0.1")
    assert c["freeze"] and c["brake"] == 0.1 and c["open"] == 0.25 and c["variant"] == "frz_b0.1"
    assert R.parse_pb("pb0.3_o0.5_r0.5_none")["brake"] is None and not R.parse_pb("pb0.3_o0.5_r0.5_b0")["freeze"]
    assert R.parse_pb("pb0.3_o0.5_r0.5_b0")["brake"] == 0.0 and R.parse_pb("pb1_o0.5_r0.5_frz")["brake"] is None
    a = argparse.Namespace(grid=True, pb=["32,1=pb1_o0.25_r0.25_none"], dissect=True)
    c16, c32, c8 = R.conditions(a, 16, 1), R.conditions(a, 32, 1), R.conditions(a, 8, 1)
    assert list(c8) == ["sel0.01"]
    assert sum(k.startswith("pb") for k in c16) == len(R.SCHEDULES) * len(R.VARIANTS) and "sel0.3" in c16
    assert c32["d96/sel1"]["d"] == 96 and c32["sel1_a0.01"]["alpha"] == 0.01 and "sel1_mlp96" not in c16
    assert R.close_step(c32["pb1_o0.25_r0.25_none"], 600) == 300 and R.close_step(c32["sel1"], 600) is None


def _cand(gmax, ok=True, h3=True):
    return {"臨界期型の合格_H1〜H3かつジニ≦0.3": ok, "H3_性能を損なわない": h3, "ジニ_シード最大": gmax}


def test_select_pb_rules():
    v = {"pb1_o0.5_r0.5_none": _cand(0.2), "pb1_o0.25_r0.25_frz_b0": _cand(0.2), "pb1_o0.25_r0.25_b0.1": _cand(0.2),
         "pb1_o0.25_r0.05_none": _cand(0.1, ok=False)}
    assert R.select_pb(v)[0] == "pb1_o0.25_r0.25_b0.1"  # 同点なら固定化期間が長く、手当てが弱い方
    v = {"pb1_o0.5_r0.5_none": _cand(0.4, ok=False, h3=False), "pb1_o0.25_r0.25_b0": _cand(0.5, ok=False)}
    k, why = R.select_pb(v)
    assert k == "pb1_o0.25_r0.25_b0" and "満たさない" in why


def test_end_to_end_small(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(R, "SCHEDULES", ((0.25, 0.25),))
    monkeypatch.setattr(R, "VARIANTS", ("none", "frz_b0.1"))
    monkeypatch.setattr(R, "CP_ETA", {"E16_L1": 0.3, "E32_L1": 1.0})
    monkeypatch.setattr(R, "DISSECT", {"sel1_mlp96": {"router_hidden": 4}})
    monkeypatch.setattr(R, "D_BIG", 12)
    args = argparse.Namespace(grid=True, pb=None, dissect=True, alpha=0.05, d=8, train_steps=8, seq_len=16,
                              n_windows=4, n_probe=2, log_every=2, lag=4, k_num=1, k_den=4, n_random=5, eps=1e-4)
    corpus = CharCorpus(synthetic_corpus(n_sentences=60, seed=0))
    subs = {}
    for name, (e, l) in (("E8_L1", (8, 1)), ("E32_L1", (32, 1))):
        subs[name] = [{k: R.run_cond(s, e, l, k, c, corpus, args) for k, c in R.conditions(args, e, l).items()}
                      for s in (0, 1)]
    assert "d12/sel1" in subs["E32_L1"][0]
    summ, v = R.analyze(subs, args)
    assert v["E32_L1"]["選択"]["キー"] in ("pb1_o0.25_r0.25_none", "pb1_o0.25_r0.25_frz_b0.1")
    assert v["E32_L1"]["H11"]["手当てあり"] == "pb1_o0.25_r0.25_frz_b0.1"
    assert set(v["E32_L1"]["H13_切り分け"]) == {"sel1", "sel1_mlp96", "d12/sel1"}
    assert "H14a_トークンごとの取り込みがある" in v["H14"] and "H14b_E32の取り込みが小さい" in v["H14"]
    assert "return" in summ["E32_L1"]["pb1_o0.25_r0.25_none"] and "adoption" in summ["E32_L1"]["sel1"]
