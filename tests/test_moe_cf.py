"""選ばれなかった専門家の反実仮想の学習信号でルータを学習する（12.6.157節）のテスト。"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from neurocortex.data import CharCorpus, synthetic_corpus
from neurocortex.experiments import run_moe_cf as R
from neurocortex.moe_cf import CounterfactualLM, forced_ce, train_cf
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


def test_default_matches_hier_model():
    for kind in ("plain", "gate", "select"):
        for layers in (1, 2):
            kw = {**KW, "kind": kind, "n_layers": layers, "probe": PROBE, "log_every": 2,
                  "brake_sched": {"rnn": (0.25, 0.25)}, "beta": 0.0}
            a, b = _make(HierBrakeLM, **kw), _make(CounterfactualLM, **kw)
            assert _train(a, train_hier) == _train(b, train_cf)
            _same(a, b)
            for p, q in zip(a.series, b.series):
                assert all(q[k] == v for k, v in p.items())


def test_forced_ce_matches_oracle():
    for kind in ("plain", "gate", "select"):
        m = _make(CounterfactualLM, kind=kind, eta=1.0)
        _train(m, train_cf, 6)
        x, y = PROBE[:, :-1], PROBE[:, 1:]
        with torch.no_grad():
            h, _ = m.rnn(m.embed(x))
            ce = forced_ce(m, h, y)
            r = m.run(x)
        o = oracle_assignment(m, x, y)
        assert ce.shape == (x.numel(), 8)
        assert abs(float(ce.min(-1).values.mean()) - o["ce_oracle"]) < 1e-6
        on = r["choice"][0].reshape(-1)
        base = F.cross_entropy(r["logits"].reshape(-1, V), y.reshape(-1), reduction="none")
        assert torch.allclose(ce.gather(-1, on.unsqueeze(-1)).squeeze(-1), base, atol=1e-5)


def test_teacher_grad_only_router():
    """補助損失の勾配はルータにだけ届く。教師の種類ごとの目標の形。"""
    x, y = PROBE[:, :-1], PROBE[:, 1:]
    for t in ("cf", "cfs", "dist", "shuf", "self"):
        m = _make(CounterfactualLM, kind="select", eta=1.0, teacher=t, lam=1.0)
        m.threshold[0] = torch.linspace(-1, 1, 8)
        loss = m.teacher_loss(x, y)
        loss.backward()
        assert loss.item() > 0 and m.routers[0].weight.grad is not None
        for p in (*m.embed.parameters(), *m.rnn.parameters(), *m.head.parameters(), *m.w1, *m.w2):
            assert p.grad is None
        assert 0 <= m.teacher_stats[-1]["agree"] <= 1


def test_teacher_targets():
    x, y = PROBE[:, :-1], PROBE[:, 1:]
    m = _make(CounterfactualLM, kind="select", eta=1.0, teacher="cf", lam=2.0)
    with torch.no_grad():
        h, _ = m.rnn(m.embed(x))
        raw = m.routers[0](h).reshape(-1, 8)
        best = forced_ce(m, h, y).argmin(-1)
    assert abs(m.teacher_loss(x, y).item() - 2.0 * F.cross_entropy(raw, best).item()) < 1e-5
    d = _make(CounterfactualLM, kind="select", eta=1.0, teacher="dist", lam=1.0)
    d.threshold[0] = torch.linspace(-3, 3, 8)
    tgt = (raw - d.threshold[0]).argmax(-1)
    assert abs(d.teacher_loss(x, y).item() - F.cross_entropy(raw, tgt).item()) < 1e-5
    s = _make(CounterfactualLM, kind="select", eta=1.0, teacher="shuf", lam=1.0, shuffle_seed=3)
    perm = torch.randperm(best.numel(), generator=torch.Generator().manual_seed(3))
    assert abs(s.teacher_loss(x, y).item() - F.cross_entropy(raw, best[perm]).item()) < 1e-5
    o = _make(CounterfactualLM, kind="select", eta=1.0, teacher="self", lam=1.0)
    o.threshold[0] = torch.linspace(-3, 3, 8)  # 閾値は使わない
    assert abs(o.teacher_loss(x, y).item() - F.cross_entropy(raw, raw.argmax(-1)).item()) < 1e-5


def test_teacher_changes_training_and_rejects():
    a = _make(CounterfactualLM, **KW)
    b = _make(CounterfactualLM, **KW, teacher="cf", lam=1.0)
    _train(a, train_cf)
    _train(b, train_cf)
    assert not torch.equal(a.routers[0].weight, b.routers[0].weight)
    assert len(b.teacher_stats) == 8 and not a.teacher_stats
    for kw in ({"teacher": "x"}, {"teacher": "cf", "n_layers": 2}):
        try:
            _make(CounterfactualLM, **kw)
        except ValueError:
            continue
        raise AssertionError(kw)


def test_conditions():
    c16, c32, c2 = R.conditions(16, 1), R.conditions(32, 1), R.conditions(32, 2)
    assert R.conditions(8, 1) == {}
    assert {"aux0", "aux0.05", "gate0.1", "sel1", "cf1", "cf0.1", "cfs1", "dist0.1", "shuf1", "self0.1", "p_cf1", "cp",
            "cp_cf1", "cp_dist0.1"} <= set(c32)
    assert c16["cf1"] == {"kind": "select", "eta": 0.3, "teacher": "cf", "lam": 1.0}
    assert c32["cp_cf1"]["freeze"] and c32["cp_cf1"]["open"] == 0.25 and "sched" not in c32["cp"]
    assert c32["p_cf1"]["kind"] == "plain"
    assert {"ord_hier", "ord_rev", "ord0_sim", "sel0.3"} <= set(c2) and "cf1" not in c2
    assert c2["ord0_rev"]["beta"] == 0.0 and c2["ord_rev"]["sched"]["experts"] == (0.125, 0.125)


def test_select_lam():
    s = {"aux0.05": {"base_ce": {"mean": 0.30}}}
    s.update({"cf0.1": {"base_ce": {"mean": 0.305}, "gini_off": {"mean": 0.2}},
              "cf1": {"base_ce": {"mean": 0.305}, "gini_off": {"mean": 0.1}}})
    assert R.select_lam(s, "cf") == 1.0
    s["cf1"]["base_ce"]["mean"] = 0.32
    assert R.select_lam(s, "cf") == 0.1
    s["cf0.1"]["base_ce"]["mean"] = 0.33
    assert R.select_lam(s, "cf") == 1.0


def test_end_to_end_small():
    torch.set_num_threads(1)
    args = argparse.Namespace(alpha=0.05, d=8, train_steps=16, seq_len=16, n_windows=4, n_probe=2, log_every=2,
                              k_num=1, k_den=4, n_random=5, eps=1e-4, tau=0.05, lam=[("cf@E16_L1", 1.0)], ref=None)
    corpus = CharCorpus(synthetic_corpus(n_sentences=60, seed=0))
    subs = {}
    for name, (e, l) in (("E16_L1", (16, 1)), ("E32_L2", (32, 2))):
        subs[name] = [{k: R.run_cond(s, e, l, k, c, corpus, args) for k, c in R.conditions(e, l).items()}
                      for s in (0, 1)]
    summ, v = R.analyze(subs, args)
    j = v["E16_L1"]
    assert j["選んだλ"]["cf"] == 1.0 and j["選んだλ"]["shuf"] == 1.0 and j["選んだλ"]["self"] == 1.0
    assert isinstance(j["H18e_反実仮想の情報が要る_cfがselfより偏らず一致率が高い"], bool)
    for k in ("H18a_反実仮想の信号で偏りとΔCE_offが下がる", "H18b_反実仮想の信号でH7*",
              "H18c_トークンの情報が要る_cfがshufより偏らず一致率が高い", "H19_蒸留で偏りとΔCE_offが下がる",
              "H18d_閉じた後の取り込み_cp_cf"):
        assert isinstance(j[k], bool)
    assert set(v["E32_L2"]) == {"β=0.1", "β=0"}
    assert isinstance(v["E32_L2"]["β=0"]["H20_逆順で偏りが小さい"], bool)
    assert "teacher_agree_last" in summ["E16_L1"]["cf1"] and "d_ce_off" in summ["E16_L1"]["sel0.3"]
