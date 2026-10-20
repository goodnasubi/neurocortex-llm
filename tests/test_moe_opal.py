"""OpAL 型の G（Go）・N（NoGo）の2系統で専門家を選ぶルータ（12.6.207節）のテスト。"""

from __future__ import annotations

import argparse
import math

import pytest
import torch
import torch.nn.functional as F

from neurocortex.data import CharCorpus, synthetic_corpus
from neurocortex.experiments import run_moe_opal as R
from neurocortex.moe_cfe import _dup_mask, _random_others, subset_ce
from neurocortex.moe_consol import train_consol
from neurocortex.moe_opal import OpALLM, OpponentRouter
from neurocortex.moe_sw import SwitchLM

V = 12
PROBE = (torch.arange(64) % V).reshape(4, 16)
KW = {"kind": "select", "eta": 0.3, "total_steps": 16, "open_frac": 0.25, "ramp_frac": 0.25, "freeze": True,
      "probe": PROBE, "log_every": 2}
R3 = {"teacher": "rand3", "lam": 1.0}
XY = (PROBE[:, :-1], PROBE[:, 1:])


def _make(cls, **kw):
    torch.manual_seed(0)
    return cls(V, d=8, n_experts=8, d_ff=8, **kw)


def _train(model, steps=16):
    return train_consol(model, torch.arange(500) % V, steps, 4, 16, 3e-3, torch.Generator().manual_seed(1), 0.0)


def _same(a, b) -> bool:
    return all(torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters()))


def test_default_matches_switch():
    """opal なしでは `SwitchLM` と同じ学習経過。"""
    for extra in ({}, R3, {**R3, "tf": True, "soft": True}, {**R3, "soft": True, "switch": "closing"}):
        a, b = _make(SwitchLM, **KW, **extra), _make(OpALLM, **KW, **extra)
        assert _train(a) == _train(b)
        assert _same(a, b) and a.series == b.series


def test_router_is_g_minus_n():
    h = torch.randn(3, 8)
    for act, g0 in (("sigmoid", 0.5), ("softplus", 1.0)):
        r = OpponentRouter(8, 5, 0.0, torch.Generator().manual_seed(0), act, 4.0)
        assert torch.allclose(r(h), torch.zeros(3, 5))  # 初期は G = N
        g, n = r.pathways(h)
        assert torch.allclose(g, torch.full((3, 5), g0)) and torch.allclose(n, torch.full((3, 5), g0))
    assert r.bg[0].item() == pytest.approx(math.log(math.e - 1))
    r = OpponentRouter(8, 5, 3.0, torch.Generator().manual_seed(0), "sigmoid", 4.0)
    g, n = r.pathways(h)
    assert torch.allclose(r(h), 4.0 * (g - n)) and (g > 0).all() and (n > 0).all()
    assert (r(h).abs() < 4.0).all()  # 発火率の飽和で選択の強さは ±β に収まる


def test_hebbian_update_matches_rule():
    """更新は Δw^G_k = α·mean_n δ_nk·G_k(h_n)·h_n、Δw^N_k = −α·mean_n δ_nk·N_k(h_n)·h_n（評価した専門家だけ）。"""
    x, y = XY
    m = _make(OpALLM, **KW, **R3, tf=True, opal="gn", opal_lr=0.5, opal_init=0.1)
    gen_state = m.sample_gen.get_state()
    assert float(m.teacher_loss(x, y)) == 0.0
    with torch.no_grad():
        h = m.rnn(m.embed(x))[0]
        hf = h.reshape(-1, 8)
        r = m.routers[0]
        chosen = (r(hf) - m.threshold[0]).argmax(-1)
        gen = torch.Generator()
        gen.set_state(gen_state)
        idx = torch.cat([chosen.unsqueeze(-1), _random_others(chosen, 8, 3, gen)], -1)
        ce = subset_ce(m, h, y, idx)
        dup = _dup_mask(idx)
        g, n = r.pathways(hf)
        wg, wn = torch.zeros(8, 8), torch.zeros(8, 8)
        for t in range(hf.shape[0]):
            ok = [j for j in range(4) if not dup[t, j]]
            v = sum(float(ce[t, j]) for j in ok) / len(ok)
            for j in ok:
                k = int(idx[t, j])
                d = v - float(ce[t, j])
                wg[k] += 0.5 * d * g[t, k] * hf[t] / hf.shape[0]
                wn[k] -= 0.5 * d * n[t, k] * hf[t] / hf.shape[0]
    assert torch.allclose(m.pending["wg"], wg, atol=1e-6) and torch.allclose(m.pending["wn"], wn, atol=1e-6)
    before = {k: p.clone() for k, p in m.routers[0].named_parameters()}
    m.homeostatic_step(torch.zeros(1, 4, 15, dtype=torch.long))
    after = dict(m.routers[0].named_parameters())
    assert torch.allclose(after["wg"], before["wg"] + wg, atol=1e-6) and m.pending is None
    assert torch.equal(m.sample_gen.get_state(), gen.get_state())


def test_opposite_directions():
    """同じトークンで δ > 0 なら G は上がり N は下がる（D1 は LTP、D2 は LTD）。"""
    m = _make(OpALLM, **KW, **R3, tf=True, opal="gn", opal_lr=1.0, opal_init=0.1)
    x, y = XY
    m.teacher_loss(x, y)
    p = m.pending
    # bg と bn の更新は符号が逆で、大きさは G と N の比（初期は G ≈ N）に従う
    nz = p["bg"].abs() > 1e-8
    assert nz.any() and (torch.sign(p["bg"][nz]) == -torch.sign(p["bn"][nz])).all()


def test_router_gets_no_lm_gradient_and_g_only_freezes_n():
    for kind in ("gn", "g"):
        m = _make(OpALLM, **KW, **R3, tf=True, opal=kind)
        n0 = (m.routers[0].wn.clone(), m.routers[0].bn.clone())
        g0 = m.routers[0].wg.clone()
        _train(m)
        r = m.routers[0]
        assert all(p.grad is None for p in r.parameters())
        assert not torch.equal(r.wg, g0)
        assert torch.equal(r.wn, n0[0]) == (kind == "g") and torch.equal(r.bn, n0[1]) == (kind == "g")
        assert len(m.opal_stats) == 16


def test_same_evaluations_as_tf_srand3():
    """評価する専門家と乱数の消費は tf_srand3 と同じ（学習則だけが違う）。前段の学習も同じ（ルータから勾配が来ない）。"""
    a = _make(OpALLM, **KW, **R3, tf=True, opal="gn")
    b = _make(SwitchLM, **KW, **R3, tf=True, soft=True)
    _train(a, 3)
    _train(b, 3)
    assert torch.equal(a.sample_gen.get_state(), b.sample_gen.get_state())


def test_invalid():
    with pytest.raises(ValueError):
        _make(OpALLM, **KW, **R3, tf=True, opal="x")
    with pytest.raises(ValueError):
        _make(OpALLM, **KW, **R3, tf=True, opal="gn", opal_act="relu")
    with pytest.raises(ValueError):
        _make(OpALLM, **KW, **R3, opal="gn")  # tf なし
    with pytest.raises(ValueError):
        _make(OpALLM, **KW, **R3, tf=True, soft=True, opal="gn")
    with pytest.raises(ValueError):
        _make(OpALLM, **KW, teacher="top3", lam=1.0, tf=True, opal="gn")


def test_conditions_and_resid():
    c = R.conditions(32, 1)
    assert set(c) == {"aux0", "aux0.05", "gate0.1", "cp_rand3", "tf_srand3", "sw_srand3", "opal", "opal_g",
                      "opal_cf", "opal_cfc"}
    assert c["opal"]["opal"] == "gn" and c["opal_g"]["opal"] == "g" and c["opal"]["tf"]
    assert c["opal"]["opal_delta"] == c["opal_g"]["opal_delta"] == "critic" and c["opal"]["critic_lr"] == 1.0
    assert c["opal_cf"]["opal_delta"] == "cf" and c["opal_cfc"]["opal_delta"] == "cfcritic"
    assert R.conditions(8, 1) == {}
    assert R.resid(0.0, R.LINE[0]) == pytest.approx(0.0)
    assert R.resid(0.1, 0.0) == pytest.approx(-(R.LINE[0] + R.LINE[1] * 0.1))


def test_small_run_and_judge():
    args = argparse.Namespace(d=8, alpha=0.05, train_steps=12, seq_len=16, n_windows=4, n_probe=2, log_every=4,
                              k_num=1, k_den=4, n_random=2, eps=1e-4, tau=0.05, rho=0.25, opal_lr=0.1,
                              opal_init=None, opal_act="sigmoid", opal_beta=4.0,
                              opal_delta="cfcritic", critic_lr=1.0)
    corpus = CharCorpus(synthetic_corpus(n_sentences=60, seed=0))
    subs = [{k: R.run_cond(seed, 16, 1, k, c, corpus, args) for k, c in R.conditions(16, 1).items()}
            for seed in (0, 1)]
    v = R.judge_scale(subs, R.conditions(16, 1), args)
    for k in ("D0_分解", "閉じる条件", "A_D_残差", "H44_閉じる前の取り込みA", "H45_閉じた後の漂流D",
              "主_AとDを同時に満たす_H44かつH45", "H46_補償関係の直線より下", "H47_対立（N）が要る",
              "H48_偏りを減らし学習を損なわない", "H49_閉じた後の取り込み_H1〜H3かつジニ≦0.3"):
        assert k in v, k
    assert isinstance(v["主_AとDを同時に満たす_H44かつH45"]["opal"], bool)
    assert "opal_stats" in subs[0]["opal"] and "opal_stats" not in subs[0]["cp_rand3"]


def test_critic_delta_updates_chosen_only():
    """修正1: 実際に選んだ専門家だけを δ = −CE_c − V(h) で更新し、クリティックも同じ δ で学ぶ。"""
    x, y = XY
    m = _make(OpALLM, **KW, **R3, tf=True, opal="gn", opal_lr=0.5, opal_delta="critic")
    gen_state = m.sample_gen.get_state()
    m.teacher_loss(x, y)
    with torch.no_grad():
        h = m.rnn(m.embed(x))[0]
        hf = h.reshape(-1, 8)
        r = m.routers[0]
        chosen = (r(hf) - m.threshold[0]).argmax(-1)
        gen = torch.Generator()
        gen.set_state(gen_state)
        idx = torch.cat([chosen.unsqueeze(-1), _random_others(chosen, 8, 3, gen)], -1)
        d = -subset_ce(m, h, y, idx)[:, 0]  # V = 0 で初期化
        g, n = r.pathways(hf)
        oh = F.one_hot(chosen, 8).float()
        wg = 0.5 * ((d.unsqueeze(-1) * g * oh).T @ hf) / hf.shape[0]
        wn = -0.5 * ((d.unsqueeze(-1) * n * oh).T @ hf) / hf.shape[0]
    assert torch.allclose(m.pending["wg"], wg, atol=1e-6) and torch.allclose(m.pending["wn"], wn, atol=1e-6)
    assert torch.allclose(m.pending_critic["critic_b"], 0.5 * d.mean())
    m.homeostatic_step(torch.zeros(1, 4, 15, dtype=torch.long))
    assert float(m.critic_b) == pytest.approx(float(0.5 * d.mean())) and m.pending_critic is None
    with pytest.raises(ValueError):
        _make(OpALLM, **KW, **R3, tf=True, opal="gn", opal_delta="td")


def test_critic_nlms_step():
    """修正2: クリティックのステップは critic_lr / mean(|h|² + 1)。"""
    x, y = XY
    m = _make(OpALLM, **KW, **R3, tf=True, opal="gn", opal_delta="cfcritic", critic_lr=1.0)
    m.teacher_loss(x, y)
    with torch.no_grad():
        hf = m.rnn(m.embed(x))[0].reshape(-1, 8)
        d = m.pending_critic["critic_b"] * float((hf.pow(2).sum(-1) + 1.0).mean()) / 0.05  # = mean δ
        assert torch.allclose(m.pending_critic["critic_b"], 0.05 * d / float((hf.pow(2).sum(-1) + 1.0).mean()))
        assert float(d) < 0.0  # 初めは V = 0 なので δ = −CE < 0


def test_cfcritic_updates_all_evaluated():
    """修正3: 評価したすべての専門家を δ_k = −CE_k − V(h) で更新し、クリティックは選んだ専門家の δ だけで学ぶ。"""
    x, y = XY
    a = _make(OpALLM, **KW, **R3, tf=True, opal="gn", opal_delta="cfcritic", critic_lr=1.0)
    b = _make(OpALLM, **KW, **R3, tf=True, opal="gn", opal_delta="critic", critic_lr=1.0)
    a.teacher_loss(x, y)
    b.teacher_loss(x, y)
    assert torch.equal(a.pending_critic["critic_b"], b.pending_critic["critic_b"])
    assert (a.pending["bg"] != 0).sum() > (b.pending["bg"] != 0).sum()
    # 初めは V = 0 なので δ_k = −CE_k < 0 で、評価したすべての専門家で G は下がり N は上がる
    nz = a.pending["bg"] != 0
    assert (a.pending["bg"][nz] < 0).all() and (a.pending["bn"][nz] > 0).all()
