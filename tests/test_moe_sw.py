"""臨界期の間と閉じた後でルータの学習則を切り替える（12.6.185節）のテスト。"""

from __future__ import annotations

import argparse

import pytest
import torch
import torch.nn.functional as F

from neurocortex.data import CharCorpus, synthetic_corpus
from neurocortex.experiments import run_moe_sw as R
from neurocortex.moe_consol import ConsolLM, train_consol
from neurocortex.moe_sw import SwitchLM
from neurocortex.moe_tf import ThreeFactorLM

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


def test_default_matches_three_factor():
    """switch・soft_after なしでは `ThreeFactorLM` と同じ学習経過。"""
    for extra in ({}, R3, {**R3, "soft": True}, {**R3, "tf": True, "soft": True}):
        a, b = _make(ThreeFactorLM, **KW, **extra), _make(SwitchLM, **KW, **extra)
        assert _train(a) == _train(b)
        assert _same(a, b) and a.series == b.series


def test_switch_closing_matches_cl_gate():
    """ハードな教師で閉じ始めから切ると、12.6.182節の cl_gate（gate_detach）と同じ学習経過。"""
    a, b = _make(ConsolLM, **KW, **R3, gate_detach=True), _make(SwitchLM, **KW, **R3, switch="closing")
    assert _train(a) == _train(b)
    assert _same(a, b) and a.series == b.series


def test_open_period_matches_gate_gradient_rule():
    """閉じ始めまでは、ゲートの勾配で学ぶ条件（cp_rand3・cp_srand3）と同じで、閉じた後に分かれる。"""
    for ref_kw, sw_kw in (({**R3, "soft": True}, {**R3, "soft": True, "switch": "closing"}),
                          ({**R3, "soft": True}, {**R3, "soft": True, "switch": "closed"}),
                          (R3, {**R3, "soft_after": True, "switch": "closing"})):
        a, b = _make(SwitchLM, **KW, **ref_kw), _make(SwitchLM, **KW, **sw_kw)
        _train(a, 5)
        _train(b, 5)
        assert not a.closing and not b.closing and _same(a, b)
        a, b = _make(SwitchLM, **KW, **ref_kw), _make(SwitchLM, **KW, **sw_kw)
        _train(a)
        _train(b)
        assert not _same(a, b)
        assert torch.equal(a.snap["open"]["router"]["weight"], b.snap["open"]["router"]["weight"])


def _router_lm_grad(m) -> float:
    x, y = XY
    m.train()
    m.zero_grad(set_to_none=True)
    logits = m(x)[0]
    F.cross_entropy(logits.reshape(-1, V), y.reshape(-1)).backward()
    g = m.routers[0].weight.grad
    return 0.0 if g is None else float(g.abs().sum())


def test_detach_timing():
    """closing は閉じ始めから、closed は閉じ終わり（s = 0）から、LM の勾配をルータに送らない。"""
    m = _make(SwitchLM, **KW, **R3, switch="closing")
    assert _router_lm_grad(m) > 0.0
    m.closing, m.strength = True, 0.5
    assert m.detach_gate() and _router_lm_grad(m) == 0.0
    m = _make(SwitchLM, **KW, **R3, switch="closed")
    m.closing, m.strength = True, 0.5
    assert not m.detach_gate() and _router_lm_grad(m) > 0.0
    m.strength = 0.0
    assert m.detach_gate() and _router_lm_grad(m) == 0.0


def test_soft_after_switches_target():
    """soft_after は閉じ始めまでハードな目標、閉じ始めからソフトな目標（乱数の消費は同じ）。"""
    x, y = XY
    m = _make(SwitchLM, **KW, **R3, soft_after=True, switch="closing")
    hard, soft = _make(SwitchLM, **KW, **R3), _make(SwitchLM, **KW, **R3, soft=True)
    assert torch.equal(m.teacher_loss(x, y), hard.teacher_loss(x, y)) and not m.soft
    soft.teacher_loss(x, y)  # 乱数の消費をそろえる
    m.closing = soft.closing = True
    assert torch.equal(m.teacher_loss(x, y), soft.teacher_loss(x, y)) and m.soft
    assert torch.equal(m.sample_gen.get_state(), soft.sample_gen.get_state())


def test_invalid():
    with pytest.raises(ValueError):
        _make(SwitchLM, **KW, **R3, switch="later")
    with pytest.raises(ValueError):
        _make(SwitchLM, **KW, **R3, soft=True, soft_after=True)
    with pytest.raises(ValueError):
        _make(SwitchLM, **KW, teacher="cf", lam=1.0, soft_after=True)


def test_conditions():
    c = R.conditions(32, 1)
    assert set(c) == {"aux0", "aux0.05", "gate0.1", "cp", "cp_rand3", "cl_gate", "tf_rand3", "tf_srand3",
                      "cp_srand3", "sw_srand3", "sw_r3s", "sw_srand3_late"}
    assert c["sw_srand3"]["switch"] == "closing" and c["sw_srand3"]["soft"] and "tf" not in c["sw_srand3"]
    assert c["sw_r3s"]["soft_after"] and "soft" not in c["sw_r3s"]
    assert c["sw_srand3_late"]["switch"] == "closed" and c["cp_srand3"]["soft"] and "switch" not in c["cp_srand3"]
    assert c["cp_rand3"]["lam"] == 0.1 and R.conditions(8, 1) == {}


def test_small_run_and_judge():
    args = argparse.Namespace(d=8, alpha=0.05, train_steps=12, seq_len=16, n_windows=4, n_probe=2, log_every=4,
                              k_num=1, k_den=4, n_random=2, eps=1e-4, tau=0.05, rho=0.25)
    corpus = CharCorpus(synthetic_corpus(n_sentences=60, seed=0))
    subs = [{k: R.run_cond(seed, 16, 1, k, c, corpus, args) for k, c in R.conditions(16, 1).items()}
            for seed in (0, 1)]
    v = R.judge_scale(subs, R.conditions(16, 1), args)
    for k in ("D0_分解", "閉じる条件", "H39_閉じる前の取り込みA", "H40_閉じた後の取り込み",
              "H41_両者より偏らず学習を損なわない", "H42_漂流Dが土台より小さい",
              "H43_閉じた後の教師はソフトがハードより偏らない_sw_r3s対cl_gate", "参考_切り替えの時期_late対sw_srand3"):
        assert k in v, k
    for k in ("H39_閉じる前の取り込みA", "H40_閉じた後の取り込み", "H41_両者より偏らず学習を損なわない"):
        assert isinstance(v[k]["主_sw_srand3"], bool)
    # 閉じ始めまでは同じなので、閉じ始めから切る条件の A は、ゲートの勾配で学ぶ同じ教師の条件と同じ
    for s in subs:
        assert R.a_d(s["sw_srand3"])[0] == pytest.approx(R.a_d(s["cp_srand3"])[0])
        assert R.a_d(s["sw_r3s"])[0] == pytest.approx(R.a_d(s["cp_rand3"])[0])
