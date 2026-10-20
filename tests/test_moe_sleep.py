"""臨界期を閉じる直前の睡眠に似た再生による、閾値の割り当てのルータのシナプスへの転写（12.6.221節）のテスト。"""

from __future__ import annotations

import argparse

import pytest
import torch

from neurocortex.experiments import run_moe_sleep as R
from neurocortex.moe_consol import ConsolLM, train_consol
from neurocortex.moe_sleep import SleepLM, slept_assignment

V = 12
PROBE = (torch.arange(64) % V).reshape(4, 16)
KW = {"kind": "select", "eta": 0.3, "total_steps": 16, "open_frac": 0.25, "ramp_frac": 0.25, "freeze": True,
      "probe": PROBE, "log_every": 2, "teacher": "rand3", "lam": 0.1}


def _make(cls, **kw):
    torch.manual_seed(0)
    return cls(V, d=8, n_experts=8, d_ff=8, **kw)


def _train(model, steps=16):
    return train_consol(model, torch.arange(500) % V, steps, 4, 16, 3e-3, torch.Generator().manual_seed(1), 0.0)


def _same(a, b):
    assert torch.equal(a.threshold, b.threshold)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_default_matches_consol():
    """睡眠・恒久化なしでは `ConsolLM` と同じ学習経過。"""
    a, b = _make(ConsolLM, **KW), _make(SleepLM, **KW)
    assert _train(a) == _train(b)
    _same(a, b)
    assert a.series == b.series and not b.replay and "slept" not in b.snap


def test_sleep_only_touches_router_and_matches_up_to_closing():
    """睡眠は閉じ始めに一度だけ入り、それまでは一度に閉じる対照と同じ。睡眠ではルータだけが変わり、sleep_w は偏りを動かさない。"""
    kw = {**KW, "ramp_frac": 0.0}
    a, b = _make(SleepLM, **kw), _make(SleepLM, sleep="w", sleep_buffer=3, sleep_steps=20, **kw)
    _train(a, 4), _train(b, 4)  # 開いている期間（4 ステップ）。まだ閉じていない
    _same(a, b)
    assert not b.closing and len(b.replay) == 3
    front = {k: v.clone() for k, v in b.state_dict().items() if not k.startswith("routers")}
    bias, weight = b.routers[0].bias.clone(), b.routers[0].weight.clone()
    b.begin_closing()
    assert "slept" in b.snap and b.sleep_stats["n_replay"] == 3
    assert torch.equal(b.routers[0].bias, bias) and not torch.equal(b.routers[0].weight, weight)
    for k, v in b.state_dict().items():
        if not k.startswith("routers"):
            assert torch.equal(v, front[k]), k
    assert all(p.grad is None for p in b.routers[0].parameters())


def test_sleep_wb_moves_bias_and_reaches_threshold_choice():
    """W と b を学ぶ睡眠は、再生した入力で閾値ありの選択に近づく（一致率が上がる）。"""
    kw = {**KW, "ramp_frac": 0.0}
    m = _make(SleepLM, sleep="wb", sleep_buffer=4, sleep_steps=300, sleep_lr=1e-2, **kw)
    m.threshold_state[0].copy_(torch.linspace(-2, 2, 8))  # 大きな閾値で閾値の有無の選択を分ける
    _train(m, 4)
    bias = m.routers[0].bias.clone()
    m.begin_closing()
    st = m.sleep_stats
    assert not torch.equal(m.routers[0].bias, bias)
    assert st["agree_after"] > st["agree_before"] and st["agree_after"] > 0.8
    assert st["loss_last"] < st["loss_first"]


def test_sleep_perm_uses_permuted_target_and_changes_result():
    kw = {**KW, "ramp_frac": 0.0}
    a = _make(SleepLM, sleep="w", sleep_buffer=2, sleep_steps=10, **kw)
    b = _make(SleepLM, sleep="perm", sleep_buffer=2, sleep_steps=10, **kw)
    for m in (a, b):
        m.threshold_state[0].copy_(torch.linspace(-1, 1, 8))
        _train(m, 4)
        m.begin_closing()
    assert not torch.equal(a.routers[0].weight, b.routers[0].weight)


def test_fold_keeps_threshold_choice_after_closing():
    """恒久化: 閉じる間に下がった分の θ を偏りに移すので、閉じ終わっても閾値ありの選択（θ 全部）が残る。"""
    m = _make(SleepLM, fold=True, **KW)
    b0 = m.routers[0].bias.clone()
    _train(m, 16)
    assert m.strength == 0.0 and torch.count_nonzero(m.threshold) == 0
    shift = m.routers[0].bias - b0
    assert shift.abs().sum() > 0
    # 閉じ始めから θ の積分は止まっているので、偏りの合計の移動は −θ_state になる（Adam の更新の分を除いて近い）
    base = _make(SleepLM, **KW)
    _train(base, 16)
    folded = m.routers[0].bias - base.routers[0].bias
    assert torch.allclose(folded, -m.threshold_state[0], atol=0.1)


def test_slept_assignment_and_validation():
    kw = {**KW, "ramp_frac": 0.0}
    m = _make(SleepLM, sleep="w", sleep_buffer=2, sleep_steps=5, **kw)
    _train(m, 8)
    r = slept_assignment(m, PROBE)
    assert set(r) == {"gini_slept_off", "used", "agree_with_on_open"} and 0 <= r["agree_with_on_open"] <= 1
    assert slept_assignment(_make(SleepLM, **KW), PROBE) == {}
    with pytest.raises(ValueError):
        _make(SleepLM, sleep="x", **KW)
    with pytest.raises(ValueError):
        _make(SleepLM, sleep="w", fold=True, **KW)
    with pytest.raises(ValueError):
        _make(SleepLM, kind="gate", eta=0.1, sleep="w")


def test_conditions_and_small_run():
    c = R.conditions()
    assert len(c) == 12 and c["abrupt"]["ramp"] == 0.0 and c["sleep_w"]["sleep"] == "w"
    assert c[R.BASE]["ramp"] == 0.25 and c["fold"]["fold"] and "teacher" not in c["cp_sleep_w"]
    from neurocortex.data import CharCorpus, synthetic_corpus
    corpus = CharCorpus(synthetic_corpus(n_sentences=60, seed=0))
    args = argparse.Namespace(tau=0.05, rho=0.25, alpha=0.05, d=8, train_steps=8, seq_len=16, n_windows=4,
                              n_probe=2, log_every=2, k_num=1, k_den=4, n_random=2, eps=1e-4, sleep_buffer=2,
                              sleep_steps=3, sleep_lr=3e-3)
    sub = []
    for seed in (0, 1):
        sub.append({k: R.run_cond(seed, k, cond, corpus, args) for k, cond in c.items()})
    assert sub[0]["sleep_w"]["slept"] and not sub[0]["abrupt"]["slept"]
    v = R.judge(sub, args)
    for k in ("H50_主_睡眠で閉じた後の取り込み", "H51_睡眠で取り込みの不足が減る", "H52_閾値の内容による", "H53_学習を損なわない"):
        assert isinstance(v[k]["合格"], bool)
