"""閉じた後に偏りが戻る原因の切り分けと、ルータのシナプス固定化（12.6.177節）のテスト。"""

from __future__ import annotations

import argparse

import pytest
import torch

from neurocortex.data import CharCorpus, synthetic_corpus
from neurocortex.experiments import run_moe_consol as R
from neurocortex.moe_consol import ConsolLM, cross_assignment, train_consol
from neurocortex.moe_react import ReactCFLM, train_react

V = 12
PROBE = (torch.arange(64) % V).reshape(4, 16)
KW = {"kind": "select", "eta": 0.3, "total_steps": 16, "open_frac": 0.25, "ramp_frac": 0.25, "freeze": True,
      "probe": PROBE, "log_every": 2}
XY = (PROBE[:, :-1], PROBE[:, 1:])


def _make(cls, **kw):
    torch.manual_seed(0)
    return cls(V, d=8, n_experts=8, d_ff=8, **kw)


def _train(model, fn, steps=16):
    return fn(model, torch.arange(500) % V, steps, 4, 16, 3e-3, torch.Generator().manual_seed(1),
              0.05 if model.kind == "plain" else 0.0)


def _same(a, b):
    assert torch.equal(a.threshold, b.threshold)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_default_matches_react():
    """手当てなしでは `ReactCFLM`＋`train_react` と同じ学習経過（教師なし・rand3）。"""
    for t in (None, "rand3"):
        kw = dict(KW)
        if t:
            kw.update(teacher=t, lam=1.0)
        a, b = _make(ReactCFLM, **kw), _make(ConsolLM, **kw)
        assert _train(a, train_react) == _train(b, train_consol)
        _same(a, b)
        assert a.series == b.series
        assert b.closing and set(b.snap) == {"open", "close", "end"}
        assert b.snap["open"]["step"] == 4 and b.snap["close"]["step"] == 8 and b.snap["open"]["strength"] == 1.0


def test_plain_matches_react():
    a, b = _make(ReactCFLM, kind="plain"), _make(ConsolLM, kind="plain")
    assert _train(a, train_react) == _train(b, train_consol)
    _same(a, b)


def test_fix_freezes_part_from_closing_start():
    base = _make(ConsolLM, **KW)
    _train(base, train_consol)
    for part in ("router", "front", "experts"):
        m = _make(ConsolLM, **KW, fix=(part,))
        _train(m, train_consol)
        snap = m.snap
        if part == "router":
            assert torch.equal(snap["close"]["router"]["weight"], snap["end"]["router"]["weight"])
            assert not torch.equal(snap["close"]["rnn"]["weight_hh_l0"], snap["end"]["rnn"]["weight_hh_l0"])
        if part == "front":
            assert torch.equal(snap["close"]["rnn"]["weight_hh_l0"], snap["end"]["rnn"]["weight_hh_l0"])
            assert torch.equal(snap["close"]["embed"]["weight"], snap["end"]["embed"]["weight"])
            assert not torch.equal(snap["close"]["router"]["weight"], snap["end"]["router"]["weight"])
        if part == "experts":
            assert torch.equal(snap["close"]["w1"], snap["end"]["w1"])
            assert not torch.equal(snap["close"]["rnn"]["weight_hh_l0"], snap["end"]["rnn"]["weight_hh_l0"])
        # 閉じ始めまでは同じ学習経過
        assert torch.equal(snap["open"]["w1"], base.snap["open"]["w1"])


def test_gate_detach_blocks_router_gradient_only_when_closing():
    m = _make(ConsolLM, **KW, gate_detach=True)
    x, y = XY
    m.train()

    def router_grad():
        m.zero_grad(set_to_none=True)
        logits, _, _ = m(x)
        torch.nn.functional.cross_entropy(logits.reshape(-1, V), y.reshape(-1)).backward()
        return m.routers[0].weight.grad

    assert router_grad() is not None and router_grad().abs().sum() > 0
    out_before = m.run(x)["logits"].detach()
    m.closing = True
    g = router_grad()
    assert g is None or float(g.abs().sum()) == 0.0
    assert m.w1[0].grad is not None and float(m.w1[0].grad.abs().sum()) > 0
    assert torch.allclose(m.run(x)["logits"], out_before)  # ゲートの値は同じ
    with pytest.raises(ValueError):
        _make(ConsolLM, kind="gate", eta=0.1, gate_detach=True)


def test_teacher_off_after_closing():
    a = _make(ConsolLM, **KW, teacher="rand3", lam=1.0)
    b = _make(ConsolLM, **KW, teacher="rand3", lam=1.0, teacher_off=True)
    _train(a, train_consol)
    _train(b, train_consol)
    assert len(b.teacher_stats) == 5  # 閉じ始め（6ステップ目）の前の5ステップだけ
    assert len(a.teacher_stats) == 16
    assert torch.equal(a.snap["open"]["router"]["weight"], b.snap["open"]["router"]["weight"])


def test_consolidation_masks_and_interpolation():
    for c in ("pnn", "meta", "stc"):
        m = _make(ConsolLM, **KW, teacher="rand3", lam=1.0, consol=c)
        _train(m, train_consol)
        f = torch.cat([v.reshape(-1) for v in m.f_mask.values()])
        assert f.min() >= 0.0 and f.max() <= 1.0
        if c == "pnn":
            assert torch.allclose(f, torch.full_like(f, 0.1))
        if c == "stc":
            assert set(f.unique().tolist()) <= {0.0, 1.0} and 0.3 < float(f.mean()) <= 0.6
        if c == "meta":
            om = torch.cat([v.reshape(-1) for v in m.omega.values()])
            assert float(om.min()) >= 0.0 and float(om.sum()) > 0
            assert torch.allclose(f, 1.0 / (1.0 + 9.0 * om / om.mean()))
        assert m.plasticity_mean() == pytest.approx(float(f.mean()))
    # 可塑性 0 の要素は閉じ終わり後に変わらない（stc）
    m = _make(ConsolLM, **KW, teacher="rand3", lam=1.0, consol="stc")
    _train(m, train_consol)
    w_close, w_end = m.snap["close"]["router"]["weight"], m.snap["end"]["router"]["weight"]
    fixed = m.f_mask["weight"] == 0
    assert torch.equal(w_close[fixed], w_end[fixed])
    assert not torch.equal(w_close[~fixed], w_end[~fixed])


def test_benna_fusi_chain():
    m = _make(ConsolLM, **KW, teacher="rand3", lam=1.0, consol="bf")
    base = _make(ConsolLM, **KW, teacher="rand3", lam=1.0)
    _train(m, train_consol)
    _train(base, train_consol)
    assert set(m.chain) == {"weight", "bias"} and m.plasticity_mean() is None
    assert torch.equal(m.snap["open"]["router"]["weight"], base.snap["open"]["router"]["weight"])
    assert not torch.equal(m.snap["end"]["router"]["weight"], base.snap["end"]["router"]["weight"])
    # 隠れた変数は閉じ始めの値から動き、u3 は u2 より遅い
    w_open = m.snap["open"]["router"]["weight"]
    u2, u3 = m.chain["weight"]
    assert float((u3 - w_open).abs().sum()) < float((u2 - w_open).abs().sum())


def test_cross_assignment():
    m = _make(ConsolLM, **KW, teacher="rand3", lam=1.0)
    _train(m, train_consol)
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    c = cross_assignment(m, PROBE)
    for k, v in m.state_dict().items():  # 状態を戻す
        assert torch.equal(v, sd[k])
    assert set(c["gini"]) == {"open", "close", "end"}
    # 対角（end, end）は学習の終わりの閾値なしの選択と同じ
    from neurocortex.moe_scale import load_stats
    _, off = load_stats(m, PROBE, use_threshold=False)
    assert c["gini"]["end"]["end"] == pytest.approx(off["gini"])
    assert 0.0 <= c["agree_open"] <= 1.0 and c["dormant_end"] >= 0 and c["expert_change"] > 0
    assert set(c["gate"]) == {"open", "close", "end"}


def test_conditions():
    c = R.conditions(32, 1)
    assert c["cp_rand3"]["lam"] == 0.1 and c["d_router"]["fix"] == ("router",) and c["d_teacher"]["teacher_off"]
    assert "teacher" not in c["cp_d_front"] and "cp_d_teacher" not in c and c["c_meta"]["consol"] == "meta"
    assert len(c) == 5 + 6 + 5 + 4 and R.conditions(8, 1) == {}


def test_small_run_and_judge():
    args = argparse.Namespace(d=8, alpha=0.05, train_steps=12, seq_len=16, n_windows=4, n_probe=2, log_every=4,
                              k_num=1, k_den=4, n_random=2, eps=1e-4, tau=0.05, rho=0.25)
    corpus = CharCorpus(synthetic_corpus(n_sentences=60, seed=0))
    subs = [{k: R.run_cond(seed, 16, 1, k, c, corpus, args) for k, c in R.conditions(16, 1).items()}
            for seed in (0, 1)]
    v = R.judge_scale(subs, R.conditions(16, 1), args)
    for k in ("D0_分解", "D1〜D6_切り分け_cp_rand3", "切り分け_参考_cp", "H30_固定化で閉じた後の取り込み",
              "H31_偏りを減らし学習を損なわない", "H32_重要度に応じた固定化が一様な安定化より偏りを減らす",
              "H33_ルータの完全な固定の効果の半分以上", "戻り_プローブ", "可塑性の平均"):
        assert k in v, k
    assert isinstance(v["H30_固定化で閉じた後の取り込み"]["合格"], bool)
    assert isinstance(v["D0_分解"]["cp_rand3"]["Aが漂流以上"], bool)
    assert R.summarize_consol(subs, "c_pnn")["all_used_seeds"] >= 0
