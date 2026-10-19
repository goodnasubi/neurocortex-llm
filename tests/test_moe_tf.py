"""ルータの学習を三要素則だけにする（12.6.181節）のテスト。"""

from __future__ import annotations

import argparse

import pytest
import torch
import torch.nn.functional as F

from neurocortex.data import CharCorpus, synthetic_corpus
from neurocortex.experiments import run_moe_tf as R
from neurocortex.moe_cfe import _dup_mask, subset_ce
from neurocortex.moe_consol import ConsolLM, train_consol
from neurocortex.moe_tf import ThreeFactorLM, subset_ce_eq

V = 12
PROBE = (torch.arange(64) % V).reshape(4, 16)
KW = {"kind": "select", "eta": 0.3, "total_steps": 16, "open_frac": 0.25, "ramp_frac": 0.25, "freeze": True,
      "probe": PROBE, "log_every": 2}
XY = (PROBE[:, :-1], PROBE[:, 1:])


def _make(cls, **kw):
    torch.manual_seed(0)
    return cls(V, d=8, n_experts=8, d_ff=8, **kw)


def _train(model, steps=16):
    return train_consol(model, torch.arange(500) % V, steps, 4, 16, 3e-3, torch.Generator().manual_seed(1), 0.0)


def test_default_matches_consol():
    """tf・soft なしでは `ConsolLM`＋`train_consol` と同じ学習経過（教師なし・rand3・cf・閉じ始めからの gate_detach）。"""
    for extra in ({}, {"teacher": "rand3", "lam": 1.0}, {"teacher": "cf", "lam": 1.0},
                  {"teacher": "rand3", "lam": 1.0, "gate_detach": True}):
        a, b = _make(ConsolLM, **KW, **extra), _make(ThreeFactorLM, **KW, **extra)
        assert _train(a) == _train(b)
        assert torch.equal(a.threshold, b.threshold)
        for pa, pb in zip(a.parameters(), b.parameters()):
            assert torch.equal(pa, pb)
        assert a.series == b.series


def _grads(m, loss):
    m.zero_grad(set_to_none=True)
    loss.backward()
    return {n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p)) for n, p in m.named_parameters()}


def test_tf_blocks_lm_gradient_to_router_from_start():
    x, y = XY
    m = _make(ThreeFactorLM, **KW, teacher="rand3", lam=1.0, tf=True)
    ref = _make(ThreeFactorLM, **KW, teacher="rand3", lam=1.0)
    m.train()
    assert not m.closing
    logits = m(x)[0]
    assert torch.allclose(logits, ref(x)[0])  # ゲートの値は同じ
    g = _grads(m, F.cross_entropy(logits.reshape(-1, V), y.reshape(-1)))
    assert float(g["routers.0.weight"].abs().sum()) == 0.0 and float(g["routers.0.bias"].abs().sum()) == 0.0
    assert float(g["w1.0"].abs().sum()) > 0 and float(g["rnn.weight_hh_l0"].abs().sum()) > 0
    # 教師の損失はルータにだけ届く
    g = _grads(m, m.teacher_loss(x, y))
    assert float(g["routers.0.weight"].abs().sum()) > 0
    assert all(float(v.abs().sum()) == 0.0 for n, v in g.items() if not n.startswith("routers"))
    with pytest.raises(ValueError):
        _make(ThreeFactorLM, kind="gate", eta=0.1, tf=True)


def test_teacher_gradient_is_local_three_factor_rule():
    """線形のルータでは教師の勾配が ΔW_j ∝ Σ_n (t_nj − p_nj)·h_n（シナプス前 × (教える信号 − シナプス後)）。"""
    x, y = XY
    for soft in (False, True):
        m = _make(ThreeFactorLM, **KW, teacher="rand3", lam=0.7, tf=True, soft=soft)
        gen = m.sample_gen.get_state()
        g = _grads(m, m.teacher_loss(x, y))
        m.sample_gen.set_state(gen)
        with torch.no_grad():
            h = m.rnn(m.embed(x))[0].reshape(-1, 8)
            raw = m.routers[0](h)
            chosen = (raw - m.threshold[0]).argmax(-1)
            r = torch.rand(chosen.numel(), 8, generator=m.sample_gen)
            r.scatter_(1, chosen.unsqueeze(-1), -1.0)
            idx = torch.cat([chosen.unsqueeze(-1), r.topk(3, dim=-1).indices], dim=-1)
            ce = subset_ce(m, h.reshape(4, 15, 8), y, idx)
            dup = _dup_mask(idx)
            if soft:
                t_sub = F.softmax((-ce / m.tau).masked_fill(dup, float("-inf")), dim=-1)
            else:
                t_sub = F.one_hot(ce.masked_fill(dup, float("inf")).argmin(-1), 4).float()
            t = torch.zeros(len(h), 8).scatter_add_(1, idx, t_sub.masked_fill(dup, 0.0))
            p = F.softmax(raw.gather(-1, idx).masked_fill(dup, float("-inf")), -1).masked_fill(dup, 0.0)
            p_full = torch.zeros(len(h), 8).scatter_add_(1, idx, p)
            dw = -0.7 * (t - p_full).T @ h / len(h)
        assert torch.allclose(g["routers.0.weight"], dw, atol=1e-6)


def test_eq_gate_evaluates_with_chosen_gate():
    x, y = XY
    m = _make(ThreeFactorLM, **KW, teacher="rand3", lam=1.0, tf=True, eq_gate=True)
    h = m.rnn(m.embed(x))[0].detach()
    idx = torch.stack([torch.arange(60) % 8, (torch.arange(60) + 3) % 8], -1)
    a, b = subset_ce_eq(m, h, y, idx), subset_ce(m, h, y, idx)
    assert torch.allclose(a[:, 0], b[:, 0]) and not torch.allclose(a[:, 1], b[:, 1])
    # 同じ専門家なら同じ CE（ゲートがそろう）
    same = torch.stack([torch.arange(60) % 8, torch.arange(60) % 8], -1)
    c = subset_ce_eq(m, h, y, same)
    assert torch.allclose(c[:, 0], c[:, 1])
    a0 = _make(ThreeFactorLM, **KW, teacher="rand3", lam=1.0)
    a0.teacher_loss(x, y)
    m.teacher_loss(x, y)
    assert torch.equal(a0.sample_gen.get_state(), m.sample_gen.get_state())
    _train(m)


def test_soft_uses_same_evaluations_and_random_stream():
    x, y = XY
    a = _make(ThreeFactorLM, **KW, teacher="rand3", lam=1.0)
    b = _make(ThreeFactorLM, **KW, teacher="rand3", lam=1.0, soft=True)
    a.teacher_loss(x, y)
    b.teacher_loss(x, y)
    assert torch.equal(a.sample_gen.get_state(), b.sample_gen.get_state())
    assert a.cost["evals"] == b.cost["evals"] and len(b.teacher_stats) == 1
    with pytest.raises(ValueError):
        _make(ThreeFactorLM, **KW, teacher="cf", lam=1.0, soft=True)


def test_tf_none_router_never_changes_and_tf_router_learns():
    m = _make(ThreeFactorLM, **KW, tf=True)
    w0 = m.routers[0].weight.detach().clone()
    _train(m)
    assert torch.equal(m.routers[0].weight, w0)
    assert set(m.snap) == {"open", "close", "end"}
    m = _make(ThreeFactorLM, **KW, teacher="rand3", lam=1.0, tf=True)
    _train(m)
    assert not torch.equal(m.routers[0].weight, w0)


def test_conditions():
    c = R.conditions(32, 1)
    assert set(c) == {"aux0", "aux0.05", "gate0.1", "cp", "cp_rand3", "cl_gate", "tf_rand3", "tf_srand3", "tf_cf",
                      "tf_none", "tf_rand3e", "cp_rand3e"}
    assert c["tf_rand3e"]["eq_gate"] and c["tf_rand3e"]["tf"] and "tf" not in c["cp_rand3e"]
    assert c["tf_rand3"]["tf"] and c["tf_rand3"]["lam"] == 0.1 and c["tf_srand3"]["soft"]
    assert c["tf_cf"]["teacher"] == "cf" and "teacher" not in c["tf_none"] and c["cl_gate"]["gate_detach"]
    assert R.conditions(8, 1) == {}


def test_small_run_and_judge():
    args = argparse.Namespace(d=8, alpha=0.05, train_steps=12, seq_len=16, n_windows=4, n_probe=2, log_every=4,
                              k_num=1, k_den=4, n_random=2, eps=1e-4, tau=0.05, rho=0.25)
    corpus = CharCorpus(synthetic_corpus(n_sentences=60, seed=0))
    subs = [{k: R.run_cond(seed, 16, 1, k, c, corpus, args) for k, c in R.conditions(16, 1).items()}
            for seed in (0, 1)]
    v = R.judge_scale(subs, R.conditions(16, 1), args)
    for k in ("D0_分解", "閉じる条件", "H34_閉じる前の取り込みAが下がる", "H35_閉じた後の取り込み",
              "H36_偏りを減らし学習を損なわない", "H37_漂流Dが小さい", "H38_教える信号の寄与_tf_rand3がtf_noneより偏らない",
              "戻り_プローブ", "教師との一致率_最後の50ステップ"):
        assert k in v, k
    assert isinstance(v["H35_閉じた後の取り込み"]["合格"], bool)
    assert isinstance(v["H34_閉じる前の取り込みAが下がる"]["主_tf_rand3"], bool)
    a, d = R.a_d(subs[0]["tf_rand3"])
    c = subs[0]["tf_rand3"]["cross"]
    assert a == pytest.approx(c["gini"]["open"]["open"] - c["on_open"])
