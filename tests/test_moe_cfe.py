"""一部の専門家だけを評価する反実仮想の教師と、価値を予測するルータ（12.6.165節）のテスト。"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from neurocortex.data import CharCorpus, synthetic_corpus
from neurocortex.experiments import run_moe_cfe as R
from neurocortex.moe_cf import CounterfactualLM, forced_ce, train_cf
from neurocortex.moe_cfe import EstimatedCFLM, _dup_mask, forced_ce_gate1, parse_teacher, subset_ce

V = 12
PROBE = (torch.arange(64) % V).reshape(4, 16)
KW = {"kind": "select", "eta": 0.3, "total_steps": 8, "open_frac": 0.25, "ramp_frac": 0.25, "freeze": True}
XY = (PROBE[:, :-1], PROBE[:, 1:])


def _train(model, steps=8):
    return train_cf(model, torch.arange(500) % V, steps, 4, 16, 3e-3, torch.Generator().manual_seed(1),
                    0.05 if model.kind == "plain" else 0.0)


def _make(cls, **kw):
    torch.manual_seed(0)
    return cls(V, d=8, n_experts=8, d_ff=8, **kw)


def _same(a, b):
    assert torch.equal(a.threshold, b.threshold)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_parse_teacher():
    assert parse_teacher("top3") == ("top", 3) and parse_teacher("rand1") == ("rand", 1)
    assert parse_teacher("val0") == ("val", 0) and parse_teacher("valall") == ("valall", 0)
    assert parse_teacher("cfg") == ("cfg", 0) and parse_teacher("cf") == ("cf", 0)
    assert parse_teacher("dval1") == ("dval", 1) and parse_teacher("dvalall") == ("dvalall", 0)
    for t in ("top0", "x", "rand", "val-1", "dval0"):
        try:
            parse_teacher(t)
        except ValueError:
            continue
        raise AssertionError(t)


def test_default_and_base_teachers_match_cf_model():
    """教師なし・既存の教師では `CounterfactualLM` と同じ学習経過。"""
    for t in (None, "cf", "cfs"):
        kw = {**KW, "probe": PROBE, "log_every": 2}
        if t:
            kw.update(teacher=t, lam=1.0)
        a, b = _make(CounterfactualLM, **kw), _make(EstimatedCFLM, **kw)
        assert _train(a) == _train(b)
        _same(a, b)
    assert b.cost["evals"] / b.cost["tokens"] == 8


def test_subset_ce_matches_forced_ce():
    for kind in ("select", "gate"):
        m = _make(EstimatedCFLM, kind=kind, eta=1.0)
        _train(m, 6)
        x, y = XY
        with torch.no_grad():
            h, _ = m.rnn(m.embed(x))
            full, full1 = forced_ce(m, h, y), forced_ce_gate1(m, h, y)
            idx = torch.randint(0, 8, (x.numel(), 3), generator=torch.Generator().manual_seed(0))
            assert torch.allclose(subset_ce(m, h, y, idx), full.gather(-1, idx), atol=1e-5)
            assert torch.allclose(subset_ce(m, h, y, idx, gate_one=True), full1.gather(-1, idx), atol=1e-5)
            r = m.run(x, gate_one=True)
            on = r["choice"][0].reshape(-1)
            base = F.cross_entropy(r["logits"].reshape(-1, V), y.reshape(-1), reduction="none")
            assert torch.allclose(full1.gather(-1, on.unsqueeze(-1)).squeeze(-1), base, atol=1e-5)


def test_dup_mask():
    idx = torch.tensor([[2, 2, 5], [1, 3, 1]])
    assert _dup_mask(idx).tolist() == [[False, True, False], [False, False, True]]


def test_partial_teachers_grad_and_cost():
    """補助損失の勾配はルータ（と価値の予測の基準）にだけ届き、top・rand は評価しなかった専門家の対数に勾配を送らない。"""
    x, y = XY
    n = y.numel()
    for t, lo, hi in (("top1", 1, 2), ("top3", 1, 4), ("rand1", 2, 2), ("rand3", 4, 4), ("val0", 1, 1),
                      ("val1", 2, 2), ("valall", 8, 8), ("cfg", 8, 8), ("dval1", 2, 2), ("dvalall", 8, 8)):
        m = _make(EstimatedCFLM, kind="select", eta=1.0, teacher=t, lam=1.0)
        m.threshold[0] = torch.linspace(-1, 1, 8)
        loss = m.teacher_loss(x, y)
        loss.backward()
        assert m.routers[0].weight.grad is not None
        for p in (*m.embed.parameters(), *m.rnn.parameters(), *m.head.parameters(), *m.w1, *m.w2):
            assert p.grad is None
        ev = m.cost["evals"] / m.cost["tokens"]
        assert lo <= ev <= hi and m.cost["tokens"] == n, (t, ev)
        assert hasattr(m, "value_w") == t.startswith("val")
    m = _make(EstimatedCFLM, kind="select", eta=1.0, teacher="rand1", lam=1.0)
    with torch.no_grad():
        h, _ = m.rnn(m.embed(x))
    raw = m.routers[0](h).reshape(-1, 8).detach().requires_grad_(True)
    m.routers[0] = _Const(raw)
    m.teacher_loss(x, y).backward()
    assert ((raw.grad != 0).sum(-1) <= 2).all()  # 評価した2つだけ


class _Const(torch.nn.Module):
    def __init__(self, raw):
        super().__init__()
        self.raw = raw

    def forward(self, h):
        return self.raw.reshape(*h.shape[:-1], -1)


def test_partial_targets():
    x, y = XY
    m = _make(EstimatedCFLM, kind="select", eta=1.0, teacher="top2", lam=2.0)
    m.threshold[0] = torch.linspace(-3, 3, 8)
    with torch.no_grad():
        h, _ = m.rnn(m.embed(x))
        raw = m.routers[0](h).reshape(-1, 8)
        full = forced_ce(m, h, y)
    c = (raw - m.threshold[0]).argmax(-1)
    idx = torch.cat([c.unsqueeze(-1), raw.topk(2, -1).indices], -1)
    dup = _dup_mask(idx)
    pos = full.gather(-1, idx).masked_fill(dup, float("inf")).argmin(-1)
    want = 2.0 * F.cross_entropy(raw.gather(-1, idx).masked_fill(dup, float("-inf")), pos)
    assert abs(m.teacher_loss(x, y).item() - want.item()) < 1e-5
    g = _make(EstimatedCFLM, kind="select", eta=1.0, teacher="cfg", lam=1.0)
    best = forced_ce_gate1(g, h, y).argmin(-1)
    assert abs(g.teacher_loss(x, y).item() - F.cross_entropy(raw, best).item()) < 1e-5


def test_value_prediction_target():
    """価値の予測: raw_e と b(h) − CE_e/τ の smooth L1（評価した専門家だけ）。b はゼロ初期化。"""
    x, y = XY
    m = _make(EstimatedCFLM, kind="select", eta=1.0, teacher="val0", lam=1.0, tau=0.5)
    with torch.no_grad():
        h, _ = m.rnn(m.embed(x))
        raw = m.routers[0](h).reshape(-1, 8)
        full = forced_ce(m, h, y)
    c = raw.argmax(-1)  # 閾値は 0
    want = F.smooth_l1_loss(raw.gather(-1, c.unsqueeze(-1)), -full.gather(-1, c.unsqueeze(-1)) / 0.5)
    assert abs(m.teacher_loss(x, y).item() - want.item()) < 1e-5
    a = _make(EstimatedCFLM, kind="select", eta=1.0, teacher="valall", lam=1.0, tau=0.5)
    assert abs(a.teacher_loss(x, y).item() - F.smooth_l1_loss(raw, -full / 0.5).item()) < 1e-5
    a.teacher_loss(x, y).backward()
    assert a.value_w.grad is not None and a.value_b.grad is not None


def test_difference_prediction_target():
    """修正1: (raw_e − raw_c) と −(CE_e − CE_c)/τ の smooth L1。全専門家の形は c の項が 0。"""
    x, y = XY
    m = _make(EstimatedCFLM, kind="select", eta=1.0, teacher="dvalall", lam=1.0, tau=0.5)
    m.threshold[0] = torch.linspace(-2, 2, 8)
    with torch.no_grad():
        h, _ = m.rnn(m.embed(x))
        raw = m.routers[0](h).reshape(-1, 8)
        full = forced_ce(m, h, y)
    c = (raw - m.threshold[0]).argmax(-1, keepdim=True)
    want = F.smooth_l1_loss(raw - raw.gather(-1, c), -(full - full.gather(-1, c)) / 0.5)
    assert abs(m.teacher_loss(x, y).item() - want.item()) < 1e-5
    d = _make(EstimatedCFLM, kind="select", eta=1.0, teacher="dval1", lam=1.0, tau=0.5, sample_seed=4)
    d.threshold[0] = m.threshold[0]
    with torch.no_grad():
        o = torch.rand(c.numel(), 8, generator=torch.Generator().manual_seed(4)).scatter_(1, c, -1.0).topk(1).indices
    want = F.smooth_l1_loss(raw.gather(-1, o) - raw.gather(-1, c), -(full.gather(-1, o) - full.gather(-1, c)) / 0.5)
    assert abs(d.teacher_loss(x, y).item() - want.item()) < 1e-5


def test_training_changes_and_rejects():
    a = _make(EstimatedCFLM, **KW)
    b = _make(EstimatedCFLM, **KW, teacher="val1", lam=1.0)
    _train(a)
    _train(b)
    assert not torch.equal(a.routers[0].weight, b.routers[0].weight) and len(b.teacher_stats) == 8
    for kw in ({"teacher": "top0"}, {"teacher": "rand1", "n_layers": 2}):
        try:
            _make(EstimatedCFLM, **kw)
        except ValueError:
            continue
        raise AssertionError(kw)


def test_conditions_and_lam():
    c16, c32 = R.conditions(16, 1), R.conditions(32, 1)
    assert R.conditions(32, 2) == {}
    assert {"aux0", "aux0.05", "gate0.1", "sel1", "cf", "top1", "top3", "rand1", "rand3", "cfg", "val0_0.1",
            "val1_1", "valall_1", "dval1_1", "dval3_0.1", "dvalall_1", "cp", "cp_cf", "cp_rand1", "cp_cfg",
            "cp_val1_0.1", "cp_dval1_1"} <= set(c32)
    assert c16["cf"]["lam"] == 1.0 and c32["cf"]["lam"] == 0.1 and c32["top1"]["lam"] == 0.1
    assert c16["val1_0.1"] == {"kind": "select", "eta": 0.3, "teacher": "val1", "lam": 0.1}
    assert c32["cp_top1"]["freeze"] and c32["cp_top1"]["open"] == 0.25
    s = {"aux0.05": {"base_ce": {"mean": 0.30}},
         "val1_0.1": {"base_ce": {"mean": 0.305}, "gini_off": {"mean": 0.2}},
         "val1_1": {"base_ce": {"mean": 0.305}, "gini_off": {"mean": 0.1}}}
    assert R.select_val_lam(s, "val1") == 1.0
    s["val1_1"]["base_ce"]["mean"] = 0.32
    assert R.select_val_lam(s, "val1") == 0.1


def test_end_to_end_small():
    torch.set_num_threads(1)
    args = argparse.Namespace(alpha=0.05, d=8, train_steps=12, seq_len=16, n_windows=4, n_probe=2, log_every=2,
                              k_num=1, k_den=4, n_random=4, eps=1e-4, tau=0.05, lam=[("val1@E16_L1", 1.0)], ref=None)
    corpus = CharCorpus(synthetic_corpus(n_sentences=60, seed=0))
    subs = {"E16_L1": [{k: R.run_cond(s, 16, 1, k, c, corpus, args) for k, c in R.conditions(16, 1).items()}
                       for s in (0, 1)]}
    summ, v = R.analyze(subs, args)
    j = v["E16_L1"]
    assert j["選んだλ_価値の予測"]["val1"] == 1.0 and j["条件のキー"]["val1"] == "val1_1"
    assert set(j["H21_一部の評価で取り込みが保たれる"]) == {"top1", "top3", "rand1", "rand3"}
    for k in ("H22_価値の予測で取り込みが保たれる_val1", "H22'_実際の結果だけ_val0", "H22_修正1_差の予測_dval1"):
        assert isinstance(j[k], bool)
    assert isinstance(j["H24_ゲートに依らない教師"]["合格"], bool)
    assert "cp_val1_1" in j["H23_閉じた後の取り込み"]
    assert j["費用"]["rand1"]["evals_per_token"] == 2.0 and j["費用"]["cf"]["evals_per_token"] == 16.0
    assert summ["E16_L1"]["sel0.3"]["cost"]["evals_per_token"]["mean"] == 0.0
