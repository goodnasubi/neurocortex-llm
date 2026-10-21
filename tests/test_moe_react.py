"""教師の割り当ての恒常性の閾値と、閉じる前の使われない専門家の再学習（12.6.171節）のテスト。"""

from __future__ import annotations

import argparse

import torch

from neurocortex.data import CharCorpus, synthetic_corpus
from neurocortex.experiments import run_moe_react as R
from neurocortex.moe_cf import train_cf
from neurocortex.moe_cfe import EstimatedCFLM, subset_ce
from neurocortex.moe_react import ReactCFLM, train_react

V = 12
PROBE = (torch.arange(64) % V).reshape(4, 16)
KW = {"kind": "select", "eta": 0.3, "total_steps": 8, "open_frac": 0.25, "ramp_frac": 0.25, "freeze": True}
XY = (PROBE[:, :-1], PROBE[:, 1:])


def _make(cls, **kw):
    torch.manual_seed(0)
    return cls(V, d=8, n_experts=8, d_ff=8, **kw)


def _train(model, fn, steps=8):
    return fn(model, torch.arange(500) % V, steps, 4, 16, 3e-3, torch.Generator().manual_seed(1),
              0.05 if model.kind == "plain" else 0.0)


def _same(a, b):
    assert torch.equal(a.threshold, b.threshold)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_default_matches_estimated_cf():
    """手当てなしでは `EstimatedCFLM`＋`train_cf` と同じ学習経過（教師なし・部分評価・全専門家の教師）。"""
    for t in (None, "rand3", "top1", "cf"):
        kw = {**KW, "probe": PROBE, "log_every": 2}
        if t:
            kw.update(teacher=t, lam=1.0)
        a, b = _make(EstimatedCFLM, **kw), _make(ReactCFLM, **kw)
        assert _train(a, train_cf) == _train(b, train_react)
        _same(a, b)
        assert a.cost["evals"] == b.cost["evals"]
        assert [s["agree"] for s in a.teacher_stats] == [s["agree"] for s in b.teacher_stats]
        assert a.series == b.series


def test_teacher_threshold_shifts_target_and_integrates():
    m = _make(ReactCFLM, **KW, teacher="rand3", lam=1.0, teacher_homeo=True)
    x, y = XY
    gen_state = m.sample_gen.get_state()
    m.teacher_threshold.copy_(torch.linspace(-50, 50, 8))  # 大きな閾値: 番号の小さい専門家が教師に選ばれる
    m.teacher_loss(x, y)
    st = m.teacher_stats[-1]
    assert 0.0 <= st["t_gini"] <= 1.0 and 0.0 <= st["t_gini_raw"] <= 1.0
    # 同じ評価の組で、閾値を掛けた目標を手で計算して積分則を確かめる
    m2 = _make(ReactCFLM, **KW, teacher="rand3", lam=1.0, teacher_homeo=True)
    m2.sample_gen.set_state(gen_state)
    th = torch.linspace(-50, 50, 8)
    with torch.no_grad():
        h, _ = m2.rnn(m2.embed(x))
        raw0 = m2.routers[0](h).reshape(-1, 8)
        chosen = (raw0 - m2.threshold[0]).argmax(-1)
        r = torch.rand(chosen.numel(), 8, generator=m2.sample_gen)
        r.scatter_(1, chosen.unsqueeze(-1), -1.0)
        idx = torch.cat([chosen.unsqueeze(-1), r.topk(3, dim=-1).indices], dim=-1)
        ce = subset_ce(m2, h, y, idx)
        hard = idx.gather(-1, (ce / m2.tau + th[idx]).argmin(-1, keepdim=True)).squeeze(-1)
    frac = torch.bincount(hard, minlength=8).float() / hard.numel()
    u = torch.full((8,), 1 / 8) * (1 - m.alpha) + m.alpha * frac
    exp = th + m.eta * (8 * u - 1)
    assert torch.allclose(m.teacher_usage, u)
    assert torch.allclose(m.teacher_threshold, exp - exp.mean(), atol=1e-5)
    assert abs(float(m.teacher_threshold.mean())) < 1e-5


def test_teacher_homeo_changes_training_and_rejects_full_teacher():
    a = _make(ReactCFLM, **KW, teacher="rand3", lam=1.0)
    b = _make(ReactCFLM, **KW, teacher="rand3", lam=1.0, teacher_homeo=True)
    _train(a, train_react, 30)
    _train(b, train_react, 30)
    assert not torch.equal(b.teacher_threshold, torch.zeros(8))
    assert not all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    for t in ("cf", "cfs", None):
        try:
            _make(ReactCFLM, **KW, teacher=t, lam=1.0, teacher_homeo=True)
        except ValueError:
            continue
        raise AssertionError(t)


def test_react_gradient_only_reaches_dormant_experts():
    m = _make(ReactCFLM, **KW, react=True)
    x, y = XY
    with torch.no_grad():
        m.off_usage.copy_(torch.tensor([0.0, 0.5, 0.01, 0.2, 0.1, 0.1, 0.05, 0.04]))  # ρ/E = 0.03125
    loss = m.react_loss(x, y)
    assert m.react_stats[-1]["dormant"] == 2 and m.react_stats[-1]["active"]
    assert float(loss.detach()) > 0
    loss.backward()
    for name, p in m.named_parameters():
        if name.startswith("w1") or name.startswith("w2"):
            g = p.grad.abs().sum(dim=(1, 2))
            assert (g[[0, 2]] > 0).all() and (g[[1, 3, 4, 5, 6, 7]] == 0).all(), name
        else:
            assert p.grad is None or float(p.grad.abs().sum()) == 0.0, name
    assert m.cost["react_evals"] == 2 * y.numel()


def test_react_loss_matches_forced_ce_on_niche():
    """再学習の損失 = 取り分のトークン（CE_c − CE_e の上位 N/E 個）を c のゲートで e に送ったときの CE の和 / N。"""
    m = _make(ReactCFLM, **KW, react=True)
    x, y = XY
    with torch.no_grad():
        m.off_usage.copy_(torch.tensor([0.0, 0.5, 0.2, 0.1, 0.1, 0.05, 0.03, 0.02]))
        m.off_usage[6] = 0.2  # 使われないのは 0 と 7
    loss = float(m.react_loss(x, y).detach())
    with torch.no_grad():
        h, _ = m.rnn(m.embed(x))
        raw0 = m.routers[0](h).reshape(-1, 8)
        chosen = (raw0 - m.threshold[0]).argmax(-1)
        n = y.numel()
        ce_c = subset_ce(m, h, y, chosen.unsqueeze(-1)).squeeze(-1)
        probs = torch.softmax(raw0, -1)
        exp = 0.0
        for e in (0, 7):
            hf = h.reshape(-1, 8)
            ye = (hf @ m.w1[0][e]).relu() @ m.w2[0][e]
            gate = probs.gather(-1, chosen.unsqueeze(-1))
            ce_e = torch.nn.functional.cross_entropy(m.head(hf + gate * ye), y.reshape(-1), reduction="none")
            exp += float(ce_e[(ce_c - ce_e).topk(n // 8).indices].sum()) / n
    assert abs(loss - exp) < 1e-5


def test_react_stops_after_close_and_changes_training():
    m = _make(ReactCFLM, **KW, react=True)
    m.strength = 0.0  # 閉じ終わった後
    with torch.no_grad():
        m.off_usage.zero_()
    loss = m.react_loss(*XY)
    assert float(loss) == 0.0 and not m.react_stats[-1]["active"]
    # 学習では、閾値が閉じ終わるまでだけ再学習が働く（閉じ方 (0.25, 0.25) の 8 ステップでは 4 ステップ目まで）
    a = _make(ReactCFLM, **{**KW, "total_steps": 40}, react=True, react_rho=8.0)  # ρ/E = 1: ほぼ全専門家が対象
    _train(a, train_react, 40)
    act = [s for s in a.react_stats if s["active"]]
    assert len(act) == 20 and all(s["step"] <= 20 for s in act)
    b = _make(ReactCFLM, **{**KW, "total_steps": 40})
    _train(b, train_react, 40)
    assert not all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def test_conditions():
    c = R.conditions(32, 1)
    assert c["rand3"] == {"kind": "select", "eta": 1.0, "teacher": "rand3", "lam": 0.1}
    assert c["rand3_ht"]["teacher_homeo"] and "react" not in c["rand3_ht"]
    assert c["cp_rand3_htre"]["teacher_homeo"] and c["cp_rand3_htre"]["react"] and c["cp_rand3_htre"]["freeze"]
    assert c["cp_re"]["react"] and "teacher" not in c["cp_re"]
    assert c["cp_rand3"]["open"] == 0.25 and c["cp_rand3"]["ramp"] == 0.25
    assert R.conditions(16, 1)["rand3"]["lam"] == 1.0 and R.conditions(8, 1) == {}


def test_small_run_and_judge(tmp_path):
    args = argparse.Namespace(d=8, alpha=0.05, train_steps=12, seq_len=16, n_windows=4, n_probe=2, log_every=4,
                              k_num=1, k_den=4, n_random=2, eps=1e-4, tau=0.05, rho=0.25, mu=1.0, ref=None)
    corpus = CharCorpus(synthetic_corpus(n_sentences=60, seed=0))
    subs = []
    for seed in (0, 1):
        runs = {k: R.run_cond(seed, 16, 1, k, c, corpus, args) for k, c in R.conditions(16, 1).items()}
        subs.append(runs)
    r = subs[0]["cp_rand3_htre"]
    assert "react" in r and "teacher_gini_last" in r and r["cost"]["evals_per_token"] == 4.0
    summ, verdict = R.analyze({"E16_L1": subs}, args)
    v = verdict["E16_L1"]
    for k in ("P1_教師の閾値で教師の偏りが下がる", "H25_教師の閾値で一致率が上がる", "H26_教師の閾値で閉じた後の取り込み",
              "H27_再学習で閉じた後の取り込み", "H28_組み合わせ", "P1_修正1_教師の負荷の移動平均のジニ",
              "H26_修正1_θ_Tを閉じ始めで止める", "H27_修正1_使われない専門家を教師で評価", "H28_修正1_組み合わせ", "H29_参考_教師なしの再学習", "戻り", "費用"):
        assert k in v, k
    assert isinstance(v["H28_組み合わせ"]["合格"], bool)
    assert "react" in summ["E16_L1"]["cp_re"]


def test_fix1_freeze_and_probe():
    """修正1: θ_T は閉じ始めから積分を止める。react_probe は使われない専門家を教師の評価の組に加える。"""
    m = _make(ReactCFLM, **{**KW, "total_steps": 40}, teacher="rand3", lam=1.0, teacher_homeo=True,
              teacher_freeze=True)
    _train(m, train_react, 40)
    th_close = m.teacher_threshold.clone()
    # 閉じ始め（10 ステップ）以後は変わらない: 同じ模型をさらに学習しても θ_T は同じ
    _train(m, train_react, 5)
    assert torch.equal(th_close, m.teacher_threshold) and not torch.equal(th_close, torch.zeros(8))
    assert all("t_gini_ema" in s for s in m.teacher_stats)
    p = _make(ReactCFLM, **KW, teacher="rand1", lam=1.0, react=True, react_probe=True)
    with torch.no_grad():
        p.off_usage.copy_(torch.tensor([0.0, 0.5, 0.01, 0.2, 0.1, 0.1, 0.05, 0.04]))
    p.teacher_loss(*XY)
    n = XY[1].numel()
    assert 2 * n < p.cost["evals"] <= 4 * n  # c と無作為な1個に、使われない専門家 0・2 が加わる（重複は1つに数える）
    q = _make(ReactCFLM, **KW, teacher="rand1", lam=1.0, react=True)
    with torch.no_grad():
        q.off_usage.copy_(p.off_usage)
    q.teacher_loss(*XY)
    assert q.cost["evals"] <= 2 * n
    for bad in ({"teacher_freeze": True}, {"react_probe": True}, {"react_probe": True, "react": True, "teacher": "cf"}):
        kw = {**KW, "teacher": "rand1", "lam": 1.0, **bad}
        try:
            _make(ReactCFLM, **kw)
        except ValueError:
            continue
        raise AssertionError(bad)
