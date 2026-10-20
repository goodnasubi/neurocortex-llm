"""文脈のゲイン変調（乗算）と加算の変調の切り分け（12.6.223〜224節）のテスト。"""

import torch

from neurocortex.context_lm import (ContextCharLM, GainCharLM, Head2CharLM, LangFamily, TrainConfig, make_languages,
                                    soft_ce, train_models)
from neurocortex.experiments.run_context_mod import analyze, rules_for, run_one


def _n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def test_bias_and_gain_have_same_per_context_parameters():
    """加算の対照は、文脈を1個増やして増えるパラメータがゲインと同じ hidden 個。"""
    for mod in ("gain", "bias"):
        d = _n_params(GainCharLM(10, 3, 5, hidden=32, mod=mod)) - _n_params(GainCharLM(10, 3, 4, hidden=32, mod=mod))
        assert d == 32
    assert _n_params(GainCharLM(10, 3, 4, hidden=32, mod="bias")) == _n_params(GainCharLM(10, 3, 4, hidden=32))
    d = _n_params(GainCharLM(10, 3, 5, hidden=32, mod="gainbias")) - _n_params(GainCharLM(10, 3, 4, hidden=32,
                                                                                         mod="gainbias"))
    assert d == 64


def test_default_gain_model_is_unchanged():
    """mod の既定（gain）は 12.6.175節と同じ乱数の使い方・パラメータ（加算は学習しない 0 の定数）。"""
    torch.manual_seed(3)
    m = GainCharLM(6, 2, 3, hidden=8)
    assert [n for n, _ in m.named_parameters()] == ["gain_logit", "emb.weight", "inp.weight", "inp.bias",
                                                    "mid.weight", "mid.bias", "out.weight", "out.bias"]
    assert not m.ctx_bias.requires_grad and torch.equal(m.biases(), torch.zeros(3, 8))
    x = torch.randint(0, 6, (4, 2))
    h1 = torch.tanh(m.inp(m.emb(x).flatten(1)))
    ref = torch.log_softmax(m.out(torch.tanh(m.mid(h1 * m.gains()[1]))), -1)
    assert torch.allclose(m(x, torch.ones(4, dtype=torch.long)), ref, atol=1e-6)


def test_bias_model_is_additive_at_the_same_place():
    torch.manual_seed(4)
    m = GainCharLM(6, 2, 3, hidden=8, mod="bias")
    with torch.no_grad():
        m.ctx_bias.normal_()
    assert torch.equal(m.gains(), torch.ones(3, 8))  # ゲインは常に 1
    x = torch.randint(0, 6, (5, 2))
    lp = m.logp_all(x)
    h1 = torch.tanh(m.inp(m.emb(x).flatten(1)))
    for c in range(3):
        ref = torch.log_softmax(m.out(torch.tanh(m.mid(h1 + m.ctx_bias[c]))), -1)
        assert torch.allclose(m(x, torch.full((5,), c)), ref, atol=1e-6)
        assert torch.allclose(lp[:, c], ref, atol=1e-6)


def test_head2_matches_logp_all_and_starts_shared():
    torch.manual_seed(5)
    m = Head2CharLM(6, 2, 3, hidden=8)
    x = torch.randint(0, 6, (5, 2))
    lp = m.logp_all(x)
    for c in range(3):
        assert torch.allclose(m(x, torch.full((5,), c)), lp[:, c], atol=1e-6)
    assert _n_params(Head2CharLM(10, 3, 5, hidden=32)) - _n_params(Head2CharLM(10, 3, 4, hidden=32)) \
        == _n_params(ContextCharLM(10, 3, 5, hidden=32)) - _n_params(ContextCharLM(10, 3, 4, hidden=32))


def test_soft_ce_uses_weighted_bias():
    """事後分布が1文脈に集中すれば、加算・両方の形の active_soft はその文脈の CE に一致する。"""
    torch.manual_seed(6)
    for mod in ("bias", "gainbias"):
        m = GainCharLM(5, 2, 3, hidden=8, mod=mod)
        with torch.no_grad():
            m.ctx_bias.normal_()
            if mod == "gainbias":
                m.gain_logit.normal_()
        x, y = torch.randint(0, 5, (9, 2)), torch.randint(0, 5, (9,))
        post = torch.eye(3)
        ce = soft_ce(m, x, y, post, "gain")
        for c in range(3):
            ref = -m(x, torch.full((9,), c)).gather(-1, y[:, None]).mean()
            assert torch.allclose(ce[c], ref, atol=1e-5)


def test_train_bias_arch_learns_context_specific_biases():
    fam = LangFamily(n_langs=3, vocab=4, order=1, diff_frac=0.7, alpha=0.1)
    g = torch.Generator().manual_seed(0)
    trans, _ = make_languages(fam, g)
    tr = train_models(fam, trans, TrainConfig(regime="joint", steps_per_lang=150, buffer=20, k=2, hidden=16,
                                              arch="bias"), g)
    assert isinstance(tr["model"], GainCharLM) and tr["model"].mod == "bias"
    assert float(tr["model"].biases().detach().std(0).mean()) > 0.01
    assert torch.equal(tr["model"].gains(), torch.ones(3, 16))
    for arch, cls in (("head2", Head2CharLM), ("gainbias", GainCharLM)):
        g = torch.Generator().manual_seed(0)
        trans, _ = make_languages(fam, g)
        tr = train_models(fam, trans, TrainConfig(regime="seq", steps_per_lang=30, buffer=10, k=2, hidden=8,
                                                  arch=arch), g)
        assert isinstance(tr["model"], cls)


def test_same_data_across_archs():
    """形を替えても、同じシードなら言語と学習のデータは同じ（文脈なしの CE の差はモデルの差だけ）。"""
    fam = LangFamily(n_langs=2, vocab=4, order=1)
    outs = []
    for arch in ("gain", "bias"):
        g = torch.Generator().manual_seed(7)
        trans, _ = make_languages(fam, g)
        tr = train_models(fam, trans, TrainConfig(regime="joint", steps_per_lang=10, buffer=5, k=2, hidden=8,
                                                  arch=arch), g)
        outs.append((trans, tr["buf_x"][0]))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])


def test_run_one_and_analyze_small():
    assert "active_soft" in rules_for("bias") and "active_soft" not in rules_for("head2")
    runs = [run_one((a, "A", reg, 100), steps=300, episodes=3, cue_ns=(2, 4))
            for a in ("head", "head2", "gain", "bias") for reg in ("joint", "seq")]
    assert all("acc_per_lang" not in r["cues"]["4"] for r in runs)
    out = analyze(runs)
    v = out["判定"]
    assert set(out["形ごと"]) == {"head", "head2", "gain", "bias"}
    assert {"M", "H1（oracle: bias − gain）", "H2（active_soft n=4: bias − gain）", "H3（12.6.176節の差の分解）",
            "H4（選択精度: gain − bias）", "参考"} <= set(v)
    h1 = v["H1（oracle: bias − gain）"]
    assert h1["回の数"] == 2 and h1["組の数"] == 2
    d = {(r["arch"], r["regime"]): r["ce_oracle"] for r in runs}
    assert abs(h1["組ごと"]["A-seq"] - (d[("bias", "seq")] - d[("gain", "seq")])) < 1e-9


def test_h1_and_h3_verdicts():
    """判定の境目: H1 は平均 +0.010・全組で正・80% の回、H3 は割合 0.5 / 0.2 で言い方を分ける。"""
    def fake(arch, fam, reg, seed, oracle):
        cues = {n: {"acc": {"active": 0.5}, "ce": {"active": 1.0, "active_soft": oracle + 0.1,
                                                     "active_mix": 1.0}} for n in ("2", "4")}
        r = {"arch": arch, "family": fam, "regime": reg, "seed": seed, "ce_oracle": oracle, "ce_nocontext": 2.0,
             "native": {"fresh": 0.3, "own_buf": 0.2}, "cues": cues}
        if arch in ("gain", "bias"):
            r["mod_stats"] = {"gain_between_ctx_sd": 0.1, "bias_between_ctx_sd": 0.1, "bias_mean_abs": 0.1}
        return r

    def build(bias_minus_gain, head_minus_gain=0.1):
        runs = []
        for fam in ("A", "B", "C"):
            for reg in ("joint", "seq"):
                for s in range(5):
                    runs += [fake("gain", fam, reg, s, 1.0), fake("bias", fam, reg, s, 1.0 + bias_minus_gain),
                             fake("head", fam, reg, s, 1.0 + head_minus_gain),
                             fake("head2", fam, reg, s, 1.0 + head_minus_gain / 2)]
        return analyze(runs)["判定"]

    v = build(0.06)
    assert v["M"]["合格"] and v["H1（oracle: bias − gain）"]["合格"] is True and v["H2（active_soft n=4: bias − gain）"]["合格"]
    assert v["H3（12.6.176節の差の分解）"]["言い方"] == "主に乗算のゲイン変調による"
    v = build(0.005)
    assert v["H1（oracle: bias − gain）"]["合格"] is False
    assert v["H3（12.6.176節の差の分解）"]["言い方"] == "主に層の数・出力の共有による（乗算に固有ではない）"
    assert build(0.03)["H3（12.6.176節の差の分解）"]["言い方"] == "乗算と層・共有の両方による"
    assert build(0.03, head_minus_gain=0.01)["H3（12.6.176節の差の分解）"]["言い方"] == "12.6.176節の差が再現せず"
    assert v["H4（選択精度: gain − bias）"]["合格"]
