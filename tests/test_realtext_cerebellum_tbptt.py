"""小脳の合成勾配（時間方向の切断、12.6.259〜260節）のテスト。コーパスは取得せず、小さな合成の文字列で動かす。"""

import math

import pytest
import torch

from neurocortex.experiments import run_realtext_cerebellum_tbptt as rct
from neurocortex.realtext_cerebellum_tbptt import (CONDS, GainEstimator, RNNCortex, TBPTTConfig, measure_rnn,
                                                   project_gain, train_tbptt)
from test_realtext_forgetting import _data


def _cfg(cond, steps=20, **kw):
    kw = {"back": 6} | kw
    return TBPTTConfig(cond=cond, steps=steps, batch=6, k=2, d=8, hidden=16, granule=64, eval_every=10,
                       eval_streams=3, eval_len=40, eval_warm=8, **kw)


def _run(cfg):
    return train_tbptt(_data(), 8, cfg, torch.Generator().manual_seed(1), torch.Generator().manual_seed(2))


def test_rnn_run_shapes_and_state_carry():
    """窓を2つに分けて状態を引き継ぐと、続けて読んだのと同じ値。"""
    m = RNNCortex(8, 3, 8, 16)
    x, c = torch.randint(0, 8, (4, 6)), torch.tensor([0, 1, 2, 0])
    lp, hs = m.run(x, c, torch.zeros(4, 16))
    assert lp.shape == (4, 6, 8) and hs.shape == (4, 6, 16)
    lp1, hs1 = m.run(x[:, :3], c, torch.zeros(4, 16))
    lp2, _ = m.run(x[:, 3:], c, hs1[:, -1])
    assert torch.allclose(torch.cat([lp1, lp2], 1), lp, atol=1e-5)


def test_gain_estimators():
    """b に比例する a では両方の推定がその係数になり、b と無関係な a では偏りのない推定だけが 0 に近い。"""
    g = torch.Generator().manual_seed(0)
    est = GainEstimator(0.9)
    for _ in range(200):
        b = torch.randn(50, generator=g)
        est.update(0.4 * b, b)
    assert est.biased == pytest.approx(0.4, abs=1e-3) and est.unbiased == pytest.approx(0.4, abs=1e-3)
    est = GainEstimator(0.99)
    for _ in range(2000):
        b = torch.randn(50, generator=g)
        est.update(5 * torch.randn(50, generator=g), b)
    assert est.unbiased < 0.15 < est.biased  # 1回ごとに切ってから平均すると上に偏る
    assert project_gain(torch.ones(3), torch.zeros(3)) == 0.0
    assert project_gain(2 * torch.ones(3), torch.ones(3)) == pytest.approx(2.0)


@pytest.mark.parametrize("cond", CONDS)
def test_conditions_run(cond):
    out = _run(_cfg(cond))
    assert all(math.isfinite(x) for x in out["final_ce"]) and len(out["curve"]) == 3
    if cond.startswith("cb"):
        assert 0.0 <= out["gain_unbiased"] <= 1.0 and 0.0 <= out["gain_biased"] <= 1.0
        assert out["cb_cos_true"] is not None
    else:
        assert out["gain_unbiased"] is None and out["sg_norm"] is None


def test_long_with_one_window_equals_tbptt():
    """逆伝播の長さが窓と同じなら long は tbptt と同じ。"""
    a = _run(_cfg("long", back=2))
    b = _run(_cfg("tbptt"))
    assert a["final_ce"] == pytest.approx(b["final_ce"], abs=1e-5)
    assert _run(_cfg("long"))["final_ce"] != pytest.approx(b["final_ce"], abs=1e-5)


def test_cb_without_learning_equals_tbptt():
    """小脳が学ばない（μ = 0、予測は 0 のまま）なら cb は tbptt と同じ。学ぶと変わる。"""
    b = _run(_cfg("tbptt"))
    assert _run(_cfg("cb", cb_mu=0.0))["final_ce"] == pytest.approx(b["final_ce"], abs=1e-5)
    assert _run(_cfg("cb"))["final_ce"] != pytest.approx(b["final_ce"], abs=1e-5)


def test_back_must_be_multiple_and_cond_checked():
    with pytest.raises(ValueError):
        _run(_cfg("long", back=5))
    with pytest.raises(ValueError):
        _run(_cfg("bp"))


def test_measure_rnn_skips_warm():
    m = RNNCortex(8, 1, 8, 16)
    s = [torch.randint(0, 8, (2, 21))]
    assert measure_rnn(m, s, 0)[0] != pytest.approx(measure_rnn(m, s, 10)[0], abs=1e-7)


def _fake(cond, s, seed, ce, k=2, **kw):
    base = {"cb_cos_true": None, "cb_cos_true_late": None, "cb_cos_boot": None, "gain_biased": None,
            "gain_unbiased": None, "sg_norm": None}
    return base | {"cond": cond, "set": s, "seed": seed, "k": k, "final_ce_mean": ce, "auc": ce + 0.1,
                   "final_loss": ce} | kw


def test_analyze_judgments():
    runs = []
    for s in ("R", "G"):
        for seed in range(5):
            e = 0.001 * seed
            g = {"cb_cos_true": 0.02, "cb_cos_true_late": 0.02, "cb_cos_boot": 0.02, "sg_norm": 0.1}
            runs += [_fake("long", s, seed, 1.60 + e), _fake("tbptt", s, seed, 1.65 + 2 * e),
                     _fake("cb", s, seed, 1.62 + e, gain_biased=0.4, gain_unbiased=0.35 + e, **g),
                     _fake("cb_shuf", s, seed, 1.66 + e, gain_biased=0.2, gain_unbiased=0.05 + e, **g),
                     _fake("cb_gain", s, seed, 1.64 + e, gain_biased=0.4, gain_unbiased=0.35, **g),
                     _fake("tbptt", s, seed, 1.70, k=1)]
    out = rct.analyze(runs)
    assert set(out) == {"K=2", "K=1"}
    j = out["K=2"]["判定"]
    assert j["V"]["合格"] and j["V"]["差"] == pytest.approx(0.05 + 0.002)
    assert j["H1"]["合格"] and j["H1"]["回復の割合"] == pytest.approx(0.032 / 0.052)
    assert j["H2"]["合格"] and j["H3"]["合格"] and j["H4"]["合格"]
    assert "V" not in out["K=1"]["判定"]
    assert out["K=2"]["表"]["cb/全体"]["n"] == 10


def test_jobs_for():
    jobs = rct.jobs_for("dev", ("R", "G"), (2, 1))
    assert len(jobs) == 2 * 2 * len(CONDS) * 5 and {j[3] for j in jobs} == set(rct.DEV_SEEDS)
    assert {j[3] for j in rct.jobs_for("eval", ("R",), (2,))} == set(rct.EVAL_SEEDS)
