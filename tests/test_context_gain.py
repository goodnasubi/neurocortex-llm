"""共有の表現への文脈のゲイン変調（12.6.175〜176節）のテスト。"""

import math

import torch

from neurocortex.context_lm import (ContextCharLM, GainCharLM, LangFamily, TrainConfig, make_languages, soft_ce,
                                    train_models)
from neurocortex.experiments.run_context_gain import FIX1_GAIN_RULES, GAIN_RULES, h0, run_one


def test_gain_model_starts_with_identical_contexts():
    """ゲインは 1 から始まるので、学習前は全文脈が同じ予測をする（文脈ごとの独立した部分がない）。"""
    torch.manual_seed(0)
    m = GainCharLM(vocab=5, k=3, n_ctx=4, hidden=16)
    assert torch.allclose(m.gains(), torch.ones(4, 16))
    x = torch.randint(0, 5, (7, 3))
    lp = m.logp_all(x)
    assert lp.shape == (7, 4, 5)
    assert torch.allclose(lp, lp[:, :1].expand_as(lp), atol=1e-6)
    assert torch.allclose(lp.exp().sum(-1), torch.ones(7, 4), atol=1e-5)


def test_gain_forward_matches_logp_all():
    torch.manual_seed(1)
    m = GainCharLM(vocab=6, k=2, n_ctx=3, hidden=8)
    with torch.no_grad():
        m.gain_logit.normal_()
    x = torch.randint(0, 6, (5, 2))
    lp = m.logp_all(x)
    for c in range(3):
        assert torch.allclose(m(x, torch.full((5,), c)), lp[:, c], atol=1e-6)
    assert ((m.gains() > 0) & (m.gains() < 2)).all()


def test_gain_is_the_only_per_context_parameter():
    """文脈を1個増やして増えるパラメータはゲインの hidden 個だけ（独立したヘッドとの違い）。"""
    def n_params(cls, n_ctx):
        return sum(p.numel() for p in cls(vocab=10, k=3, n_ctx=n_ctx, hidden=32).parameters())
    assert n_params(GainCharLM, 5) - n_params(GainCharLM, 4) == 32
    assert n_params(ContextCharLM, 5) - n_params(ContextCharLM, 4) == 32 + 32 * 10 + 10


def test_train_gain_arch_learns_context_specific_gains():
    fam = LangFamily(n_langs=3, vocab=4, order=1, diff_frac=0.7, alpha=0.1)
    g = torch.Generator().manual_seed(0)
    trans, _ = make_languages(fam, g)
    tr = train_models(fam, trans, TrainConfig(regime="joint", steps_per_lang=150, buffer=20, k=2, hidden=16,
                                              arch="gain"), g)
    assert isinstance(tr["model"], GainCharLM) and isinstance(tr["plain"], GainCharLM)
    assert tr["plain"].gain_logit.shape == (1, 16)
    assert float(tr["model"].gains().std(0).mean()) > 0.01  # 文脈の間でゲインが分かれる


def test_head_arch_is_default_and_unchanged():
    fam = LangFamily(n_langs=2, vocab=4, order=1)
    g = torch.Generator().manual_seed(5)
    trans, _ = make_languages(fam, g)
    tr = train_models(fam, trans, TrainConfig(regime="seq", steps_per_lang=20, buffer=10, k=2, hidden=8), g)
    assert isinstance(tr["model"], ContextCharLM)


def test_run_one_small():
    # バッファ 2000 個×2 を取れるだけの学習文が要るので、学習は 300 ステップ/言語
    r = run_one(("A", "seq", 0), steps=300, episodes=4, cue_ns=(2, 8))
    assert set(r["cues"]) == {"2", "8"}
    assert set(r["cues"]["8"]["acc"]) == set(GAIN_RULES)
    assert all(math.isfinite(v) for v in r["cues"]["2"]["ce"].values())
    assert set(r["gain_stats"]) == {"mean_abs_dev", "between_ctx_sd"}


def test_soft_ce_with_one_hot_posterior_equals_single_context():
    """事後分布が1つの文脈に集中すれば、重みづけたゲイン・予測の混合とも、その文脈で予測した CE と同じ。"""
    torch.manual_seed(2)
    m = GainCharLM(vocab=5, k=2, n_ctx=3, hidden=8)
    with torch.no_grad():
        m.gain_logit.normal_()
        x = torch.randint(0, 5, (30, 2))
        y = torch.randint(0, 5, (30,))
        post = torch.eye(3)
        want = torch.stack([-m(x, torch.full((30,), c)).gather(-1, y[:, None]).mean() for c in range(3)])
        assert torch.allclose(soft_ce(m, x, y, post, "gain"), want, atol=1e-5)
        assert torch.allclose(soft_ce(m, x, y, post, "mix"), want, atol=1e-5)
        # 予測の混合の CE は、文脈ごとの CE の事後確率の重みづけ平均以下（log の凹性）
        half = torch.tensor([[0.5, 0.5, 0.0]])
        assert float(soft_ce(m, x, y, half, "mix")) <= float(want[:2].mean()) + 1e-6


def test_soft_rules_do_not_change_original_rules():
    vals = []
    for rules in (GAIN_RULES, FIX1_GAIN_RULES):
        vals.append(run_one(("A", "joint", 1), steps=300, episodes=4, cue_ns=(2,), rules=rules)["cues"]["2"])
    assert all(vals[0]["acc"][r] == vals[1]["acc"][r] and vals[0]["ce"][r] == vals[1]["ce"][r] for r in GAIN_RULES)
    assert vals[1]["acc"]["active_soft"] == vals[1]["acc"]["active"]
    assert math.isfinite(vals[1]["ce"]["active_soft"]) and math.isfinite(vals[1]["ce"]["active_mix"])


def test_h0_judgement():
    def run(fam, seed, ce_soft):
        return {"family": fam, "regime": "joint", "seed": seed, "ce_nocontext": 1.5,
                "cues": {"4": {"ce": {"active_soft": ce_soft}}}}
    ok = h0([run("A", s, 1.4) for s in range(10)] + [run("B", s, 1.45) for s in range(10)], "active_soft")
    assert ok["合格"] and ok["下回った回"] == "20/20"
    bad = h0([run("A", s, 1.4) for s in range(10)] + [run("B", s, 1.6 if s < 3 else 1.45) for s in range(10)],
             "active_soft")
    assert not bad["合格"]  # 3/20 が悪い（85% < 90%）
