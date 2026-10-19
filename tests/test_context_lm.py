"""文字 LM の文脈推定（12.6.173〜174節）のテスト。"""

import math

import pytest
import torch

from neurocortex.context_inference import cue_information
from neurocortex.context_lm import (RULES, LangFamily, TrainConfig, batched_information, evaluate, make_languages,
                                    sample_text, select_contexts, state_index, train_models, windows)
from neurocortex.experiments.run_context_lm import analyze


def test_languages_differ_only_in_masked_states():
    fam = LangFamily(n_langs=4, vocab=5, order=2, diff_frac=0.4)
    trans, masks = make_languages(fam, torch.Generator().manual_seed(0))
    assert trans.shape == (4, 25, 5)
    assert torch.allclose(trans.sum(-1), torch.ones(4, 25))
    assert int(masks[0].sum()) == 0 and all(int(m.sum()) == 10 for m in masks[1:])
    for j in range(1, 4):
        assert torch.equal(trans[j][~masks[j]], trans[0][~masks[j]])
        assert not torch.allclose(trans[j][masks[j]], trans[0][masks[j]])


def test_languages_reproducible():
    fam = LangFamily()
    a, _ = make_languages(fam, torch.Generator().manual_seed(3))
    b, _ = make_languages(fam, torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


def test_state_index_and_windows():
    x = torch.tensor([[1, 2, 3], [0, 4, 1]])
    assert state_index(x, 5, 2).tolist() == [2 * 5 + 3, 4 * 5 + 1]
    text = torch.arange(6)[None]
    xs, ys = windows(text, 3)
    assert xs.tolist() == [[0, 1, 2], [1, 2, 3], [2, 3, 4]] and ys.tolist() == [3, 4, 5]


def test_sample_text_follows_deterministic_language():
    fam = LangFamily(n_langs=1, vocab=3, order=1)
    trans = torch.zeros(3, 3)
    trans[0, 1] = trans[1, 2] = trans[2, 0] = 1.0  # 0 → 1 → 2 → 0
    text = sample_text(trans, fam, 5, 12, torch.Generator().manual_seed(0))
    assert text.shape == (5, 12)
    assert torch.equal(text[:, 1:], (text[:, :-1] + 1) % 3)


def test_batched_information_matches_cue_information():
    g = torch.Generator().manual_seed(0)
    probs = torch.softmax(torch.randn(3, 6, 4, 5, generator=g), -1)
    prior = torch.softmax(torch.randn(3, 4, generator=g), -1)
    got = batched_information(probs, prior)
    for e in range(3):
        assert torch.allclose(got[e], cue_information(probs[e], prior[e]), atol=1e-6)


def _toy_logp(e_n: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """候補 4 個・文脈 2 個・V = 2。候補 0〜2 は両文脈が同じ予測、候補 3 だけが食い違う（文脈1が正しい）。"""
    p = torch.full((e_n, 4, 2, 2), 0.5)
    p[:, 3, 0] = torch.tensor([0.95, 0.05])
    p[:, 3, 1] = torch.tensor([0.05, 0.95])
    answers = torch.ones(e_n, 4, dtype=torch.long)
    return torch.log(p), answers


def test_active_selects_the_informative_cue():
    logp, ans = _toy_logp()
    chosen_p, info_p = select_contexts(logp, ans, 1, active=False)
    chosen_a, info_a = select_contexts(logp, ans, 1, active=True)
    assert (chosen_a == 1).all()  # 食い違う候補を選んで正しい文脈を当てる
    assert (info_a > info_p + 0.5).all()
    assert torch.allclose(info_p, torch.zeros_like(info_p), atol=1e-6)


def test_passive_is_sum_of_loglik_over_first_cues():
    g = torch.Generator().manual_seed(1)
    logp = torch.log_softmax(torch.randn(5, 6, 3, 4, generator=g), -1)
    ans = torch.randint(0, 4, (5, 6), generator=g)
    chosen, _ = select_contexts(logp, ans, 3, active=False)
    probs = 0.98 * logp.exp() + 0.02 / 4
    ll = torch.log(probs.gather(-1, ans[:, :, None, None].expand(5, 6, 3, 1)).squeeze(-1))[:, :3].sum(1)
    assert torch.equal(chosen, ll.argmax(-1))


def test_active_with_all_candidates_observed_equals_passive():
    """候補を全て観測すれば、選ぶ順によらず事後分布は同じ（観測の数 = 候補の数）。"""
    g = torch.Generator().manual_seed(2)
    logp = torch.log_softmax(torch.randn(7, 4, 3, 4, generator=g), -1)
    ans = torch.randint(0, 4, (7, 4), generator=g)
    assert torch.equal(select_contexts(logp, ans, 4, active=True)[0], select_contexts(logp, ans, 4, active=False)[0])


@pytest.mark.parametrize("regime", ["joint", "seq"])
def test_train_and_evaluate_small(regime):
    fam = LangFamily(n_langs=3, vocab=4, order=1, diff_frac=0.5)
    g = torch.Generator().manual_seed(0)
    trans, _ = make_languages(fam, g)
    cfg = TrainConfig(regime=regime, steps_per_lang=60, buffer=20, k=2, hidden=16)
    tr = train_models(fam, trans, cfg, g)
    assert all(b.shape == (20, 2) for b in tr["buf_x"]) and all(h.shape == (20, 2) for h in tr["hold_x"])
    ev = evaluate(fam, trans, tr, cfg, (1, 2), g, episodes=10, test_chains=3)
    assert set(ev["cues"]) == {"1", "2"}
    for n in ("1", "2"):
        assert set(ev["cues"][n]["acc"]) == set(RULES)
        assert all(0.0 <= a <= 1.0 for a in ev["cues"][n]["acc"].values())
        assert all(math.isfinite(c) for c in ev["cues"][n]["ce"].values())
    assert ev["entropy"] <= ev["ce_oracle"] + 1.0
    assert set(ev["native"]) == {"own_buf", "other_buf", "holdout", "fresh"}


def test_train_reproducible_and_rejects_bad_regime():
    fam = LangFamily(n_langs=2, vocab=4, order=1)
    cfg = TrainConfig(regime="seq", steps_per_lang=20, buffer=10, k=2, hidden=8)
    outs = []
    for _ in range(2):
        g = torch.Generator().manual_seed(5)
        trans, _ = make_languages(fam, g)
        outs.append(train_models(fam, trans, cfg, g)["final_loss"])
    assert outs[0] == outs[1]
    with pytest.raises(ValueError):
        train_models(fam, trans, TrainConfig(regime="other"), torch.Generator())


def test_analyze_classifies_and_judges():
    def run(fam, regime, seed, gap, gain):
        cues = {}
        for n in ("1", "2", "4", "8", "16"):
            acc = {r: 0.6 for r in RULES}
            acc["active"] = 0.9
            acc["active_replay"] = 0.9 + gain
            ce = {r: 2.0 for r in RULES}
            ce["active"] = 1.6
            ce["active_replay"] = 1.6 - gain
            cues[n] = {"acc": acc, "ce": ce, "info": {r: 0.5 for r in RULES}}
        return {"family": fam, "regime": regime, "seed": seed, "cues": cues, "ce_oracle": 1.5, "ce_nocontext": 2.2,
                "entropy": 1.4, "native": {"own_buf": 0.1, "other_buf": 0.1 + gap, "holdout": 0.1 + gap,
                                           "fresh": 0.1 + gap}}
    runs = [run("A", "joint", s, 0.0, 0.0) for s in range(3)] + [run("C", "seq", s, 0.2, 0.05) for s in range(3)]
    res = analyze(runs)
    assert res["組"]["A-joint"]["分類"] == "上書きなし" and res["組"]["C-seq"]["分類"] == "上書きあり"
    v = res["判定"]
    assert v["H0"]["合格"] and v["H1"]["合格"]
    assert v["H2"]["H2a 合格（上書きありで利得・特異性）"] is True
    assert v["H2"]["H2b 合格（上書きなしで利得なし）"] is True
