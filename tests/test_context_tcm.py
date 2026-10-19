"""時間文脈のタグ（TCM）による海馬の痕跡の文脈依存の想起（12.6.179〜180節）のテスト。"""

import torch

from neurocortex.context_lm import (LangFamily, TrainConfig, make_languages, tcm_drift, tcm_items, tcm_recall,
                                    tcm_tags, train_models, windows)
from neurocortex.experiments import run_context_gain, run_context_tcm


def test_drift_is_leaky_integration_of_orthogonal_items():
    items = torch.tensor([[0, 2, 2]])
    t = tcm_drift(items, 3, 0.5)
    assert torch.allclose(t[0, 0], torch.tensor([0.5, 0.0, 0.0]))
    assert torch.allclose(t[0, 1], torch.tensor([0.25, 0.0, 0.5]))
    assert torch.allclose(t[0, 2], torch.tensor([0.125, 0.0, 0.75]))
    assert torch.equal(tcm_items(torch.tensor([[1, 2, 0]]), 3), torch.tensor([[5, 6]]))


def test_tags_align_with_windows_and_exclude_the_answer():
    g = torch.Generator().manual_seed(0)
    text = torch.randint(0, 4, (3, 12), generator=g)
    k, beta = 3, 0.1
    x, _ = windows(text, k)
    tags = tcm_tags(text, k, 4, beta)
    assert tags.shape == (x.shape[0], 16)
    n_w = text.shape[1] - k
    for c in range(3):
        for w in range(n_w):
            prefix = text[c:c + 1, :w + k]  # 窓の最後の文字まで（答え text[c, w + k] は含まない）
            want = tcm_drift(tcm_items(prefix, 4), 16, beta)[0, -1]
            assert torch.equal(x[c * n_w + w], text[c, w:w + k])
            assert torch.allclose(tags[c * n_w + w], want)


def test_recall_retrieves_traces_from_the_matching_context():
    """文字の集合が違う2つの話し手: 今の文脈に近いタグの痕跡は、同じ話し手の痕跡になる。"""
    g = torch.Generator().manual_seed(1)
    text0 = torch.randint(0, 3, (4, 60), generator=g)  # 文字 0〜2
    text1 = torch.randint(3, 6, (4, 60), generator=g)  # 文字 3〜5
    tags = torch.cat([tcm_tags(text0, 3, 6, 0.05), tcm_tags(text1, 3, 6, 0.05)])
    n0 = tags.shape[0] // 2
    cue = torch.randint(3, 6, (5, 8, 3), generator=g)  # 話し手1のラベルなしの窓
    idx = tcm_recall(cue, tags, 6, 0.05, 10)
    assert idx.shape == (5, 10)
    assert (idx >= n0).all()


def test_tags_do_not_change_training_or_storage():
    """タグの計算は乱数を使わないので、学習・保存は tcm_beta によらず同じ。"""
    fam = LangFamily(n_langs=2, vocab=4, order=1, diff_frac=0.5, alpha=0.3)
    res = []
    for beta in (0.0, 0.05):
        g = torch.Generator().manual_seed(3)
        trans, _ = make_languages(fam, g)
        res.append(train_models(fam, trans, TrainConfig(regime="seq", steps_per_lang=20, buffer=15, k=3, hidden=8,
                                                        arch="gain", tcm_beta=beta), g))
    assert res[0]["buf_tag"] == [None, None]
    assert all(t.shape == (15, 16) for t in res[1]["buf_tag"] + res[1]["hold_tag"])
    for key in ("buf_x", "buf_y", "hold_x"):
        assert all(torch.equal(a, b) for a, b in zip(res[0][key], res[1][key]))
    for p, q in zip(res[0]["model"].parameters(), res[1]["model"].parameters()):
        assert torch.equal(p, q)


def test_tcm_rules_do_not_change_existing_rules():
    """TCM の規則（専用の乱数）を加えても、12.6.175節の規則の値は同じ。純度は 0〜1（バッファ 2000 個に足りる学習文）。"""
    job = ("A", "seq", 7)
    base = run_context_gain.run_one(job, steps=300, episodes=4, cue_ns=(2,))
    tcm = run_context_tcm.run_one(job, steps=300, episodes=4, cue_ns=(2,))
    for rule in run_context_gain.GAIN_RULES:
        assert base["cues"]["2"]["acc"][rule] == tcm["cues"]["2"]["acc"][rule]
        assert base["cues"]["2"]["ce"][rule] == tcm["cues"]["2"]["ce"][rule]
    for rule in ("active_replay_tcm", "active_holdout_tcm"):
        assert 0.0 <= tcm["cues"]["2"]["acc"][rule] <= 1.0
    assert all(0.0 <= v <= 1.0 for v in tcm["cues"]["2"]["tcm_purity"].values())


def _run(fam, regime, seed, acc, own_kl=0.3, fresh_kl=0.3):
    cues = {}
    for n in ("1", "2", "4", "8", "16"):
        a = {"passive": 0.5, "uniform": 0.5, "uniform_replay": 0.5, "active_holdout": 0.7, **acc}
        cues[n] = {"acc": a, "ce": {r: 1.0 - v for r, v in a.items()}, "info": {r: 0.1 for r in a},
                   "tcm_purity": {"replay_tcm": 0.8, "holdout_tcm": 0.8}}
    return {"family": fam, "regime": regime, "seed": seed, "ce_oracle": 0.5, "ce_nocontext": 1.0, "entropy": 0.4,
            "native": {"own_buf": own_kl, "other_buf": 1.0, "holdout": fresh_kl, "fresh": fresh_kl}, "cues": cues}


def test_h2_tcm_verdict():
    good = {"active": 0.80, "active_replay": 0.75, "active_replay_own": 0.84, "active_replay_tcm": 0.83,
            "active_holdout_tcm": 0.80, "active_replay_cued": 0.0}
    flat = {"active": 0.80, "active_replay": 0.80, "active_replay_own": 0.80, "active_replay_tcm": 0.80,
            "active_holdout_tcm": 0.80}
    good.pop("active_replay_cued")
    runs = [_run("C", "seq", s, good, 0.3, 0.45) for s in range(10)] + [_run("A", "joint", s, flat) for s in range(10)]
    h = run_context_tcm.analyze_tcm(runs)["判定"]["H2_TCM"]
    assert h["上書きあり"] == ["C-seq"] and h["上書きなし"] == ["A-joint"]
    assert h["H2-T1"]["合格"] and h["H2-T2"]["合格"] and h["H2b"]["合格"] and h["H2c"]["合格"]
    assert h["失敗が解けた"]
    harmful = dict(good, active_replay_tcm=0.70)
    runs2 = [_run("C", "seq", s, harmful, 0.3, 0.45) for s in range(10)] + runs[10:]
    h2 = run_context_tcm.analyze_tcm(runs2)["判定"]["H2_TCM"]
    assert not h2["H2-T1"]["合格"] and not h2["失敗が解けた"]
