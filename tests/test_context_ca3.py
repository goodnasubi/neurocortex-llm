"""海馬 CA3 のパターン補完による、少ない手がかりからの文脈依存の想起（12.6.183〜184節）のテスト。"""

import torch

from neurocortex.context_lm import ca3_complete, tcm_recall, tcm_tags
from neurocortex.experiments import run_context_ca3, run_context_tcm


def test_completion_without_iterations_is_identity():
    t = torch.tensor([[3.0, 4.0, 0.0]])
    tags = torch.eye(3)
    assert torch.allclose(ca3_complete(t, tags, 0, 1), torch.tensor([[0.6, 0.8, 0.0]]))


def test_completion_moves_to_the_centroid_of_the_nearby_cluster():
    """2つのまとまりのタグ: 部分的な手がかりは、近いまとまりの重心へ補完され、そこで止まる（アトラクタ）。"""
    a = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
    b = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    tags = torch.cat([a, b])
    centroid_a = torch.nn.functional.normalize(torch.nn.functional.normalize(a, dim=-1).mean(0), dim=-1)
    s1 = ca3_complete(torch.tensor([[0.3, 0.2, 0.0, 0.0]]), tags, 1, 3)
    s5 = ca3_complete(torch.tensor([[0.3, 0.2, 0.0, 0.0]]), tags, 5, 3)
    assert torch.allclose(s1[0], centroid_a, atol=1e-6)
    assert torch.allclose(s5[0], centroid_a, atol=1e-6)


def test_completion_purifies_recall_from_a_few_cues():
    """2つの話し手の文字の分布が重なる場合: 短い手がかりでの想起の純度は、補完で下がらず、平均では上がる。"""
    g = torch.Generator().manual_seed(4)
    p0 = torch.tensor([0.35, 0.35, 0.1, 0.1, 0.05, 0.05])
    p1 = torch.tensor([0.05, 0.05, 0.1, 0.1, 0.35, 0.35])
    text0 = torch.multinomial(p0, 8 * 200, replacement=True, generator=g).reshape(8, 200)
    text1 = torch.multinomial(p1, 8 * 200, replacement=True, generator=g).reshape(8, 200)
    beta = 0.02
    tags = torch.cat([tcm_tags(text0, 3, 6, beta, 1), tcm_tags(text1, 3, 6, beta, 1)])
    n0 = tags.shape[0] // 2
    cue = torch.multinomial(p1, 50 * 2 * 3, replacement=True, generator=g).reshape(50, 2, 3)
    pur0 = float((tcm_recall(cue, tags, 6, beta, 8, 1) >= n0).float().mean())
    pur3 = float((tcm_recall(cue, tags, 6, beta, 8, 1, 3, 100) >= n0).float().mean())
    assert pur3 >= pur0 and pur3 > 0.9


def test_ca3_rules_do_not_change_existing_rules():
    """CA3 の規則（専用の乱数）を加えても、12.6.179節の規則（修正1）の値は同じ。純度は 0〜1。"""
    job = ("C", "seq", 7)
    tcm = run_context_tcm.run_one(job, steps=300, episodes=4, cue_ns=(2,), beta=run_context_tcm.FIX1_BETA,
                                  item=run_context_tcm.FIX1_ITEM)
    ca3 = run_context_ca3.run_one(job, steps=300, episodes=4, cue_ns=(2,))
    for rule in run_context_tcm.TCM_ALL_RULES:
        assert tcm["cues"]["2"]["acc"][rule] == ca3["cues"]["2"]["acc"][rule]
        assert tcm["cues"]["2"]["ce"][rule] == ca3["cues"]["2"]["ce"][rule]
    for key in ("replay_ca3", "holdout_ca3", "replay_ca3_s1", "replay_ca3_s10"):
        assert 0.0 <= ca3["cues"]["2"]["tcm_purity"][key] <= 1.0
        assert 0.0 <= ca3["cues"]["2"]["acc"]["active_" + key] <= 1.0


def _run(fam, regime, seed, acc, purity, own_kl=0.3, fresh_kl=0.3):
    cues = {}
    for n in ("1", "2", "4", "8", "16"):
        a = {"passive": 0.5, "uniform": 0.5, "uniform_replay": 0.5, "active_holdout": 0.7,
             "active_holdout_tcm": 0.8, **acc}
        cues[n] = {"acc": a, "ce": {r: 1.0 - v for r, v in a.items()}, "info": {r: 0.1 for r in a},
                   "tcm_purity": purity}
    return {"family": fam, "regime": regime, "seed": seed, "ce_oracle": 0.5, "ce_nocontext": 1.0, "entropy": 0.4,
            "native": {"own_buf": own_kl, "other_buf": 1.0, "holdout": fresh_kl, "fresh": fresh_kl}, "cues": cues}


def test_h3_verdict():
    pur = {"replay_tcm": 0.7, "holdout_tcm": 0.7, "replay_ca3": 0.9, "holdout_ca3": 0.9, "replay_ca3_s1": 0.85,
           "replay_ca3_s10": 0.9}
    good = {"active": 0.80, "active_replay": 0.75, "active_replay_own": 0.84, "active_replay_tcm": 0.81,
            "active_replay_ca3": 0.83, "active_holdout_ca3": 0.80, "active_replay_ca3_s1": 0.82,
            "active_replay_ca3_s10": 0.83}
    flat = {r: 0.80 for r in good}
    runs = [_run("C", "seq", s, good, pur, 0.3, 0.45) for s in range(10)] + \
        [_run("A", "joint", s, flat, pur) for s in range(10)]
    h = run_context_ca3.analyze_ca3(runs)["判定"]["H3_CA3"]
    assert h["上書きあり"] == ["C-seq"] and h["上書きなし"] == ["A-joint"]
    assert h["H3-T1"]["合格"] and h["H3-T2"]["合格"] and h["H3b"]["合格"] and h["H3-P"]["合格"]
    assert h["失敗が解けた"] and h["補完による解決"]
    weak = dict(good, active_replay_ca3=0.81)  # 補完なしと同じ: 取り戻した割合 25%、補完の寄与なし
    runs2 = [_run("C", "seq", s, weak, pur, 0.3, 0.45) for s in range(10)] + runs[10:]
    h2 = run_context_ca3.analyze_ca3(runs2)["判定"]["H3_CA3"]
    assert not h2["H3-T2"]["合格"] and not h2["H3-P"]["合格"] and not h2["失敗が解けた"]
