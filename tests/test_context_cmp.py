"""CA3 のパターン補完の強さを CA1 の比較器の食い違いで決める（12.6.191〜192節）のテスト。"""

import math

import torch

from neurocortex.context_lm import ca1_mismatch, ca3_complete
from neurocortex.experiments import run_context_ach, run_context_cmp


def _two_clusters():
    a = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
    b = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    return torch.cat([a, b])


def test_mismatch_is_relative_drop_of_agreement_with_input():
    """δ = 1 − cos(s_S, t̂)/cos(s_1, t̂)。1回の反復では 0、0〜1 に収まり、定義どおりの値。"""
    tags = _two_clusters()
    t = torch.tensor([[0.3, 0.2, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2], [0.0, 0.1, 0.0, 0.5]])
    assert torch.allclose(ca1_mismatch(t, tags, 1, 3), torch.zeros(3), atol=1e-6)
    d = ca1_mismatch(t, tags, 3, 3)
    t_hat = torch.nn.functional.normalize(t, dim=-1)
    c1 = (ca3_complete(t, tags, 1, 3) * t_hat).sum(-1)
    c3 = (ca3_complete(t, tags, 3, 3) * t_hat).sum(-1)
    assert torch.allclose(d, (1 - c3 / c1).clamp(0, 1), atol=1e-6)
    assert bool(((d >= 0) & (d <= 1)).all())


def test_mismatch_grows_when_attractor_drifts_away_from_input():
    """入力のすぐ近くにまとまりがあれば食い違いは小さく、補完が入力から離れた重心へ引くほど大きい。"""
    near = torch.tensor([[1.0, 0.0, 0.0], [0.99, 0.1, 0.0], [0.99, 0.0, 0.1]])
    far = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.1], [0.0, 1.0, 0.2], [0.1, 1.0, 0.0]])
    t = torch.tensor([[1.0, 0.0, 0.0]])
    assert float(ca1_mismatch(t, near, 3, 3)) < 0.01
    assert float(ca1_mismatch(t, far, 3, 3)) > float(ca1_mismatch(t, near, 3, 3)) + 0.1


def test_cmp_rules_do_not_change_existing_rules():
    """比較器の規則（専用の乱数）を加えても、12.6.187節の規則の値は同じ。並べ替えは λ の平均を変えない。"""
    job = ("C", "seq", 7)
    ach = run_context_ach.run_one(job, steps=300, episodes=4, cue_ns=(2,))
    cmp = run_context_cmp.run_one(job, steps=300, episodes=4, cue_ns=(2,))
    for rule in run_context_ach.ACH_ALL_RULES:
        assert ach["cues"]["2"]["acc"][rule] == cmp["cues"]["2"]["acc"][rule]
        assert ach["cues"]["2"]["ce"][rule] == cmp["cues"]["2"]["ce"][rule]
    lam = cmp["cues"]["2"]["ach_lambda"]
    assert abs(lam["replay_cmp"] - lam["replay_cmp_shuf"]) < 1e-6
    assert lam["replay_cmp_u"] <= lam["replay_cmp"] + 1e-6  # u ≦ 1 を掛ける
    for key in ("replay_cmp", "holdout_cmp", "replay_cmp_shuf", "replay_cmp_u"):
        assert 0.0 <= lam[key] <= 1.0
        assert 0.0 <= cmp["cues"]["2"]["tcm_purity"][key] <= 1.0


def _run(fam, regime, seed, acc, own_kl=0.3, fresh_kl=0.3):
    keys = ("replay_tcm", "holdout_tcm", "replay_ca3", "holdout_ca3", "replay_ca3_s1", "replay_ca3_s10",
            "replay_ach", "holdout_ach", "replay_ach_hard", "replay_ach_shuf", "replay_ach_inv",
            "replay_cmp", "holdout_cmp", "replay_cmp_shuf", "replay_cmp_u")
    pur = {k: 0.8 for k in keys}
    lam = {k: 0.3 for k in keys if "ach" in k or "cmp" in k}
    cues = {}
    for n in ("1", "2", "4", "8", "16"):
        a = {"passive": 0.5, "uniform": 0.5, "uniform_replay": 0.5, "active_holdout": 0.7,
             "active_holdout_tcm": 0.8, **{r: v(seed) if callable(v) else v for r, v in acc.items()}}
        cues[n] = {"acc": a, "ce": {r: 1.0 - v for r, v in a.items()}, "info": {r: 0.1 for r in a},
                   "tcm_purity": pur, "ach_lambda": lam}
    return {"family": fam, "regime": regime, "seed": seed, "ce_oracle": 0.5, "ce_nocontext": 1.0, "entropy": 0.4,
            "native": {"own_buf": own_kl, "other_buf": 1.0, "holdout": fresh_kl, "fresh": fresh_kl}, "cues": cues}


def test_h5_verdict_requires_consistent_episode_gain_over_shuffle():
    over = {"active": 0.80, "active_replay": 0.75, "active_replay_own": 0.84, "active_replay_tcm": 0.81,
            "active_replay_ca3": 0.83, "active_holdout_ca3": 0.80, "active_replay_ca3_s1": 0.82,
            "active_replay_ca3_s10": 0.83, "active_replay_ach": 0.83, "active_holdout_ach": 0.80,
            "active_replay_ach_hard": 0.82, "active_replay_ach_shuf": 0.83, "active_replay_ach_inv": 0.81,
            "active_replay_cmp": 0.83, "active_holdout_cmp": 0.80, "active_replay_cmp_shuf": 0.82,
            "active_replay_cmp_u": 0.83}
    none_ok = {r: 0.80 for r in over}
    # 比較器は害なし、並べ替えは害（ばらつきつき: 差は回ごとに 0.008〜0.012 で一貫して正）
    none_ok.update(active_replay_ca3=0.78, active_replay_cmp_shuf=lambda s: 0.79 + 0.0005 * (s % 5 - 2))
    runs = [_run("C", "seq", s, over, 0.3, 0.45) for s in range(10)] + \
        [_run("A", "joint", s, none_ok) for s in range(10)]
    h = run_context_cmp.analyze_cmp(runs)["判定"]["H5_CMP"]
    assert h["上書きあり"] == ["C-seq"] and h["上書きなし"] == ["A-joint"]
    for key in ("H5-T1", "H5-T2", "H5b", "H5-C", "H5-G", "H5-Gover"):
        assert h[key]["合格"], key
    assert h["失敗が解けた"] and h["比較器のゲートによる解決"]
    assert h["H5-G"]["cmp − cmp_shuf（上書きなし、8・16、1回ごと）"]["t"] > run_context_cmp.T_MIN
    # 平均は正でも回ごとの符号がばらつく（t ≦ 2）なら H5-G は不合格
    noisy = dict(none_ok, active_replay_cmp_shuf=lambda s: 0.80 + (0.02 if s % 2 else -0.019))
    runs2 = runs[:10] + [_run("A", "joint", s, noisy) for s in range(10)]
    h2 = run_context_cmp.analyze_cmp(runs2)["判定"]["H5_CMP"]
    d = h2["H5-G"]["cmp − cmp_shuf（上書きなし、8・16、1回ごと）"]
    assert d["平均"] < 0.001 and d["t"] < run_context_cmp.T_MIN and not h2["H5-G"]["合格"]
    assert not h2["比較器のゲートによる解決"]


def test_t_value():
    assert math.isclose(run_context_cmp._t([1.0, 2.0, 3.0]), 2.0 / (1.0 / math.sqrt(3)))
    assert run_context_cmp._t([0.5, 0.5]) == math.inf
