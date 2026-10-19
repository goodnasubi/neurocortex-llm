"""CA3 のパターン補完の強さを想起の不確かさでゲートする（12.6.187〜188節）のテスト。"""

import torch

from neurocortex.context_lm import ca3_complete, recall_uncertainty
from neurocortex.experiments import run_context_ach, run_context_ca3


def test_uncertainty_is_normalized_entropy_of_recalled_contexts():
    srcs = torch.tensor([0, 0, 1, 1, 2, 3])
    idx = torch.tensor([[0, 1, 0, 1], [0, 2, 4, 5]])  # 1つの文脈だけ / 4文脈に均等
    u = recall_uncertainty(idx, srcs, 4)
    assert torch.allclose(u, torch.tensor([0.0, 1.0]), atol=1e-6)


def test_gate_interpolates_between_no_completion_and_full_completion():
    """λ = 1 は 12.6.183節の補完、λ = 0 は補完なし（入力のまま）。間の λ は入力とまとまりの重心の間に止まる。"""
    a = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
    b = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    tags = torch.cat([a, b])
    t = torch.tensor([[0.3, 0.2, 0.0, 0.0], [0.3, 0.2, 0.0, 0.0], [0.3, 0.2, 0.0, 0.0]])
    full = ca3_complete(t, tags, 3, 3)
    gated = ca3_complete(t, tags, 3, 3, torch.tensor([1.0, 0.0, 0.5]))
    t_hat = torch.nn.functional.normalize(t[0], dim=-1)
    assert torch.allclose(gated[0], full[0], atol=1e-6)
    assert torch.allclose(gated[1], t_hat, atol=1e-6)
    cos_in, cos_full = float(gated[2] @ t_hat), float(gated[2] @ full[0])
    assert cos_in < 1 - 1e-4 and cos_full < 1 - 1e-4


def test_ach_rules_do_not_change_existing_rules():
    """ゲートの規則（専用の乱数）を加えても、12.6.183節の規則の値は同じ。λ・純度は 0〜1、hard の λ は 0 か 1 の平均。"""
    job = ("C", "seq", 7)
    ca3 = run_context_ca3.run_one(job, steps=300, episodes=4, cue_ns=(2,))
    ach = run_context_ach.run_one(job, steps=300, episodes=4, cue_ns=(2,))
    for rule in run_context_ca3.CA3_ALL_RULES:
        assert ca3["cues"]["2"]["acc"][rule] == ach["cues"]["2"]["acc"][rule]
        assert ca3["cues"]["2"]["ce"][rule] == ach["cues"]["2"]["ce"][rule]
    lam = ach["cues"]["2"]["ach_lambda"]
    assert abs(lam["replay_ach"] - lam["replay_ach_shuf"]) < 1e-6  # 並べ替えは平均を変えない
    assert abs(lam["replay_ach"] + lam["replay_ach_inv"] - 1.0) < 1e-6
    for key, val in lam.items():
        assert 0.0 <= val <= 1.0
        assert 0.0 <= ach["cues"]["2"]["tcm_purity"][key] <= 1.0


def _run(fam, regime, seed, acc, own_kl=0.3, fresh_kl=0.3):
    pur = {k: 0.8 for k in ("replay_tcm", "holdout_tcm", "replay_ca3", "holdout_ca3", "replay_ca3_s1",
                            "replay_ca3_s10", "replay_ach", "holdout_ach", "replay_ach_hard", "replay_ach_shuf",
                            "replay_ach_inv")}
    lam = {k: 0.3 for k in ("replay_ach", "holdout_ach", "replay_ach_hard", "replay_ach_shuf", "replay_ach_inv")}
    cues = {}
    for n in ("1", "2", "4", "8", "16"):
        a = {"passive": 0.5, "uniform": 0.5, "uniform_replay": 0.5, "active_holdout": 0.7,
             "active_holdout_tcm": 0.8, **acc}
        cues[n] = {"acc": a, "ce": {r: 1.0 - v for r, v in a.items()}, "info": {r: 0.1 for r in a},
                   "tcm_purity": pur, "ach_lambda": lam}
    return {"family": fam, "regime": regime, "seed": seed, "ce_oracle": 0.5, "ce_nocontext": 1.0, "entropy": 0.4,
            "native": {"own_buf": own_kl, "other_buf": 1.0, "holdout": fresh_kl, "fresh": fresh_kl}, "cues": cues}


def test_h4_verdict():
    over = {"active": 0.80, "active_replay": 0.75, "active_replay_own": 0.84, "active_replay_tcm": 0.81,
            "active_replay_ca3": 0.83, "active_holdout_ca3": 0.80, "active_replay_ca3_s1": 0.82,
            "active_replay_ca3_s10": 0.83, "active_replay_ach": 0.83, "active_holdout_ach": 0.80,
            "active_replay_ach_hard": 0.82, "active_replay_ach_shuf": 0.83, "active_replay_ach_inv": 0.81}
    none_ok = {r: 0.80 for r in over}
    none_ok.update(active_replay_ca3=0.78, active_replay_ach_shuf=0.79)  # 常に補完・並べ替えは害、ゲートは害なし
    runs = [_run("C", "seq", s, over, 0.3, 0.45) for s in range(10)] + \
        [_run("A", "joint", s, none_ok) for s in range(10)]
    h = run_context_ach.analyze_ach(runs)["判定"]["H4_ACh"]
    assert h["上書きあり"] == ["C-seq"] and h["上書きなし"] == ["A-joint"]
    for key in ("H4-T1", "H4-T2", "H4b", "H4-C", "H4-G"):
        assert h[key]["合格"], key
    assert h["失敗が解けた"] and h["ゲートによる解決"]
    none_bad = dict(none_ok, active_replay_ach=0.785)  # ゲートしても上書きなしで害（−0.015）
    runs2 = runs[:10] + [_run("A", "joint", s, none_bad) for s in range(10)]
    h2 = run_context_ach.analyze_ach(runs2)["判定"]["H4_ACh"]
    assert not h2["H4b"]["合格"] and not h2["H4-G"]["合格"] and not h2["失敗が解けた"]
