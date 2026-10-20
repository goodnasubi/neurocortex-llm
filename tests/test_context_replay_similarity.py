"""類似度に基づく選択的リプレイ（12.6.227節）。"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.basal_ganglia_core import ParitySpec  # noqa: E402
from neurocortex.context_inference import run_context_sequence  # noqa: E402
from neurocortex.experiments import run_context_replay_similarity as exp  # noqa: E402

SPECS = [ParitySpec(n_inputs=4, relevant=p) for p in [(0, 1), (1, 2), (2, 3)]]
KW = dict(pretrain_steps=150, steps=150, eval_n=50, cue_n=8, batch=16, novel_patience=3, selection_trials=2,
          diagnostics=True, reliability_holdout=40, episodic_hold=20,
          writeback="none", writeback_at=2, writeback_steps=5, writeback_eval=100,
          writeback_keep=True, replay_rule="ach_c", ach_ema=0.9)


def _run(**kw):
    torch.manual_seed(0)
    return run_context_sequence("inferred_gated", SPECS, 16, torch.Generator().manual_seed(1),
                                torch.Generator().manual_seed(2), **{**KW, **kw})


def test_default_unchanged_and_before_rule_identical():
    """replay_sim = "" は従来と同じ。類似度の重みは規則を適用する前（タスク 0・1）の学習を変えない。"""
    a, b = _run(), _run(replay_sim="")
    assert a["R"] == b["R"] and a["diag"]["upd"] == b["diag"]["upd"]
    assert "sim" not in a["diag"]["replay_mix"][2]
    h = _run(replay_sim="hid")
    assert h["diag"]["upd"][:300] == a["diag"]["upd"][:300] and h["R"][:2] == a["R"][:2]


def test_similarity_direction_and_shuffle():
    """hid は似た痕跡（z > 0）を、hid_neg は似ていない痕跡（z < 0）を引く。並べ替えは向きを持たず、β = 0 は一様。"""
    z = {s: _run(replay_sim=s)["diag"]["replay_mix"][2]["sim"] for s in ("hid", "hid_neg", "hid_shuf", "inp")}
    assert z["hid"] > 0.3 and z["hid_neg"] < -0.3 and z["inp"] > 0.3
    assert abs(z["hid_shuf"]) < z["hid"]
    assert abs(_run(replay_sim="hid", replay_sim_beta=0.0)["diag"]["replay_mix"][2]["sim"]) < z["hid"]
    assert _run(replay_sim="hid")["diag"]["replay_mix"][2]["fam"] > 0  # 固定化の分が実際にある


def test_rejects_invalid():
    with pytest.raises(ValueError):
        _run(replay_sim="foo")
    with pytest.raises(ValueError):
        _run(replay_sim="hid", replay_rule="keep_unknown")
    with pytest.raises(ValueError):
        _run(replay_sim="hid", replay_rule="need")
    with pytest.raises(ValueError):
        _run(replay_sim="hid", replay_sim_beta=-1.0)


def test_experiment_configs_and_jobs():
    """新しい条件は ach_c と replay_sim だけが違い、30組 × 条件の数の仕事を作る。"""
    for w in exp.NEW:
        d = {k: v for k, v in exp.CONFIGS[w].items() if k not in ("replay_sim", "replay_sim_beta")}
        assert d == exp.CONFIGS["ach_c"] and exp.CONFIGS[w]["replay_sim_beta"] == exp.BETA
    assert len(exp.jobs()) == 30 * len(exp.NEW)


def _fake(rule_vals: dict) -> dict:
    """verdict の入力の形（over_14bit・not_over・all）を作る。"""
    o, n, a = {}, {}, {}
    for w, (end, new, mn, bad) in rule_vals.items():
        o[f"end_{w}"], n[f"end_{w}"] = end, 0.0
        a.update({f"new_{w}": new, f"min_new_{w}": mn, f"bad_frac_{w}": bad / 150, f"bad_{w}": bad})
    return {"over_14bit": o, "not_over": n, "all": a}


def test_verdict():
    """H は ach_c_hid が D1〜D3 と S（並べ替えより一般化 +0.02 以上で悪い回が増えない）を満たすとき。"""
    base = {"ach_c": (0.09, 0.0, -0.04, 2), "ach_c_hid_neg": (0.05, 0.0, -0.01, 0),
            "ach_c_hid_shuf": (0.09, 0.0, -0.04, 2), "ach_c_inp": (0.09, 0.0, -0.04, 2)}
    v = exp.verdict(_fake({**base, "ach_c_hid": (0.12, 0.0, -0.03, 1)}))
    assert v["H"] and v["ach_c_hid"]["S"] and v["ach_c_hid"]["D4"] and not v["H_neg"]
    v = exp.verdict(_fake({**base, "ach_c_hid": (0.105, 0.0, -0.03, 1)}))  # D1 は合格、S は不合格
    assert v["ach_c_hid"]["success"] and not v["ach_c_hid"]["S"] and not v["H"]
    v = exp.verdict(_fake({**base, "ach_c_hid": (0.20, 0.0, -0.03, 3)}))  # D2(c) 不合格
    assert not v["H"]
