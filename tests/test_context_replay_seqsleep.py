"""覚醒の後の NREM/REM の睡眠（REM は新皮質の自己生成。12.6.231節）。"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.basal_ganglia_core import ParitySpec  # noqa: E402
from neurocortex.context_inference import SEQ_RULES, run_context_sequence  # noqa: E402
from neurocortex.experiments import run_context_replay_seqsleep as exp  # noqa: E402

# 新しいタスクの文脈が確立して痕跡を持つ小さな設定（12.6.167節のテストと同じ）
SPECS = [ParitySpec(n_inputs=4, relevant=p) for p in [(0, 1), (1, 2), (2, 3)]]
KW = dict(pretrain_steps=150, steps=150, eval_n=50, cue_n=8, batch=16, novel_patience=3, selection_trials=2,
          diagnostics=True, reliability_holdout=40, episodic_hold=20, writeback="none", writeback_at=2,
          writeback_steps=5, writeback_eval=100, writeback_keep=True)


def _run(**kw):
    torch.manual_seed(0)
    return run_context_sequence("inferred_gated", SPECS, 16, torch.Generator().manual_seed(1),
                                torch.Generator().manual_seed(2), **{**KW, **kw})


def test_no_sleep_equals_ach_c():
    """睡眠のステップが 0 なら、覚醒中は ach_c と同じなので学習も同じになる（写しも乱数を消費しない）。"""
    ref = _run(replay_rule="ach_c")
    for r in SEQ_RULES:
        x = _run(replay_rule=r, sleep_every=20, sleep_steps=0)
        assert x["R"] == ref["R"] and x["diag"]["ctx_acc"] == ref["diag"]["ctx_acc"]
        assert x["diag"]["replay_mix"][2]["fam"] == ref["diag"]["replay_mix"][2]["fam"]


def test_sleep_phases_change_learning_after_rule_only(monkeypatch):
    """睡眠の段階は規則の適用の後（新しいタスク）にだけ置かれ、同じ数のステップを学ぶ。自己生成の標的を使う規則だけが
    睡眠の段階ごとにモデルを写す。NREM の有無で学習が変わる（この小さな設定は正確度が飽和し、自己生成の標的と保存した
    ラベルがほぼ一致するので、REM の標的による違いは計測では確かめない）。"""
    import neurocortex.context_inference as ci
    snaps = {"n": 0}
    real = ci.copy.deepcopy

    def counting(x):
        snaps["n"] += 1
        return real(x)

    monkeypatch.setattr(ci.copy, "deepcopy", counting)
    rs, n_snap = {}, {}
    for r in (*SEQ_RULES, "ach_c_nrem2"):
        snaps["n"] = 0
        rs[r] = _run(replay_rule=r, sleep_every=20, sleep_steps=4)
        n_snap[r] = snaps["n"]
    ref = _run(replay_rule="ach_c")
    steps = {r: x["diag"]["replay_mix"][2]["sleep"] for r, x in rs.items()}
    assert len(set(steps.values())) == 1 and steps["ach_c_seq"] > 0 and steps["ach_c_seq"] % 4 == 0
    for x in rs.values():
        assert x["R"][:2] == ref["R"][:2] and x["diag"]["writeback"] == ref["diag"]["writeback"]
        assert x["diag"]["replay_mix"][2]["fam"] > 0.0
    n_phase = steps["ach_c_seq"] // 4
    assert n_snap["ach_c_seq"] == n_snap["ach_c_seq_rem"] == n_snap["ach_c_seq_mix"] == n_phase
    assert n_snap["ach_c_seq_lab"] == n_snap["ach_c_nrem2"] == 0
    rem = _run(replay_rule="ach_c_seq_rem", sleep_every=10, sleep_steps=20)["diag"]["ctx_acc"][2]
    assert rem != _run(replay_rule="ach_c_seq", sleep_every=10, sleep_steps=20)["diag"]["ctx_acc"][2]


def test_rejects_invalid():
    with pytest.raises(ValueError):
        _run(replay_rule="ach_c_seq", sleep_every=0)
    with pytest.raises(ValueError):
        _run(replay_rule="ach_c_seq_x")


def test_experiment_configs_and_jobs():
    """新しい条件は ach_c_nrem2 と replay_rule だけが違い、30組 × 条件の数の仕事を作る。"""
    for w in exp.NEW:
        d = {k: v for k, v in exp.CONFIGS[w].items() if k != "replay_rule"}
        assert d == {k: v for k, v in exp.CONFIGS["ach_c_nrem2"].items() if k != "replay_rule"}
        assert exp.CONFIGS[w]["replay_rule"] == w
    assert len(exp.jobs()) == 30 * len(exp.NEW)
    assert exp.DEV_SEEDS == list(range(100, 105)) and exp.EVAL_SEEDS == list(range(0, 5))


def _fake(rule_vals: dict) -> dict:
    o, n, a = {}, {}, {}
    for w, (end, new, mn, bad) in rule_vals.items():
        o[f"end_{w}"], n[f"end_{w}"] = end, 0.0
        a.update({f"new_{w}": new, f"min_new_{w}": mn, f"bad_frac_{w}": bad / 150, f"bad_{w}": bad})
    return {"over_14bit": o, "not_over": n, "all": a}


def test_verdict():
    base = {"ach_c": (0.09, 0.0, -0.04, 2), "ach_c_nrem2": (0.11, -0.06, -0.40, 20),
            "ach_c_seq_lab": (0.11, -0.06, -0.40, 20), "ach_c_seq_rem": (0.08, 0.0, -0.03, 1)}
    v = exp.verdict(_fake({**base, "ach_c_seq": (0.12, -0.01, -0.04, 1)}))
    assert v["H"] and v["ach_c_seq"]["R"] and v["ach_c_seq"]["N"] and v["ach_c_seq"]["O"] and v["ach_c_seq"]["D4"]
    v = exp.verdict(_fake({**base, "ach_c_seq": (0.12, -0.05, -0.30, 15)}))  # 損傷が残る
    assert not v["H"] and not v["ach_c_seq"]["R"] and not v["ach_c_seq"]["O"] and v["ach_c_seq"]["N"]
    v = exp.verdict(_fake({**base, "ach_c_seq": (0.09, 0.0, -0.03, 1)}))  # D1 不合格、NREM の上積みなし
    assert not v["H"] and not v["ach_c_seq"]["N"]
