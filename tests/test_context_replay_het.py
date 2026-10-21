"""ヘテロシナプスの競合による共有の可塑性の配分（12.6.237節）。"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.basal_ganglia_core import ParitySpec  # noqa: E402
from neurocortex.context_inference import run_context_sequence  # noqa: E402
from neurocortex.experiments import run_context_replay_het as exp  # noqa: E402

# 12.6.233節のテストと同じ小さな設定
SPECS = [ParitySpec(n_inputs=4, relevant=p) for p in [(0, 1), (1, 2), (2, 3)]]
KW = dict(pretrain_steps=150, steps=150, eval_n=50, cue_n=8, batch=16, novel_patience=3, selection_trials=2,
          diagnostics=True, reliability_holdout=40, episodic_hold=20, writeback="none", writeback_at=2,
          writeback_steps=5, writeback_eval=100, writeback_keep=True)


def _run(**kw):
    torch.manual_seed(0)
    return run_context_sequence("inferred_gated", SPECS, 16, torch.Generator().manual_seed(1),
                                torch.Generator().manual_seed(2), **{**KW, **kw})


@pytest.mark.parametrize("rule", ["ach_c", ""])
def test_het_default_is_plain(rule):
    """het_gain = 1 は従来と同じ（乱数の消費も同じ）で、診断も付かない。"""
    a = _run(replay_rule=rule)
    b = _run(replay_rule=rule, het_gain=1.0, het_mode="perm")
    assert a["R"] == b["R"] and a["diag"]["ctx_acc"] == b["diag"]["ctx_acc"]
    assert "het" not in b["diag"]


@pytest.mark.parametrize("rule", ["ach_c", ""])
def test_het_only_after_writeback_and_changes_learning(rule):
    """競合は書き戻しの時点（新しいタスク）からだけ働き、それまでの学習は変えない。ach_c（規則あり）でも keep
    （規則なしで加えた痕跡を一様にリプレイ）でも働き、逆向きのシナプスの一部を弱める。"""
    ref = _run(replay_rule=rule)
    x = _run(replay_rule=rule, het_gain=0.0)
    assert x["R"][:2] == ref["R"][:2] and x["diag"]["writeback"] == ref["diag"]["writeback"]
    h = x["diag"]["het"][2]
    assert h["n"] > 0 and 0.0 < h["frac"] == h["conflict"] < 1.0
    assert set(x["diag"]["het"]) == {2}
    assert x["diag"]["upd"] != ref["diag"]["upd"]


def test_het_perm_keeps_count_but_moves_positions():
    """perm は弱める数を競合と同じに保つ（割合が逆向きの割合と一致する）が、位置が違うので学習が変わる。
    学習の乱数の消費は変えないので、書き戻しまでは同じ。"""
    c = _run(replay_rule="ach_c", het_gain=0.0)
    p = _run(replay_rule="ach_c", het_gain=0.0, het_mode="perm")
    assert p["R"][:2] == c["R"][:2]
    hp = p["diag"]["het"][2]
    assert hp["frac"] == pytest.approx(hp["conflict"])
    assert p["diag"]["upd"] != c["diag"]["upd"]


def test_het_wta_weakens_the_smaller_signal():
    """修正1の wta は逆向きのシナプスのうち痕跡側が小さい所だけ痕跡を弱め（割合は逆向きの割合より小さい）、残りは
    新しい入力の側を弱めるので、conflict とも違う学習になる。"""
    c = _run(replay_rule="ach_c", het_gain=0.0)
    w = _run(replay_rule="ach_c", het_gain=0.0, het_mode="wta")
    assert w["R"][:2] == c["R"][:2]
    h = w["diag"]["het"][2]
    assert 0.0 < h["frac"] < h["conflict"]
    assert w["diag"]["upd"] != c["diag"]["upd"]


def test_het_partial_gain_between():
    """0 < γ < 1 は γ = 0 とも γ = 1 とも違う学習になる。"""
    r1, r0, rh = (_run(replay_rule="ach_c", het_gain=g)["diag"]["upd"] for g in (1.0, 0.0, 0.5))
    assert rh != r0 and rh != r1


@pytest.mark.parametrize("kw", [dict(het_gain=1.5), dict(het_gain=-0.1), dict(het_mode="x"),
                                dict(het_gain=0.0, ctx_units=4)])
def test_invalid_args(kw):
    with pytest.raises(ValueError):
        _run(replay_rule="ach_c", **kw)


def test_experiment_config_jobs_and_verdict():
    assert exp.HET == {"het_gain": 0.0}
    for c in exp.NEW:
        assert exp.CONFIGS[c]["het_gain"] == 0.0
    assert exp.CONFIGS["ach_c_het_perm"]["het_mode"] == "perm"
    assert exp.CONFIGS["ach_c_het"]["replay_rule"] == exp.CONFIGS["ach_c"]["replay_rule"] == "ach_c"
    assert "replay_rule" not in exp.CONFIGS["keep_het"] or not exp.CONFIGS["keep_het"]["replay_rule"]
    assert not set(exp.DEV_SEEDS) & set(exp.EVAL_SEEDS)
    o = {"end_keep": 0.14, "end_ach_c": 0.08, "end_keep_het": 0.15, "end_ach_c_het": 0.12, "end_ach_c_het_perm": 0.09}
    n = {f"end_{w}": 0.01 for w in ("keep", "ach_c", *exp.NEW)}
    a = {"new_keep": -0.03, "new_ach_c": 0.0, "new_keep_het": -0.01, "new_ach_c_het": -0.01,
         "new_ach_c_het_perm": -0.01, "bad_ach_c": 2, "bad_keep_het": 1, "bad_ach_c_het": 1, "bad_ach_c_het_perm": 1,
         "bad_keep": 9}
    for w in ("keep", "ach_c", *exp.NEW):
        a.setdefault(f"min_new_{w}", -0.04)
        a.setdefault(f"bad_frac_{w}", a.get(f"bad_{w}", 0) / 150)
    v = exp.verdict({"over_14bit": o, "not_over": n, "all": a})
    # 直線: end = 0.08 + 0.06·(new − 0)/(−0.03)。new −0.01 → 0.10、残差は keep_het +0.05、ach_c_het +0.02、perm −0.01
    assert v["keep_het"]["residual"] == pytest.approx(0.05) and v["keep_het"]["F"]
    assert v["ach_c_het"]["residual"] == pytest.approx(0.02)
    assert v["ach_c_het_perm"]["residual"] == pytest.approx(-0.01)
    assert v["F"] and v["H"] and v["ach_c_het"]["D4"] and v["P"]
    assert v["H_fix"] is None and v["F_fix"] is None  # 修正の条件がなければ判定しない
    assert all(exp.CONFIGS[c]["het_mode"] == "wta" and exp.CONFIGS[c]["het_gain"] == 0.0 for c in exp.FIXES)
    o["end_ach_c_het_wta"], n["end_ach_c_het_wta"] = 0.13, 0.01  # 残差 +0.03 → F_fix 合格
    a.update(new_ach_c_het_wta=-0.01, min_new_ach_c_het_wta=-0.04, bad_ach_c_het_wta=1, bad_frac_ach_c_het_wta=1 / 150)
    assert exp.verdict({"over_14bit": o, "not_over": n, "all": a})["F_fix"] is True
    o["end_ach_c_het_perm"] = 0.115  # 残差 +0.015 → 差 +0.005 で P 不合格
    assert exp.verdict({"over_14bit": o, "not_over": n, "all": a})["P"] is False
