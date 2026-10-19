"""連続的な新しさの信号と、新しい記憶を優先する睡眠の段階（12.6.167節）。"""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.basal_ganglia_core import ParitySpec  # noqa: E402
from neurocortex.context_inference import run_context_sequence  # noqa: E402

SPECS = [ParitySpec(n_inputs=8, relevant=p) for p in [(0, 1), (2, 3), (4, 5)]]
KW = dict(pretrain_steps=60, steps=60, eval_n=50, cue_n=8, batch=16, novel_patience=3, selection_trials=10,
          diagnostics=True, reliability_holdout=40, episodic_hold=20,
          writeback="none", writeback_at=2, writeback_steps=5, writeback_eval=100)


# 新しいタスクの文脈が確立し、痕跡を持つ小さな設定
SPECS2 = [ParitySpec(n_inputs=4, relevant=p) for p in [(0, 1), (1, 2), (2, 3)]]
KW2 = {**KW, "pretrain_steps": 150, "steps": 150, "selection_trials": 2}


def _run(specs=SPECS, base_kw=KW, **kw):
    torch.manual_seed(0)
    return run_context_sequence("inferred_gated", specs, 16, torch.Generator().manual_seed(1),
                                torch.Generator().manual_seed(2), **{**base_kw, **kw})


def _run2(**kw):
    return _run(SPECS2, KW2, **kw)


def test_ach_c_familiarity_sets_kept_share():
    """規則はタスク writeback_at から適用する。加えた痕跡はなじみの分（round(f·batch) 個）からしか取らないので、
    その割合は f の平均を超えない。損失の移動平均が遅い（ach_ema が 1 に近い）ほど f は低く、痕跡の固定化は遅れる。"""
    keep = _run2(writeback_keep=True)
    fast = _run2(writeback_keep=True, replay_rule="ach_c", ach_ema=0.0)
    slow = _run2(writeback_keep=True, replay_rule="ach_c", ach_ema=0.999)
    for r in (fast, slow):
        assert r["R"][:2] == keep["R"][:2] and r["diag"]["writeback"] == keep["diag"]["writeback"]
        assert list(r["diag"]["replay_mix"]) == [2]
        m = r["diag"]["replay_mix"][2]
        assert 0.0 <= m["fam"] <= 1.0 and m["kept"] <= m["fam"] + 1e-6 and 15 <= m["size"] <= 16
    mf, ms = fast["diag"]["replay_mix"][2], slow["diag"]["replay_mix"][2]
    assert mf["fam"] > ms["fam"] and mf["kept"] > ms["kept"]


def test_ach_c_without_familiarity_equals_no_kept_traces():
    """驚きが続く（f が常に 0）設定では、痕跡を加えない場合と同じ学習になる。"""
    base = _run()
    r = _run(writeback_keep=True, replay_rule="ach_c")
    assert r["diag"]["replay_mix"][2]["fam"] == 0.0
    assert r["R"] == base["R"] and r["diag"]["ctx_acc"] == base["diag"]["ctx_acc"]


def test_nrem_alternates_new_and_old_traces():
    """nrem は覚醒中に加えた痕跡を除き、新しい文脈の痕跡がなければ睡眠の段階を置かない。睡眠のステップが 0 なら、
    痕跡を加えない場合と同じ学習になる。"""
    base = _run()
    r0 = _run(writeback_keep=True, replay_rule="nrem", sleep_every=20, sleep_steps=0)
    assert r0["R"] == base["R"] and r0["diag"]["ctx_acc"] == base["diag"]["ctx_acc"]
    r = _run(writeback_keep=True, replay_rule="nrem", sleep_every=20, sleep_steps=7)
    m = r["diag"]["replay_mix"][2]
    assert m["kept"] == 0.0
    # この小さな設定では新しい文脈の痕跡が保存されないので、睡眠の段階は置かれず、痕跡を加えない場合と同じになる
    assert m["sleep"] == 0 and r["R"] == base["R"]
    # 12.6.163節の sleep は新しい文脈の痕跡がなくても睡眠の段階を置く（従来の挙動を変えていない）
    assert _run(writeback_keep=True, replay_rule="sleep", sleep_every=20, sleep_steps=7)["diag"]["replay_mix"][2]["sleep"] == 14


def test_nrem_sleeps_when_new_context_has_traces():
    """新しい文脈が確立して痕跡を持つ設定では、睡眠の段階が置かれ、学習が変わる。"""
    base = _run2()
    r = _run2(writeback_keep=True, replay_rule="nrem", sleep_every=20, sleep_steps=4)
    m = r["diag"]["replay_mix"][2]
    assert m["sleep"] > 0 and m["sleep"] % 4 == 0 and m["kept"] == 0.0
    assert r["R"][:2] == base["R"][:2]
    assert r["R"][2] != base["R"][2] or r["diag"]["ctx_acc"][2] != base["diag"]["ctx_acc"][2]
    c = _run2(writeback_keep=True, replay_rule="ach_c_nrem", sleep_every=20, sleep_steps=4)["diag"]["replay_mix"][2]
    assert c["sleep"] == m["sleep"] and c["fam"] > 0.0
    # 修正1の nrem2: 加えた痕跡がなければ nrem と同じ（NREM は新しい文脈、REM は古い文脈の痕跡）
    nb = _run2(replay_rule="nrem", sleep_every=20, sleep_steps=4)
    n2b = _run2(replay_rule="nrem2", sleep_every=20, sleep_steps=4)
    assert n2b["R"] == nb["R"] and n2b["diag"]["ctx_acc"] == nb["diag"]["ctx_acc"]
    n2 = _run2(writeback_keep=True, replay_rule="nrem2", sleep_every=20, sleep_steps=4)
    assert n2["diag"]["replay_mix"][2]["sleep"] == m["sleep"]
    # （この小さな設定は正確度が飽和するので、加えた痕跡による違いは計測では確かめない）


def test_new_rules_reject_bad_arguments():
    for kw in (dict(replay_rule="ach_c", ach_ema=1.0), dict(replay_rule="ach_c", ach_ema=-0.1),
               dict(replay_rule="nrem", sleep_every=0), dict(replay_rule="ach_c_nrem", replay_need=0.0)):
        with pytest.raises(ValueError):
            _run(**kw)


def test_experiment_jobs_analyze_and_verdict(tmp_path):
    from neurocortex.experiments.run_context_replay_novelty import (CONFIGS, DEV_SEEDS, EVAL_SEEDS, FAMILIES, FIX1,
                                                                    NEW_RULES, analyze, jobs)
    assert len(jobs()) == len(FAMILIES) * 2 * 3 * (len(CONFIGS) - len(FIX1)) and set(NEW_RULES) < set(CONFIGS)
    assert len(jobs(FIX1)) == len(FAMILIES) * 2 * 3 * len(FIX1)
    assert len(DEV_SEEDS) == len(EVAL_SEEDS) == 10 and not set(DEV_SEEDS) & set(EVAL_SEEDS)

    def row(end, new, seed):
        return {"seed": seed, "pre_nat": [0.4, 0.9], "end_nat": end, "gap": [0.3, 0.0], "exp_end": end[0],
                "new_nat": new, "new_exp": new, "selection_acc": 0.9, "mix_own": 0.1, "mix_kept": 0.3,
                "mix_fam": 0.5, "sleep_steps": 0}

    seeds = range(100, 110)
    runs = {"flip7_m1/reset_L0/base": {"per_seed": [row([0.3, 0.9], 0.8, s) for s in seeds]},
            "flip7_m1/reset_L0/ach_need2": {"per_seed": [row([0.5, 0.9], 0.8 - 0.3 * (s < 102), s) for s in seeds]},
            "flip7_m1/reset_L0/ach_c": {"per_seed": [row([0.6, 0.9], 0.8 - 0.2 * (s == 100), s) for s in seeds]},
            "flip7_m1/split_L0/base": {"per_seed": [row([0.3, 0.9], 0.8, s) for s in seeds]},
            "flip7_m1/split_L0/ach_c": {"per_seed": [row([0.6, 0.9], 0.8, s) for s in seeds]},
            "flip7_m1/split_L0/ach_need2": {"per_seed": [row([0.45, 0.9], 0.8, s) for s in seeds]},
            "parity/reset_L0/base": {"per_seed": [{**row([0.9, 0.9], 0.9, s), "gap": [0.0, 0.0]} for s in seeds]},
            "parity/reset_L0/ach_c": {"per_seed": [{**row([0.9, 0.9], 0.9, s), "gap": [0.0, 0.0]} for s in seeds]}}
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"runs": runs}), encoding="utf-8")
    res = analyze([p])
    a = res["all"]
    assert res["pre_identical"] and res["over_14bit"]["n"] == 2 and res["split_all"]["n"] == 1
    assert a["runs_ach_c"] == 30 and a["bad_ach_c"] == 1 and a["bad_ach_need2"] == 2
    assert a["min_new_ach_c"] == pytest.approx(-0.02) and a["argmin_new_ach_c"] == "flip7_m1/reset_L0"
    v = res["verdict"]
    # ach_c: 上書きありの end は +0.15 ≧ 0.10、平均・最悪の組は基準内だが、−0.10 未満の回が 1/30 > 1%
    assert v["ach_c"]["B1"] and v["ach_c"]["B3"] and not v["ach_c"]["B2"] and not v["ach_c"]["success"]
    assert v["ach_c"]["B4"]  # −0.10 未満の回が ach_need2 の半分以下（1 ≦ 2 / 2）
    sp = v["ach_c"]["H3_split"]  # 分離の機構の組だけ: 上書きなしの組がないので B3 は判定しない
    assert sp["B1"] and sp["B2"] and sp["B3"] is None and not sp["success"]
    assert v["H"] is False and v["H3"] is False and v["H1"] is False
