"""連続的な新しさの信号と、文脈固有・共有の可塑性の分離の組み合わせ（12.6.169節）。"""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.basal_ganglia_core import ParitySpec  # noqa: E402
from neurocortex.context_inference import run_context_sequence  # noqa: E402

# 新しいタスクの文脈が確立し、痕跡を持つ小さな設定（12.6.167節のテストと同じ）
SPECS = [ParitySpec(n_inputs=4, relevant=p) for p in [(0, 1), (1, 2), (2, 3)]]
KW = dict(pretrain_steps=150, steps=150, eval_n=50, cue_n=8, batch=16, novel_patience=3, selection_trials=2,
          diagnostics=True, reliability_holdout=40, episodic_hold=20,
          writeback="none", writeback_at=2, writeback_steps=5, writeback_eval=100)


def _run(condition="inferred_gated", **kw):
    torch.manual_seed(0)
    return run_context_sequence(condition, SPECS, 16, torch.Generator().manual_seed(1),
                                torch.Generator().manual_seed(2), **{**KW, **kw})


def test_nov_split_only_from_writeback_task_on_new_contexts():
    """利得はタスク writeback_at から、その時点になかった（新しい）文脈で学習するステップにだけかかる。それまでの
    学習と書き戻しの前の測定は変えない。共有の利得の平均は [nov_shared_gain, 1] に入る。"""
    base = _run(writeback_keep=True, replay_rule="ach_c")
    r = _run(writeback_keep=True, replay_rule="ach_c", nov_split=True)
    assert r["R"][:2] == base["R"][:2] and r["diag"]["writeback"] == base["diag"]["writeback"]
    assert "nov_split" not in base["diag"] and list(r["diag"]["nov_split"]) == [2]
    d = r["diag"]["nov_split"][2]
    assert d["steps"] > 0 and 0.2 - 1e-9 <= d["g_shared"] <= 1.0 + 1e-9
    assert r["diag"]["ctx_acc"][2] != base["diag"]["ctx_acc"][2]
    # 新しい文脈で学習しないステップには利得をかけないので、利得をかけたステップは学習のステップ数以下
    assert d["steps"] <= KW["steps"]


def test_nov_split_gain_follows_familiarity():
    """なじみの度合い f が 0 のまま（損失の移動平均がほとんど動かない）なら、共有の利得は常に nov_shared_gain。
    速くなじむ（ach_ema = 0）と、平均の利得は 1 に近づく。"""
    slow = _run(nov_split=True, ach_ema=0.9999)["diag"]["nov_split"][2]
    fast = _run(nov_split=True, ach_ema=0.0)["diag"]["nov_split"][2]
    assert slow["g_shared"] == pytest.approx(0.2) and fast["g_shared"] > slow["g_shared"]


def test_nov_split_floor_and_no_effect_cases():
    """下限（12.6.132節）で共有の利得が 1 になる。oracle では効かない。既定（False）は診断を加えない。"""
    r = _run(nov_split=True, split_shared_lr_floor=0.05, ach_ema=0.9999)
    assert r["diag"]["nov_split"][2]["g_shared"] == pytest.approx(1.0)
    o, ob = _run("oracle", nov_split=True, writeback=""), _run("oracle", writeback="")
    assert o["R"] == ob["R"] and "nov_split" not in o["diag"]


def test_nov_split_shared_gain_zero_freezes_shared_weights():
    """共有の利得 0・新しい文脈の列の利得 1 で f が 0 のままなら、新しい文脈で学習するステップでは共有の重みが
    動かないので、新しいタスクは元の写像のまま（f が 0 でない利得の版より新しいタスクの正確度が上がらない）。"""
    frozen = _run(nov_split=True, nov_shared_gain=0.0, nov_ctx_gain=1.0, ach_ema=0.9999)
    free = _run()
    assert frozen["diag"]["nov_split"][2]["g_shared"] == pytest.approx(0.0)
    assert frozen["R"][2][2] <= free["R"][2][2]


def test_nov_split_rejects_bad_arguments():
    for kw in (dict(nov_split=True, nov_ctx_gain=0.5), dict(nov_split=True, nov_shared_gain=1.5),
               dict(nov_split=True, nov_shared_gain=-0.1)):
        with pytest.raises(ValueError):
            _run(**kw)


def test_experiment_jobs_analyze_and_verdict(tmp_path):
    from neurocortex.experiments.run_context_replay_split import (CONFIGS, DEV_SEEDS, EVAL_SEEDS, FAMILIES, NEW,
                                                                  REUSED, analyze, jobs)
    assert len(jobs()) == len(FAMILIES) * 2 * 3 * len(NEW) and set(NEW) | set(REUSED) == set(CONFIGS)
    assert len(DEV_SEEDS) == len(EVAL_SEEDS) == 10 and not set(DEV_SEEDS) & set(EVAL_SEEDS)
    assert CONFIGS["ach_c_nov"]["nov_split"] and CONFIGS["ach_c_nov"]["replay_rule"] == "ach_c"
    assert "replay_rule" not in CONFIGS["base_nov"] and not CONFIGS["base_nov"].get("writeback_keep")

    def row(end, new, seed, gap=(0.3, 0.0)):
        return {"seed": seed, "pre_nat": [0.4, 0.9], "end_nat": end, "gap": list(gap), "exp_end": end[0],
                "new_nat": new, "new_exp": new, "selection_acc": 0.9, "mix_own": 0.1, "mix_kept": 0.3,
                "mix_fam": 0.5, "nov_steps": 100, "nov_g_shared": 0.4}

    seeds = range(100, 110)
    old = {"flip7_m1/reset_L0/base": {"per_seed": [row([0.3, 0.9], 0.8, s) for s in seeds]},
           "flip7_m1/reset_L0/ach_c": {"per_seed": [row([0.38, 0.9], 0.8 - 0.2 * (s < 102), s) for s in seeds]},
           "parity/split_L0/base": {"per_seed": [row([0.9, 0.9], 0.9, s, (0.0, 0.0)) for s in seeds]},
           "parity/split_L0/ach_c": {"per_seed": [row([0.9, 0.9], 0.9, s, (0.0, 0.0)) for s in seeds]}}
    new = {"flip7_m1/reset_L0/ach_c_nov": {"per_seed": [row([0.6, 0.9], 0.8 - 0.2 * (s == 100), s) for s in seeds]},
           "flip7_m1/reset_L0/base_nov": {"per_seed": [row([0.32, 0.9], 0.8, s) for s in seeds]},
           "flip7_m1/reset_L0/ach_c_nov_ctx": {"per_seed": [row([0.38, 0.9], 0.8, s) for s in seeds]},
           "parity/split_L0/ach_c_nov": {"per_seed": [row([0.9, 0.9], 0.9, s, (0.0, 0.0)) for s in seeds]}}
    p_old, p_new = tmp_path / "old.json", tmp_path / "new.json"
    p_old.write_text(json.dumps({"runs": old}), encoding="utf-8")
    p_new.write_text(json.dumps({"runs": new}), encoding="utf-8")
    res = analyze([p_old, p_new])
    a, o = res["all"], res["over_14bit"]
    assert res["pre_identical"] and o["n"] == 1 and res["not_over"]["n"] == 1 and res["split_all"]["n"] == 1
    assert a["runs_ach_c_nov"] == 20 and a["bad_ach_c_nov"] == 1 and a["bad_ach_c"] == 2
    assert o["end_ach_c_nov"] == pytest.approx(0.15) and o["end_ach_c"] == pytest.approx(0.04)
    v = res["verdict"]
    # ach_c_nov: 上書きありの end +0.15 ≧ 0.10、−0.10 未満の回が 1/20 > 1% で D2 は不合格
    assert v["ach_c_nov"]["D1"] and v["ach_c_nov"]["D3"] and not v["ach_c_nov"]["D2"] and not v["H"]
    # D4: ach_c からの上積み（+0.11 ≧ 0.02）かつ大きく損なった回が増えない（1 ≦ 2）。D5: base_nov との差 +0.14
    assert v["ach_c_nov"]["D4"] and v["ach_c_nov"]["D5"]
    # 切り分けの ach_c_nov_ctx は上積み 0 なので D4 不合格。上書きなしの組がないので D3 は判定しない
    assert not v["ach_c_nov_ctx"]["D4"] and v["ach_c_nov_ctx"]["D3"] is None
    assert v["H_fix"] is None
