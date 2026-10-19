"""海馬の痕跡のリプレイの配分（12.6.163節）。"""

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


def _run(**kw):
    torch.manual_seed(0)
    return run_context_sequence("inferred_gated", SPECS, 16, torch.Generator().manual_seed(1),
                                torch.Generator().manual_seed(2), **{**KW, **kw})


def test_rules_apply_only_from_writeback_and_change_later_learning():
    """規則はタスク writeback_at から適用する。それまでの学習と書き戻しの前後の測定は一様なリプレイと同じ。"""
    keep = _run(writeback_keep=True)
    assert "replay_mix" not in keep["diag"]
    for rule in ("gain", "need", "sleep"):
        r = _run(writeback_keep=True, replay_rule=rule, sleep_every=20)
        assert r["R"][:2] == keep["R"][:2] and r["diag"]["writeback"] == keep["diag"]["writeback"]
        assert list(r["diag"]["replay_mix"]) == [2]
        assert r["R"][2] != keep["R"][2] or r["diag"]["ctx_acc"][2] != keep["diag"]["ctx_acc"][2]


def test_need_share_and_gain_prefers_kept_traces():
    """need は古い文脈のリプレイを半分にし、残りをそのステップの文脈から取る（痕跡がまだなければ取らない）。
    gain は損失の大きい痕跡（加えた痕跡）を一様より多く取る。"""
    m = _run(writeback_keep=True, replay_rule="need")["diag"]["replay_mix"][2]
    assert 8 <= m["size"] <= 16 and m["own"] * m["size"] <= 8 + 1e-6 and m["sleep"] == 0
    # この小さな設定では新しい文脈の痕跡が保存されないので、ほとんどのステップで古い文脈の 8 個だけを取る
    assert _run(writeback_keep=True)["diag"]["buffer"]["ctx"].max() < 2 and m["size"] < 9
    uni = _run(writeback_keep=True, replay_rule="gain", replay_beta=1.0)["diag"]["replay_mix"][2]
    pri = _run(writeback_keep=True, replay_rule="gain", replay_beta=0.0)["diag"]["replay_mix"][2]
    assert 0.0 < uni["kept"] < 1.0 and pri["kept"] > uni["kept"]


def test_sleep_separates_kept_traces_from_wake_replay():
    """sleep は覚醒中に加えた痕跡をリプレイしない。睡眠のステップが 0 なら、痕跡を加えない場合と同じ学習になる。"""
    base = _run()
    r0 = _run(writeback_keep=True, replay_rule="sleep", sleep_every=20, sleep_steps=0)
    assert r0["R"] == base["R"] and r0["diag"]["ctx_acc"] == base["diag"]["ctx_acc"]
    assert r0["diag"]["replay_mix"][2]["kept"] == 0.0
    r = _run(writeback_keep=True, replay_rule="sleep", sleep_every=20, sleep_steps=7)
    m = r["diag"]["replay_mix"][2]
    assert m["kept"] == 0.0 and m["sleep"] == 2 * 7  # ステップ 20・40 の2回
    assert r["R"][2] != base["R"][2] or r["diag"]["ctx_acc"][2] != base["diag"]["ctx_acc"][2]


def test_ach_replays_kept_traces_only_without_surprise():
    """ach は驚きのあるステップで加えた痕跡を除く。驚きがない（store）ステップでは一様（keep と同じ割合）に取る。"""
    uni = _run(writeback_keep=True, replay_rule="gain", replay_beta=1.0)["diag"]["replay_mix"][2]  # 一様なリプレイ
    m = _run(writeback_keep=True, replay_rule="ach")["diag"]["replay_mix"][2]
    mn = _run(writeback_keep=True, replay_rule="ach_need")["diag"]["replay_mix"][2]
    assert m["size"] == 16 and 0.0 <= m["kept"] < 0.5 and mn["size"] < 9
    # この小さな設定の新しいタスクは学習が終わらず驚きが続くので、加えた痕跡はほとんどリプレイされない
    assert m["kept"] < uni["kept"]


def test_replay_rule_rejects_bad_arguments():
    for kw in (dict(replay_rule="bad"), dict(replay_rule="gain", replay_beta=1.5),
               dict(replay_rule="need", replay_need=1.0), dict(replay_rule="sleep", sleep_every=0),
               dict(replay_rule="sleep", sleep_steps=-1)):
        with pytest.raises(ValueError):
            _run(**kw)


def test_experiment_jobs_analyze_and_verdict(tmp_path):
    from neurocortex.experiments.run_context_replay_alloc import CONFIGS, FAMILIES, RULES, analyze, jobs
    from neurocortex.experiments.run_context_replay_alloc import FIX1
    assert len(jobs()) == len(FAMILIES) * 2 * 3 * (len(CONFIGS) - len(FIX1)) and set(RULES) < set(CONFIGS)
    assert len(jobs(FIX1)) == len(FAMILIES) * 2 * 3 * len(FIX1)

    def row(end, new, seed=100):
        return {"seed": seed, "pre_nat": [0.4, 0.9], "end_nat": end, "gap": [0.2, 0.0], "exp_end": end[0],
                "new_nat": new, "new_exp": new, "selection_acc": 0.9, "mix_own": 0.1, "mix_kept": 0.3}

    runs = {"flip7_m1/reset_L0/base": {"per_seed": [row([0.3, 0.9], 0.8)]},
            "flip7_m1/reset_L0/keep": {"per_seed": [row([0.5, 0.9], 0.6)]},
            "flip7_m1/reset_L0/gain": {"per_seed": [row([0.5, 0.9], 0.79)]},
            "flip7_m1/reset_L0/sleep": {"per_seed": [row([0.36, 0.9], 0.8)]},
            "flip7_m1/reset_L0/sleep_buf": {"per_seed": [row([0.34, 0.9], 0.8)]},
            "parity/reset_L0/base": {"per_seed": [{**row([0.9, 0.9], 0.9), "gap": [0.0, 0.0]}]},
            "parity/reset_L0/gain": {"per_seed": [{**row([0.88, 0.9], 0.9), "gap": [0.0, 0.0]}]}}
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"runs": runs}), encoding="utf-8")
    res = analyze([p])
    o, a = res["over_14bit"], res["all"]
    assert res["pre_identical"] and o["n"] == 1 and res["not_over"]["n"] == 1
    assert o["end_gain"] == pytest.approx(0.1) and o["over_end_gain"] == pytest.approx(0.2)
    assert a["min_new_keep"] == pytest.approx(-0.2) and a["argmin_new_keep"] == "flip7_m1/reset_L0"
    v = res["verdict"]
    assert v["gain"]["A1"] and v["gain"]["A2"] and not v["gain"]["A3"] and v["gain"]["A4"] and not v["gain"]["success"]
    assert not v["sleep"]["A1"] and v["sleep"]["A2"] and v["sleep"]["A3"] is None and not v["sleep"]["A5"]
