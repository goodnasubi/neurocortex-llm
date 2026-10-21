"""不確かさによる新しさの信号（12.6.225節）。"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.basal_ganglia_core import ParitySpec  # noqa: E402
from neurocortex.context_inference import run_context_sequence  # noqa: E402
from neurocortex.experiments import run_context_replay_uncertainty as ru  # noqa: E402

SPECS = [ParitySpec(n_inputs=8, relevant=p) for p in [(0, 1), (2, 3), (4, 5)]]
KW = dict(pretrain_steps=60, steps=60, eval_n=50, cue_n=8, batch=16, novel_patience=3, selection_trials=10,
          diagnostics=True, reliability_holdout=40, episodic_hold=20,
          writeback="none", writeback_at=2, writeback_steps=5, writeback_eval=100)
SPECS2 = [ParitySpec(n_inputs=4, relevant=p) for p in [(0, 1), (1, 2), (2, 3)]]
KW2 = {**KW, "pretrain_steps": 150, "steps": 150, "selection_trials": 2}


def _run(specs=SPECS2, base_kw=KW2, **kw):
    torch.manual_seed(0)
    return run_context_sequence("inferred_gated", specs, 16, torch.Generator().manual_seed(1),
                                torch.Generator().manual_seed(2), **{**base_kw, **kw})


def test_ach_c_unchanged_by_new_estimates():
    """不確かさの推定を加えても、ach_c の学習（12.6.167節）は変わらない。"""
    a = _run(writeback_keep=True, replay_rule="ach_c")
    b = _run(writeback_keep=True, replay_rule="ach_c", ach_ema=0.98)
    assert a["R"] == b["R"] and a["diag"]["replay_mix"] == b["diag"]["replay_mix"]


@pytest.mark.parametrize("rule", ["ach_u", "ach_v", "ach_uv"])
def test_uncertainty_rules_share_ach_c_allocation(rule):
    """書き戻しの前は従来と同じで、加えた痕跡はなじみの分からしか取らない（kept ≦ f の平均）。"""
    keep = _run(writeback_keep=True)
    r = _run(writeback_keep=True, replay_rule=rule, ach_ema=0.0)
    assert r["R"][:2] == keep["R"][:2] and r["diag"]["writeback"] == keep["diag"]["writeback"]
    m = r["diag"]["replay_mix"][2]
    assert 0.0 <= m["fam"] <= 1.0 and m["kept"] <= m["fam"] + 1e-6


def test_volatility_delays_familiarity():
    """損失の揺らぎを足す ach_v のなじみの度合いは、同じ損失の平均を使う ach_c を超えない。"""
    c = _run(writeback_keep=True, replay_rule="ach_c")["diag"]["replay_mix"][2]
    v = _run(writeback_keep=True, replay_rule="ach_v")["diag"]["replay_mix"][2]
    assert v["fam"] <= c["fam"] + 1e-6


def test_surprise_only_equals_no_kept_traces():
    """驚きが続く（f が常に 0）設定では、どの規則も痕跡を加えない場合と同じ学習になる。"""
    base = _run(SPECS, KW)
    for rule in ("ach_u", "ach_v", "ach_uv"):
        r = _run(SPECS, KW, writeback_keep=True, replay_rule=rule)
        assert r["diag"]["replay_mix"][2]["fam"] == 0.0
        assert r["R"] == base["R"]


def _res(end, bad, new=-0.0, worst=-0.01, intact=0.01, runs=150):
    o, n, a = {}, {}, {}
    for w in ru.RULES:
        o[f"end_{w}"] = end.get(w, 0.08)
        n[f"end_{w}"] = intact
        a[f"new_{w}"] = new
        a[f"min_new_{w}"] = worst
        a[f"bad_{w}"] = bad.get(w, 2)
        a[f"bad_frac_{w}"] = bad.get(w, 2) / runs
    return {"over_14bit": o, "not_over": n, "all": a}


def test_verdict_criteria():
    """B1〜B3 は 12.6.167節と同じ。B6 は ach_c からのパレート改善。"""
    v = ru.verdict(_res({"ach_u": 0.11}, {"ach_u": 1, "ach_v": 1, "ach_uv": 4}))
    assert v["ach_u"]["success"] and v["ach_u"]["B6"] and v["H_u"] and v["H"] and v["H_P"]
    assert not v["ach_v"]["success"] and v["ach_v"]["B6"]  # 一般化は同じで保護が半分以下（2 → 1）
    assert not v["ach_uv"]["B6"]  # 保護が悪い
    v = ru.verdict(_res({"ach_u": 0.095}, {"ach_u": 3, "ach_v": 2, "ach_uv": 2}))
    assert not v["H"] and not v["H_P"]  # +0.015 の一般化は基準（+0.02）未満、保護も変わらない
