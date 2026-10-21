"""カスケード＋新しさによる捕獲の確証実験と、捕獲の駆動の対照（12.6.235節）。"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.basal_ganglia_core import ParitySpec  # noqa: E402
from neurocortex.context_inference import run_context_sequence  # noqa: E402
from neurocortex.experiments import run_context_cascade_capture_confirm as exp  # noqa: E402

# 12.6.233節のテストと同じ小さな設定
SPECS = [ParitySpec(n_inputs=4, relevant=p) for p in [(0, 1), (1, 2), (2, 3)]]
KW = dict(pretrain_steps=150, steps=150, eval_n=50, cue_n=8, batch=16, novel_patience=3, selection_trials=2,
          diagnostics=True, reliability_holdout=40, episodic_hold=20, writeback="none", writeback_at=2,
          writeback_steps=5, writeback_eval=100, writeback_keep=True, replay_rule="ach_c", cascade_levels=4,
          cascade_g=0.2, cascade_capture=0.1)


def _run(**kw):
    torch.manual_seed(0)
    return run_context_sequence("inferred_gated", SPECS, 16, torch.Generator().manual_seed(1),
                                torch.Generator().manual_seed(2), **{**KW, **kw})


def test_drive_recorded_and_replaying_it_reproduces():
    """覚醒中の各捕獲の 1 − f が記録され、同じ列を駆動として与えると同じ学習になる。別の列では変わる。"""
    a = _run()
    d = a["diag"]["cascade_drive"]
    assert len(d) == KW["steps"] and all(0.0 <= v <= 1.0 for v in d)
    b = _run(cascade_capture_drive=d)
    assert b["R"] == a["R"] and b["diag"]["cascade"] == a["diag"]["cascade"] and b["diag"]["cascade_drive"] == d
    c = _run(cascade_capture_drive=[0.0])
    assert c["diag"]["cascade"][2] != a["diag"]["cascade"][2]
    assert c["diag"]["cascade_drive"] == [0.0] * len(d)  # 長さを越えたら先頭に戻る
    assert "cascade_drive" not in _run(cascade_capture=0.0)["diag"]


@pytest.mark.parametrize("kw", [dict(cascade_capture=0.0, cascade_capture_drive=[0.5]),
                                dict(cascade_capture_drive=[]), dict(cascade_capture_drive=[1.2])])
def test_invalid_drive(kw):
    with pytest.raises(ValueError):
        _run(**kw)


def test_control_drives():
    drive = [1.0, 0.8, 0.5, 0.2, 0.0, 0.0]
    c = exp.control_drives(drive, 3)
    assert sorted(c["ach_c_cas_cap_perm"]) == sorted(drive) and c["ach_c_cas_cap_perm"] != drive
    assert c["ach_c_cas_cap_mean"] == [pytest.approx(2.5 / 6)] * 6
    assert exp.control_drives(drive, 3) == c  # シードで決まる


def test_run_one_capctl(monkeypatch):
    """capctl は ach_c_cas_cap を測り、記録した駆動の並べ替え・平均で対照の2つを測る。"""
    calls = []

    def fake(job, kw):
        calls.append((job[3], kw.get("cascade_capture_drive")))
        drv = kw.get("cascade_capture_drive") or [1.0, 0.0, 0.5]
        return {"key": f"{job[0]}/{job[1]}_L{job[2]}/{job[3]}", "row": {"seed": job[4]}, "drive": drv}

    monkeypatch.setattr(exp, "run_job", fake)
    out = exp.run_one(("parity", "reset", 0, "capctl", 200))
    assert [x["key"].rsplit("/", 1)[1] for x in out] == ["ach_c_cas_cap", *exp.CONTROLS]
    assert calls[0] == ("ach_c_cas_cap", None)
    assert sorted(calls[1][1]) == [0.0, 0.5, 1.0] and calls[2][1] == [0.5] * 3
    assert all(x["row"]["drive_len_match"] for x in out[1:]) and "drive" not in out[0]
    assert len(exp.run_one(("parity", "reset", 0, "keep", 200))) == 1


def test_phases_and_seeds():
    assert exp.CONFIRM_SEEDS == list(range(200, 210))
    assert not set(exp.CONFIRM_SEEDS) & set(exp.DEV_SEEDS + exp.EVAL_SEEDS)
    assert len(exp.jobs("confirm", exp.CONFIRM_SEEDS)) == 30 * 4 * 10
    assert len(exp.jobs("dev", exp.DEV_SEEDS)) == 30 * 5
    assert exp.CONFIGS["ach_c_cas_cap"]["cascade_capture"] == 0.1


def test_verdict_and_paired():
    o = {"end_keep": 0.14, "end_ach_c": 0.08, "end_ach_c_cas_cap": 0.11, "end_ach_c_cas_cap_perm": 0.10,
         "end_ach_c_cas_cap_mean": 0.105}
    n = {f"end_{w}": 0.01 for w in exp.RULES}
    a = {"new_keep": -0.03, "new_ach_c": 0.0, "new_ach_c_cas_cap": 0.0, "new_ach_c_cas_cap_perm": -0.01,
         "new_ach_c_cas_cap_mean": 0.0}
    for w in exp.RULES:
        a.update({f"min_new_{w}": -0.04, f"bad_{w}": 1, f"bad_frac_{w}": 1 / 300})
    v = exp.verdict({"over_14bit": o, "not_over": n, "all": a})
    # 直線: end = 0.08 + 0.06·new/(−0.03)。cap の残差 +0.03、perm 0.10 − 0.10 = 0、mean +0.025
    assert v["ach_c_cas_cap"]["residual"] == pytest.approx(0.03) and v["C"]
    assert v["N_each"] == {"ach_c_cas_cap_perm": True, "ach_c_cas_cap_mean": False} and v["N"] is False
    p = exp.paired({0: 0.1, 1: 0.3, 2: -0.1})
    assert p["n"] == 3 and p["pos"] == 2 and p["mean"] == pytest.approx(0.1)
    assert p["t"] == pytest.approx(0.1 / (0.2 / 3 ** 0.5))
