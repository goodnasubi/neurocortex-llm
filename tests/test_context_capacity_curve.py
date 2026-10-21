"""共有の隠れ層の幅と交換の曲線（12.6.241節）。"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.experiments import run_context_capacity_curve as exp  # noqa: E402
from neurocortex.experiments import run_context_replay_cascade as cas  # noqa: E402


def test_configs_only_change_width():
    assert exp.WIDTHS == (64, 128) and exp.BASE_W == 32
    for w in exp.WIDTHS:
        for r in exp.RULES0:
            c = dict(exp.CONFIGS[f"{r}_h{w}"])
            assert c.pop("hidden") == w and c == exp.CONFIGS[r]
    assert not set(exp.DEV_SEEDS) & set(exp.EVAL_SEEDS)
    assert exp.FIXES == ()


def test_run_job_hidden_default_and_width(monkeypatch):
    """run_job は hidden を取り除いて幅として渡し、既定は 32（従来と同じ）。渡した設定の dict は変えない。"""
    seen = []

    def fake(cond, specs, hidden, *a, **kw):
        seen.append((hidden, "hidden" in kw))
        raise RuntimeError("stop")

    monkeypatch.setattr(cas, "run_context_sequence", fake)
    kw = {"hidden": 128}
    for k in ({}, kw):
        with pytest.raises(RuntimeError):
            cas.run_job(("parity", "reset", 0, "x", 0), k)
    assert seen == [(32, False), (128, False)] and kw == {"hidden": 128}


def _row(seed, end, new, gap):
    return {"seed": seed, "end_nat": end, "new_nat": new, "gap": gap, "pre_nat": [0.0, 0.0], "mix_kept": None}


def _runs(cfgs: dict) -> dict:
    """cfgs: 設定の名前 → (end, new)。2組（flip_m1 の上書きあり、parity の上書きなし）× 2シード。"""
    runs = {}
    for cb, gap in (("flip_m1/reset_L0", [0.2, 0.2]), ("parity/reset_L0", [0.0, 0.0])):
        for cfg, (end, new) in cfgs.items():
            runs[f"{cb}/{cfg}"] = {"per_seed": [_row(s, [end, end], new, gap) for s in (0, 1)]}
    return runs


def test_analyze_frames_and_verdict(tmp_path):
    # 幅 32: base (0, 0)、keep (+0.14, −0.03)、ach_c (+0.08, 0)。直線 end = 0.08 − 2·new
    ref = _runs({"base": (0.0, 0.0), "keep": (0.14, -0.03), "ach_c": (0.08, 0.0)})
    # 幅 64: base_h64 が幅 32 より end +0.05（abs の枠でだけ効く）。in の枠で keep (+0.15, −0.02) → 残差 +0.03、
    # ach_c (+0.09, 0) → +0.01。幅 128: in で keep (+0.16, −0.01) → +0.06、ach_c (+0.10, 0) → +0.02
    new = _runs({"base_h64": (0.05, 0.0), "keep_h64": (0.20, -0.02), "ach_c_h64": (0.14, 0.0),
                 "base_h128": (0.05, 0.0), "keep_h128": (0.21, -0.01), "ach_c_h128": (0.15, 0.0)})
    p0, p1 = tmp_path / "ref.json", tmp_path / "new.json"
    p0.write_text(json.dumps({"seeds": [0, 1, 2], "runs": ref}))
    p1.write_text(json.dumps({"seeds": [0, 1], "runs": new}))
    res = exp.analyze([p0, p1])
    assert res["seeds"] == [0, 1]
    pts = res["points"]
    assert pts["32"]["keep"]["end"] == pytest.approx(0.14) and pts["32"]["ach_c"]["new"] == pytest.approx(0.0)
    assert pts["64_in"]["keep"]["residual"] == pytest.approx(0.03)
    assert pts["64_in"]["ach_c"]["residual"] == pytest.approx(0.01)
    assert pts["64_abs"]["ach_c"]["end"] == pytest.approx(0.14) and pts["64_abs"]["ach_c"]["residual"] == pytest.approx(0.06)
    assert pts["128_in"]["keep"]["residual"] == pytest.approx(0.06)
    v = res["verdict"]
    assert v["res_64_in"] == pytest.approx(0.02) and v["res_128_in"] == pytest.approx(0.04)
    assert v["H"] and v["A"] and v["M"]
    # K: 代価 ach_c − keep の new は幅 32 で 0.03、幅 128 で 0.01（半分以下）
    assert v["cost_32"] == pytest.approx(0.03) and v["cost_128"] == pytest.approx(0.01) and v["K"]
    assert res["pre_identical"] == {"32": True, "64": True, "128": True}


def test_verdict_negative():
    def pt(end, new, res):
        return {"end": end, "new": new, "residual": res, "crit": {"success": False}}
    points = {"32": {"keep": pt(0.14, -0.03, None), "ach_c": pt(0.08, 0.0, None)},
              "64_in": {"keep": pt(0.14, -0.03, 0.01), "ach_c": pt(0.08, 0.0, -0.01)},
              "64_abs": {"keep": pt(0.14, -0.03, 0.03), "ach_c": pt(0.08, 0.0, 0.02)},
              "128_in": {"keep": pt(0.14, -0.025, -0.01), "ach_c": pt(0.08, 0.0, 0.0)},
              "128_abs": {"keep": pt(0.14, -0.03, 0.03), "ach_c": pt(0.08, 0.0, 0.0)}}
    v = exp.verdict(points)
    assert v["H"] is False and v["A"] is False and v["M"] is False and v["K"] is False
    assert v["joint"] == []
    points["128_abs"]["ach_c"]["residual"] = 0.01  # 平均 +0.02 → A 合格
    assert exp.verdict(points)["A"] is True


def test_residual_none_when_missing():
    line = ({"end": 0.1, "new": -0.03}, {"end": 0.05, "new": -0.03})
    assert exp.residual(line, {"end": 0.1, "new": 0.0}) is None  # 横軸が同じ
    assert exp.residual(({"end": 0.1, "new": -0.03}, {"end": None, "new": 0.0}), {"end": 0.1, "new": 0.0}) is None
