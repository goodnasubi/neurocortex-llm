"""容量の効果と表現の重なり・k-WTA（12.6.243節）。"""

import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex import context_inference as ci  # noqa: E402
from neurocortex.experiments import run_context_capacity_overlap as exp  # noqa: E402
from neurocortex.experiments import run_context_replay_cascade as cas  # noqa: E402


def test_configs():
    """密な層は 12.6.241節と同じ設定に rep_diag だけを足し、k-WTA は k = W/4。"""
    for r in exp.RULES0:
        for w in exp.DENSE:
            c = dict(exp.CONFIGS[f"{r}_h{w}"])
            assert c.pop("hidden") == w and c.pop("rep_diag") is True and c == exp.SPLIT_CONFIGS[r]
        for w, k in exp.SPARSE.items():
            c = dict(exp.CONFIGS[f"{r}_k{w}"])
            assert k == w // 4 and c.pop("kwta") == k and c.pop("hidden") == w and c.pop("rep_diag") is True
            assert c == exp.SPLIT_CONFIGS[r]
    assert not set(exp.DEV_SEEDS) & set(exp.EVAL_SEEDS) and exp.FIXES == ()


def test_kwta_backbone_keeps_top_k():
    torch.manual_seed(0)
    fc1 = nn.Linear(5, 8)
    bb = ci.KWTABackbone(fc1, 3)
    x = torch.randn(4, 5)
    h = bb(x)
    z = fc1(x)
    assert ((h != 0).sum(-1) == 3).all()
    top = z.topk(3, dim=-1).indices
    assert torch.allclose(h.gather(-1, top), torch.tanh(z).gather(-1, top))
    h.sum().backward()  # 勾配は勝者のユニットだけを通る
    assert fc1.weight.grad is not None


def test_rep_stats():
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    s = ci.rep_stats({0: a}, {0: a, 1: b})
    assert s["drift"] == {0: pytest.approx(0.0)}
    assert s["overlap"][0][1] == pytest.approx(0.5) and s["overlap"][1][1] == pytest.approx(1.0)
    # 1 個だけ活動する 2 ユニットの疎さは 1/2
    assert s["sparse"][0] == pytest.approx(0.5) and s["active"][1] == pytest.approx(0.5)


def test_kwta_validation():
    spec = ci.FlipSpec()
    g = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError):
        ci.run_context_sequence("inferred_gated", [spec, spec], 8, g, g, kwta=8)
    with pytest.raises(ValueError):
        ci.run_context_sequence("inferred_gated", [spec, spec], 8, g, g, rep_diag=True)


def test_rep_diag_does_not_change_learning():
    """rep_diag の診断は学習の乱数と更新を変えない。k-WTA は同じ初期化から別の結果になる。"""
    specs = [ci.FlipSpec(relevant=(0, 1)), ci.FlipSpec(relevant=(2, 3)), ci.FlipSpec(relevant=(4, 5))]
    kw = dict(steps=30, pretrain_steps=30, diagnostics=True, writeback="none", writeback_keep=True, writeback_at=2,
              writeback_steps=10, writeback_eval=100, eval_n=100, recon_lr=0.01, recon_batch=8, episodic_hold=10)

    def run(**extra):
        torch.manual_seed(0)
        return ci.run_context_sequence("inferred_gated", specs, 16, torch.Generator().manual_seed(1),
                                       torch.Generator().manual_seed(2), **kw, **extra)

    a, b, c = run(), run(rep_diag=True), run(rep_diag=True, kwta=4)
    assert a["R"] == b["R"] and a["assign"] == b["assign"]
    rep = b["diag"]["rep"]
    assert set(rep) >= {"drift", "overlap", "sparse", "active", "gcos_tr", "gcos_old"}
    assert all(rep[f"gcos_{n}"] is None or -1.0 <= rep[f"gcos_{n}"] <= 1.0 for n in ("tr", "old"))
    assert all(v > 0.5 for v in rep["sparse"].values())
    assert all(v <= 4 / 16 + 1e-6 for v in c["diag"]["rep"]["active"].values())


def _row(seed, end, new, gap, g=None, ov=None):
    return {"seed": seed, "end_nat": end, "new_nat": new, "gap": gap, "pre_nat": [0.0, 0.0], "mix_kept": None,
            "gcos_tr": g, "rep_overlap": ov, "rep_drift": 0.1, "rep_sparse": 0.5, "rep_active": 0.5, "gcos_old": -0.1}


def _runs(cfgs: dict) -> dict:
    """cfgs: 設定の名前 → (end, new, gcos_tr, rep_overlap)。2組（上書きあり・なし）× 2シード。"""
    runs = {}
    for cb, gap in (("flip_m1/reset_L0", [0.2, 0.2]), ("parity/reset_L0", [0.0, 0.0])):
        for cfg, (end, new, g, ov) in cfgs.items():
            runs[f"{cb}/{cfg}"] = {"per_seed": [_row(s, [end, end], new, gap, g, ov) for s in (0, 1)]}
    return runs


def test_analyze_and_verdict(tmp_path):
    # 幅 32 の直線: keep (+0.14, −0.03)、ach_c (+0.08, 0) → end = 0.08 − 2·new
    runs = _runs({
        "base_h32": (0.0, 0.0, None, 0.8), "keep_h32": (0.14, -0.03, -0.10, 0.8), "ach_c_h32": (0.08, 0.0, -0.1, 0.8),
        "base_h64": (0.0, 0.0, None, 0.8), "keep_h64": (0.15, -0.02, -0.08, 0.8), "ach_c_h64": (0.09, 0.0, -0.1, 0.8),
        "base_h128": (0.0, 0.0, None, 0.8), "keep_h128": (0.16, -0.01, 0.06, 0.8), "ach_c_h128": (0.10, 0.0, 0.0, 0.8),
        # k-WTA 幅 32: base_k32 が end −0.05。in で keep (+0.17, −0.02) → +0.05、ach_c (+0.11, 0) → +0.03
        "base_k32": (-0.05, 0.0, None, 0.6), "keep_k32": (0.12, -0.02, -0.05, 0.6), "ach_c_k32": (0.06, 0.0, 0.0, 0.6),
        # 幅 128 の k-WTA: in の残差 keep +0.06、ach_c +0.02（密な層と同じ → Q 不合格）
        "base_k128": (0.0, 0.0, None, 0.7), "keep_k128": (0.16, -0.01, 0.0, 0.7), "ach_c_k128": (0.10, 0.0, 0.0, 0.7),
    })
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"seeds": [0, 1], "runs": runs}))
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"runs": {k.replace("_h32", ""): v for k, v in runs.items() if k.endswith("_h32")}}))
    res = exp.analyze([p], [ref])
    pts = res["points"]
    assert pts["k32_in"]["keep"]["residual"] == pytest.approx(0.05)
    assert pts["k32_in"]["ach_c"]["residual"] == pytest.approx(0.03)
    assert pts["k32_abs"]["keep"]["end"] == pytest.approx(0.12)
    assert res["reproduce"] == {"n": 12, "identical": 12}
    assert res["rep"]["keep_h32"]["abs_gcos_tr"] == pytest.approx(0.10)
    v = res["verdict"]
    assert v["res_k32_in"] == pytest.approx(0.04) and v["S"]
    assert v["res_h128"] == pytest.approx(0.04) and v["res_k128_in"] == pytest.approx(0.04) and not v["Q"]
    assert v["O"]  # |g| 0.10 > 0.08 > 0.06 ≤ 0.08
    assert v["P"]  # |g| 0.05 ≤ 0.08、重なり 0.6 < 0.8


def test_verdict_negative_and_missing():
    pts = {"h128": {"keep": {"residual": 0.0, "crit": {"success": False}},
                    "ach_c": {"residual": 0.0, "crit": {"success": False}}}}
    rep = {"keep_h32": {"abs_gcos_tr": 0.1, "rep_overlap": 0.5}, "keep_h64": {"abs_gcos_tr": 0.12},
           "keep_h128": {"abs_gcos_tr": 0.05}, "keep_k32": {"abs_gcos_tr": 0.09, "rep_overlap": 0.4}}
    v = exp.verdict(pts, rep)
    assert v["S"] is None and v["Q"] is None and v["O"] is False and v["P"] is False


def test_run_job_passes_kwta_and_rep(monkeypatch):
    seen = []

    def fake(cond, specs, hidden, *a, **kw):
        seen.append((hidden, kw.get("kwta"), kw.get("rep_diag")))
        raise RuntimeError("stop")

    monkeypatch.setattr(cas, "run_context_sequence", fake)
    with pytest.raises(RuntimeError):
        cas.run_job(("parity", "reset", 0, "keep_k32", 0), exp.CONFIGS["keep_k32"])
    assert seen == [(32, 8, True)]
