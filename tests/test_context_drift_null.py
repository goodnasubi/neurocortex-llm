"""表現の動きの読み出しの零空間への分解（12.6.247節）。"""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex import context_inference as ci  # noqa: E402
from neurocortex.experiments import run_context_capacity_overlap as ov  # noqa: E402
from neurocortex.experiments import run_context_drift_null as exp  # noqa: E402


def test_configs():
    """12.6.243節の密な層の設定に null_diag だけを足す。"""
    assert set(exp.NEW) == {f"{r}_h{w}" for r in ov.RULES0 for w in ov.DENSE}
    for k, c in exp.CONFIGS.items():
        c = dict(c)
        assert c.pop("null_diag") is True and c == ov.CONFIGS[k]
    assert not set(exp.DEV_SEEDS) & set(exp.EVAL_SEEDS) and exp.FIXES == ()


def _head(d, c=0.0):
    """ロジットの差の重みが d、バイアスの差が c の 2クラスの読み出し。"""
    d = torch.tensor(d)
    return [torch.stack([torch.zeros_like(d), d]), torch.tensor([0.0, c])]


def test_null_stats_null_drift():
    """読み出しに直交する動きは出力に効かない（f_pot = 0、pot_rel = 0）。"""
    h0 = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    h1 = h0 + torch.tensor([0.0, 2.0])
    y = torch.tensor([1, 0])
    s = ci.null_stats(h0, h1, _head([1.0, 0.0]), _head([1.0, 0.0]), y)
    assert s["d_tot"] == pytest.approx(4.0)
    assert s["f_pot"] == pytest.approx(0.0) and s["e_iso"] == pytest.approx(0.0) and s["pot_rel"] == pytest.approx(0.0)
    assert s["harm_rep"] == pytest.approx(0.0) and s["harm_ro"] == pytest.approx(0.0)
    assert s["acc0"] == 1.0 and s["acc1"] == 1.0


def test_null_stats_potent_drift_and_compensation():
    """出力に効く方向の動きを読み出しが打ち消すと comp = 1、損なう向きなら harm_rep < 0。"""
    h0 = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    h1 = h0 * 0.5  # 古いマージンを半分にする（正解の向きに対して損なう）
    y = torch.tensor([1, 0])
    s = ci.null_stats(h0, h1, _head([1.0, 0.0]), _head([2.0, 0.0]), y)
    assert s["f_pot"] == pytest.approx(1.0) and s["e_iso"] == pytest.approx(2.0)
    assert s["pot_rel"] == pytest.approx(0.5)
    assert s["harm_rep"] == pytest.approx(-0.5) and s["harm_ro"] == pytest.approx(0.5)
    assert s["comp"] == pytest.approx(1.0)
    s2 = ci.null_stats(h0, h1, _head([1.0, 0.0]), _head([1.0, 0.0]), y)
    assert s2["comp"] == pytest.approx(0.0)


def test_null_diag_validation_and_no_effect():
    spec = ci.FlipSpec()
    g = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError):
        ci.run_context_sequence("inferred_gated", [spec, spec], 8, g, g, null_diag=True)
    specs = [ci.FlipSpec(relevant=(0, 1)), ci.FlipSpec(relevant=(2, 3)), ci.FlipSpec(relevant=(4, 5))]
    kw = dict(steps=30, pretrain_steps=30, diagnostics=True, writeback="none", writeback_keep=True, writeback_at=2,
              writeback_steps=10, writeback_eval=100, eval_n=100, recon_lr=0.01, recon_batch=8, episodic_hold=10,
              rep_diag=True)

    def run(**extra):
        torch.manual_seed(0)
        return ci.run_context_sequence("inferred_gated", specs, 16, torch.Generator().manual_seed(1),
                                       torch.Generator().manual_seed(2), **kw, **extra)

    a, b = run(), run(null_diag=True)
    assert a["R"] == b["R"] and a["assign"] == b["assign"] and a["diag"]["rep"] == b["diag"]["rep"]
    raw = b["diag"]["rep_raw"]
    assert raw["head0"][0].shape == (2, 16) and set(raw["h0"]) <= set(raw["h1"])
    assert not torch.equal(raw["head0"][0], raw["head1"][0])  # 新しいタスクの学習で読み出しも動く


def test_spearman():
    assert exp.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert exp.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert exp.spearman([1, 1, 2], [0, 0, 1]) == pytest.approx(1.0)
    assert exp.spearman([1, 2], [1, 2]) is None


def _row(seed, d_tot, pot_rel, e_iso, comp, harm, old):
    return {"seed": seed, "end_nat": [old, old], "pre_nat": [0.0, 0.0], "new_nat": 0.0, "rep_drift": 0.1,
            "gcos_tr": 0.0, "nl_d_tot": d_tot, "nl_pot_rel": pot_rel, "nl_e_iso": e_iso, "nl_comp": comp,
            "nl_harm_rep": harm, "nl_harm_ro": 0.0, "nl_f_pot": 0.1, "nl_e_rep": 1.0, "nl_f_pot1": 0.1,
            "nl_f_h0": 0.1, "nl_acc0": 0.9, "nl_acc1": 0.9}


def test_analyze_and_verdict(tmp_path):
    runs = {}
    for r in ov.RULES0:
        for w, (d_tot, pot, e, comp) in {32: (10.0, 0.8, 1.2, 0.2), 64: (20.0, 0.7, 0.9, 0.4),
                                         128: (40.0, 0.6, 0.8, 0.6)}.items():
            runs[f"flip_m1/reset_L0/{r}_h{w}"] = {"per_seed": [
                _row(s, d_tot + s, pot, e, comp, h, h) for s, h in ((0, -0.2), (1, 0.1), (2, 0.3))]}
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"seeds": [0, 1, 2], "runs": runs}))
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"runs": runs}))
    res = exp.analyze([p], [ref])
    assert res["reproduce"] == {"n": 27, "identical": 27}
    assert res["summary"]["keep_h32"]["nl_d_tot"] == pytest.approx(11.0)
    assert res["summary"]["keep_h32"]["old_change"] == pytest.approx(0.2 / 3)
    ps = res["per_seed"]["keep/nl_d_tot"]
    assert ps["mean"] == pytest.approx(30.0) and ps["n_pos"] == 3
    v = res["verdict"]
    assert v["Z"] and v["E"] and v["K"] and v["F"] and v["F_rho"] == pytest.approx(1.0)


def test_verdict_negative():
    summ = {f"{r}_h{w}": {"nl_d_tot": 1.0, "nl_pot_rel": 0.5 + w / 1000, "nl_e_iso": 1.5, "nl_comp": 0.1,
                          "nl_harm_rep": 0.0, "nl_harm_ro": 0.0} for r in ov.RULES0 for w in ov.DENSE}
    v = exp.verdict(summ, {}, set())
    assert v["Z"] is False and v["E"] is False and v["K"] is False and v["F"] is None
