"""文脈でゲートした文脈固有の部分ネットワーク（12.6.219節）。"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.basal_ganglia_core import ParitySpec  # noqa: E402
from neurocortex.context_inference import run_context_sequence  # noqa: E402
from neurocortex.context_subnet import GatedModel, n_params  # noqa: E402
from neurocortex.experiments import run_context_replay_subnet as exp  # noqa: E402

SPECS = [ParitySpec(n_inputs=4, relevant=p) for p in [(0, 1), (1, 2), (2, 3)]]
KW = dict(pretrain_steps=150, steps=150, eval_n=50, cue_n=8, batch=16, novel_patience=3, selection_trials=2,
          diagnostics=True, reliability_holdout=40, episodic_hold=20,
          writeback="none", writeback_at=2, writeback_steps=5, writeback_eval=100)


def _run(condition="inferred_gated", hidden=16, **kw):
    torch.manual_seed(0)
    return run_context_sequence(condition, SPECS, hidden, torch.Generator().manual_seed(1),
                                torch.Generator().manual_seed(2), **{**KW, **kw})


def _x(n_bits, C, ctx, n=8):
    oh = torch.zeros(n, C)
    oh[:, ctx] = 1.0
    return torch.cat([torch.randint(0, 2, (n, n_bits)).float(), oh], dim=1)


def test_gate_isolates_context_subnets():
    """文脈 c の入力の損失は、他の枠の部分ネットワークのパラメータに勾配を流さない。c の枠には流れる。"""
    torch.manual_seed(0)
    m = GatedModel(4, 3, 8, 5)
    loss = F.cross_entropy(m(_x(4, 3, 1)), torch.zeros(8, dtype=torch.long))
    loss.backward()
    for c in range(3):
        g = [q.grad[ix].abs().sum() for q, ix in m.ctx_slices(c)]
        assert all(x > 0 for x in g) if c == 1 else all(x == 0 for x in g)
    assert m.backbone.fc1.weight.grad.abs().sum() > 0


def test_param_count_matches_wide_control():
    """部分ネットワーク（C 枠 × K）のパラメータ数は、共有の隠れ層を H + C·K に広げた対照と一致する。"""
    from neurocortex.hippocampus_replay import make_model
    for fam in ("flip_m1", "parity", "flip8_m1"):
        C = exp.n_slots(fam)
        n_in = exp.specs_for(fam)[0].n_inputs
        assert n_params(GatedModel(n_in, C, exp.HIDDEN, exp.K)) == n_params(make_model(n_in + C, exp.HIDDEN + C * exp.K))


def test_default_unchanged_and_subnet_changes_learning():
    """ctx_units = 0（既定）は従来と同じ。ctx_units > 0 では結果が変わり、oracle でも動く。none では拒否する。"""
    a, b = _run(), _run(ctx_units=0)
    assert a["R"] == b["R"] and a["assign"] == b["assign"]
    s = _run(ctx_units=4)
    assert s["diag"]["upd"] != a["diag"]["upd"]  # 正確度は飽和しうるので、更新の大きさの列で比べる
    assert len(_run("oracle", writeback="", ctx_units=4)["R"]) == 3
    with pytest.raises(ValueError):
        _run("none", writeback="", ctx_units=4)
    with pytest.raises(ValueError):
        _run(ctx_units=-1)


def test_nov_gain_applies_to_new_context_subnet():
    """nov_split の新しい文脈の利得は、その文脈の部分ネットワークにもかかる（列の利得 1 と比べて結果が変わる）。
    共有の利得 0 では、新しい文脈で学習するステップの共有の重みは動かない（利得をかけたステップがある）。"""
    kw = dict(writeback_keep=True, replay_rule="ach_c", nov_split=True, ach_ema=0.9999, ctx_units=4)
    g2 = _run(**kw)
    g1 = _run(**{**kw, "nov_ctx_gain": 1.0})
    assert g2["diag"]["nov_split"][2]["steps"] > 0
    assert g2["R"][:2] == g1["R"][:2]
    pre = [u for u in g2["diag"]["upd"] if u[0] < 2]
    assert pre == [u for u in g1["diag"]["upd"] if u[0] < 2]
    assert [u for u in g2["diag"]["upd"] if u[0] == 2] != [u for u in g1["diag"]["upd"] if u[0] == 2]


def test_verdict_logic():
    """判定（12.6.219節）: D1〜D3、D4、D5、G、C、H・H_sub。"""
    def grp(end, bad, new=0.0, mn=0.0):
        o = {}
        for w, e in end.items():
            o.update({f"end_{w}": e, f"new_{w}": new, f"min_new_{w}": mn, f"bad_{w}": bad[w],
                      f"bad_frac_{w}": bad[w] / 150})
        return o
    end = {"ach_c": 0.08, "ach_c_nov": 0.14, "ach_c_sub_nov": 0.16, "ach_c_sub": 0.09, "base_sub_nov": 0.10,
           "ach_c_wide_nov": 0.15}
    bad = {"ach_c": 3, "ach_c_nov": 14, "ach_c_sub_nov": 1, "ach_c_sub": 1, "base_sub_nov": 5, "ach_c_wide_nov": 10}
    o = grp(end, bad)
    n = grp({w: 0.0 for w in end}, bad)
    v = exp.verdict({"over_14bit": o, "not_over": n, "all": o})
    m = v["ach_c_sub_nov"]
    assert v["H"] and m["D1"] and m["D2"] and m["D3"] and m["D4"] and m["D5"] and m["G"] and m["C"]
    assert v["H_sub"] is False and v["H_fix"] is None
    o2 = {**o, "bad_ach_c_wide_nov": 1, "end_base_sub_nov": 0.12}
    v2 = exp.verdict({"over_14bit": o2, "not_over": n, "all": o2})
    assert v2["ach_c_sub_nov"]["G"] is False and v2["ach_c_sub_nov"]["D5"] is False


def test_zero_readout_starts_from_shared_layer():
    """修正1: 読み出しを 0 で始めると、どの枠でも初期の予測は共有の層だけで決まる（枠によらず同じ）。
    乱数の消費は変えないので、共有の層と部分ネットワークの入力重みは既定と同じ。"""
    torch.manual_seed(0)
    a = GatedModel(4, 3, 8, 5)
    torch.manual_seed(0)
    z = GatedModel(4, 3, 8, 5, zero_readout=True)
    assert torch.equal(a.sub.weight, z.sub.weight) and torch.equal(a.backbone.fc1.weight, z.backbone.fc1.weight)
    bits = torch.randint(0, 2, (8, 4)).float()
    outs = [z(torch.cat([bits, F.one_hot(torch.full((8,), c), 3).float()], 1)) for c in range(3)]
    shared = [z.head(torch.cat([z.backbone(torch.cat([bits, F.one_hot(torch.full((8,), c), 3).float()], 1)),
                                torch.zeros(8, 15)], 1)) for c in range(3)]
    assert all(torch.allclose(o, s) for o, s in zip(outs, shared))
