"""海馬のバッファの候補の切り分けと、対立仮説が複数残る場面の検証（12.6.151節）。"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.basal_ganglia_core import ParitySpec  # noqa: E402
from neurocortex.context_inference import FlipSpec, make_batch, run_context_sequence, task_labels  # noqa: E402

SPECS = [ParitySpec(n_inputs=8, relevant=p) for p in [(0, 1), (2, 3), (4, 5)]]
KW = dict(pretrain_steps=60, steps=60, eval_n=50, cue_n=8, batch=16, novel_patience=3, selection_trials=10,
          diagnostics=True)
ACT = dict(cue_select="evidence", evidence_lapse=0.02, evidence_alpha=0.001, active_pool=4)


def _run(**kw):
    torch.manual_seed(0)
    return run_context_sequence("inferred_gated", SPECS, 16, torch.Generator().manual_seed(1),
                                torch.Generator().manual_seed(2), **{**KW, **kw})


def test_holdout_storage_does_not_change_training_or_selection():
    """保留サンプルの保存は学習・乱数の消費・選択に影響しない（12.6.147節の値がそのまま再現する）。"""
    base = _run(**ACT)
    held = _run(**ACT, reliability_holdout=20)
    assert held["assign"] == base["assign"] and held["R"] == base["R"]
    assert held["selection_choices"] == base["selection_choices"] and held["selection_cues"] == base["selection_cues"]
    for kw in (dict(active_source="holdout"), dict(active_verify=2, verify_mode="all"),
               dict(active_verify=1, verify_mode="each")):
        r = _run(**ACT, reliability_holdout=20, **kw)
        assert r["assign"] == base["assign"] and r["n_contexts"] >= 2
        assert all(1 <= n <= KW["cue_n"] for row in r["selection_cues"] for n in row)


def test_verify_top2_default_unchanged():
    a = _run(**ACT, active_verify=2)
    b = _run(**ACT, active_verify=2, verify_mode="top2")
    assert a["selection_choices"] == b["selection_choices"] and a["selection_cues"] == b["selection_cues"]


@pytest.mark.parametrize("mode,n", [("all", 2), ("each", 1)])
def test_verify_modes_only_add_cues(mode, n):
    """検証は閾値に達した後に手がかりを足すだけなので、決定までの手がかりの数は active より減らない。"""
    a = _run(**ACT)
    v = _run(**ACT, active_verify=n, verify_mode=mode)
    na = [x for row in a["selection_cues"] for x in row]
    nv = [x for row in v["selection_cues"] for x in row]
    assert all(y >= x for x, y in zip(na, nv)) and sum(nv) > sum(na)


def test_rival_share_and_order():
    r = _run(**ACT)
    used = set(range(r["n_contexts"]))
    for shares, orders in zip(r["selection_rival_share"], r["selection_order"]):
        for s, o in zip(shares, orders):
            if o is None:
                assert s is None
                continue
            assert 0.0 <= s <= 1.0 + 1e-9 and set(o) == used and len(o) == len(used)
            assert s >= 1.0 / (len(used) - 1) - 1e-6  # 2番目の文脈は先頭以外で最大
    assert "selection_rival_share" not in _run(cue_select="batch")


def test_origin_acc_structure():
    r = _run(**ACT, reliability_holdout=20)
    oa = r["diag"]["origin_acc"]
    assert len(oa) == len(SPECS) and all(len(row) == r["n_contexts"] for row in oa)
    vals = [v for row in oa for d in row for v in d.values()]
    assert all(v is None or 0.0 <= v <= 1.0 for v in vals) and any(v is not None for v in vals)
    assert any(d["holdout"] is not None for row in oa for d in row)
    no_hold = _run(**ACT)["diag"]["origin_acc"]
    assert all(d["holdout"] is None for row in no_hold for d in row)


@pytest.mark.parametrize("bad", [dict(active_source="holdout"), dict(active_verify=2, verify_mode="x")])
def test_invalid_args(bad):
    with pytest.raises(ValueError):
        _run(**{**ACT, **bad})


def test_flip7_specs_and_jobs():
    from neurocortex.experiments.run_context_replay_verify import (REPLAY_SELECTS, VERIFY_SELECTS, jobs,
                                                                     specs_for)
    specs = specs_for("flip7_m1")
    assert len(specs) == 7 and all(isinstance(s, FlipSpec) and s.n_inputs == 14 for s in specs)
    bits, _ = make_batch(specs[0], 4000, torch.Generator().manual_seed(0))
    labels = [task_labels(s, bits) for s in specs]
    for i in range(7):
        for k in range(i + 1, 7):  # どの2タスクも入力の約半分で食い違う
            assert abs(float((labels[i] != labels[k]).float().mean()) - 0.5) < 0.05
    assert len(jobs("replay", REPLAY_SELECTS)) == 6 * 2 * 3 * 3 * len(REPLAY_SELECTS)
    assert len(jobs("verify", VERIFY_SELECTS)) == 2 * 3 * 3 * len(VERIFY_SELECTS)
    assert REPLAY_SELECTS["gen_holdout"]["active_source"] == "holdout"
    assert VERIFY_SELECTS["verify_each"]["verify_mode"] == "each"


def test_frontier_interpolation():
    from neurocortex.experiments.run_context_replay_verify import frontier
    pts = [(5.0, 0.9), (3.0, 0.8), (8.0, 0.95)]
    assert abs(frontier(pts, 4.0) - 0.85) < 1e-9
    assert frontier(pts, 1.0) == 0.8 and frontier(pts, 12.0) == 0.95  # 範囲の外は端の値
