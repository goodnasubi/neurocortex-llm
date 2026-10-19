"""テスト時の再固定化・隠れ層の表現での読み出し・保留サンプルの正確さによる調停（12.6.159節）。"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.basal_ganglia_core import ParitySpec  # noqa: E402
from neurocortex.context_inference import (EPISODIC_EXTRA_READOUTS, EPISODIC_READOUTS,  # noqa: E402
                                           RECON_READOUTS, episodic_readout, episodic_readout_dist,
                                           run_context_sequence)

SPECS = [ParitySpec(n_inputs=8, relevant=p) for p in [(0, 1), (2, 3), (4, 5)]]
KW = dict(pretrain_steps=60, steps=60, eval_n=50, cue_n=8, batch=16, novel_patience=3, selection_trials=10,
          diagnostics=True, reliability_holdout=40, episodic_eval=200, episodic_hold=20)


def _run(**kw):
    torch.manual_seed(0)
    return run_context_sequence("inferred_gated", SPECS, 16, torch.Generator().manual_seed(1),
                                torch.Generator().manual_seed(2), **{**KW, **kw})


def test_readout_dist_matches_hamming_readout():
    """距離をハミング距離で与えれば episodic_readout と一致する（表現での検索の尺度合わせの基準）。"""
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 2, (5, 6), generator=g).float() * 2 - 1
    sb = torch.randint(0, 2, (7, 6), generator=g).float() * 2 - 1
    sl = torch.randint(0, 2, (7,), generator=g)
    p = torch.softmax(torch.randn(5, 2, generator=g), -1)
    d = (6 - x @ sb.T) / 2
    for rule in ("mix", "switch"):
        assert torch.allclose(episodic_readout_dist(p, d, sl, rule=rule), episodic_readout(p, x, sb, sl, rule=rule))
    assert torch.equal(episodic_readout_dist(p, torch.zeros(5, 0), sl[:0]), p)
    with pytest.raises(ValueError):
        episodic_readout_dist(p, d, sl, rule="bad")


def test_extra_and_recon_do_not_change_training_selection_or_readouts():
    base = _run()
    ext = _run(episodic_extra=True, recon_steps=5)
    assert ext["assign"] == base["assign"] and ext["R"] == base["R"]
    assert ext["selection_choices"] == base["selection_choices"]
    d, d0 = ext["diag"], base["diag"]
    T, n_ctx = len(SPECS), ext["n_contexts"]
    keys = set(EPISODIC_READOUTS) | set(EPISODIC_EXTRA_READOUTS) | set(RECON_READOUTS)
    for row, row0 in zip(d["episodic_acc"], d0["episodic_acc"]):
        for accs, accs0 in zip(row, row0):
            assert set(accs) == keys and all(0.0 <= v <= 1.0 for v in accs.values())
            assert all(accs[k] == accs0[k] for k in EPISODIC_READOUTS)  # 12.6.155節の値は変わらない
    assert len(d["episodic_arb"]) == T and all(len(r) == n_ctx for r in d["episodic_arb"])
    assert all(set(a.values()) <= {"param", "mix", "switch"} for r in d["episodic_arb"] for a in r)
    for name in ("recon_all", "recon_inter"):
        side = d["recon_side"][name]
        assert len(side) == n_ctx and all(len(m) == T and all(len(v) == n_ctx for v in m) for m in side)


def test_recon_zero_steps_equals_param_and_arb_picks_from_candidates():
    """学び直しのステップが 0 なら複製は元のモデルと同じで、recon_* は param に等しい。調停の正確度は候補のどれか。"""
    r = _run(episodic_extra=True, recon_steps=0)
    for row in r["diag"]["episodic_acc"]:
        for accs in row:
            assert not set(RECON_READOUTS) & set(accs)
    # recon_steps > 0 の複製を作ったうえで、学習率をほぼ 0 にすれば param と一致する
    r = _run(episodic_extra=True, recon_steps=3, recon_lr=1e-12)
    for row, arbs in zip(r["diag"]["episodic_acc"], r["diag"]["episodic_arb"]):
        for accs, arb in zip(row, arbs):
            assert all(accs[k] == accs["param"] for k in RECON_READOUTS)
            assert accs["arb_bits"] == {"param": accs["param"], "mix": accs["mix_all"],
                                        "switch": accs["switch_all"]}[arb["bits"]]
            if arb["rep"] == "param":
                assert accs["arb_rep"] == accs["param"]
    assert all(abs(v) < 1e-9 for m in r["diag"]["recon_side"]["recon_all"] for row in m for v in row)


def test_recon_changes_the_copy_only():
    """十分な学習率では学び直した複製の予測は元のモデルと変わる（recon_all が param と異なる組がある）が、
    最終モデル（param・R）は変わらない。不正な引数は拒否する。"""
    r = _run(episodic_extra=True, recon_steps=20)
    accs = [a for row in r["diag"]["episodic_acc"] for a in row]
    assert any(a["recon_all"] != a["param"] for a in accs)
    assert [a["param"] for a in accs] == [a["param"] for row in _run()["diag"]["episodic_acc"] for a in row]
    with pytest.raises(ValueError):
        _run(recon_steps=-1)
    with pytest.raises(ValueError):
        _run(recon_steps=1, recon_lr=0.0)


def test_experiment_jobs_and_analyze(tmp_path):
    import json

    from neurocortex.experiments.run_context_reconsolidation import FAMILIES, READOUTS, analyze, jobs
    assert len(jobs()) == len(FAMILIES) * 2 * 3
    summ = {"nat_own_buf": 1.0, "nat_fresh": 0.8, "selection_acc": 0.9, "cover_nat": 0.01,
            **{f"exp_{k}": 0.6 + 0.01 * i for i, k in enumerate(READOUTS)},
            **{f"nat_{k}": 0.6 for k in READOUTS}, "side_recon_all": -0.03, "side_recon_inter": -0.01}
    runs = {"flip7_m1/reset_L0/base": {"summary": {k: {"mean": v} for k, v in summ.items()}},
            "parity/split_L0/base": {"summary": {k: {"mean": v} for k, v in {**summ, "nat_fresh": 0.99}.items()}}}
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"runs": runs}), encoding="utf-8")
    res = analyze([p])
    assert res["over_14bit"]["n"] == 1 and res["not_over"]["n"] == 1 and res["all"]["n"] == 2
    assert res["over_14bit"]["d_mix_all"] == pytest.approx(0.01)
    assert res["over_14bit"]["side_recon_all"] == pytest.approx(-0.03)
