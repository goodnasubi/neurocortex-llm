"""学び直しの共有の重みへの書き戻しと、バッファでの学び直しが効く理由の切り分け（12.6.161節）。"""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.basal_ganglia_core import ParitySpec  # noqa: E402
from neurocortex.context_inference import RECON_EXTRA_READOUTS, RECON_READOUTS, run_context_sequence  # noqa: E402

SPECS = [ParitySpec(n_inputs=8, relevant=p) for p in [(0, 1), (2, 3), (4, 5)]]
KW = dict(pretrain_steps=60, steps=60, eval_n=50, cue_n=8, batch=16, novel_patience=3, selection_trials=10,
          diagnostics=True, reliability_holdout=40, episodic_hold=20)


def _run(**kw):
    torch.manual_seed(0)
    return run_context_sequence("inferred_gated", SPECS, 16, torch.Generator().manual_seed(1),
                                torch.Generator().manual_seed(2), **{**KW, **kw})


def test_writeback_none_keeps_trajectory_and_records_measurement():
    """"none" は測定だけで、学習・選択・R は書き戻しを指定しない場合と同じ。前後の測定は一致する。"""
    base = _run()
    r = _run(writeback="none", writeback_at=2, writeback_steps=5, writeback_eval=100)
    assert r["R"] == base["R"] and r["assign"] == base["assign"]
    assert r["selection_choices"] == base["selection_choices"]
    assert r["diag"]["ctx_acc"] == base["diag"]["ctx_acc"]
    w = r["diag"]["writeback"]
    assert w["at"] == 2 and w["pre"] == w["post"]
    assert len(w["pre"]["fresh"]) == 2 and all(len(x) == len(w["used"]) for x in w["pre"]["fresh"])
    assert all(len(x) == len(w["used"]) for x in w["pre"]["own_buf"])


def test_writeback_changes_shared_weights_but_not_before():
    """書き戻しは共有の重みを変え（その後の R が変わる）、書き戻しの前の学習と測定は "none" と同じ。"""
    kw = dict(writeback_at=2, writeback_steps=30, writeback_eval=100)
    none = _run(writeback="none", **kw)
    for w in ("all", "inter"):
        r = _run(writeback=w, **kw)
        assert r["R"][:2] == none["R"][:2] and r["diag"]["ctx_acc"][:2] == none["diag"]["ctx_acc"][:2]
        assert r["diag"]["writeback"]["pre"] == none["diag"]["writeback"]["pre"]
        assert r["diag"]["writeback"]["post"] != r["diag"]["writeback"]["pre"]
    # 学習率がほぼ 0 なら書き戻しの前後は変わらない
    r = _run(writeback="inter", recon_lr=1e-12, **kw)
    assert r["diag"]["writeback"]["post"] == r["diag"]["writeback"]["pre"]


def test_writeback_keep_adds_hold_samples_to_replay_after_measurement():
    """修正1: 書き戻しの後に保留サンプルをバッファに加える。書き戻しまでと前後の測定は変わらず、その後の学習は変わる。"""
    kw = dict(writeback="none", writeback_at=2, writeback_steps=5, writeback_eval=100)
    base = _run(**kw)
    r = _run(writeback_keep=True, **kw)
    assert r["diag"]["writeback"] == base["diag"]["writeback"] and r["R"][:2] == base["R"][:2]
    b0, b1 = base["diag"]["buffer"], r["diag"]["buffer"]
    n_add = int((b1["src"] == -1).sum())
    assert n_add > 0 and b1["labels"].shape[0] == b0["labels"].shape[0] + n_add
    assert all(int(((b1["src"] == -1) & (b1["ctx"] == c)).sum()) <= 20 for c in b1["ctx"].unique())
    assert r["R"][2] != base["R"][2] or r["diag"]["ctx_acc"][2] != base["diag"]["ctx_acc"][2]


def test_writeback_rejects_bad_arguments():
    with pytest.raises(ValueError):
        _run(writeback="bad", writeback_at=1)
    with pytest.raises(ValueError):
        _run(writeback="inter", writeback_at=0)
    with pytest.raises(ValueError):
        _run(writeback="inter", writeback_at=3)
    with pytest.raises(ValueError):
        torch.manual_seed(0)
        run_context_sequence("oracle", SPECS, 16, torch.Generator().manual_seed(1), torch.Generator().manual_seed(2),
                             pretrain_steps=10, steps=10, writeback="inter", writeback_at=1)


def test_recon_extra_does_not_change_other_readouts():
    """切り分けの変種は他の再固定化の値を変えない。誤り駆動の変種は、誤りがなければ param に等しい。"""
    kw = dict(episodic_eval=200, recon_steps=10)
    base = _run(**kw)
    r = _run(recon_extra=True, **kw)
    for row, row0 in zip(r["diag"]["episodic_acc"], base["diag"]["episodic_acc"]):
        for a, a0 in zip(row, row0):
            assert set(RECON_EXTRA_READOUTS) <= set(a) and not set(RECON_EXTRA_READOUTS) & set(a0)
            assert all(a[k] == a0[k] for k in a0)
    mg = r["diag"]["recon_margin"]
    assert len(mg) == r["n_contexts"]
    for row in r["diag"]["episodic_acc"]:
        for c, a in enumerate(row):
            if mg[c]["acc"] == 1.0:  # バッファを全部当てていれば誤り駆動の学び直しは何も変えない
                assert a["recon_buf_err"] == a["param"]
    assert all(0.0 <= a[k] <= 1.0 for row in r["diag"]["episodic_acc"] for a in row
               for k in RECON_READOUTS + RECON_EXTRA_READOUTS)


def test_experiment_jobs_and_analyze(tmp_path):
    from neurocortex.experiments.run_context_writeback import CONFIGS, FAMILIES, FIX1, NEW_TASK, analyze, jobs
    assert len(jobs()) == len(FAMILIES) * 2 * 3 * (len(CONFIGS) - len(FIX1)) and set(NEW_TASK) == set(FAMILIES)
    assert len(jobs(FIX1)) == len(FAMILIES) * 2 * 3 * len(FIX1)

    def wb(post, end, new):
        return {"seed": 100, "pre_nat": [0.4, 0.9], "post_nat": post, "end_nat": end, "gap": [0.2, 0.0],
                "exp_end": end[0], "new_nat": new, "new_exp": new, "selection_acc": 0.9}

    runs = {"flip7_m1/reset_L0/wb_none": {"per_seed": [wb([0.4, 0.9], [0.3, 0.9], 0.8)]},
            "flip7_m1/reset_L0/wb_all": {"per_seed": [wb([0.6, 0.8], [0.4, 0.85], 0.8)]},
            "flip7_m1/reset_L0/wb_inter": {"per_seed": [wb([0.6, 0.9], [0.5, 0.9], 0.78)]},
            "flip7_m1/reset_L0/sep": {"per_seed": [{"gap_combo": 0.2, "exp_param": 0.4, "nat_param": 0.4,
                                                    **{f"exp_{k}": 0.5 for k in RECON_READOUTS + RECON_EXTRA_READOUTS},
                                                    **{f"nat_{k}": 0.5 for k in RECON_READOUTS + RECON_EXTRA_READOUTS},
                                                    "buf_loss": 0.1, "buf_acc": 1.0, "buf_p_correct": 0.95}]}}
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"runs": runs}), encoding="utf-8")
    res = analyze([p])
    o = res["over_14bit"]
    assert res["pre_identical"] and o["n"] == 1
    assert o["post_inter"] == pytest.approx(0.1) and o["end_inter"] == pytest.approx(0.1)
    assert o["intact_post_all"] == pytest.approx(-0.1) and o["over_end_inter"] == pytest.approx(0.2)
    assert o["new_inter"] == pytest.approx(-0.02) and o["d_recon_buf_err"] == pytest.approx(0.1)
