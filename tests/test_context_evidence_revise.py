"""文脈の信頼性による閾値と、決定後の乗り換え（12.6.139節）。"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.basal_ganglia_core import ParitySpec  # noqa: E402
from neurocortex.context_inference import run_context_sequence  # noqa: E402

SPECS = [ParitySpec(n_inputs=8, relevant=p) for p in [(0, 1), (2, 3), (4, 5)]]
KW = dict(pretrain_steps=60, steps=60, eval_n=50, cue_n=8, batch=16, novel_patience=3, selection_trials=10,
          diagnostics=True)
EV = dict(cue_select="evidence", evidence_lapse=0.02, evidence_alpha=0.001)


def _run(**kw):
    torch.manual_seed(0)
    return run_context_sequence("inferred_gated", SPECS, 16, torch.Generator().manual_seed(1),
                                torch.Generator().manual_seed(2), **{**KW, **kw})


def test_defaults_unchanged_and_new_args_ignored_without_evidence():
    a = _run()
    b = _run(evidence_reliability=True, evidence_com=True, com_alpha=0.3, com_window=2)
    assert a["R"] == b["R"] and a["assign"] == b["assign"] and a["selection_choices"] == b["selection_choices"]
    assert "selection_observed" not in a and "cues_observed" not in a["diag"]
    e = _run(**EV)
    e2 = _run(**EV, com_alpha=0.3, com_window=2)  # 変種を有効にしなければ 12.6.135節と同じ
    assert e["R"] == e2["R"] and e["selection_choices"] == e2["selection_choices"]


def test_revised_path_without_variants_matches_fixed_evidence():
    """信頼性・乗り換えを無効にした新しい経路の判定は、12.6.135節の判定と同じ。"""
    e = _run(**EV)
    c = _run(**EV, evidence_com=True, com_alpha=1e-300)  # 境界が事実上無限（対数オッズ 690）→ 乗り換えない
    assert e["assign"] == c["assign"] and e["R"] == c["R"]
    assert e["selection_choices"] == c["selection_choices"] == c["selection_first"]
    assert e["selection_cues"] == c["selection_cues"]
    assert all(o == KW["cue_n"] for row in c["selection_observed"] for o in row)
    assert all(s == 0 for row in c["selection_switches"] for s in row)


def test_variants_do_not_change_training():
    a = _run()
    for kw in (dict(evidence_reliability=True), dict(evidence_com=True), dict(evidence_reliability=True, evidence_com=True)):
        r = _run(**EV, **kw)
        assert r["assign"] == a["assign"] and r["n_contexts"] == a["n_contexts"] >= 2


def test_change_of_mind_ends_at_max_without_threshold_when_bound_low():
    """乗り換えの境界をほぼ 0 にすると、最後の手がかりの時点で対数尤度の和が最大の文脈（lapse なしなら従来の選択）に乗り換わる。"""
    a = _run()
    c = _run(cue_select="evidence", evidence_lapse=0.0, evidence_alpha=0.99, evidence_com=True,
             com_alpha=0.5 - 1e-9)
    assert all(n == 1 for row in c["selection_cues"] for n in row)  # 1個目で決める
    assert all(o == KW["cue_n"] for row in c["selection_observed"] for o in row)
    assert c["selection_choices"] == a["selection_choices"]


def test_com_window_limits_observed():
    c = _run(cue_select="evidence", evidence_lapse=0.02, evidence_alpha=0.99, evidence_com=True, com_window=2)
    for row_n, row_o in zip(c["selection_cues"], c["selection_observed"]):
        for n, o in zip(row_n, row_o):
            assert o == min(KW["cue_n"], n + 2)
    assert len(c["diag"]["cues_observed"]) == len(SPECS) and len(c["diag"]["switches"][-1]) == len(SPECS)


def test_reliability_recorded_and_slows_decision():
    e = _run(**EV)
    r = _run(**EV, evidence_reliability=True)
    rel = r["diag"]["reliability"][-1]
    assert set(rel) == set(range(r["n_contexts"])) and all(0.0 <= v <= 1.0 for v in rel.values())
    # 閾値は信頼性 ≦ 1 で厳しくなる方向にしか変わらず、lapse も下限以上なので、決定は平均で遅くなるか同じ
    mean = lambda x: sum(sum(row) for row in x) / sum(len(row) for row in x)  # noqa: E731
    assert mean(r["selection_cues"]) >= mean(e["selection_cues"]) - 1e-9


@pytest.mark.parametrize("bad", [dict(com_alpha=0.0), dict(com_alpha=1.0), dict(com_window=-1)])
def test_invalid_args(bad):
    with pytest.raises(ValueError):
        _run(**EV, **bad)


def test_experiment_jobs_cover_design():
    from neurocortex.experiments.run_context_evidence_revise import CUE_NS, FAMILIES, LS, SELECTS, SUPPLEMENT, jobs
    assert set(CUE_NS) == {1, 2, 4, 8, 16} and set(LS) == {0, 250, 500}
    assert set(SELECTS) >= {"batch", "evidence_fix2", "rel", "com", "rel_com"}
    assert len(jobs()) == (len(FAMILIES) + len(SUPPLEMENT)) * len(LS) * len(CUE_NS) * 5
