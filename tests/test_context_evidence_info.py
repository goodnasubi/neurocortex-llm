"""手がかりの情報量による証拠の重み付けと、能動的な手がかりの選択（12.6.143節）。"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.basal_ganglia_core import ParitySpec  # noqa: E402
from neurocortex.context_inference import cue_information, run_context_sequence  # noqa: E402

SPECS = [ParitySpec(n_inputs=8, relevant=p) for p in [(0, 1), (2, 3), (4, 5)]]
KW = dict(pretrain_steps=60, steps=60, eval_n=50, cue_n=8, batch=16, novel_patience=3, selection_trials=10,
          diagnostics=True)
EV = dict(cue_select="evidence", evidence_lapse=0.02, evidence_alpha=0.001)


def _run(**kw):
    torch.manual_seed(0)
    return run_context_sequence("inferred_gated", SPECS, 16, torch.Generator().manual_seed(1),
                                torch.Generator().manual_seed(2), **{**KW, **kw})


def test_cue_information_agreement_and_disagreement():
    agree = torch.tensor([[[0.2, 0.8], [0.0, 1.0]]])  # 同じ正解を予測（確信度だけ違う）
    disagree = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    same = torch.tensor([[[0.3, 0.7], [0.3, 0.7]]])
    assert float(cue_information(same)) == pytest.approx(0.0, abs=1e-6)
    assert float(cue_information(disagree)) == pytest.approx(1.0, abs=1e-6)
    assert 0.0 < float(cue_information(agree)) < 0.3
    # 事後分布が一方の文脈に集中すると情報量は 0
    assert float(cue_information(disagree, torch.tensor([1.0, 0.0]))) == pytest.approx(0.0, abs=1e-6)


def test_defaults_unchanged_and_training_unaffected():
    a = _run(**EV)
    assert "selection_info" not in a and "cue_info" not in a["diag"]
    for kw in (dict(cue_weight="info"), dict(active_pool=4), dict(cue_weight="info", active_pool=4)):
        r = _run(**EV, **kw)
        assert r["assign"] == a["assign"] and r["n_contexts"] == a["n_contexts"] >= 2
        assert len(r["selection_info"]) == len(SPECS) and len(r["diag"]["cue_info"][-1]) == len(SPECS)
        assert all(1 <= n <= KW["cue_n"] for row in r["selection_cues"] for n in row)
    with pytest.raises(ValueError):  # cue_select="batch" では使えない
        _run(cue_weight="info", active_pool=4)


def test_active_pool_one_without_threshold_matches_full_evidence():
    """候補が従来の手がかりだけ（P = 1）で打ち切りがなければ、観測の順序だけが違い、選択は全手がかりの蓄積と同じ。"""
    e = _run(cue_select="evidence", evidence_lapse=0.02, evidence_alpha=0.0)
    r = _run(cue_select="evidence", evidence_lapse=0.02, evidence_alpha=0.0, active_pool=1)
    assert r["selection_choices"] == e["selection_choices"]
    assert all(n == KW["cue_n"] for row in r["selection_cues"] for n in row)


def test_active_picks_more_informative_cues():
    """打ち切りなしで、能動的に選んだ手がかりは従来の手がかりより一様事前のもとでの情報量が多い。"""
    ev = {**EV, "evidence_alpha": 0.0}
    r = _run(**ev, cue_weight="info", active_pool=4)
    e = _run(**ev, cue_weight="info")  # 能動なし（情報量の記録のため重みだけ有効）
    mean = lambda x: sum(sum(row) for row in x) / sum(len(row) for row in x)  # noqa: E731
    assert mean(r["selection_info"]) > mean(e["selection_info"])


def test_info_weight_ignores_agreeing_cues():
    """全文脈が同じ正解を予測する手がかり（正確度 0.8 の文脈と確信した文脈）の重みは、予測が食い違う手がかりの 1/5 未満。"""
    probs = torch.tensor([[[0.2, 0.8], [0.01, 0.99]], [[0.9, 0.1], [0.1, 0.9]]])
    w = cue_information(probs)
    assert float(w[0]) < 0.2 * float(w[1])
    assert math.isfinite(float(w.sum()))


@pytest.mark.parametrize("bad", [dict(cue_weight="x"), dict(active_pool=-1),
                                 dict(cue_weight="info", evidence_com=True),
                                 dict(active_pool=4, evidence_reliability=True)])
def test_invalid_args(bad):
    with pytest.raises(ValueError):
        _run(**EV, **bad)


def test_experiment_jobs_cover_design():
    from neurocortex.experiments.run_context_evidence_info import CUE_NS, FAMILIES, LS, SELECTS, SUPPLEMENT, jobs
    assert set(CUE_NS) == {1, 2, 4, 8, 16} and set(LS) == {0, 250, 500}
    assert set(SELECTS) >= {"batch", "evidence_fix2", "info", "active", "active_info"}
    assert len(jobs()) == (len(FAMILIES) + len(SUPPLEMENT)) * len(LS) * len(CUE_NS) * 5


def test_info_post_weights_training_unaffected_and_valid():
    """修正1（重みを観測直前の事後分布で測る）は学習に影響せず、手がかり1個では一様事前の重みと同じ選択になる。"""
    a = _run(**EV)
    r = _run(**EV, cue_weight="info_post")
    assert r["assign"] == a["assign"]
    assert all(1 <= n <= KW["cue_n"] for row in r["selection_cues"] for n in row)
    one = dict(cue_n=1)
    assert _run(**EV, **one, cue_weight="info_post")["selection_choices"] == \
        _run(**EV, **one, cue_weight="info")["selection_choices"]
