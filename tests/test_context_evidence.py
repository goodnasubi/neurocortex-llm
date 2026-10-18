"""テスト時の文脈選択を証拠の蓄積にする（12.6.135節）。"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.basal_ganglia_core import ParitySpec  # noqa: E402
from neurocortex.context_inference import run_context_sequence  # noqa: E402

SPECS = [ParitySpec(n_inputs=8, relevant=p) for p in [(0, 1), (2, 3), (4, 5)]]
# 文脈が複数作られる程度に学習させる（12.6.115節のゲート付き推定）
KW = dict(pretrain_steps=60, steps=60, eval_n=50, cue_n=8, batch=16, novel_patience=3, selection_trials=10,
          diagnostics=True)


def _run(**kw):
    torch.manual_seed(0)
    return run_context_sequence("inferred_gated", SPECS, 16, torch.Generator().manual_seed(1),
                                torch.Generator().manual_seed(2), **{**KW, **kw})


def test_default_is_batch_and_evidence_args_ignored():
    a = _run()
    b = _run(cue_select="batch", evidence_lapse=0.5, evidence_alpha=0.5)
    assert a["R"] == b["R"] and a["assign"] == b["assign"]
    assert a["selection_choices"] == b["selection_choices"]
    assert all(n == KW["cue_n"] for row in a["diag"]["cues_used"] for n in row)


def test_evidence_does_not_change_training_and_matches_batch_without_early_stop():
    a = _run()
    e = _run(cue_select="evidence", evidence_lapse=0.0, evidence_alpha=0.0)
    assert a["assign"] == e["assign"] and a["n_contexts"] == e["n_contexts"] >= 2
    # 手がかりの生成は共通なので、打ち切らなかった試行では lapse なしの蓄積は平均損失の最小と同じ文脈を選ぶ
    n_full = 0
    for ca, ce, ne in zip(a["selection_choices"], e["selection_choices"], e["selection_cues"]):
        for x, y, n in zip(ca, ce, ne):
            if n == KW["cue_n"]:
                n_full += 1
                assert x == y
    assert n_full > 0


def test_evidence_threshold_controls_cues_used():
    early = _run(cue_select="evidence", evidence_alpha=0.99)  # 事後確率 0.01 で決める → 1個目で決定
    assert all(n == 1 for row in early["selection_cues"] for n in row)
    late = _run(cue_select="evidence", evidence_alpha=1e-6)
    cues = [n for row in late["selection_cues"] for n in row]
    assert all(1 <= n <= KW["cue_n"] for n in cues) and sum(cues) / len(cues) > 1
    assert len(late["diag"]["cues_used"]) == len(SPECS) and all(len(r) == len(SPECS) for r in late["diag"]["cues_used"])


@pytest.mark.parametrize("bad", [dict(cue_select="x"), dict(evidence_lapse=-0.1), dict(evidence_lapse=1.5),
                                 dict(evidence_alpha=1.0), dict(evidence_alpha=-0.1)])
def test_invalid_args(bad):
    with pytest.raises(ValueError):
        _run(**bad)


def test_experiment_jobs_cover_design():
    from neurocortex.experiments.run_context_evidence import CUE_NS, FAMILIES, LS, MECHS, SELECTS, jobs
    assert set(CUE_NS) == {1, 2, 4, 8, 16} and set(LS) == {0, 250, 500}
    assert {m for m, _ in MECHS} == {"split", "reset"} and "batch" in SELECTS and "evidence" in SELECTS
    assert len(jobs()) == len(FAMILIES) * len(MECHS) * len(LS) * len(CUE_NS) * 3
