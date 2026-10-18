"""能動的な手がかりの選択の候補の数・自分で候補を作る形・検証つきの打ち切り（12.6.147節）。"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.basal_ganglia_core import ParitySpec  # noqa: E402
from neurocortex.basal_ganglia_core import make_batch as parity_batch  # noqa: E402
from neurocortex.context_inference import CueSpec, FlipSpec, make_batch, run_context_sequence, task_labels  # noqa: E402

SPECS = [ParitySpec(n_inputs=8, relevant=p) for p in [(0, 1), (2, 3), (4, 5)]]
KW = dict(pretrain_steps=60, steps=60, eval_n=50, cue_n=8, batch=16, novel_patience=3, selection_trials=10,
          diagnostics=True)
ACT = dict(cue_select="evidence", evidence_lapse=0.02, evidence_alpha=0.001, active_pool=4)


def _run(**kw):
    torch.manual_seed(0)
    return run_context_sequence("inferred_gated", SPECS, 16, torch.Generator().manual_seed(1),
                                torch.Generator().manual_seed(2), **{**KW, **kw})


def test_task_labels_match_make_batch():
    for spec in (FlipSpec(8, flip=(2, 3)), CueSpec(16, (0, 1), (8, 9), 0.9), ParitySpec(8, (2, 3))):
        bits, labels = make_batch(spec, 200, torch.Generator().manual_seed(3))
        assert torch.equal(task_labels(spec, bits), labels)
    # ParitySpec の make_batch は従来の乱数消費のまま
    a = make_batch(SPECS[0], 30, torch.Generator().manual_seed(5))
    b = parity_batch(SPECS[0], 30, torch.Generator().manual_seed(5))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_defaults_unchanged_and_training_unaffected():
    base = _run(**ACT)
    same = _run(**ACT, active_source="task", active_verify=0)
    assert same["selection_choices"] == base["selection_choices"] and same["selection_cues"] == base["selection_cues"]
    for kw in (dict(active_source="uniform"), dict(active_source="replay"), dict(active_source="search"),
               dict(active_verify=2), dict(active_pool=16)):
        r = _run(**{**ACT, **kw})
        assert r["assign"] == base["assign"] and r["n_contexts"] == base["n_contexts"] >= 2  # 学習は同じ
        assert all(1 <= n <= KW["cue_n"] for row in r["selection_cues"] for n in row)


def test_verify_observes_at_least_as_many_cues():
    """検証は閾値に達した後に手がかりを足すだけなので、決定までの手がかりの数は減らない（上限は cue_n）。"""
    a = _run(**ACT)
    v = _run(**ACT, active_verify=2)
    na = [n for row in a["selection_cues"] for n in row]
    nv = [n for row in v["selection_cues"] for n in row]
    assert sum(nv) > sum(na) and all(y >= x for x, y in zip(na, nv))


def test_search_picks_more_informative_cues_than_uniform():
    """打ち切りなしで、山登りで作った手がかりは一様に作った候補から選んだ手がかりより情報量が多い（以上）。"""
    ev = {**ACT, "evidence_alpha": 0.0}
    s = _run(**ev, active_source="search")
    u = _run(**ev, active_source="uniform")
    mean = lambda x: sum(sum(row) for row in x) / sum(len(row) for row in x)  # noqa: E731
    assert mean(s["selection_info"]) >= mean(u["selection_info"]) - 1e-6


@pytest.mark.parametrize("bad", [dict(active_source="x"), dict(active_verify=-1),
                                 dict(active_pool=0, active_source="search"), dict(active_pool=0, active_verify=2)])
def test_invalid_args(bad):
    with pytest.raises(ValueError):
        _run(**{**ACT, **bad})


def test_experiment_jobs():
    from neurocortex.experiments.run_context_active_generate import SELECTS, jobs
    j = jobs()
    assert len(j) == 5 * 3 * 5 * len(SELECTS)
    assert SELECTS["active"]["active_pool"] == 4 and SELECTS["active_verify"]["active_verify"] == 2
    assert SELECTS["gen_search"]["active_source"] == "search" and SELECTS["active_cautious"]["evidence_alpha"] == 1e-5
