import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.basal_ganglia_core import ParitySpec  # noqa: E402
from neurocortex.context_inference import assignment_purity, run_context_sequence  # noqa: E402

SPECS = [ParitySpec(n_inputs=8, relevant=p) for p in [(0, 1), (2, 3), (4, 5)]]
KW = dict(pretrain_steps=4, steps=4, eval_n=20, cue_n=4, batch=8)


def _run(cond, **kw):
    torch.manual_seed(0)
    return run_context_sequence(cond, SPECS, 8, torch.Generator().manual_seed(1), torch.Generator().manual_seed(2),
                                **{**KW, **kw})


@pytest.mark.parametrize("cond", ["oracle", "none", "inferred", "random"])
def test_shapes_and_reproducibility(cond):
    a, b = _run(cond), _run(cond)
    assert a["R"] == b["R"] and len(a["R"]) == 3 and all(len(r) == 3 for r in a["R"])
    assert len(a["assign"]) == 12


def test_oracle_uses_true_task_as_context():
    assert [c for _, c in _run("oracle")["assign"]] == [0] * 4 + [1] * 4 + [2] * 4


def test_inferred_never_sees_task_identity():
    # inferred は τ を極端に大きくすると驚きが起きず、全タスクを文脈0で学習する（タスク ID を使っていない）
    r = _run("inferred", tau=1e9)
    assert {c for _, c in r["assign"]} == {0} and r["n_contexts"] == 1


def test_inferred_no_switch_when_never_established():
    # τ を極端に小さくすると、どの文脈も「確立済み」にならず切り替えも起きない
    r = _run("inferred", tau=-1.0)
    assert r["n_contexts"] == 1


def test_assignment_purity():
    p = assignment_purity([(0, 0), (0, 0), (1, 1), (1, 0)], 2)
    assert p["purity"] == pytest.approx(0.75) and p["distinct"] == 1.0


def test_cue_size_does_not_change_training():
    a = _run("inferred", cue_n=1, selection_trials=3)
    b = _run("inferred", cue_n=8, selection_trials=3)
    assert a["assign"] == b["assign"]
    assert 0.0 <= a["selection_acc"] <= 1.0
    assert _run("oracle", selection_trials=3)["selection_acc"] is None
