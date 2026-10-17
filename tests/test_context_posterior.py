import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.basal_ganglia_core import ParitySpec  # noqa: E402
from neurocortex.context_posterior import run_posterior_sequence  # noqa: E402

SPECS = [ParitySpec(n_inputs=8, relevant=p) for p in [(0, 1), (2, 3), (4, 5)]]
KW = dict(pretrain_steps=4, steps=4, eval_n=20, cue_n=4, batch=8)


def _run(**kw):
    torch.manual_seed(0)
    return run_posterior_sequence(SPECS, 8, torch.Generator().manual_seed(1), torch.Generator().manual_seed(2),
                                  **{**KW, **kw})


def test_shapes_and_reproducibility():
    a, b = _run(), _run()
    assert a["R"] == b["R"] and len(a["R"]) == 3 and all(len(r) == 3 for r in a["R"])


def test_no_new_context_while_first_not_established():
    # τ を負にすると最初の文脈が確立せず、新しい文脈は作られない（タスク ID を使っていない）
    assert _run(tau=-1.0)["n_contexts"] == 1


def test_transition_runs():
    assert len(_run(transition_steps=3)["R"]) == 3
