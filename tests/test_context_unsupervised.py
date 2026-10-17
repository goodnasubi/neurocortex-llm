import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.context_inference import CueSpec, make_batch  # noqa: E402
from neurocortex.context_unsupervised import run_unsupervised_sequence  # noqa: E402

SPECS = [CueSpec(16, p, (8 + 2 * i, 9 + 2 * i), 0.9) for i, p in enumerate([(0, 1), (2, 3), (4, 5)])]
KW = dict(pretrain_steps=4, steps=4, eval_n=20, cue_n=4, batch=16)


def _run(**kw):
    torch.manual_seed(0)
    return run_unsupervised_sequence(SPECS, 8, torch.Generator().manual_seed(1), torch.Generator().manual_seed(2),
                                     **{**KW, **kw})


def test_cue_spec_biases_only_its_cue_bits():
    bits, _ = make_batch(SPECS[1], 4000, torch.Generator().manual_seed(0))
    frac = (bits > 0).float().mean(dim=0)
    assert abs(frac[10] - 0.9) < 0.03 and abs(frac[11] - 0.9) < 0.03 and abs(frac[8] - 0.5) < 0.03


def test_shapes_and_reproducibility():
    a, b = _run(), _run()
    assert a["R"] == b["R"] and len(a["R"]) == 3 and 0 <= a["sample_purity"] <= 1


def test_allocates_context_per_distinct_input_distribution():
    # 入力分布だけで文脈が分かれる（ラベルは推定に使わない）
    assert _run(steps=20, pretrain_steps=20)["n_contexts"] >= 2
