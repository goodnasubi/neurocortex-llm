"""選んだ文脈の予測と記憶に基づく予測の組み合わせ（12.6.155節）。"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.basal_ganglia_core import ParitySpec  # noqa: E402
from neurocortex.context_inference import (EPISODIC_READOUTS, episodic_readout,  # noqa: E402
                                           run_context_sequence)

SPECS = [ParitySpec(n_inputs=8, relevant=p) for p in [(0, 1), (2, 3), (4, 5)]]
KW = dict(pretrain_steps=60, steps=60, eval_n=50, cue_n=8, batch=16, novel_patience=3, selection_trials=10,
          diagnostics=True)


def _run(**kw):
    torch.manual_seed(0)
    return run_context_sequence("inferred_gated", SPECS, 16, torch.Generator().manual_seed(1),
                                torch.Generator().manual_seed(2), **{**KW, **kw})


def _bits(rows):
    return torch.tensor(rows, dtype=torch.float32)


def test_readout_exact_match_and_far_traces():
    """完全に一致する痕跡があれば記憶の重みは κ = 1 で 0.5 以上、遠い痕跡だけなら p_param にほぼ等しい。"""
    x = _bits([[1, 1, 1, 1], [-1, -1, -1, -1]])
    p = torch.tensor([[0.9, 0.1], [0.9, 0.1]])
    sb, sl = _bits([[1, 1, 1, 1]]), torch.tensor([1])
    mix = episodic_readout(p, x, sb, sl, rule="mix")
    sw = episodic_readout(p, x, sb, sl, rule="switch")
    assert torch.allclose(mix[0], torch.tensor([0.45, 0.55]))  # w = 1/2
    assert torch.allclose(sw[0], torch.tensor([0.0, 1.0]))
    assert torch.allclose(mix[1], p[1], atol=1e-3) and torch.equal(sw[1], p[1])  # 距離4: s = e^-8
    assert torch.allclose(mix.sum(1), torch.ones(2))


def test_readout_empty_store_and_errors():
    x, p = _bits([[1, -1]]), torch.tensor([[0.3, 0.7]])
    assert torch.equal(episodic_readout(p, x, torch.zeros(0, 2), torch.zeros(0, dtype=torch.long)), p)
    with pytest.raises(ValueError):
        episodic_readout(p, x, x, torch.tensor([0]), rule="bad")
    with pytest.raises(ValueError):
        episodic_readout(p, x, x, torch.tensor([0]), lam=0.0)
    with pytest.raises(ValueError):
        _run(episodic_eval=-1)


def test_readout_weights_by_proximity():
    """近い痕跡ほど重い: 距離0のラベル0と距離1のラベル1が1個ずつなら、記憶の予測はラベル0に傾く。"""
    x = _bits([[1, 1, 1]])
    sb, sl = _bits([[1, 1, 1], [1, 1, -1]]), torch.tensor([0, 1])
    pe = episodic_readout(torch.tensor([[0.5, 0.5]]), x, sb, sl, lam=0.5, kappa=1e-9)
    s1 = torch.exp(torch.tensor(-2.0))
    assert torch.allclose(pe[0], torch.tensor([1 / (1 + s1), s1 / (1 + s1)]))


def test_episodic_eval_does_not_change_training_or_selection():
    base = _run(reliability_holdout=40)
    epi = _run(reliability_holdout=40, episodic_eval=200, episodic_hold=20)
    assert epi["assign"] == base["assign"] and epi["R"] == base["R"]
    assert epi["selection_choices"] == base["selection_choices"]
    d = epi["diag"]
    T, n_ctx = len(SPECS), epi["n_contexts"]
    assert len(d["episodic_acc"]) == T and all(len(row) == n_ctx for row in d["episodic_acc"])
    for row in d["episodic_acc"]:
        for accs in row:
            assert set(accs) == set(EPISODIC_READOUTS) and all(0.0 <= v <= 1.0 for v in accs.values())
    assert all(0.0 <= v <= 1.0 for row in d["episodic_cover"] for v in row)
    assert all(v is None or 0.0 <= v <= 1.0 for row in d["store_purity"] for v in row)


def test_hold_prefix_equals_smaller_holdout():
    """保留サンプルの先頭 episodic_hold 個は reliability_holdout = episodic_hold の保留サンプルと同じなので、
    mix_all・mix_hold は保留サンプルを増やしても変わらない（mix_all200 は増やした分を使う）。"""
    small = _run(reliability_holdout=20, episodic_eval=200, episodic_hold=20)["diag"]["episodic_acc"]
    large = _run(reliability_holdout=60, episodic_eval=200, episodic_hold=20)["diag"]["episodic_acc"]
    for rs, rl in zip(small, large):
        for a, b in zip(rs, rl):
            for k in ("param", "mix_all", "switch_all", "mix_buf", "mix_hold", "mix_nogate"):
                assert a[k] == b[k]
            assert a["mix_all200"] == a["mix_all"]  # 保留サンプルが先頭と同じ数なら同じ痕跡


def test_experiment_jobs_and_analyze(tmp_path):
    import json

    from neurocortex.experiments.run_context_episodic_readout import FAMILIES, SELECTS, analyze, jobs
    js = jobs(list(SELECTS))
    assert len(js) == len(FAMILIES) * 2 * 3 * len(SELECTS)
    summ = {"nat_own_buf": 1.0, "nat_fresh": 0.8, "selection_acc": 0.9, "cover_nat": 0.3, "purity_nat": 1.0,
            **{f"exp_{k}": 0.6 + 0.01 * i for i, k in enumerate(EPISODIC_READOUTS)},
            **{f"nat_{k}": 0.6 for k in EPISODIC_READOUTS}}
    runs = {"flip_m1/reset_L0/cue16/gen_replay": {"summary": {k: {"mean": v} for k, v in summ.items()}}}
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"runs": runs}), encoding="utf-8")
    res = analyze([p])
    assert res["flip_m1_reset"]["n"] == 1 and res["over"]["n"] == 1 and res["not_over"]["n"] == 0
    assert res["flip_m1_reset"]["d_mix_all"] == pytest.approx(0.01)
