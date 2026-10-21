import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import step44_continual_analysis as an
import step44_continual_forgetting as cf
from colab_step44_full_scale_experiment import WordChunkDataset


def test_ce_forgetting_known_values():
    CE = [[2.0, 9.0, 9.0], [3.0, 2.0, 9.0], [4.0, 2.5, 2.0]]
    m = cf.ce_forgetting(CE)
    # F_ce: j=0 → 4-min(2,3)=2, j=1 → 2.5-2=0.5 → 1.25。dCE: (4-2, 2.5-2)/2 = 1.25
    assert m["F_ce"] == pytest.approx(1.25) and m["dCE"] == pytest.approx(1.25)
    assert m["ACC_ce"] == pytest.approx((4 + 2.5 + 2) / 3) and m["LA_ce"] == pytest.approx(2.0)


def test_task_order_rotation_and_vocab():
    langs = list(cf.LANGS)
    assert cf.task_order(langs, 0) == langs and cf.task_order(langs, 1)[0] == langs[1]
    assert sorted(cf.task_order(langs, 7)) == sorted(langs)
    v = cf.build_shared_vocab({"b": ["x", "x", "y"], "a": ["z", "z", "x"]}, 1)
    assert v == ["<unk>", "x", "z"]  # 各言語の最頻語の和集合。先頭は <unk>


def _data(V=30, K=3, T=16, n=8):
    g = torch.Generator().manual_seed(0)
    tasks = {}
    for k in range(K):
        ids = torch.randint(1, V, (n * T,), generator=g)
        tasks[f"l{k}"] = {"train": WordChunkDataset(ids, T), "eval": WordChunkDataset(ids.clone(), T)}
    return {"vocab_size": V, "tasks": tasks, "langs": list(tasks)}


ARGS = SimpleNamespace(max_seq_length=16, hidden_size=16, num_layers=1, nhead=2, batch_size=4, lr=1e-3,
                       grad_accum=1, epochs=1)


@pytest.mark.parametrize("phase", ["V", "A", "C"])
def test_run_continual_smoke(phase):
    r = cf.run_continual(phase, 1, _data(), ARGS, log=lambda s: None)
    assert len(r["ce"]) == 3 and r["order"][0] == "l1"
    assert all(0.0 <= a <= 1.0 for row in r["acc"] for a in row) and r["metrics"]["F_ce"] == r["metrics"]["F_ce"]
    assert (r["n_params"] > r["n_params_backbone"]) == (phase != "V")


def test_deterministic_and_same_order_across_phases():
    a = cf.run_continual("V", 2, _data(), ARGS, log=lambda s: None)
    b = cf.run_continual("V", 2, _data(), ARGS, log=lambda s: None)
    assert a["ce"] == b["ce"]  # 決定的
    assert cf.run_continual("A", 2, _data(), ARGS, log=lambda s: None)["order"] == a["order"]


def test_analysis(tmp_path):
    for s, base in enumerate([1.0, 1.1, 0.9]):
        for p, off in zip("VABC", [0.0, -1.0, 0.0, 1.0]):
            (tmp_path / f"phase{p}_seed{s}.json").write_text(json.dumps({
                "order": ["a", "b"], "n_params": 1, "train_seconds": 1.0,
                "metrics": {m: base + (off if m in ("F_ce", "F_acc") else 0.0) + 0.01 * s for m in an.METRICS}}))
    out = an.analyze(tmp_path, [0, 1, 2])
    assert out["comparisons"]["V->A_F_ce"]["verdict_paired"] == "改善"
    assert out["comparisons"]["V->C_F_ce"]["verdict_paired"] == "悪化"
    assert out["h1"]["A"] == "支持" and out["h1"]["C"].startswith("逆")
    assert an.verdict(1.0, 0.1, lower_better=False) == "改善" and an.verdict(0.1, 0.1) == "差なし"
    assert "H1" in an.to_markdown(out)
