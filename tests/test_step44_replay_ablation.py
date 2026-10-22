import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import step44_continual_forgetting as cf
import step44_continual_replay as cr
import step44_replay_ablation_analysis as ab
from colab_step44_full_scale_experiment import WordChunkDataset


def _data(V=30, K=3, T=16, n=8):
    g = torch.Generator().manual_seed(0)
    tasks = {}
    for k in range(K):
        ids = torch.randint(1, V, (n * T,), generator=g)
        tasks[f"l{k}"] = {"train": WordChunkDataset(ids, T), "eval": WordChunkDataset(ids.clone(), T)}
    return {"vocab_size": V, "tasks": tasks, "langs": list(tasks)}


def _args(**kw):
    a = dict(max_seq_length=16, hidden_size=16, num_layers=1, nhead=2, batch_size=4, lr=1e-3, grad_accum=1, epochs=1,
             memory_per_task=3, replay_size=2)
    a.update(kw)
    return SimpleNamespace(**a)


def test_c1_matches_compute_and_first_task():
    d = _data()
    a = cf.run_continual("A", 2, d, _args(), log=lambda s: None)
    r = cr.run_continual_replay(2, d, _args(), log=lambda s: None)
    c = cr.run_continual_replay(2, d, _args(replay_source="current", label="C1"), log=lambda s: None)
    assert c["phase"] == "C1" and c["replay_source"] == "current"
    assert c["n_replayed_per_task"] == r["n_replayed_per_task"]  # 計算量（追加断片数）が R と同一
    assert c["ce"][0] == a["ce"][0]  # 最初のタスクは A と同一
    assert c["stored_chunk_idx"] == []  # メモリを使わない
    assert c["ce"][1] != a["ce"][1] and c["ce"][1] != r["ce"][1]
    c2 = cr.run_continual_replay(2, d, _args(replay_source="current", label="C1"), log=lambda s: None)
    assert c2["ce"] == c["ce"]  # 決定的


def test_r_unchanged_by_new_options():
    d = _data()
    r = cr.run_continual_replay(1, d, _args(), log=lambda s: None)
    r2 = cr.run_continual_replay(1, d, _args(replay_source="memory", label="R"), log=lambda s: None)
    assert r["ce"] == r2["ce"] and r["phase"] == r2["phase"] == "R"


def test_unigram_ce():
    tr = torch.tensor([[1, 1, 2, 0]])
    ev = torch.tensor([[0, 1, 2]])
    # counts+1: [2,3,2] / 7
    exp = -(math.log(3 / 7) + math.log(2 / 7)) / 2
    assert abs(ab.unigram_ce(tr, ev, 3) - exp) < 1e-12
    ex = ab.excess_1gram({"order": ["x", "y"], "ce": [[0, 0], [5.0, 4.0]]}, {"x": 4.5, "y": 4.5})
    assert ex["per_lang"] == {"x": 0.5, "y": -0.5} and ex["past"] == 0.5 and ex["all"] == 0.0


def test_h1_compute():
    c = lambda p, u: {"verdict_paired": p, "verdict_unpaired": u}
    assert ab.h1_compute(c("悪化", "悪化")) == "支持"
    assert "傾向" in ab.h1_compute(c("悪化", "差なし"))
    assert "不支持" in ab.h1_compute(c("差なし", "差なし"))


def _write(dirp, p, s, f, la, row0, cfg=None):
    m = {k: 1.0 for k in ab.ca.METRICS}
    m.update(F_ce=f, LA_ce=la)
    (dirp / f"phase{p}_seed{s}.json").write_text(json.dumps({
        "order": ["a", "b"], "n_params": 1, "train_seconds": 1.0, "metrics": m, "ce": [row0, [6.0, 5.0]],
        "config": cfg or {"lr": 5e-4, "memory_per_task": 8}, "env": {"torch": "x"}}))


def test_analysis(tmp_path):
    nd, rd, bd = tmp_path / "n", tmp_path / "r", tmp_path / "b"
    for x in (nd, rd, bd):
        x.mkdir()
    for s, base in enumerate([3.8, 3.6, 4.0]):
        _write(bd, "V", s, base, 5.0, [9, 9])
        _write(bd, "A", s, base - 0.01 * s, 5.0, [5.0, 9.0])
        _write(rd, "R", s, 0.08 + 0.01 * s, 5.0, [5.0, 9.0], cfg={"lr": 5e-4, "memory_per_task": 64})
        _write(nd, "C1", s, base + 0.1 + 0.02 * s, 5.0, [5.0, 9.0])
        _write(nd, "Rmid", s, 0.3 + 0.01 * s, 5.0, [5.0, 9.0])
        _write(nd, "Rsmall", s, 0.6 - 0.01 * s, 5.0, [5.0, 9.0])
    out = ab.analyze(nd, rd, bd, [0, 1, 2], uni={"a": 5.5, "b": 5.0})
    assert out["H1_verdict"] == "支持" and out["H2_verdict"] == "支持"
    assert out["reuse_issues"] == [] and set(out["first_row_ce_maxdiff_vs_A"].values()) == {0.0}
    assert out["dose_response"]["n_seeds_monotone_decreasing"] == 3
    assert abs(out["excess_1gram"]["R"]["past_mean"] - 0.5) < 1e-12  # 最終行 [6,5] の過去言語 a: 6-5.5
    assert "H2" in ab.to_markdown(out)
    _write(nd, "Rmid", 0, 0.3, 5.0, [5.0, 9.0], cfg={"lr": 1e-3})
    assert any("config.lr" in i for i in ab.analyze(nd, rd, bd, [0, 1, 2])["reuse_issues"])
