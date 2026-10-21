import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import step44_continual_forgetting as cf
import step44_continual_replay as cr
import step44_replay_analysis as ra
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


def test_memory_store_and_sample():
    ds = WordChunkDataset(torch.arange(10 * 4), 4)  # 10 本
    m = cr.EpisodicMemory(3)
    idx = m.store(ds, torch.Generator().manual_seed(1))
    assert len(m) == 3 and len(set(idx.tolist())) == 3  # 非復元
    assert torch.equal(m.slots[0], ds.chunks[idx])
    m.store(ds, torch.Generator().manual_seed(2))
    s = m.sample(50, torch.Generator().manual_seed(3))
    assert s.shape == (50, 4)
    allm = torch.cat(m.slots)
    assert all(any(torch.equal(r, x) for x in allm) for r in s)  # 再生はメモリ内の断片だけ
    big = cr.EpisodicMemory(100)
    big.store(ds, torch.Generator().manual_seed(1))
    assert len(big) == 10  # 断片数を超えない


def test_concat_replay():
    b = {"input_ids": torch.zeros(4, 5, dtype=torch.long)}
    out = cr.concat_replay(b, torch.ones(2, 5, dtype=torch.long))
    assert out["input_ids"].shape == (6, 5) and torch.equal(out["labels"], out["input_ids"])


def test_first_task_identical_to_A_and_replay_zero_equals_A():
    d = _data()
    a = cf.run_continual("A", 2, d, _args(), log=lambda s: None)
    r = cr.run_continual_replay(2, d, _args(), log=lambda s: None)
    assert r["ce"][0] == a["ce"][0]  # 最初のタスクはメモリが空で A と同一
    assert r["order"] == a["order"] and r["n_params"] == a["n_params"]
    assert r["n_replayed_per_task"] == [0, 2 * 8 // 4 * 1, 2 * 8 // 4 * 1]  # 2 本 × 2 更新（8 本 / batch 4）
    assert r["ce"][1] != a["ce"][1]  # 2 番目以降は再生で変わる
    r0 = cr.run_continual_replay(2, d, _args(replay_size=0), log=lambda s: None)
    assert r0["ce"] == a["ce"]  # 再生なしなら A と完全に同一（メモリの乱数は全体の乱数と独立）


def test_deterministic():
    d = _data()
    x = cr.run_continual_replay(1, d, _args(), log=lambda s: None)
    y = cr.run_continual_replay(1, d, _args(), log=lambda s: None)
    assert x["ce"] == y["ce"] and x["stored_chunk_idx"] == y["stored_chunk_idx"]
    assert x["metrics"]["F_ce"] == x["metrics"]["F_ce"]


def _write(dirp, p, s, f, la, row0, cfg=None, env=None):
    m = {k: 1.0 for k in ra.ca.METRICS}
    m.update(F_ce=f, LA_ce=la)
    (dirp / f"phase{p}_seed{s}.json").write_text(json.dumps({
        "order": ["a", "b"], "n_params": 1, "train_seconds": 1.0, "metrics": m, "ce": [row0, [0, 0]],
        "config": cfg or {"lr": 5e-4}, "env": env or {"torch": "x"}}))


def test_analysis(tmp_path):
    rd, bd = tmp_path / "r", tmp_path / "b"
    rd.mkdir(), bd.mkdir()
    for s, base in enumerate([3.8, 3.6, 4.0]):
        _write(bd, "V", s, base, 5.0, [9, 9])
        _write(bd, "A", s, base - 0.01 * s, 5.0, [5.0, 9.0])
        _write(rd, "R", s, base - 1.0 + 0.01 * s, 5.0 + 0.001 * s, [5.0, 9.0])
    out = ra.analyze(rd, bd, [0, 1, 2])
    assert out["H1_verdict"] == "支持" and out["comparisons"]["V->R_F_ce"]["verdict_unpaired"] == "改善"
    assert out["reuse_issues"] == [] and out["R_vs_A_first_row_ce_maxdiff"] == 0.0
    assert "H1" in ra.to_markdown(out)
    _write(bd, "A", 0, 3.8, 5.0, [5.0, 9.0], cfg={"lr": 1e-3})
    assert any("config.lr" in i for i in ra.analyze(rd, bd, [0, 1, 2])["reuse_issues"])
