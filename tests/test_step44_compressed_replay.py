import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import step44_compressed_replay as cr2
import step44_continual_forgetting as cf
import step44_continual_replay as cr
from colab_step44_full_scale_experiment import BrainInspiredLLMFullScale, WordChunkDataset

pytest.importorskip("faiss")


def _data(V=30, K=3, T=16, n=8):
    g = torch.Generator().manual_seed(0)
    tasks = {}
    for k in range(K):
        ids = torch.randint(1, V, (n * T,), generator=g)
        tasks[f"l{k}"] = {"train": WordChunkDataset(ids, T), "eval": WordChunkDataset(ids.clone(), T)}
    return cr2.add_unigram({"vocab_size": V, "tasks": tasks, "langs": list(tasks)})


def _args(**kw):
    a = dict(max_seq_length=16, hidden_size=16, num_layers=2, nhead=2, batch_size=4, lr=1e-3, grad_accum=1, epochs=1,
             memory_per_task=3, replay_size=2, layer=None, pq_m=4, pq_nbits=2)
    a.update(kw)
    return SimpleNamespace(**a)


def _model(phase="A", layers=2):
    torch.manual_seed(0)
    cfg = cf.make_config(phase, _args(num_layers=layers))
    return BrainInspiredLLMFullScale(cfg, vocab_size=30, hidden_size=16, num_layers=layers, nhead=2)


@pytest.mark.parametrize("layer", [0, 1, 2])
def test_lower_upper_split_equals_forward(layer):
    m = _model().eval()
    x = torch.randint(0, 30, (3, 16))
    full = m(input_ids=x, labels=x, attention_mask=torch.ones_like(x))
    up = cr2.forward_upper(m, cr2.forward_lower(m, x, layer), layer, x)
    assert torch.allclose(up["logits"], full["logits"], atol=1e-5)
    assert torch.allclose(up["loss"], full["loss"], atol=1e-5)  # 補助ヘッドの損失を含めて同じ


def test_eval_decomp_matches_eval_task():
    m = _model()
    ds = WordChunkDataset(torch.randint(0, 30, (16 * 5,)), 16)
    d = cr2.eval_task_decomp(m, ds, batch_size=2)
    ce, acc = cf.eval_task(m, ds, batch_size=2)
    assert d["ce"] == ce and d["acc"] == acc  # 同じ計算順でビット一致
    assert d["ce_marg"] >= 0 and abs(d["gain"] - (d["ce_marg"] - d["ce"])) < 1e-12


def test_decomp_forgetting_sign():
    row = lambda m, g: {"ce_marg": m, "gain": g}
    rows = [[row(5, 1), row(9, 0)], [row(7, 0.5), row(5, 1)]]
    f = cr2.decomp_forgetting(rows)
    assert f["dCE_marg"] == 2 and f["dG"] == -0.5  # dCE = 2.5 = dCE_marg − dG


def test_shuffle_within_keeps_counts():
    c = torch.randint(0, 30, (4, 16))
    s = cr2.shuffle_within(c, torch.Generator().manual_seed(1))
    assert all(torch.equal(a.sort().values, b.sort().values) for a, b in zip(c, s))
    assert not torch.equal(c, s)


def test_hidden_memory_store_sample_bytes():
    m = _model()
    ds = WordChunkDataset(torch.randint(0, 30, (16 * 8,)), 16)
    mem = cr2.HiddenMemory(per_task=3, layer=1, pq_m=4, pq_nbits=2)
    idx = mem.store(m, ds, torch.Generator().manual_seed(5))
    raw = cr.EpisodicMemory(3)
    assert torch.equal(raw.store(ds, torch.Generator().manual_seed(5)), idx)  # R と同じ断片を選ぶ
    assert mem.hidden[0].shape == (3, 16, 16) and torch.equal(mem.labels[0], ds.chunks[idx])
    assert mem.nbytes[0] == 3 * 16 * 1 + 3 * 16 * 2 + 16 * 4 * 2  # 符号 1 B + ID 2 B/位置 + fp16 符号帳
    assert 0 <= mem.recon_rel_err[0] < 1
    h, y = mem.sample(5, torch.Generator().manual_seed(0))
    assert h.shape == (5, 16, 16) and y.shape == (5, 16)
    assert all(len(v) == 1 and v[0] >= 0 for v in [mem.drift(m)])


def test_V_and_R_reproduce_44_9_and_44_10():
    d = _data()
    v_old = cf.run_continual("V", 3, d, _args(), log=lambda s: None)
    v_new = cr2.run("V", 3, d, _args(), log=lambda s: None)
    assert v_new["ce"] == v_old["ce"]
    r_old = cr.run_continual_replay(3, d, _args(), log=lambda s: None)
    r_new = cr2.run("R", 3, d, _args(mode="R"), log=lambda s: None)
    assert r_new["ce"] == r_old["ce"] and r_new["n_replayed_per_task"] == r_old["n_replayed_per_task"]
    assert r_new["memory_bytes_per_task"] == [3 * 16 * 2] * 3 and r_new["memory_bytes_total"] == 2 * 3 * 16 * 2


@pytest.mark.parametrize("mode,layer", [("S", None), ("C", None), ("H", 1), ("H", None)])
def test_other_modes_run_first_task_equals_A(mode, layer):
    d = _data()
    a = cf.run_continual("A", 2, d, _args(), log=lambda s: None)
    r = cr2.run(mode, 2, d, _args(layer=layer), log=lambda s: None)
    assert r["ce"][0] == a["ce"][0]  # 最初のタスクは再生なしで A と同一
    assert r["n_replayed_per_task"] == [0, 4, 4]
    assert r["ce"][1] != a["ce"][1]  # 2 番目から再生が効く
    if mode == "H":
        assert r["layer"] == (2 if layer is None else layer) and len(r["drift_final"]) == 3


def test_tag_of():
    a = _args(mode="H", layer=2, memory_per_task=64, replay_size=16, pq_m=16, pq_nbits=8, num_layers=4)
    assert cr2.tag_of(a) == "H2_M64_r16_pq16x8"
    assert cr2.tag_of(_args(mode="V")) == "V"
    assert cr2.tag_of(_args(mode="R", memory_per_task=4)) == "R_M4_r2"


def _fake_run(tag, seed, f, la, g=0.5, nbytes=10, h=False):
    m = {k: 0.1 for k in cr2_analysis().METRICS}
    m.update({"F_ce": f, "LA_ce": la, "G_last_past": g})
    r = {"order": ["a", "b"], "metrics": m, "memory_bytes_total": nbytes, "train_seconds": 1.0}
    if h:
        r.update({"pq_recon_rel_err": [0.1, 0.1], "drift_final": [0.2, 0.0]})
    return r


def cr2_analysis():
    import step44_compressed_analysis as an
    return an


def test_analysis_hypotheses(tmp_path):
    import json
    an = cr2_analysis()
    F = {"V": 4.0, "R": 0.1, "S": 0.3, "C": 3.9, "H2": 2.0, "H4": 1.0, "R_M4": 1.5, "R_r2": 0.5}
    for i, s in enumerate(range(5)):
        for k, tag in an.COND.items():
            j = 0.01 * ((i * 7 + len(k)) % 5)
            g = 0.8 + j if k == "R" else 0.5 + j
            r = _fake_run(tag, s, F[k] + j, 5.0 + j, g=g, nbytes=100 if k.startswith("H") else 6, h=k.startswith("H"))
            (tmp_path / f"{tag}_seed{s}.json").write_text(json.dumps(r))
    out = an.analyze(tmp_path, list(range(5)))
    h = out["hypotheses"]
    assert h["H1_V->H2"] == "支持"
    assert h["H2_R->H2"].startswith("R が H を支配") and h["H2_bytes_ratio_H2/R"] == 100 / 6
    assert abs(h["H3a_share_freq_mean"] - 3.7 / 3.9) < 0.02 and h["H3a"].startswith("大部分")
    assert h["H3b"].startswith("R は頻度を超えて")
    assert h["H4_C->R"].startswith("軽減は再生の内容")
    assert h["H5_V->R_M4"] == "支持"
    assert "| F_ce |" in an.to_markdown(out)
