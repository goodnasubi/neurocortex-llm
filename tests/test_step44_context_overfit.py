"""ステップ44.14: 文脈への過適合への対策（X4: 露出制限、D4: 多様性最大化の保存選択、C4: 切り出し再生）のテスト。

- 多様性選択: 決定的で、重なりの大きい断片群からは分散して選び、無作為抽出より平均ペアワイズ Jaccard が下がる。
- 切り出し: 窓は保存断片の連続部分列で、長さと決定性が保たれる。
- X4/D4/C4 の run: 最初のタスクは A と同一、X4 の再生数はプール 4 本あたり 1 本、
  44.14 の追加記録（断片別再生回数・再生トークン数・保存断片の評価と般化ギャップ）が入り、
  チェックポイントからの再開がビット一致する。
- 解析: s14_H0〜s14_H4 の判定キーが事前登録どおりに働く。
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import step44_compressed_replay as cr2
import step44_continual_forgetting as cf
from colab_step44_full_scale_experiment import WordChunkDataset


def _data(V=30, K=3, T=16, n=8):
    g = torch.Generator().manual_seed(0)
    tasks = {}
    for k in range(K):
        ids = torch.randint(1, V, (n * T,), generator=g)
        tasks[f"l{k}"] = {"train": WordChunkDataset(ids, T), "eval": WordChunkDataset(ids.clone(), T)}
    return cr2.add_unigram({"vocab_size": V, "tasks": tasks, "langs": list(tasks)})


def _args(**kw):
    a = dict(max_seq_length=16, hidden_size=16, num_layers=2, nhead=2, batch_size=4, lr=1e-3, grad_accum=1,
             epochs=1, memory_per_task=4, replay_size=3, layer=None, pq_m=4, pq_nbits=2, prefix_len=4,
             window_len=8)
    a.update(kw)
    return SimpleNamespace(**a)


# ---------------------------------------------------------------- 多様性選択と切り出し（単位）

def _clustered_chunks():
    """16 本のほぼ同一な断片（語集合が大きく重なる）と、互いに素な 4 本の断片。"""
    rows = []
    for i in range(16):
        r = [1, 2, 3, 4, 5, 6, 7, 8]
        r[i % 8] = 40 + i
        rows.append(r)
    for g in range(4):
        rows.append([100 * (g + 1) + j for j in range(1, 9)])
    return torch.tensor(rows)


def test_select_diverse_is_deterministic_and_spreads():
    chunks = _clustered_chunks()
    i1 = cr2.select_diverse_indices(chunks, 4, torch.Generator().manual_seed(7))
    i2 = cr2.select_diverse_indices(chunks, 4, torch.Generator().manual_seed(7))
    assert torch.equal(i1, i2) and len(set(i1.tolist())) == 4
    div = cr2.mean_pairwise_jaccard(chunks[i1])
    rnd_idx = torch.randperm(len(chunks), generator=torch.Generator().manual_seed(7))[:4]
    rnd = cr2.mean_pairwise_jaccard(chunks[rnd_idx])
    assert div < 0.3 and div < rnd  # 貪欲遠点は重なりの小さい組を選び、無作為抽出より多様になる


def test_crop_windows_are_contiguous_slices():
    chunks = torch.arange(5 * 16).reshape(5, 16)
    w1 = cr2.crop_windows(chunks, 8, torch.Generator().manual_seed(3))
    w2 = cr2.crop_windows(chunks, 8, torch.Generator().manual_seed(3))
    assert w1.shape == (5, 8) and torch.equal(w1, w2)  # 同じ生成器なら決定的
    for src, win in zip(chunks, w1):
        s = int(win[0]) - int(src[0])
        assert 0 <= s <= 8 and torch.equal(win, src[s:s + 8])  # 元の断片の連続部分列
    assert torch.equal(cr2.crop_windows(chunks, 64, torch.Generator().manual_seed(3)), chunks)  # 窓 >= 長さなら無変更


# ---------------------------------------------------------------- run の振る舞い

@pytest.mark.parametrize("mode", ["X4", "D4", "C4"])
def test_new_modes_first_task_equals_A(mode):
    d = _data()
    a = cf.run_continual("A", 2, d, _args(), log=lambda s: None)
    r = cr2.run(mode, 2, d, _args(), log=lambda s: None)
    assert r["ce"][0] == a["ce"][0]  # 最初のタスクは再生なしで A と同一
    assert r["memory_bytes_per_task"] == [4 * 16 * 2] * 3  # R_M4 と同じ保存の会計（生の単語 ID）
    assert len(r["stored_eval"]) == 3 and set(r["stored_gap"]) == {"gap_ce_past", "gap_gain_past"}
    assert len(r["stored_pairwise_jaccard"]) == 3


def test_X4_replay_size_follows_pool():
    d = _data()
    r = cr2.run("X4", 2, d, _args(), log=lambda s: None)
    # プール 4 本あたり 1 本: フェーズごとに 1・2 本。更新は 2 回/フェーズ（8 断片 ÷ バッチ 4）
    assert r["replay_sizes_per_task"] == [0, 1, 2]
    assert r["n_replayed_per_task"] == [0, 2, 4]
    counts = r["replay_counts_per_task"]
    assert counts[0] == [] and len(counts[1]) == 1 and len(counts[1][0]) == 4
    assert sum(counts[1][0]) == 2 and sum(sum(t) for t in counts[2]) == 4


def test_replay_counts_include_duplicate_draws():
    # r (6) > プール (4) なら 1 回の抽出に必ず重複が入る。回数は重複も含めて数えること
    d = _data()
    r = cr2.run("D4", 2, d, _args(replay_size=6), log=lambda s: None)
    assert r["n_replayed_per_task"] == [0, 12, 12]  # r=6 × 2 更新
    assert sum(r["replay_counts_per_task"][1][0]) == 12
    assert sum(sum(t) for t in r["replay_counts_per_task"][2]) == 12


def test_D4_and_C4_replay_records():
    d = _data()
    rd = cr2.run("D4", 2, d, _args(), log=lambda s: None)
    assert rd["n_replayed_per_task"] == [0, 6, 6]  # r=3 × 2 更新
    assert sum(sum(t) for t in rd["replay_counts_per_task"][2]) == 6
    rc = cr2.run("C4", 2, d, _args(), log=lambda s: None)
    assert rc["window_len"] == 8
    assert rc["replay_tokens_per_task"] == [0, 48, 48]  # 窓 8 トークン × 6 本（他条件の半分に相当）


@pytest.mark.parametrize("mode", ["X4", "D4", "C4"])
def test_new_modes_resume_from_checkpoint_is_bit_identical(tmp_path, mode):
    d = _data()
    ck = str(tmp_path / "ck.pt")
    full = cr2.run(mode, 4, d, _args(), log=lambda s: None, ckpt=ck)
    resumed = cr2.run(mode, 4, d, _args(), log=lambda s: None, ckpt=ck)
    assert resumed["ce"] == full["ce"] and resumed["ce_marg"] == full["ce_marg"]
    assert resumed["replay_counts_per_task"] == full["replay_counts_per_task"]
    assert resumed["stored_gap"] == full["stored_gap"]
    plain = cr2.run(mode, 4, d, _args(), log=lambda s: None)
    assert plain["ce"] == full["ce"]


def test_tag_of_44_14_modes():
    assert cr2.tag_of(_args(mode="X4", memory_per_task=4, replay_size=16)) == "X4_M4_rdiv4"
    assert cr2.tag_of(_args(mode="D4", memory_per_task=4, replay_size=16)) == "D4_M4_r16"
    assert cr2.tag_of(_args(mode="C4", memory_per_task=4, replay_size=16)) == "C4_M4_r16"


# ---------------------------------------------------------------- 解析の s14 判定

def _fake_run(tag, seed, f, g_gain):
    import step44_compressed_analysis as an
    m = {k: 0.1 for k in an.METRICS}
    m.update({"F_ce": f, "LA_ce": 5.0, "G_last_past": g_gain})
    return {"order": ["a", "b"], "metrics": m, "memory_bytes_total": 6, "train_seconds": 1.0}


def test_analysis_s14_hypotheses(tmp_path):
    import json

    import step44_compressed_analysis as an
    G = {"V": 0.5, "R": 0.4, "R_M4": -0.4, "X4": 0.0, "D4": 0.2, "C4": 0.1}
    F = {"V": 4.0, "R": 0.1, "R_M4": 1.5, "X4": 1.2, "D4": 0.9, "C4": 1.1}
    tags = {"V": "V", "R": "R_M64_r16", "R_M4": "R_M4_r16", **an.COND14}
    for i, s in enumerate(range(5)):
        for k, tag in tags.items():
            j = 0.01 * ((i * 7 + len(k)) % 5)
            (tmp_path / f"{tag}_seed{s}.json").write_text(json.dumps(_fake_run(tag, s, F[k] + j, G[k] + j)))
    out = an.analyze(tmp_path, list(range(5)))
    h = out["hypotheses"]
    assert h["s14_H0_reproduced"] is True and h["s14_H0"].startswith("文脈への過適合を再現した")
    assert h["s14_H1"].startswith("保存の多様性最大化は、保存数が少なくても文脈への過適合を防ぐ")
    assert h["s14_H2"].startswith("再生のたびに提示文脈を変えることは、文脈への過適合を防ぐ")
    assert h["s14_H3"].startswith("病理の主因は 1 断片あたりの逐語再生の回数である")
    assert h["s14_H3_interpretation"].startswith("いずれかの対策で回復した")
    assert h["s14_H4_D4_marg_worsened"] is False and "s14_H4_C4_dG" in h
    assert "| F_ce |" in an.to_markdown(out)
