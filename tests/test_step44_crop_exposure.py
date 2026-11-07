"""ステップ44.15: 切り出し再生と回数制限の併用（CX4）のテスト。

- CX4 の run: 最初のタスクは A と同一、再生数は X4 と同じプール 4 本あたり 1 本、
  切り出しが適用され（窓の長さの記録・再生トークン数が X4 の半分）、
  結果に replay_rule と window_len の両方が記録され、
  チェックポイントからの再開がビット一致する。
- 解析: s15_H0〜s15_H4 の判定キーが事前登録どおりに働く（H2 の実用候補の分岐を含む）。
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


# ---------------------------------------------------------------- CX4 の run の振る舞い

def test_CX4_first_task_equals_A():
    d = _data()
    a = cf.run_continual("A", 2, d, _args(), log=lambda s: None)
    r = cr2.run("CX4", 2, d, _args(), log=lambda s: None)
    assert r["ce"][0] == a["ce"][0]  # 最初のタスクは再生なしで A と同一
    assert r["memory_bytes_per_task"] == [4 * 16 * 2] * 3  # R_M4 と同じ保存の会計（生の単語 ID）
    assert len(r["stored_eval"]) == 3 and set(r["stored_gap"]) == {"gap_ce_past", "gap_gain_past"}
    assert len(r["stored_pairwise_jaccard"]) == 3


def test_CX4_replay_size_follows_pool_like_X4():
    d = _data()
    r = cr2.run("CX4", 2, d, _args(), log=lambda s: None)
    # プール 4 本あたり 1 本（X4 と同じ規則）: フェーズごとに 1・2 本。更新は 2 回/フェーズ（8 断片 ÷ バッチ 4）
    assert r["replay_sizes_per_task"] == [0, 1, 2]
    assert r["n_replayed_per_task"] == [0, 2, 4]
    x = cr2.run("X4", 2, d, _args(), log=lambda s: None)
    assert r["replay_sizes_per_task"] == x["replay_sizes_per_task"]
    counts = r["replay_counts_per_task"]
    assert counts[0] == [] and len(counts[1]) == 1 and len(counts[1][0]) == 4
    assert sum(counts[1][0]) == 2 and sum(sum(t) for t in counts[2]) == 4
    # 実再生回数/断片は合計を断片数で割った水準になる（フェーズ 2: 2 回 ÷ 4 本、フェーズ 3: 4 回 ÷ 8 本）
    assert sum(counts[1][0]) / len(counts[1][0]) == 0.5


def test_CX4_records_rule_window_and_half_tokens():
    d = _data()
    r = cr2.run("CX4", 2, d, _args(), log=lambda s: None)
    assert r["replay_rule"] == "pool//4"  # X4 の記録
    assert r["window_len"] == 8  # C4 の記録（_args の window_len）
    # 切り出しが適用される: 再生トークン数は窓 8 トークン × 再生本数で、X4（全長 16）の半分
    assert r["replay_tokens_per_task"] == [0, 16, 32]
    x = cr2.run("X4", 2, d, _args(), log=lambda s: None)
    assert "window_len" not in x
    assert r["replay_tokens_per_task"] == [t // 2 for t in x["replay_tokens_per_task"]]
    c = cr2.run("C4", 2, d, _args(), log=lambda s: None)
    assert "replay_rule" not in c and c["window_len"] == 8


def test_CX4_resume_from_checkpoint_is_bit_identical(tmp_path):
    d = _data()
    ck = str(tmp_path / "ck.pt")
    full = cr2.run("CX4", 4, d, _args(), log=lambda s: None, ckpt=ck)
    resumed = cr2.run("CX4", 4, d, _args(), log=lambda s: None, ckpt=ck)
    assert resumed["ce"] == full["ce"] and resumed["ce_marg"] == full["ce_marg"]
    assert resumed["replay_counts_per_task"] == full["replay_counts_per_task"]
    assert resumed["replay_tokens_per_task"] == full["replay_tokens_per_task"]
    assert resumed["stored_gap"] == full["stored_gap"]
    plain = cr2.run("CX4", 4, d, _args(), log=lambda s: None)
    assert plain["ce"] == full["ce"]


def test_tag_of_CX4():
    assert cr2.tag_of(_args(mode="CX4", memory_per_task=4, replay_size=16)) == "CX4_M4_rdiv4"


# ---------------------------------------------------------------- 解析の s15 判定

def _fake_run(tag, seed, f, g_gain):
    import step44_compressed_analysis as an
    m = {k: 0.1 for k in an.METRICS}
    m.update({"F_ce": f, "LA_ce": 5.0, "G_last_past": g_gain})
    return {"order": ["a", "b"], "metrics": m, "memory_bytes_total": 6, "train_seconds": 1.0}


def _write_fake(tmp_path, G, F):
    import json

    import step44_compressed_analysis as an
    tags = {"V": "V", "R": "R_M64_r16", "R_M4": "R_M4_r16", **an.COND14, **an.COND15}
    for i, s in enumerate(range(5)):
        for k, tag in tags.items():
            if k not in G:
                continue
            j = 0.01 * ((i * 7 + len(k)) % 5)
            (tmp_path / f"{tag}_seed{s}.json").write_text(json.dumps(_fake_run(tag, s, F[k] + j, G[k] + j)))
    return an


def test_analysis_s15_hypotheses_candidate_CX4(tmp_path):
    an = _write_fake(tmp_path,
                     G={"V": 0.5, "R": 0.4, "R_M4": -0.4, "X4": 0.1, "C4": -0.2, "CX4": 0.2},
                     F={"V": 4.0, "R": 0.1, "R_M4": 1.5, "X4": 1.2, "C4": 1.1, "CX4": 1.0})
    out = an.analyze(tmp_path, list(range(5)))
    h = out["hypotheses"]
    assert h["s15_H0_reproduced"] is True and h["s15_H0"].startswith("文脈への過適合を再現した")
    assert h["s15_H1_prevented"] is True
    assert h["s15_H1"].startswith("切り出し再生と回数制限の併用は、保存数が少なくても文脈への過適合を防ぐ")
    # H1 成立かつ F_ce が X4 以下 → 実用候補は CX4
    assert h["s15_H2_candidate"] == "CX4" and h["s15_H2"].startswith("CX4 を残課題 (3) へ持っていく実用候補とする")
    # H3: 交互作用 (CX4 − X4) − (C4 − R_M4) = (0.2−0.1) − (−0.2−(−0.4)) = −0.1（seed ごとの揺れを除いた水準）
    assert h["s15_H3_interaction_G_last_past_mean"] == pytest.approx(-0.1, abs=0.05)
    assert h["s15_H3_interaction_F_ce_mean"] == pytest.approx((1.0 - 1.2) - (1.1 - 1.5), abs=0.05)
    assert h["s15_H4_CX4_marg_worsened"] is False and "s15_H4_CX4_dG" in h
    assert "s15_H4_CX4_vs_X4_dCE_marg" in h and "s15_H4_CX4_vs_X4_dG" in h
    assert "| F_ce |" in an.to_markdown(out)


def test_analysis_s15_H2_candidate_stays_X4_when_F_worse(tmp_path):
    an = _write_fake(tmp_path,
                     G={"V": 0.5, "R": 0.4, "R_M4": -0.4, "X4": 0.1, "C4": -0.2, "CX4": 0.2},
                     F={"V": 4.0, "R": 0.1, "R_M4": 1.5, "X4": 0.8, "C4": 1.1, "CX4": 1.0})
    h = an.analyze(tmp_path, list(range(5)))["hypotheses"]
    assert h["s15_H1_prevented"] is True  # 文脈は防いだ
    assert h["s15_H2_F_not_worse_than_X4"] is False
    assert h["s15_H2_candidate"] == "X4" and "実用候補は X4 のまま" in h["s15_H2"]


def test_analysis_s15_H1_rejected_when_gain_negative(tmp_path):
    an = _write_fake(tmp_path,
                     G={"V": 0.5, "R": 0.4, "R_M4": -0.4, "X4": 0.1, "C4": -0.2, "CX4": -0.1},
                     F={"V": 4.0, "R": 0.1, "R_M4": 1.5, "X4": 1.2, "C4": 1.1, "CX4": 1.0})
    h = an.analyze(tmp_path, list(range(5)))["hypotheses"]
    assert h["s15_H1_prevented"] is False and "とは言えない" in h["s15_H1"]
    assert h["s15_H2_candidate"] == "X4" and "併用は棄却し実用候補は X4" in h["s15_H2"]
