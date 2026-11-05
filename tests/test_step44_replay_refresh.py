"""ステップ44.13: 再固定化（HR）と生成的リプレイ（G）のテスト。

- refresh の正しさ: 作り直し直後、保存符号を復元した隠れ状態は現在のモデルの forward_lower 出力と
  PQ 誤差の範囲で一致し、drift（古さ）は PQ の水準まで下がる。重みと保存の大きさは変わらない。
- 生成の決定性: 同じ seed・同じモデル状態では生成断片が完全一致する。prefix の保存会計は 2048 B/タスク。
- HR/G の run: 最初のタスクは A と同一、追加の記録項目（refresh_drift・gen_quality）が入り、
  チェックポイントからの再開がビット一致する。
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import step44_compressed_replay as cr2
import step44_continual_forgetting as cf
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
    a = dict(max_seq_length=16, hidden_size=16, num_layers=4, nhead=2, batch_size=4, lr=1e-3, grad_accum=1,
             epochs=1, memory_per_task=3, replay_size=2, layer=None, pq_m=4, pq_nbits=2, prefix_len=4)
    a.update(kw)
    return SimpleNamespace(**a)


def _model(seed=0, layers=4):
    torch.manual_seed(seed)
    cfg = cf.make_config("A", _args(num_layers=layers))
    return BrainInspiredLLMFullScale(cfg, vocab_size=30, hidden_size=16, num_layers=layers, nhead=2)


def test_refresh_restores_drift_to_pq_level():
    ds = WordChunkDataset(torch.randint(0, 30, (16 * 8,)), 16)
    m_old, m_new = _model(0), _model(1)  # 初期化の違う 2 つのネットワーク（学習で変わった新皮質の代わり）
    mem = cr2.HiddenMemory(per_task=3, layer=2, pq_m=4, pq_nbits=2)
    mem.store(m_old, ds, torch.Generator().manual_seed(5))
    drift_before = mem.drift(m_new)
    assert all(v > 0.3 for v in drift_before)  # 別のネットワークから見ると保存表現は古い
    nbytes_before = list(mem.nbytes)
    weights_before = {k: v.clone() for k, v in m_new.state_dict().items()}
    mem.refresh(m_new)
    drift_after = mem.drift(m_new)
    # 作り直し直後の古さは、そのタスクの PQ 復元誤差そのもの（現在のネットワークで作った表現との差は符号化誤差だけ）。
    # 小規模設定（d=16, PQ 4×2bit）では PQ 誤差自体が大きいので、水準の絶対値でなく誤差との一致で見る。
    for after, err in zip(drift_after, mem.recon_rel_err):
        assert abs(after - err) < 1e-5
        assert after < 0.5
    assert all(a < b for a, b in zip(drift_after, drift_before))
    assert mem.nbytes == nbytes_before  # 符号帳は置き換わるだけで、保存の大きさは不変
    assert all(torch.equal(weights_before[k], v) for k, v in m_new.state_dict().items())  # 重みは変えない
    # 復元した隠れ状態が現在のモデルの forward_lower 出力と PQ 誤差の範囲で一致する（drift と同じ eval で測る）
    m_new.eval()
    cur = cr2.forward_lower(m_new, mem.labels[0], 2)
    rel = float(((cur - mem.hidden[0]) ** 2).sum() / (cur ** 2).sum())
    assert abs(rel - mem.recon_rel_err[0]) < 1e-5


def test_generative_memory_prefix_bytes_and_determinism():
    ds = WordChunkDataset(torch.randint(0, 30, (16 * 80,)), 16)
    mem = cr2.GenerativeMemory(per_task=64, prefix_len=16)
    idx = mem.store(ds, torch.Generator().manual_seed(5))
    assert mem.nbytes[0] == 64 * 16 * 2 == 2048  # prefix の保存会計（R の 1/8）
    assert torch.equal(mem.prefixes[0], ds.chunks[idx])  # prefix_len = 系列長なら断片そのもの

    m = _model(0)
    mem2 = cr2.GenerativeMemory(per_task=3, prefix_len=4)
    mem2.store(ds, torch.Generator().manual_seed(5))
    assert torch.equal(mem2.prefixes[0], ds.chunks[torch.randperm(len(ds),
                       generator=torch.Generator().manual_seed(5))[:3]][:, :4])
    g1 = mem2.generate(m, 16, torch.Generator().manual_seed(900))
    g2 = mem2.generate(m, 16, torch.Generator().manual_seed(900))
    assert torch.equal(g1, g2) and g1.shape == (3, 16)  # 同じ seed・同じモデルでは完全一致
    assert torch.equal(g1[:, :4], mem2.prefixes[0])  # 先頭は保存した prefix のまま
    s = mem2.sample(5, torch.Generator().manual_seed(0))
    assert s.shape == (5, 16)


@pytest.mark.parametrize("mode", ["HR", "G"])
def test_new_modes_run_first_task_equals_A(mode):
    d = _data()
    a = cf.run_continual("A", 2, d, _args(), log=lambda s: None)
    r = cr2.run(mode, 2, d, _args(), log=lambda s: None)
    assert r["ce"][0] == a["ce"][0]  # 最初のタスクは再生なしで A と同一
    assert r["n_replayed_per_task"] == [0, 4, 4]
    assert r["ce"][1] != a["ce"][1]
    if mode == "HR":
        assert r["layer"] == 2 and len(r["drift_final"]) == 3  # 設計で第 2 層に固定
        assert len(r["refresh_drift"]) == 2  # フェーズ境界（タスク 1, 2）ごとに作り直し
        for e in r["refresh_drift"]:
            assert all(x <= y + 1e-9 for x, y in zip(e["after"], e["before"]))  # 作り直しで古さが増えない
        assert r["memory_bytes_per_task"][0] == 3 * 16 * 1 + 3 * 16 * 2 + 16 * 4 * 2  # H と同じ会計
    else:
        assert r["prefix_len"] == 4 and len(r["gen_quality"]) == 2
        assert all(set(e["per_task"]) == set(r["order"][:e["task"]]) for e in r["gen_quality"])
        assert all(0 < q["unigram_ce"] and 0 <= q["freq_l1_vs_train"] <= 2
                   for e in r["gen_quality"] for q in e["per_task"].values())
        assert r["memory_bytes_per_task"] == [3 * 4 * 2] * 3  # prefix だけの会計


@pytest.mark.parametrize("mode", ["HR", "G"])
def test_new_modes_resume_from_checkpoint_is_bit_identical(tmp_path, mode):
    d = _data()
    ck = str(tmp_path / "ck.pt")
    full = cr2.run(mode, 4, d, _args(), log=lambda s: None, ckpt=ck)
    resumed = cr2.run(mode, 4, d, _args(), log=lambda s: None, ckpt=ck)
    assert resumed["ce"] == full["ce"] and resumed["ce_marg"] == full["ce_marg"]
    assert resumed["memory_bytes_per_task"] == full["memory_bytes_per_task"]
    if mode == "HR":
        assert resumed["refresh_drift"] == full["refresh_drift"]
    else:
        assert resumed["gen_quality"] == full["gen_quality"]
    plain = cr2.run(mode, 4, d, _args(), log=lambda s: None)
    assert plain["ce"] == full["ce"]


def test_tag_of_new_modes():
    a = _args(mode="HR", layer=None, memory_per_task=64, replay_size=16, pq_m=16, pq_nbits=8, num_layers=4)
    assert cr2.tag_of(a) == "HR2_M64_r16_pq16x8"  # 層が tag から分かる（第 2 層固定）
    assert cr2.tag_of(_args(mode="G", memory_per_task=64, replay_size=16)) == "G_M64_r16"
