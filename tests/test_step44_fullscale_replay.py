"""ステップ44.16: フル規模での再検証の実行系と解析のテスト。

- 露出正規化スケジューラ: 合計がちょうど Q になり、状態の保存・復元で列が続く。
- WikiText の文書分割: 学習と評価で文書が重ならない（リークなし）、評価は固定量、分割は連続。
- corpus_frac（パイロット）: 学習だけが縮み、評価集合は変わらない。
- 起動時の git クリーンチェック: null／+dirty を拒否する。
- 小さな合成データでの run: 露出監査が目標 E に一致し、段階途中のチェックポイントからの
  再開が中断しない場合とビット一致する。
- 解析: s16 の判定キー（H0-F の関門、H1-F、H2-F の実用候補）が事前登録どおりに働き、
  真正性チェック違反では判定を出さない。
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import step44_fullscale_analysis as an
import step44_fullscale_replay as fs


# ---------------------------------------------------------------- スケジューラ

def test_scheduler_distributes_exact_total():
    for total, steps in [(10, 4), (62, 3), (0, 5), (7, 7), (1000, 1600)]:
        s = fs.ExposureScheduler(total, steps)
        ks = [s.next() for _ in range(steps)]
        assert sum(ks) == total and s.given == total
        assert all(k >= 0 for k in ks)


def test_scheduler_state_roundtrip_continues_sequence():
    a = fs.ExposureScheduler(62, 5)
    head = [a.next(), a.next()]
    b = fs.ExposureScheduler.from_state(a.state())
    tail = [b.next() for _ in range(3)]
    c = fs.ExposureScheduler(62, 5)
    assert head + tail == [c.next() for _ in range(5)]


def test_literal_rule_presentations_grows_with_steps():
    # 字面どおりの移植では提示回数が更新数に比例して暴れる（設計 §2.5 の棄却理由）
    assert fs.literal_rule_presentations("R_M4", 1600, 4) == 6400.0
    assert fs.literal_rule_presentations("CX4", 1600, 12) == pytest.approx(400.0)
    assert fs.literal_rule_presentations("CX4", 1600, 0) == 0.0


# ---------------------------------------------------------------- 文書分割

def _docs(n=16, len_doc=64, vocab=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(1, vocab, (len_doc,), generator=g) for _ in range(n)]


def test_wikitext_documents_split_on_headings():
    lines = [" = Alpha = ", "body a", "more a", " = Beta = ", "body b"]
    docs = fs.wikitext_documents(lines)
    assert len(docs) == 2 and "body a" in docs[0] and "body b" in docs[1]


def test_split_documents_no_leak_and_fixed_eval():
    docs = _docs()
    sp = fs.split_documents(docs, seq_len=16, eval_tokens=32)
    seen = []
    for name in sp["order"]:
        s = sp["split"][name]
        assert not set(s["train_doc_ids"]) & set(s["eval_doc_ids"])  # 同一文書が両側に無い
        assert s["eval_tokens"] == 32  # 評価は固定量に切る
        seen += s["train_doc_ids"] + s["eval_doc_ids"]
    assert sorted(seen) == list(range(len(docs)))  # 全文書がどこか1か所にだけ入る
    # 部分コーパスは連続（文書番号が単調）
    parts = [sp["split"][n]["train_doc_ids"] + sp["split"][n]["eval_doc_ids"] for n in sp["order"]]
    assert all(max(p) < min(q) for p, q in zip(parts, parts[1:]))


def test_split_documents_corpus_frac_shrinks_train_only():
    docs = _docs()
    full = fs.split_documents(docs, seq_len=16, eval_tokens=32)
    frac = fs.split_documents(docs, seq_len=16, eval_tokens=32, corpus_frac=0.5)
    for name in full["order"]:
        f, p = full["split"][name], frac["split"][name]
        assert p["train_tokens"] < f["train_tokens"]
        assert p["eval_doc_ids"] == f["eval_doc_ids"] and p["eval_tokens"] == f["eval_tokens"]


def test_build_data_provenance():
    docs = _docs()
    data = fs.build_data(docs, vocab_size=30, seq_len=16, eval_tokens=32)
    prov = data["provenance"]
    assert len(prov["documents_sha256"]) == 64 and prov["n_documents"] == 16
    assert set(data["unigram_ce"]) == set(data["order"])


# ---------------------------------------------------------------- git クリーンチェック

def test_check_clean_tree(monkeypatch):
    monkeypatch.setattr(fs, "env_info", lambda: {"git_commit": None})
    with pytest.raises(RuntimeError):
        fs.check_clean_tree()
    monkeypatch.setattr(fs, "env_info", lambda: {"git_commit": "abc123+dirty"})
    with pytest.raises(RuntimeError):
        fs.check_clean_tree()
    assert fs.check_clean_tree(allow=True) == "abc123+dirty"  # 明示の許可がある場合だけ通す
    monkeypatch.setattr(fs, "env_info", lambda: {"git_commit": "abc123"})
    assert fs.check_clean_tree() == "abc123"


# ---------------------------------------------------------------- run のスモークと再開

def _data(vocab=30):
    return fs.build_data(_docs(), vocab_size=vocab, seq_len=16, eval_tokens=32)


def _args(**kw):
    a = dict(hidden_size=16, num_layers=2, nhead=2, max_seq_length=16, batch_size=4, grad_accum=1,
             lr=1e-3, device="cpu", fp16=False, passes=1, eval_batch_size=2, ckpt_every_steps=0,
             allow_unclean_tree=True)
    a.update(kw)
    return SimpleNamespace(**a)


def test_run_V_and_CX4_smoke_and_exposure_audit(monkeypatch):
    monkeypatch.setattr(fs, "WINDOW_LEN", 8)
    d = _data()
    v = fs.run("V", 990, d, _args(), log=lambda s: None)
    assert v["metrics"]["G_diag"] == v["metrics"]["G_diag"]  # 有限値が入る
    assert v["exposure_audit"]["target_E"] is None and v["stored_eval"] is None
    r = fs.run("CX4", 990, d, _args(), log=lambda s: None)
    assert r["window_len"] == 8 and r["exposure_E"] == 15.5
    audit = r["exposure_audit"]["per_stage"]
    assert audit[0]["pool"] == 0
    for st in audit[1:]:
        # 総提示回数 Q = round(E × プール本数) がそのまま配られ、1断片平均は E に一致する
        assert st["total_presentations"] == round(15.5 * st["pool"])
        assert st["mean_per_fragment"] == pytest.approx(15.5)
    assert r["n_presented_per_task"] == [st["total_presentations"] for st in audit]
    assert r["config_sha256"] == fs.config_sha256(r["config"])
    assert len(r["data_provenance"]["documents_sha256"]) == 64


def test_run_resume_mid_stage_is_bit_identical(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "WINDOW_LEN", 8)
    d = _data()
    args = _args(ckpt_every_steps=1)
    plain = fs.run("CX4", 7, d, args, log=lambda s: None)
    ck = str(tmp_path / "ck.pt")
    calls = {"n": 0}
    orig = fs.write_checkpoint

    def flaky(path, state):
        calls["n"] += 1
        if calls["n"] == 6:  # 段階1の途中の保存で落ちたことにする
            raise RuntimeError("simulated crash")
        return orig(path, state)

    monkeypatch.setattr(fs, "write_checkpoint", flaky)
    with pytest.raises(RuntimeError):
        fs.run("CX4", 7, d, args, log=lambda s: None, ckpt=ck)
    monkeypatch.setattr(fs, "write_checkpoint", orig)
    resumed = fs.run("CX4", 7, d, args, log=lambda s: None, ckpt=ck)
    assert resumed["ce"] == plain["ce"] and resumed["ce_marg"] == plain["ce_marg"]
    assert resumed["replay_counts_per_task"] == plain["replay_counts_per_task"]
    assert resumed["replay_tokens_per_task"] == plain["replay_tokens_per_task"]
    assert resumed["n_presented_per_task"] == plain["n_presented_per_task"]


def test_tag_of_matches_analysis():
    for mode, tag in an.COND16.items():
        assert fs.tag_of(mode) == tag


# ---------------------------------------------------------------- 解析の s16 判定

def _fake_run(mode, seed, f, g_gain, commit="abc123"):
    m = {k: 0.1 for k in an.METRICS}
    m.update({"F_ce": f, "G_last_past": g_gain, "G_diag": 0.5})
    cfg = {"step": "44.16", "mode": mode, "seed": seed}
    E = fs.EXPOSURE_E.get(mode)
    audit = {"target_E": E, "per_stage": []}
    presented = []
    for i in range(4):
        pool = fs.MEMORY_PER_TASK.get(mode, 0) * i
        total = int(round(E * pool)) if E else 0
        presented.append(total)
        audit["per_stage"].append({"stage": i, "pool": pool, "total_presentations": total,
                                   "mean_per_fragment": float(E) if pool else 0.0,
                                   "max_per_fragment": total})
    return {"mode": mode, "seed": seed, "metrics": m, "train_seconds": 10.0,
            "env": {"git_commit": commit}, "config": cfg, "config_sha256": fs.config_sha256(cfg),
            "started_utc": "2026-11-08T00:00:00+00:00", "finished_utc": "2026-11-08T01:00:00+00:00",
            "exposure_audit": audit, "n_presented_per_task": presented}


def _write_fake(tmp_path, G, F, commit="abc123"):
    for mode, tag in an.COND16.items():
        if mode not in G:
            continue
        for i, s in enumerate(range(200, 206)):
            j = 0.01 * ((i * 7 + len(mode)) % 5)
            (tmp_path / f"{tag}_seed{s}.json").write_text(
                json.dumps(_fake_run(mode, s, F[mode] + j, G[mode] + j, commit=commit)))


def test_analysis_s16_candidate_CX4(tmp_path):
    _write_fake(tmp_path,
                G={"V": 0.5, "R": 0.4, "R_M4": -0.4, "X4": 0.05, "CX4": 0.2},
                F={"V": 4.0, "R": 0.1, "R_M4": 1.5, "X4": 1.2, "CX4": 1.0})
    out = an.analyze(tmp_path, list(range(200, 206)))
    h = out["hypotheses"]
    assert h["s16_H0F_reproduced"] is True
    assert h["s16_H1F_prevented"] is True
    assert h["s16_H1F_share_CX4"] == pytest.approx((0.2 - -0.4) / (0.5 - -0.4), abs=0.05)
    assert h["s16_H2F_candidate"] == "CX4"
    assert "| CX4 |" in an.to_markdown(out)


def test_analysis_s16_H0F_gate_blocks_judgment(tmp_path):
    _write_fake(tmp_path,
                G={"V": 0.5, "R": 0.4, "R_M4": 0.1, "X4": 0.05, "CX4": 0.2},  # 病理が出ない
                F={"V": 4.0, "R": 0.1, "R_M4": 1.5, "X4": 1.2, "CX4": 1.0})
    h = an.analyze(tmp_path, list(range(200, 206)))["hypotheses"]
    assert h["s16_H0F_reproduced"] is False and "判定不能" in h["s16_H0F"]
    assert "s16_H1F_prevented" not in h


def test_analysis_s16_validation_blocks_on_dirty_commit(tmp_path):
    _write_fake(tmp_path,
                G={"V": 0.5, "R": 0.4, "R_M4": -0.4, "X4": 0.05, "CX4": 0.2},
                F={"V": 4.0, "R": 0.1, "R_M4": 1.5, "X4": 1.2, "CX4": 1.0},
                commit="abc123+dirty")
    out = an.analyze(tmp_path, list(range(200, 206)))
    assert "s16_validation" in out["hypotheses"] and "s16_H0F_reproduced" not in out["hypotheses"]
    assert any("dirty" in p for ps in out["validation"].values() for p in ps)
