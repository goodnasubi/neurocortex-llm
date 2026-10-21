"""実テキスト・自己回帰の文字 Transformer への比較器ゲートの移植（12.6.213〜214節）のテスト。
コーパスは取得せず、小さな合成の文字列で動かす。"""

import math

import torch

from neurocortex.context_lm import ca1_mismatch, ca3_complete, tcm_drift
from neurocortex.context_realtext import (REAL_RULES, LangData, NgramSpeaker, RealConfig, TransformerContextLM,
                                          build_vocab, complete, drift_last, encode, evaluate_realtext, item_table,
                                          items_of, mismatch, normalize_text, now_context, trace_tags,
                                          train_realtext)
from neurocortex.experiments import run_context_realtext as rr


def _tags(n=60, d=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, d, generator=g) * (torch.rand(n, d, generator=g) < 0.4)
    x[: n // 2, : d // 2] += 0.5  # 2つのまとまり
    x[n // 2:, d // 2:] += 0.5
    return x


def test_fast_completion_and_mismatch_match_context_lm():
    """速い版の補完・比較器は context_lm.ca3_complete・ca1_mismatch と同じ値（疎な積でも同じ）。"""
    tags = _tags()
    t = torch.rand(7, 12, generator=torch.Generator().manual_seed(1))
    x_hat = torch.nn.functional.normalize(tags, dim=-1)
    lam = torch.rand(7, generator=torch.Generator().manual_seed(2))
    for iters, lm in ((0, None), (1, None), (3, None), (3, lam)):
        ref = ca3_complete(t, tags, iters, 5, lm)
        assert torch.allclose(complete(t, x_hat, iters, 5, lm), ref, atol=1e-5)
        assert torch.allclose(complete(t, x_hat, iters, 5, lm, x_sp=x_hat.to_sparse_csr()), ref, atol=1e-5)
    assert torch.allclose(mismatch(t, x_hat, 3, 5), ca1_mismatch(t, tags, 3, 5), atol=1e-5)
    assert torch.allclose(mismatch(t, x_hat, 3, 5, x_hat.to_sparse_csr()), ca1_mismatch(t, tags, 3, 5), atol=1e-5)


def test_drift_matches_tcm_drift():
    """最後の時点の漂流・段落の中のタグ・今の文脈は context_lm.tcm_drift と同じ値。"""
    items = torch.randint(0, 9, (3, 15), generator=torch.Generator().manual_seed(0))
    assert torch.allclose(drift_last(items, 9, 0.1), tcm_drift(items, 9, 0.1)[:, -1], atol=1e-6)
    v, dim = 5, 7
    table = item_table(v, dim)
    pas = torch.randint(0, v, (4, 20), generator=torch.Generator().manual_seed(3))
    rows, ends = torch.tensor([0, 2, 3, 1]), torch.tensor([5, 19, 2, 10])
    tags = trace_tags(pas, rows, ends, v, table, dim, 0.1)
    full = tcm_drift(items_of(pas, v, table), dim, 0.1)  # 位置 i の文脈は full[:, i − 2]
    for r, (row, e) in enumerate(zip(rows.tolist(), ends.tolist())):
        assert torch.allclose(tags[r], full[row, e - 2], atol=1e-6)
    cue = torch.randint(0, v, (2, 3, 4), generator=torch.Generator().manual_seed(4))
    ref = tcm_drift(items_of(cue, v, table).reshape(2, -1), dim, 0.1)[:, -1]
    assert torch.allclose(now_context(cue, v, table, dim, 0.1), ref, atol=1e-6)


def test_speaker_is_a_distribution_and_follows_counts():
    """話し手の n-gram は確率分布で、学習文の決まった続きを強く予測し、見たことのない文脈では下の次数に退く。"""
    ids = torch.tensor([1, 2, 3, 4] * 50)
    spk = NgramSpeaker(ids, 6, order=3)
    p = spk.probs(torch.tensor([[1, 2], [2, 3], [5, 5]]))
    assert torch.allclose(p.sum(-1), torch.ones(3), atol=1e-5)
    assert float(p[0, 3]) > 0.9 and float(p[1, 4]) > 0.9
    assert torch.allclose(p[2], spk.uni, atol=1e-6)  # 1文字の文脈 5 も見たことがない → 一様との補間
    s = spk.sample(torch.tensor([[1, 2]] * 20), torch.Generator().manual_seed(0))
    assert (s == 3).float().mean() > 0.8


def test_vocab_and_encode():
    text = normalize_text("Ab  ab\nAB  zz")
    assert text == "ab ab ab zz"
    voc = build_vocab([text], min_frac=0.2)
    assert set(voc) == {"a", "b", " "}
    assert encode("az", voc).tolist() == [voc.index("a") + 1, 0]


def test_transformer_is_causal_and_logp_all_matches_forward():
    """因果的: 後の文字を変えても前の位置の予測は変わらない。logp_all は文脈ごとの forward と同じ。"""
    torch.manual_seed(0)
    m = TransformerContextLM(7, 3, 8, d=16, layers=1, heads=2, ff=32, hidden=16).eval()
    with torch.no_grad():
        x = torch.randint(0, 7, (2, 8))
        y = x.clone()
        y[:, 5:] = (y[:, 5:] + 1) % 7
        c = torch.tensor([0, 2])
        assert torch.allclose(m.seq_logp(x, c)[:, :5], m.seq_logp(y, c)[:, :5], atol=1e-5)
        w = x[:, :4]
        la = m.logp_all(w)
        for k in range(3):
            assert torch.allclose(la[:, k], m(w, torch.full((2,), k)), atol=1e-5)
        assert torch.allclose(m.seq_logp(x, c)[:, 3], m(w, c), atol=1e-5)


def _toy_data(n_l=3, length=6000):
    g = torch.Generator().manual_seed(0)
    data = []
    for j in range(n_l):  # 言語ごとに違う周期の文字列＋雑音
        base = torch.arange(length) % (3 + j) + 1
        noise = torch.randint(1, 8, (length,), generator=g)
        ids = torch.where(torch.rand(length, generator=g) < 0.2, noise, base)
        data.append(LangData(str(j), ids[:4000], ids[4000:5200], ids[5200:]))
    return data


def test_train_and_evaluate_smoke():
    """小さな設定で joint・seq が動き、λ の並べ替えは λ の平均を変えず、値の範囲が正しい。"""
    data = _toy_data()
    spk = [NgramSpeaker(d.speaker, 8, 3) for d in data]
    for regime, rw in (("joint", False), ("seq", False), ("seq", True)):
        cfg = RealConfig(regime=regime, replay_window=rw, steps_per_lang=20, batch=8, replay_batch=4, seq_len=12, text_len=100,
                         buffer=30, d=16, layers=1, heads=2, hidden=16, tcm_dim=64, order=[2, 0, 1])
        g = torch.Generator().manual_seed(5)
        tr = train_realtext(data, 8, cfg, g)
        assert tr["bank"]["buf"][0]["x"].shape == (30, cfg.k)
        assert torch.equal(tr["bank"]["buf"][0]["x"], tr["bank"]["buf"][0]["seg"][:, cfg.seq_len - cfg.k:cfg.seq_len])
        ev = evaluate_realtext(data, spk, tr, cfg, 8, (2,), g, episodes=5)
        res = ev["cues"]["2"]
        assert set(res["acc"]) == set(REAL_RULES)
        assert all(0.0 <= a <= 1.0 for a in res["acc"].values())
        assert abs(res["lambda"]["active_replay_cmp"] - res["lambda"]["active_replay_cmp_shuf"]) < 1e-6
        assert abs(res["lambda"]["active_replay_ach"] - res["lambda"]["active_replay_ach_shuf"]) < 1e-6
        assert all(0.0 <= p <= 1.0 for p in res["purity"].values())
        assert math.isfinite(ev["ce_oracle"])


def _run(set_name, regime, seed, acc, own=1.0, hold=1.0):
    cues = {n: {"acc": dict(acc[n]) if n in acc else dict(acc["*"]), "purity": {}, "lambda": {}} for n in rr.NS}
    return {"set": set_name, "regime": regime, "seed": seed, "native": {"own_buf": own, "holdout": hold, "fresh": hold},
            "ce_oracle": 1.0, "ce_matrix": [[1.0, 2.0], [2.0, 1.0]], "speaker_ce": [1.0], "cues": cues}


def test_analyze_verdicts():
    """組の分類（holdout − own_buf）と H8 の判定。比較器が並べ替えより一貫して良ければ H8-G は合格。"""
    base = {r: 0.80 for r in REAL_RULES}
    runs = []
    for s in range(5):
        a = dict(base, active_replay_own=0.84, active_replay_tcm=0.80, active_replay_ca3=0.81,
                 active_replay_ach=0.81, active_replay_cmp=0.83 + 0.001 * s, active_replay_cmp_shuf=0.82,
                 active_holdout_cmp=0.80)
        runs.append(_run("R", "seq", s, {"*": a}, own=1.0, hold=1.2))
        runs.append(_run("R", "joint", s, {"*": dict(base)}, own=1.0, hold=1.0))
    v = rr.analyze(runs)["判定"]
    assert v["上書きあり"] == ["R-seq"] and v["上書きなし"] == ["R-joint"]
    assert v["合否"]["H8-T1"] and v["合否"]["H8-T2"] and v["合否"]["H8b"]
    assert v["合否"]["H8-C"] and v["合否"]["H8-K"] and v["合否"]["H8-G"]
    assert v["利得が移った"] and v["比較器が最良のまま"]
    # 並べ替えとの差の符号がばらつくと H8-G は不合格
    for i, r in enumerate(runs):
        if r["regime"] == "seq":
            for n in rr.NS:
                r["cues"][n]["acc"]["active_replay_cmp_shuf"] = 0.83 + (0.01 if i % 4 == 0 else -0.002)
    assert rr.analyze(runs)["判定"]["合否"]["H8-G"] is False


def test_fix1_changes_only_seq_replay():
    """修正1（リプレイを問う窓にする）は joint の学習を変えない（joint の回を再利用できる）。"""
    data = _toy_data()
    kw = dict(regime="joint", steps_per_lang=5, batch=8, replay_batch=4, seq_len=12, text_len=100, buffer=10, d=16,
              layers=1, heads=2, hidden=16, tcm_dim=64)
    a = train_realtext(data, 8, RealConfig(**kw), torch.Generator().manual_seed(1))
    b = train_realtext(data, 8, RealConfig(replay_window=True, **kw), torch.Generator().manual_seed(1))
    for pa, pb in zip(a["model"].parameters(), b["model"].parameters()):
        assert torch.equal(pa, pb)
