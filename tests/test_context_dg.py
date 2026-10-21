"""歯状回のパターン分離による CA3 の補完（12.6.197〜198節）のテスト。"""

import torch
from torch import nn

from neurocortex.context_lm import (ca1_mismatch, ca3_complete, ca3_recall_codes, code_overlap, dg_codes, dg_weights,
                                    rp_codes)
from neurocortex.experiments import run_context_cmp, run_context_dg


def _data(seed=0):
    g = torch.Generator().manual_seed(seed)
    tags = torch.rand(400, 40, generator=g) ** 4
    t = torch.rand(6, 40, generator=g) ** 8
    return g, tags, t


def test_raw_codes_reproduce_existing_completion_and_mismatch():
    """符号を生のタグにすると、ca3_complete + 余弦の上位と同じ順位、ca1_mismatch と同じ δ になる。"""
    g, tags, t = _data()
    x = nn.functional.normalize(tags, dim=-1)
    for lam in (None, torch.rand(6, generator=g)):
        s = ca3_complete(t, tags, 3, 20, lam)
        ref = (s @ x.T).topk(30).indices
        idx, delta = ca3_recall_codes(t, tags, x, 3, 20, 30, lam)
        assert torch.equal(idx, ref)
    _, delta = ca3_recall_codes(t, tags, x, 3, 20, 30)
    assert torch.allclose(delta, ca1_mismatch(t, tags, 3, 20), atol=1e-6)
    # 反復0回は貫通路の入力だけ（補完なし）
    idx0, d0 = ca3_recall_codes(t, tags, x, 0, 20, 30)
    assert torch.equal(idx0, (nn.functional.normalize(t, dim=-1) @ x.T).topk(30).indices)
    assert torch.allclose(d0, torch.zeros(6))


def test_sparse_dg_codes_match_dense_and_are_k_wta():
    """歯状回の符号は行ごとに高々 a 個が正で正規化され、疎な演算と密な演算で同じ想起になる。"""
    g, tags, t = _data(1)
    w = dg_weights(40, 160, 0.1, g)
    ind, val, dim = dg_codes(tags, w, 8)
    assert dim == 160 and ind.shape == (400, 8)
    assert bool((ind[:, 1:] > ind[:, :-1]).all())  # CSR のため昇順
    assert torch.allclose(val.norm(dim=-1), torch.ones(400), atol=1e-5)
    dense = torch.zeros(400, 160).scatter_(1, ind, val)
    # 上位 a 個は入力の上位 a 個の細胞
    h = nn.functional.normalize(tags, dim=-1) @ w.T
    top = h.topk(9, dim=-1)
    assert torch.allclose(dense.gather(1, top.indices[:, :8]) > 0, (top.values[:, :8] > top.values[:, 8:]))
    lam = torch.rand(6, generator=g)
    for lm in (None, lam):
        a, da = ca3_recall_codes(t, tags, (ind, val, dim), 3, 20, 30, lm)
        b, db = ca3_recall_codes(t, tags, dense, 3, 20, 30, lm)
        assert torch.equal(a, b) and torch.allclose(da, db, atol=1e-6)


def test_dg_separates_overlapping_clusters_but_random_projection_does_not():
    """共通の成分の大きい2つのまとまり: 歯状回の符号では違うまとまりの重なりが下がり（比 between/within が小さい）、
    同じ数の単位の線形の乱数射影では生のタグとほぼ同じ。"""
    g = torch.Generator().manual_seed(3)
    common = torch.rand(60, generator=g)
    proto = torch.rand(2, 60, generator=g) ** 3
    srcs = torch.arange(600) % 2
    tags = common + 0.8 * proto[srcs] + 0.3 * torch.rand(600, 60, generator=g)
    w = dg_weights(60, 1200, 0.05, g)
    raw = code_overlap(nn.functional.normalize(tags, dim=-1), srcs, 300, torch.Generator().manual_seed(0))
    dg = code_overlap(dg_codes(tags, w, 60), srcs, 300, torch.Generator().manual_seed(0))
    rp = code_overlap(rp_codes(tags, w, 60, 0.05), srcs, 300, torch.Generator().manual_seed(0))
    assert dg["ratio"] < raw["ratio"] - 0.1
    assert dg["ratio"] < rp["ratio"] - 0.1
    assert raw["between"] > 0.9  # 生のタグは重なっている


def test_dg_rules_do_not_change_existing_rules():
    """歯状回の規則（専用の乱数）を加えても、12.6.191節の規則の値は同じ。λ・純度は 0〜1。"""
    job = ("C", "seq", 7)
    cmp = run_context_cmp.run_one(job, steps=300, episodes=4, cue_ns=(2,))
    dg = run_context_dg.run_one(job, steps=300, episodes=4, cue_ns=(2,))
    for rule in run_context_cmp.CMP_ALL_RULES:
        assert cmp["cues"]["2"]["acc"][rule] == dg["cues"]["2"]["acc"][rule]
        assert cmp["cues"]["2"]["ce"][rule] == dg["cues"]["2"]["ce"][rule]
    lam = dg["cues"]["2"]["ach_lambda"]
    for key in ("replay_dgcmp", "holdout_dgcmp", "replay_rpcmp"):
        assert 0.0 <= lam[key] <= 1.0
    for key in ("replay_dg", "replay_dgcmp", "holdout_dgcmp", "replay_rp", "replay_rpcmp"):
        assert 0.0 <= dg["cues"]["2"]["tcm_purity"][key] <= 1.0
    assert set(dg["dg_overlap"]) == {"raw", "dg", "rp"}


def _run(fam, regime, seed, acc, own_kl=0.3, fresh_kl=0.3):
    keys = ("replay_tcm", "holdout_tcm", "replay_ca3", "holdout_ca3", "replay_ca3_s1", "replay_ca3_s10",
            "replay_ach", "holdout_ach", "replay_ach_hard", "replay_ach_shuf", "replay_ach_inv",
            "replay_cmp", "holdout_cmp", "replay_cmp_shuf", "replay_cmp_u",
            "replay_dg", "replay_dgcmp", "holdout_dgcmp", "replay_rp", "replay_rpcmp")
    pur = {k: 0.8 for k in keys}
    lam = {k: 0.3 for k in keys if "ach" in k or "cmp" in k}
    cues = {}
    for n in ("1", "2", "4", "8", "16"):
        a = {"passive": 0.5, "uniform": 0.5, "uniform_replay": 0.5, "active_holdout": 0.7,
             "active_holdout_tcm": 0.8, **{r: v(seed) if callable(v) else v for r, v in acc.items()}}
        cues[n] = {"acc": a, "ce": {r: 1.0 - v for r, v in a.items()}, "info": {r: 0.1 for r in a},
                   "tcm_purity": pur, "ach_lambda": lam}
    ov = {"raw": {"within": 0.9, "between": 0.85, "ratio": 0.94}, "dg": {"within": 0.3, "between": 0.1, "ratio": 0.33},
          "rp": {"within": 0.8, "between": 0.75, "ratio": 0.93}}
    return {"family": fam, "regime": regime, "seed": seed, "ce_oracle": 0.5, "ce_nocontext": 1.0, "entropy": 0.4,
            "native": {"own_buf": own_kl, "other_buf": 1.0, "holdout": fresh_kl, "fresh": fresh_kl}, "cues": cues,
            "dg_overlap": ov}


def test_h6_verdict():
    base = {"active": 0.80, "active_replay": 0.75, "active_replay_own": 0.84, "active_replay_tcm": 0.81,
            "active_replay_ca3": 0.83, "active_holdout_ca3": 0.80, "active_replay_ca3_s1": 0.82,
            "active_replay_ca3_s10": 0.83, "active_replay_ach": 0.83, "active_holdout_ach": 0.80,
            "active_replay_ach_hard": 0.82, "active_replay_ach_shuf": 0.83, "active_replay_ach_inv": 0.81,
            "active_replay_cmp": 0.83, "active_holdout_cmp": 0.80, "active_replay_cmp_shuf": 0.82,
            "active_replay_cmp_u": 0.83, "active_replay_dg": 0.82, "active_replay_dgcmp": 0.83,
            "active_holdout_dgcmp": 0.80, "active_replay_rp": 0.82, "active_replay_rpcmp": 0.82}
    none_ok = {r: 0.80 for r in base}
    # 族 B: 比較器は −0.014 の害、歯状回は害なし、乱数射影は害（回ごとに 0.008〜0.012 で一貫）
    b = dict(none_ok, active_replay_cmp=0.786, active_replay_rpcmp=lambda s: 0.79 + 0.0005 * (s % 5 - 2))
    runs = [_run("C", "seq", s, base, 0.3, 0.45) for s in range(10)] + \
        [_run("A", "joint", s, none_ok) for s in range(10)] + [_run("B", "joint", s, b) for s in range(10)]
    h = run_context_dg.analyze_dg(runs)["判定"]["H6_DG"]
    assert h["上書きあり"] == ["C-seq"] and set(h["上書きなし"]) == {"A-joint", "B-joint"}
    for key in ("H6-T1", "H6-T2", "H6-K", "H6b", "H6-B", "H6-R", "H6-P"):
        assert h[key]["合格"], key
    assert h["族Bの害を除き利得を保った"] and h["分離による解決"]
    # 歯状回で族 B の害が残る（−0.012）と H6-B は不合格
    b2 = dict(b, active_replay_dgcmp=0.788)
    runs2 = runs[:20] + [_run("B", "joint", s, b2) for s in range(10)]
    h2 = run_context_dg.analyze_dg(runs2)["判定"]["H6_DG"]
    assert not h2["H6-B"]["合格"] and not h2["族Bの害を除き利得を保った"]
    # 利得を半分しか保たないと H6-K は不合格（H6-T2 は合格のまま）
    over2 = dict(base, active_replay_dgcmp=0.815, active_holdout_dgcmp=0.79, active_replay_own=0.825)
    runs3 = [_run("C", "seq", s, over2, 0.3, 0.45) for s in range(10)] + runs[10:]
    h3 = run_context_dg.analyze_dg(runs3)["判定"]["H6_DG"]
    assert h3["H6-T2"]["合格"] and not h3["H6-K"]["合格"] and not h3["分離による解決"]
