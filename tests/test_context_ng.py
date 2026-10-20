"""歯状回の成体神経新生による符号化（12.6.203〜204節）のテスト。"""

import torch
from torch import nn

from neurocortex.context_lm import ca3_recall_codes, dg_codes, dg_weights, ng_codes, storage_times
from neurocortex.experiments import run_context_dg, run_context_ng
from test_context_dg import _run as _dg_run


def _young(codes, m):
    """各痕跡の符号のうち、値が正の若い細胞（番号 ≥ m）の集合。"""
    ind, val, _ = codes
    return [set(ind[i][(ind[i] >= m) & (val[i] > 0)].tolist()) for i in range(ind.shape[0])]


def test_storage_times():
    """seq では言語 j の痕跡の時刻は [j, j + 1)、joint では言語によらず [0, 言語の数)。"""
    src = torch.arange(4).repeat_interleave(500)
    t = storage_times(src, "seq", 4, torch.Generator().manual_seed(0))
    assert bool(((t >= src) & (t < src + 1)).all())
    tj = storage_times(src, "joint", 4, torch.Generator().manual_seed(0))
    assert bool(((tj >= 0) & (tj < 4)).all())
    means = [float(tj[src == j].mean()) for j in range(4)]
    assert max(means) - min(means) < 0.3  # 言語と無関係


def test_ng_codes_shape_and_temporal_sharing():
    """符号は行ごとに正規化され、列の番号は昇順。成熟した部分は固定の歯状回の上位と同じ細胞。若い細胞は、時刻の近い
    痕跡どうしで多く共有され、若い期間 τ より離れた痕跡どうしでは共有されない。"""
    g = torch.Generator().manual_seed(1)
    tags = torch.rand(600, 40, generator=g) ** 4
    times = torch.linspace(0, 3, 600)
    w = dg_weights(40, 160, 0.1, g)
    codes = ng_codes(tags, times, w, 16, 20, 0.5, 1.0, 0.1, 0.0, torch.Generator().manual_seed(2))
    ind, val, dim = codes
    assert ind.shape == (600, 16) and bool((ind[:, 1:] >= ind[:, :-1]).all())
    assert torch.allclose(val.norm(dim=-1), torch.ones(600), atol=1e-5)
    m_ind, _, _ = dg_codes(tags, w, 8)
    mature = [set(ind[i][ind[i] < 160].tolist()) for i in range(600)]
    assert all(mature[i] == set(m_ind[i].tolist()) for i in range(600))
    young = _young(codes, 160)
    assert all(len(y) >= 6 for y in young)
    near = sum(len(young[i] & young[i + 5]) for i in range(0, 590, 7))
    far = sum(len(young[i] & young[i + 300]) for i in range(0, 300, 7))  # Δt = 1.5 > τ
    assert near > 0 and far == 0


def test_hebb_makes_concurrent_traces_share_young_cells():
    """若い細胞は広く発火する（若い細胞の半分）ので、Hebb 則は若い細胞の結合をその時期の入力の平均へそろえる。その結果、
    同じ時期に保存した痕跡どうしの若い細胞の共有が Hebb 則なしより増え（時間的に近い出来事の統合）、内容（まとまり）に
    よる共有の差は小さくなる。"""
    g = torch.Generator().manual_seed(3)
    proto = torch.rand(2, 60, generator=g) ** 6
    src = torch.arange(800) % 2
    tags = proto[src] + 0.05 * torch.rand(800, 60, generator=g)
    times = torch.linspace(0, 0.5, 800)
    w = dg_weights(60, 240, 0.1, g)

    def share(eta):
        young = _young(ng_codes(tags, times, w, 24, 40, 0.5, 1.0, 0.1, eta, torch.Generator().manual_seed(4)), 240)
        same = sum(len(young[i] & young[i + 2]) for i in range(400, 790, 3))
        diff = sum(len(young[i] & young[i + 1]) for i in range(400, 790, 3))
        return same, diff

    s0, d0 = share(0.0)
    s1, d1 = share(0.05)
    assert s1 + d1 > s0 + d0
    assert s1 - d1 < s0 - d0


def test_ng_codes_work_with_ca3_recall():
    """神経新生の符号でも、疎な演算と密な演算の想起が一致する（空き列の重複は値0）。"""
    g = torch.Generator().manual_seed(5)
    tags = torch.rand(300, 40, generator=g) ** 4
    t = torch.rand(4, 40, generator=g) ** 8
    times = torch.rand(300, generator=g) * 0.2  # 初めの方は若い細胞が足りない痕跡もある
    codes = ng_codes(tags, times, dg_weights(40, 160, 0.1, g), 16, 20, 0.5, 1.0, 0.1, 0.01, g)
    ind, val, dim = codes
    dense = torch.zeros(300, dim).scatter_add_(1, ind, val)
    for lam in (None, torch.rand(4, generator=g)):
        a, da = ca3_recall_codes(t, tags, codes, 3, 10, 20, lam)
        b, db = ca3_recall_codes(t, tags, nn.functional.normalize(dense, dim=-1), 3, 10, 20, lam)
        assert torch.equal(a, b) and torch.allclose(da, db, atol=1e-6)


def test_ng_rules_do_not_change_existing_rules():
    """神経新生の規則（専用の乱数）を加えても、12.6.197節までの規則の値は同じ。λ・純度は 0〜1。"""
    job = ("C", "seq", 7)
    dg = run_context_dg.run_one(job, steps=300, episodes=4, cue_ns=(2,))
    ng = run_context_ng.run_one(job, steps=300, episodes=4, cue_ns=(2,))
    for rule in run_context_dg.DG_ALL_RULES:
        assert dg["cues"]["2"]["acc"][rule] == ng["cues"]["2"]["acc"][rule]
        assert dg["cues"]["2"]["ce"][rule] == ng["cues"]["2"]["ce"][rule]
    assert dg["dg_overlap"] == ng["dg_overlap"]
    for key in ("replay_ngcmp", "holdout_ngcmp", "replay_ngshufcmp", "replay_ngnohebbcmp"):
        assert 0.0 <= ng["cues"]["2"]["ach_lambda"][key] <= 1.0
        assert 0.0 <= ng["cues"]["2"]["tcm_purity"][key] <= 1.0
    assert set(ng["ng_overlap"]) == {"ng", "ngshuf", "ngnohebb"}
    # seq では時刻が言語とそろうので、神経新生の符号は並べ替えより言語を分ける
    assert ng["ng_overlap"]["ng"]["ratio"] < ng["ng_overlap"]["ngshuf"]["ratio"]


def _run(fam, regime, seed, acc, own_kl=0.3, fresh_kl=0.3, ng_ratio=0.2):
    r = _dg_run(fam, regime, seed, acc, own_kl, fresh_kl)
    keys = ("replay_ng", "replay_ngcmp", "holdout_ngcmp", "replay_ngshuf", "replay_ngshufcmp", "replay_ngnohebbcmp")
    for c in r["cues"].values():
        c["tcm_purity"] = dict(c["tcm_purity"], **{k: 0.8 for k in keys})
        c["ach_lambda"] = dict(c["ach_lambda"], **{k: 0.3 for k in keys if k.endswith("cmp")})
    shuf = 0.6 if regime == "seq" else ng_ratio
    r["ng_overlap"] = {"ng": {"within": 0.4, "between": 0.4 * ng_ratio, "ratio": ng_ratio},
                       "ngshuf": {"within": 0.3, "between": 0.3 * shuf, "ratio": shuf},
                       "ngnohebb": {"within": 0.3, "between": 0.1, "ratio": 0.33}}
    return r


def test_h7_verdict():
    base = {"active": 0.80, "active_replay": 0.75, "active_replay_own": 0.84, "active_replay_tcm": 0.81,
            "active_replay_ca3": 0.83, "active_holdout_ca3": 0.80, "active_replay_ca3_s1": 0.82,
            "active_replay_ca3_s10": 0.83, "active_replay_ach": 0.83, "active_holdout_ach": 0.80,
            "active_replay_ach_hard": 0.82, "active_replay_ach_shuf": 0.83, "active_replay_ach_inv": 0.81,
            "active_replay_cmp": 0.82, "active_holdout_cmp": 0.80, "active_replay_cmp_shuf": 0.82,
            "active_replay_cmp_u": 0.83, "active_replay_dg": 0.82, "active_replay_dgcmp": 0.82,
            "active_holdout_dgcmp": 0.80, "active_replay_rp": 0.82, "active_replay_rpcmp": 0.82,
            "active_replay_ng": 0.83, "active_replay_ngcmp": lambda s: 0.835 + 0.002 * (s % 3 - 1),
            "active_holdout_ngcmp": 0.80, "active_replay_ngshuf": 0.82, "active_replay_ngshufcmp": 0.82,
            "active_replay_ngnohebbcmp": 0.83}
    none_ok = {r: 0.80 for r in base}
    runs = [_run("C", "seq", s, base, 0.3, 0.45) for s in range(10)] + \
        [_run("A", "joint", s, none_ok) for s in range(10)] + [_run("B", "joint", s, none_ok) for s in range(10)]
    h = run_context_ng.analyze_ng(runs)["判定"]["H7_NG"]
    assert h["上書きあり"] == ["C-seq"]
    for key in ("H7-R", "H7-S", "H7-T1", "H7-T2", "H7b", "H7-D", "H7-C", "H7-B", "H7-P"):
        assert h[key]["合格"], key
    assert h["神経新生の時間的な割り当てで古い痕跡と新しい痕跡を分けた"]
    # 並べ替えの対照と同じなら H7-S は不合格で、結論も言えない
    same = dict(base, active_replay_ngshufcmp=base["active_replay_ngcmp"])
    runs2 = [_run("C", "seq", s, same, 0.3, 0.45) for s in range(10)] + runs[10:]
    h2 = run_context_ng.analyze_ng(runs2)["判定"]["H7_NG"]
    assert h2["H7-R"]["合格"] and not h2["H7-S"]["合格"]
    assert not h2["神経新生の時間的な割り当てで古い痕跡と新しい痕跡を分けた"]
    # 上書きのない組で害（A-joint で −0.025、上書きなしの平均 −0.0125）があると H7b は不合格
    harm = dict(none_ok, active_replay_ngcmp=0.775)
    runs3 = runs[:10] + [_run("A", "joint", s, harm) for s in range(10)] + runs[20:]
    h3 = run_context_ng.analyze_ng(runs3)["判定"]["H7_NG"]
    assert not h3["H7b"]["合格"] and not h3["神経新生の時間的な割り当てで古い痕跡と新しい痕跡を分けた"]
    # joint で神経新生の符号が並べ替えと違う（時刻が言語と無関係なのに分かれる）と操作の確認は不合格
    runs4 = runs[:10] + [_run("A", "joint", s, none_ok, ng_ratio=0.2) for s in range(10)]
    for r in runs4[10:]:
        r["ng_overlap"]["ngshuf"]["ratio"] = 0.5
    h4 = run_context_ng.analyze_ng(runs4 + runs[20:])["判定"]["H7_NG"]
    assert not h4["H7-P"]["合格"]
