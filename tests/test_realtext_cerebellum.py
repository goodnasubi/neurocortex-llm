"""小脳の合成勾配（12.6.257〜258節）のテスト。コーパスは取得せず、小さな合成の文字列で動かす。"""

import math

import pytest
import torch

from neurocortex.context_realtext import TransformerContextLM
from neurocortex.experiments import run_realtext_cerebellum as rcb
from neurocortex.realtext_cerebellum import (CONDS, CBConfig, Cerebellum, lower, lower_params, tangent,
                                             train_cerebellum, upper_logp)
from test_realtext_forgetting import _data


def _cfg(cond, steps=20):
    return CBConfig(cond=cond, steps=steps, batch=8, seq_len=8, d=16, layers=2, heads=2, hidden=16,
                    feedback_every=5, granule=64, eval_every=10, eval_segments=10)


def test_split_matches_seq_logp():
    """下の皮質と上の皮質を続けると seq_logp と同じ値。"""
    m = TransformerContextLM(8, 3, 8, d=16, layers=2, heads=2, hidden=16).eval()
    x = torch.randint(0, 8, (4, 8))
    c = torch.tensor([0, 1, 2, 1])
    assert torch.allclose(upper_logp(m, lower(m, x, 1), c, 1), m.seq_logp(x, c), atol=1e-5)
    names = {id(p) for p in lower_params(m, 1)}
    assert id(m.tok.weight) in names and id(m.body.layers[0].linear1.weight) in names
    assert id(m.body.layers[1].linear1.weight) not in names and id(m.head_w) not in names


def test_tangent_removes_radial_component():
    g, h = torch.randn(3, 5, 7), torch.randn(3, 5, 7)
    t = tangent(g, h)
    assert torch.allclose((t * h).sum(-1), torch.zeros(3, 5), atol=1e-5)
    assert torch.allclose(tangent(t, h), t, atol=1e-6)


def test_cerebellum_zero_init_sparsity_and_lms():
    """プルキンエ細胞は 0 で始まり、顆粒細胞は約 25% が発火し、LMS は固定の標的への誤差を減らす。"""
    cb = Cerebellum(8, 512, 0.25, torch.Generator().manual_seed(0), vocab=5)
    h, y = torch.randn(64, 8), torch.randint(0, 5, (64,))
    assert torch.all(cb(h, y) == 0)
    frac = float((cb.granule_code(h, y) > 0).float().mean())
    assert 0.15 < frac < 0.35
    target = torch.randn(64, 8)
    errs = []
    for _ in range(300):
        pred = cb.lms(h, y, target, 0.5)
        errs.append(float(((target - pred) ** 2).mean()))
    assert errs[0] == pytest.approx(float((target ** 2).mean()))  # 返すのは学ぶ前の予測
    assert errs[-1] < 0.5 * errs[0] and all(b <= a + 1e-6 for a, b in zip(errs, errs[1:]))  # 安定に減る
    assert not torch.equal(cb.granule_code(h, y), cb.granule_code(h))  # 結果の入力は顆粒細胞の符号を変える


@pytest.mark.parametrize("cond", CONDS)
def test_conditions_run_and_count_lower_updates(cond):
    out = train_cerebellum(_data(), 8, _cfg(cond), torch.Generator().manual_seed(1), torch.Generator().manual_seed(2))
    expect = {"bp": 20, "frozen": 0, "sparse": 4, "sparse10": 4}.get(cond, 20)
    assert out["lower_updates"] == expect
    assert all(math.isfinite(x) for x in out["final_ce"])
    assert [r[0] for r in out["curve"]] == [0, 10, 20]
    if cond.startswith("cb"):
        assert out["cb_cos"] is not None and -1 <= out["cb_cos"] <= 1
    else:
        assert out["cb_cos"] is None


def test_frozen_keeps_lower_cortex_and_conditions_differ():
    """frozen は下の皮質を変えない。同じシードでも条件ごとに結果が違い、評価の断片は同じ（初期の CE が一致）。"""
    snap = {}
    orig = lower_params

    def spy(model, split):
        snap["m"] = model
        snap["init"] = [p.detach().clone() for p in orig(model, split)]
        return orig(model, split)

    import neurocortex.realtext_cerebellum as rc
    rc.lower_params = spy
    try:
        train_cerebellum(_data(), 8, _cfg("frozen"), torch.Generator().manual_seed(1), torch.Generator().manual_seed(2))
    finally:
        rc.lower_params = orig
    for a, b in zip(snap["init"], orig(snap["m"], 1)):
        assert torch.equal(a, b)
    runs = {c: train_cerebellum(_data(), 8, _cfg(c), torch.Generator().manual_seed(1), torch.Generator().manual_seed(2))
            for c in ("bp", "sparse", "cb", "cb_shuf")}
    assert len({r["curve"][0][1] for r in runs.values()}) == 1
    assert len({r["final_ce_mean"] for r in runs.values()}) == 4


def test_analyze_judgments():
    runs = []
    ce = {"bp": 1.50, "frozen": 1.60, "sparse": 1.58, "sparse10": 1.70, "cb": 1.54, "cb_shuf": 1.57, "cb_state": 1.59}
    cos = {"cb": 0.2, "cb_shuf": 0.1, "cb_state": 0.0}
    for s in ("R", "G"):
        for k in range(5):
            for c, v in ce.items():
                runs.append({"set": s, "seed": k, "cond": c, "final_ce_mean": v + 0.002 * k * (c == "bp"),
                             "auc": v + 0.1, "final_loss": v, "cb_cos": cos.get(c), "cb_cos_late": cos.get(c)})
    j = rcb.analyze(runs)["判定"]
    assert j["V"]["合格"] and j["V2"]["合格"]
    assert j["H1"]["合格"] and j["H1"]["回復の割合"] > 0.25
    assert j["H2"]["合格"] and j["H3"]["合格"]
    worse = [r | {"final_ce_mean": 1.62} if r["cond"] == "cb" else r for r in runs]
    assert not rcb.analyze(worse)["判定"]["H1"]["合格"]


def test_jobs():
    jobs = rcb.jobs_for("dev", ("R", "G"))
    assert len(jobs) == 2 * len(CONDS) * 5 and {j[2] for j in jobs} == set(rcb.DEV_SEEDS)
    assert {j[2] for j in rcb.jobs_for("eval", ("R",), ["cb"])} == set(rcb.EVAL_SEEDS)


def test_reliability_gain():
    """修正1: 利得は 0〜1 で、小脳の条件でだけ記録され、利得なしの cb と結果が違う。"""
    cfg = _cfg("cb", steps=30)
    cfg.reliability = True
    out = train_cerebellum(_data(), 8, cfg, torch.Generator().manual_seed(1), torch.Generator().manual_seed(2))
    assert out["gain_mean"] is not None and 0.0 <= out["gain_mean"] <= 1.0
    base = train_cerebellum(_data(), 8, _cfg("cb", steps=30), torch.Generator().manual_seed(1),
                            torch.Generator().manual_seed(2))
    assert base["gain_mean"] is None and base["final_ce_mean"] != out["final_ce_mean"]
    sp = _cfg("sparse", steps=30)
    sp.reliability = True
    assert train_cerebellum(_data(), 8, sp, torch.Generator().manual_seed(1),
                            torch.Generator().manual_seed(2))["gain_mean"] is None
