"""タスクの番号を与えない形の忘却指標（12.6.239〜240節）のテスト。コーパスは取得せず、小さな合成の文字列で動かす。"""

import math

import torch

from neurocortex.context_realtext import TransformerContextLM
from neurocortex.experiments import run_realtext_forgetting_taskfree as tf
from neurocortex.realtext_forgetting import ForgetConfig, infer_logp, measure, measure_infer, train_forgetting
from test_realtext_forgetting import _data


def _model(n_ctx=3, v=8, t=8):
    torch.manual_seed(0)
    m = TransformerContextLM(v, n_ctx, t, 16, 1, 2, hidden=16)
    with torch.no_grad():
        m.ctx.weight.normal_()
    return m.eval()


def _xy(v=8, t=8, n=5):
    s = torch.randint(1, v, (n, t + 1), generator=torch.Generator().manual_seed(1))
    return s[:, :-1], s[:, 1:]


def test_single_context_equals_its_head():
    """候補の文脈が1つなら、推定の予測はそのヘッドの予測と一致し、事後分布は 1。"""
    m, (x, y) = _model(), _xy()
    mix, post = infer_logp(m, x, y, [1])
    assert torch.allclose(mix, m.seq_logp(x, torch.full((x.shape[0],), 1)), atol=1e-5)
    assert torch.allclose(post, torch.ones_like(post))


def test_first_position_is_uniform_mixture_and_posterior_is_bayesian():
    """最初の位置は一様な混合、最後の事後分布は全位置の尤度の積に比例する。"""
    m, (x, y) = _model(), _xy()
    seen = [0, 2]
    mix, post = infer_logp(m, x, y, seen)
    lps = torch.stack([m.seq_logp(x, torch.full((x.shape[0],), c)) for c in seen])
    assert torch.allclose(mix[:, 0], torch.logsumexp(lps[:, :, 0], 0) - math.log(2), atol=1e-5)
    ll = lps.gather(-1, y[None, :, :, None].expand(2, -1, -1, 1))[..., 0].sum(-1)
    assert torch.allclose(post, torch.softmax(ll, 0).T, atol=1e-5)
    assert torch.allclose(mix.exp().sum(-1), torch.ones(mix.shape[:2]), atol=1e-4)


def test_infer_ce_bounded_by_oracle_plus_log_k():
    """混合の CE は、正しいヘッドの CE + log K を超えない（混合は正しいヘッドを重み 1/K 以上で含む）。"""
    m = _model()
    segs = [torch.randint(1, 8, (6, 9), generator=torch.Generator().manual_seed(j)) for j in range(3)]
    _, ce_o = measure(m, segs)
    _, ce_i, ident = measure_infer(m, segs, [0, 1, 2])
    assert all(ci <= co + math.log(3) + 1e-5 for ci, co in zip(ce_i, ce_o))
    assert all(0.0 <= a <= 1.0 for a in ident)
    _, _, ident2 = measure_infer(m, segs, [0, 1])
    assert ident2[2] == 0.0  # まだ学んでいない文脈は同定されない


def _cfg(rule, sh=False):
    return ForgetConfig(rule=rule, steps_per_lang=15, batch=8, replay_batch=4, seq_len=8, buffer=20, d=16,
                        layers=1, heads=2, hidden=16, eval_segments=10, shared_head=sh)


def test_train_records_infer_rows_and_shared_head():
    """文脈のモデルは推定の行列と同定の正確度を持ち、最初の行（1言語）では推定と oracle が一致する。
    共有のヘッドのモデルは文脈を1つだけ持ち、推定の行を持たない。"""
    out = train_forgetting(_data(), 8, _cfg("replay"), [2, 0, 1], torch.Generator().manual_seed(5),
                           torch.Generator().manual_seed(6))
    assert len(out["R_ce_inf"]) == 3 and set(out["metrics_nce_inf"]) == {"ACC", "BWT", "F", "LA"}
    assert abs(out["R_ce_inf"][0][0] - out["R_ce"][0][0]) < 1e-5 and out["R_id_inf"][0][0] == 1.0
    sh = train_forgetting(_data(), 8, _cfg("replay", True), [2, 0, 1], torch.Generator().manual_seed(5),
                          torch.Generator().manual_seed(6))
    assert "R_ce_inf" not in sh and len(sh["R_ce"]) == 3
    j = train_forgetting(_data(), 8, _cfg("joint"), [2, 0, 1], torch.Generator().manual_seed(5),
                         torch.Generator().manual_seed(6))
    assert set(j["metrics_nce_inf"]) == {"ACC"}


def _fake(cond, s, seed, acc, bwt, la, inf_shift=0.0):
    sh = cond.endswith("_sh")
    m = {"ACC": acc, "BWT": bwt, "F": -bwt, "LA": la}
    r = {"cond": cond, "set": s, "seed": seed, "shared_head": sh, "metrics_nce": m}
    if not sh:
        r["metrics_nce_inf"] = {k: v + (inf_shift if k == "ACC" else 0.0) for k, v in m.items()}
        r["R_id_inf"] = [[1.0, 0.9]]
    return r


def test_analyze_judgments():
    runs = []
    for s in ("R", "G"):
        for seed in range(5):
            e = 0.01 * seed
            runs += [_fake("naive", s, seed, -2.4 + e, -1.4 + e, -1.36), _fake("replay", s, seed, -1.5 + e, -0.1, -1.38, -0.03),
                     _fake("ewc", s, seed, -1.7 - e, -0.02, -1.68), _fake("naive_sh", s, seed, -2.5 - e, -1.6 - e, -1.3),
                     _fake("replay_sh", s, seed, -1.6 - e, -0.2, -1.4)]
    j = tf.analyze(runs)["判定"]
    assert j["V"]["合格"] and j["H1"]["合格"] and j["H2"]["合格"] and j["H4"]["合格"]
    assert j["H3"]["合格"] and abs(j["H3"]["差"] + 0.03) < 1e-9
    assert j["H5"]["合格"] and j["H6"]["合格"]


def test_jobs():
    assert len(tf.jobs_for("dev", ("R", "G"))) == 2 * len(tf.CONDS) * 5
    assert {j[2] for j in tf.jobs_for("eval", ("R",))} == set(range(5))
    assert tf.CONDS["ewc"][1] == 0.3
