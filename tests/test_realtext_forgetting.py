"""実言語モデルでの継続学習の忘却指標（12.6.229〜230節）のテスト。コーパスは取得せず、小さな合成の文字列で動かす。"""

import torch

from neurocortex.context_realtext import LangData, TransformerContextLM
from neurocortex.experiments import run_realtext_forgetting as rf
from neurocortex.realtext_forgetting import ForgetConfig, fisher_diag, measure, train_forgetting


def _data(n_l=3, v=7, n=4000, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for j in range(n_l):  # 言語ごとに違う周期の文字列（少しの雑音）
        base = torch.arange(n) % (3 + j) + 1
        noise = torch.randint(1, v, (n,), generator=g)
        ids = torch.where(torch.rand(n, generator=g) < 0.1, noise, base)
        out.append(LangData(f"l{j}", ids[: n // 2], ids[n // 2: 3 * n // 4], ids[3 * n // 4:]))
    return out


def _cfg(rule, lam=0.0):
    return ForgetConfig(rule=rule, steps_per_lang=15, batch=8, replay_batch=4, seq_len=8, buffer=20, d=16,
                        layers=1, heads=2, hidden=16, ewc_lambda=lam, fisher_batches=4, fisher_batch=2,
                        eval_segments=10)


def _run(rule, lam=0.0, order=(2, 0, 1)):
    return train_forgetting(_data(), 8, _cfg(rule, lam), list(order), torch.Generator().manual_seed(5),
                            torch.Generator().manual_seed(6))


def test_matrix_shape_and_metrics():
    """順に学ぶ規則は T×T の行列と4指標を返し、joint は最後の行だけを返す。"""
    out = _run("naive")
    assert len(out["R_acc"]) == 3 and all(len(r) == 3 for r in out["R_ce"])
    assert set(out["metrics_nce"]) == {"ACC", "BWT", "F", "LA"}
    assert all(0.0 <= a <= 1.0 for r in out["R_acc"] for a in r)
    j = _run("joint")
    assert len(j["R_acc"]) == 1 and set(j["metrics_acc"]) == {"ACC"}


def test_first_language_is_identical_across_rules():
    """最初の言語の学習まではリプレイも罰則もないので、naive・replay・ewc の最初の行は一致する。
    λ = 0 の ewc は naive と全体が一致する（評価は学習の乱数を使わない）。"""
    nv, rp, ew = _run("naive"), _run("replay"), _run("ewc", 0.5)
    assert nv["R_ce"][0] == rp["R_ce"][0] == ew["R_ce"][0]
    assert nv["R_ce"] == _run("ewc", 0.0)["R_ce"]
    assert nv["R_ce"][1] != rp["R_ce"][1] and nv["R_ce"][1] != ew["R_ce"][1]


def test_order_permutes_columns():
    """R は学習の順に並べた行列（R[i][i] は i 番目に学んだ言語）。"""
    out = _run("naive", order=(1, 2, 0))
    assert out["order"] == [1, 2, 0]
    model = TransformerContextLM(8, 3, 8, 16, 1, 2, hidden=16)
    segs = [d.test[:90].reshape(10, 9) for d in _data()]
    acc, ce = measure(model, segs)
    assert len(acc) == len(ce) == 3 and all(c > 0 for c in ce)


def test_fisher_is_nonnegative_and_nonzero():
    model = TransformerContextLM(8, 3, 8, 16, 1, 2, hidden=16)
    segs = _data()[0].train[:900].reshape(100, 9)
    fs = fisher_diag(model, segs, 0, 3, 4, torch.Generator().manual_seed(0))
    flat = torch.cat([f.reshape(-1) for f in fs])
    assert (flat >= 0).all() and float(flat.sum()) > 0
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in model.parameters())


def test_strong_ewc_reduces_drift():
    """大きな λ の EWC は、最初の言語の CE の悪化を naive より小さくする。"""
    nv, ew = _run("naive"), _run("ewc", 1e3)
    assert ew["R_ce"][-1][0] - ew["R_ce"][0][0] < nv["R_ce"][-1][0] - nv["R_ce"][0][0]


def _fake(rule, set_name, seed, bwt, acc, la, lam=0.0):
    return {"rule": rule, "set": set_name, "seed": seed, "lam": lam,
            "metrics_nce": {"ACC": acc, "BWT": bwt, "F": -bwt, "LA": la},
            "metrics_acc": {"ACC": acc, "BWT": bwt, "F": -bwt, "LA": la}}


def test_analyze_judgements():
    runs = []
    for s in ("R", "G"):
        for seed in range(3):
            e = 0.01 * seed
            runs += [_fake("naive", s, seed, -0.5 - e, -2.0 - e, -1.5),
                     _fake("replay", s, seed, -0.1, -1.7 + e, -1.52 - e),
                     _fake("ewc", s, seed, -0.4, -1.9 - e, -1.5, 0.03),
                     _fake("ewc", s, seed, -0.45, -1.95, -1.5, 0.3),
                     _fake("replay_ewc", s, seed, -0.05, -1.6 + 2 * e, -1.55, 0.03),
                     {"rule": "joint", "set": s, "seed": seed, "lam": 0.0, "metrics_nce": {"ACC": -1.5},
                      "metrics_acc": {"ACC": 0.5}}]
    j = rf.analyze(runs)["判定"]
    assert j["ewc_best"] == "ewc@0.03" and j["V"]["R"]["合格"]
    assert j["H1"]["合格"] and j["H2"]["合格"] and j["H3"]["合格"]
    assert j["H4"]["合格"] and j["H5"]["合格"]
    assert abs(j["closure"]["割合"] - (-1.69 + 2.01) / (-1.5 + 2.01)) < 1e-6


def test_jobs():
    assert len(rf.jobs_for("dev", ("R", "G"), None)) == 2 * 5 * 5
    ev = rf.jobs_for("eval", ("R",), 0.03)
    assert {(r, x) for _, r, _, x in ev} == {("naive", 0.0), ("replay", 0.0), ("ewc", 0.03), ("replay_ewc", 0.03),
                                             ("joint", 0.0)}
    assert {sd for *_, sd, _ in ev} == set(rf.EVAL_SEEDS)
