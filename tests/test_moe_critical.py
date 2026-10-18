"""臨界期型の閉じ方とゲートの大きさに左右されない取り込みの指標（12.6.145節）のテスト。"""

from __future__ import annotations

import argparse

import torch

from neurocortex.data import CharCorpus, synthetic_corpus
from neurocortex.experiments import run_moe_critical as R
from neurocortex.moe_absorb import AnnealedScaledRoutedCharLM, anneal_strength
from neurocortex.moe_critical import CriticalPeriodLM, absorption_metrics, close_strength
from neurocortex.moe_scale import ScaledRoutedCharLM, train_scaled

V = 12


def test_close_strength_schedule():
    for t in range(0, 11):
        assert close_strength(t, 10, 0.5, 0.5) == anneal_strength(t, 10, 0.5)
    assert close_strength(3, 10, None) == 1.0
    s = [close_strength(t, 20, 0.25, 0.25) for t in range(21)]
    assert s[:6] == [1.0] * 6 and abs(s[7] - 0.6) < 1e-12 and s[10:] == [0.0] * 11  # 固定化期間は 0


def _train(model, steps=8):
    data = torch.arange(500) % V
    return train_scaled(model, data, steps, 4, 16, 3e-3, torch.Generator().manual_seed(1),
                        0.05 if model.kind == "plain" else 0.0)


def test_default_matches_scaled_model_even_with_probe():
    probe = (torch.arange(64) % V).reshape(4, 16)
    for kind in ("plain", "gate", "select"):
        for layers in (1, 2):
            torch.manual_seed(0)
            a = ScaledRoutedCharLM(V, d=8, n_experts=8, d_ff=8, n_layers=layers, kind=kind, eta=0.1)
            torch.manual_seed(0)
            b = CriticalPeriodLM(V, d=8, n_experts=8, d_ff=8, n_layers=layers, kind=kind, eta=0.1,
                                 total_steps=8, probe=probe, log_every=2, vec_every=4)
            assert _train(a) == _train(b)
            assert torch.equal(a.threshold, b.threshold)
            for pa, pb in zip(a.parameters(), b.parameters()):
                assert torch.equal(pa, pb)
            if kind != "plain":
                assert [p["step"] for p in b.series] == [1, 2, 4, 6, 8]
                assert "theta_state" in b.series[2] and "theta_state" not in b.series[1]


def test_half_schedule_matches_annealed_model():
    torch.manual_seed(0)
    a = AnnealedScaledRoutedCharLM(V, d=8, n_experts=8, d_ff=8, kind="select", eta=0.1, total_steps=8,
                                   anneal_start=0.5)
    torch.manual_seed(0)
    b = CriticalPeriodLM(V, d=8, n_experts=8, d_ff=8, kind="select", eta=0.1, total_steps=8, open_frac=0.5,
                         ramp_frac=0.5)
    assert _train(a) == _train(b)
    assert torch.equal(a.threshold, b.threshold) and torch.equal(a.threshold_state, b.threshold_state)


def test_consolidation_keeps_threshold_off():
    torch.manual_seed(0)
    b = CriticalPeriodLM(V, d=8, n_experts=8, d_ff=8, kind="select", eta=0.3, total_steps=8, open_frac=0.25,
                         ramp_frac=0.25, probe=(torch.arange(64) % V).reshape(4, 16), log_every=1)
    _train(b)
    assert [p["s"] for p in b.series][4:] == [0.0] * 4
    assert all(p["th_range"] == 0.0 for p in b.series[4:])
    assert all(p["agree"] == 1.0 and p["gini_on"] == p["gini_off"] for p in b.series[4:])


def test_absorption_metrics():
    torch.manual_seed(0)
    m = CriticalPeriodLM(V, d=8, n_experts=8, d_ff=8, kind="select", eta=0.1)
    x = (torch.arange(64) % V).reshape(4, 16)
    y = ((torch.arange(64) + 1) % V).reshape(4, 16)
    a = absorption_metrics(m, x, y)
    assert a["agree"] == 1.0 and a["d_ce_off"] == 0.0 and a["agree_state"] == 1.0
    r = m.run(x)
    assert abs(a["contrib"] - float(r["contrib"].mean().detach())) < 1e-7
    m.threshold_state[0, int(r["choice"][0, 0, 0])] = 100.0  # 選ばれていた専門家を状態の上で強く抑える
    m.threshold.copy_(0.5 * m.threshold_state)
    a = absorption_metrics(m, x, y)
    assert a["agree"] < 1.0 and a["agree_state"] < 1.0
    assert abs(a["d_ce_off_norm"] - a["d_ce_off"] / a["contrib"]) < 1e-12
    assert torch.count_nonzero(m.threshold - 0.5 * m.threshold_state) == 0  # 測定の後に閾値は元に戻る


def test_rho_detects_absorbed_offsets():
    torch.manual_seed(0)
    probe = (torch.arange(64) % V).reshape(4, 16)
    m = CriticalPeriodLM(V, d=8, n_experts=8, d_ff=8, kind="select", eta=0.1, probe=probe)
    st = torch.linspace(-1, 1, 8)
    m.threshold_state[0] = st
    with torch.no_grad():
        m.routers[0].bias.sub_(st)  # 重みが −θ の分だけ対数を動かした（取り込んだ）状態
    assert m.snapshot()["rho"] > 0.99


def _cand(gini_max, ok=True, h3=True, gini=0.2):
    return {"H1_崩壊を防ぐ": ok, "H2_ハブの非対称性を消す": ok, "H3_性能を損なわない": h3, "ジニ": gini,
            "ジニ_シード最大": gini_max}


def test_select_cp_rules():
    v = {"cp0.1_o0.5_r0.05": _cand(0.25), "cp0.03_o0.5_r0.05": _cand(0.25), "cp0.03_o0.25_r0.05": _cand(0.25),
         "cp1_o0.25_r0.05": _cand(0.2, gini=0.31), "cp0.3_o0.25_r0.05": _cand(0.1, ok=False)}
    assert R.select_cp(v)[0] == "cp0.03_o0.25_r0.05"  # 同じシード最大のジニなら η が小さく、固定化期間が長い
    v = {"cp0.1_o0.5_r0.05": _cand(0.4, ok=False), "cp0.3_o0.5_r0.05": _cand(0.35, ok=False, h3=False)}
    k, why = R.select_cp(v)
    assert k == "cp0.1_o0.5_r0.05" and "満たさない" in why


def test_conditions_and_parse():
    a = argparse.Namespace(sel_etas=[0.01], extra_etas=[1.0], ann_etas=[0.1], long=True, grid=True,
                           grid_scales=["16,1"], cp=["32,1=cp0.3_o0.25_r0.05"], train_steps=600)
    c16, c32 = R.conditions(a, 16, 1), R.conditions(a, 32, 1)
    assert sum(k.startswith("cp") for k in c16) == len(R.SCHEDULES) * len(R.CP_ETAS)
    assert [k for k in c32 if k.startswith("cp")] == ["cp0.3_o0.25_r0.05"] and "sel1" in c32 and "sel1" not in c16
    assert c32["cp0.3_o0.25_r0.05"] == {"kind": "select", "eta": 0.3, "open": 0.25, "ramp": 0.05}
    assert c32["long/sel0.01"]["steps"] == 2400 and c16["ann0.1"]["open"] == 0.5


def test_end_to_end_small(monkeypatch):
    torch.set_num_threads(1)
    args = argparse.Namespace(sel_etas=[0.01], extra_etas=[], ann_etas=[0.1], long=True, grid=True,
                              grid_scales=["16,1"], cp=None, alpha=0.05, d=8, train_steps=8, seq_len=16,
                              n_windows=4, n_probe=2, log_every=2, k_num=1, k_den=4, n_random=5, eps=1e-4)
    monkeypatch.setattr(R, "SCHEDULES", ((0.25, 0.25),))
    monkeypatch.setattr(R, "CP_ETAS", (0.1,))
    corpus = CharCorpus(synthetic_corpus(n_sentences=60, seed=0))
    sub = {}
    for name, (e, l) in (("E8_L1", (8, 1)), ("E16_L1", (16, 1))):
        sub[name] = [{k: R.run_cond(s, e, l, k, c, corpus, args) for k, c in R.conditions(args, e, l).items()}
                     for s in (0, 1)]
    ref = R.absorb_summary(sub["E8_L1"], "sel0.01")["d_ce_off_norm"]["mean"]
    r = R.analyze(sub["E16_L1"], args, "E16_L1", ref)
    v = r["判定"]
    assert v["2_選択"]["キー"] == "cp0.1_o0.25_r0.25"
    assert "H7*" in v["1_指標の測り直し"]["long/sel0.01"] and "sel0.01" in v["3_時系列"]
    assert r["要約"]["取り込みの指標"]["cp0.1_o0.25_r0.25"]["agree"]["mean"] == 1.0
    assert "theta_state" in sub["E16_L1"][0]["cp0.1_o0.25_r0.25"]["series"][-1]  # 最後のステップはベクトルを残す
