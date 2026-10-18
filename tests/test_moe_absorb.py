"""大きな規模の恒常性 MoE で重みへの取り込みを促す（12.6.141節）のテスト。"""

from __future__ import annotations

import argparse

import torch

from neurocortex.data import CharCorpus, synthetic_corpus
from neurocortex.experiments import run_moe_absorb as R
from neurocortex.moe_absorb import AnnealedScaledRoutedCharLM, anneal_strength
from neurocortex.moe_scale import ScaledRoutedCharLM, train_scaled

V = 12


def test_anneal_strength_schedule():
    assert anneal_strength(7, 10, None) == 1.0
    assert anneal_strength(5, 10, 0.5) == 1.0
    assert abs(anneal_strength(8, 10, 0.5) - 0.4) < 1e-12
    assert anneal_strength(10, 10, 0.5) == 0.0


def _models(kind: str, **kw):
    torch.manual_seed(0)
    a = ScaledRoutedCharLM(V, d=8, n_experts=8, d_ff=8, kind=kind, eta=0.1)
    torch.manual_seed(0)
    b = AnnealedScaledRoutedCharLM(V, d=8, n_experts=8, d_ff=8, kind=kind, eta=0.1, **kw)
    return a, b


def test_without_anneal_matches_scaled_model():
    data = torch.randint(0, V, (400,))
    for kind in ("plain", "gate", "select"):
        a, b = _models(kind, total_steps=6)
        ca = train_scaled(a, data, 6, 4, 16, 3e-3, torch.Generator().manual_seed(1), 0.05 if kind == "plain" else 0.0)
        cb = train_scaled(b, data, 6, 4, 16, 3e-3, torch.Generator().manual_seed(1), 0.05 if kind == "plain" else 0.0)
        assert ca == cb
        assert torch.equal(a.threshold, b.threshold)
        for pa, pb in zip(a.parameters(), b.parameters()):
            assert torch.equal(pa, pb)


def test_anneal_turns_threshold_off_at_the_end():
    data = torch.randint(0, V, (400,))
    _, b = _models("select", total_steps=8, anneal_start=0.5)
    seen = []
    orig = b.homeostatic_step

    def step(choice):
        orig(choice)
        seen.append((b.strength, b.threshold.clone(), b.threshold_state.clone()))

    b.homeostatic_step = step
    train_scaled(b, data, 8, 4, 16, 3e-3, torch.Generator().manual_seed(1))
    assert [s for s, _, _ in seen] == [1.0, 1.0, 1.0, 1.0, 0.75, 0.5, 0.25, 0.0]
    for s, th, st in seen:
        assert torch.allclose(th, s * st)
    assert torch.count_nonzero(b.threshold) == 0 and torch.count_nonzero(b.threshold_state) > 0


def _args(**kw) -> argparse.Namespace:
    a = dict(sel_etas=[0.01, 0.1], ann_etas=[0.01], anneal_start=0.5, long_eta=0.01, alpha=0.05, d=8,
             train_steps=4, seq_len=16, n_windows=4, k_num=1, k_den=4, n_random=5, eps=1e-4)
    a.update(kw)
    return argparse.Namespace(**a)


def test_conditions_long_training_only_above_eight_experts():
    args = _args()
    c8, c16 = R.conditions(args, 8), R.conditions(args, 16)
    assert not any(k.startswith("long/") for k in c8)
    assert {k for k in c16 if k.startswith("long/")} == {"long/aux0", "long/aux0.05", "long/gate0.1", "long/sel0.01"}
    assert all(c16[k]["steps"] == 8 for k in c16 if k.startswith("long/"))
    assert c16["ann0.01"]["anneal"] and c16["long/sel0.01"]["kind"] == "select"


def test_select_eta_b_uses_absorption():
    def cond(ratio, ce, off_gini, off_dce):
        return {"zero": {"ratio_of_means": ratio, "base_ce": {"mean": ce}},
                "threshold_off": {"gini": {"mean": off_gini}, "d_ce": {"mean": off_dce}}}
    s = {"aux0.05": {"zero": {"base_ce": {"mean": 0.32}}},
         "sel0.01": cond(1.2, 0.32, 0.5, 0.1), "sel0.1": cond(1.1, 0.32, 0.2, 0.01), "sel0.3": cond(1.0, 0.32, 0.1, 0.0)}
    assert R.select_eta_b(s, [0.01, 0.1, 0.3])[0] == 0.1
    s["sel0.1"] = cond(1.1, 0.32, 0.35, 0.01)
    s["sel0.3"] = cond(1.0, 0.32, 0.31, 0.03)
    assert R.select_eta_b(s, [0.01, 0.1, 0.3])[0] == 0.3  # 取り込みの条件なし → 外したときのジニが最小


def test_end_to_end_small():
    torch.set_num_threads(1)
    args = _args()
    corpus = CharCorpus(synthetic_corpus(n_sentences=60, seed=0))
    sub = []
    for seed in (0, 1):
        sub.append({k: R.run_cond(seed, 16, 1, c, corpus, args) for k, c in R.conditions(args, 16).items()})
    r = R.analyze(sub, args, None)
    v = r["判定"]
    assert {"a_学習量", "b_η選択", "c_閾値を弱める_η0.01", "参考_12.6.138節と同じ_sel0.01"} <= set(v)
    assert v["c_閾値を弱める_η0.01"]["H4_閾値の幅が小さい"]  # 学習の終わりに閾値は 0
    assert r["要約"]["ann0.01"]["threshold_range"]["mean"] == 0.0
    assert r["要約"]["ann0.01"]["threshold_state_range"] > 0
    assert "long" in r["要約"] and "取り込みが起きた_H7かつH1〜H3" in v["a_学習量"]
