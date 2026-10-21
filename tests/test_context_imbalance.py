"""偏った記憶での比較器のエピソードごとの寄与（12.6.215〜216節）のテスト。コーパスは取得せず、小さな合成の文字列で動かす。"""

import torch

from neurocortex.context_realtext import REAL_RULES, NgramSpeaker, RealConfig, evaluate_realtext, train_realtext
from neurocortex.experiments import run_context_imbalance as ri

from test_context_realtext import _toy_data


def _trained(regime="seq"):
    data = _toy_data()
    spk = [NgramSpeaker(d.speaker, 8, 3) for d in data]
    cfg = RealConfig(regime=regime, replay_window=True, steps_per_lang=10, batch=8, replay_batch=4, seq_len=12,
                     text_len=100, buffer=30, d=16, layers=1, heads=2, hidden=16, tcm_dim=64, order=[2, 0, 1])
    g = torch.Generator().manual_seed(5)
    return data, spk, cfg, train_realtext(data, 8, cfg, g), g


def test_default_evaluate_unchanged_by_new_options():
    """mem_keep・detail を与えなければ、12.6.214節の評価と同じ値（detail は値を変えず項目を足すだけ）。"""
    data, spk, cfg, tr, g = _trained()
    state = g.get_state()
    a = evaluate_realtext(data, spk, tr, cfg, 8, (2,), g, episodes=5)
    g.set_state(state)
    b = evaluate_realtext(data, spk, tr, cfg, 8, (2,), g, episodes=5, detail=True)
    assert a["cues"]["2"]["acc"] == b["cues"]["2"]["acc"]
    assert "acc_lang" not in a["cues"]["2"] and "acc_lang" in b["cues"]["2"]
    for r in REAL_RULES:  # 言語ごとの精度の平均は全体の精度
        assert abs(sum(b["cues"]["2"]["acc_lang"][r]) / 3 - b["cues"]["2"]["acc"][r]) < 1e-6


def test_imbalanced_memory_keeps_counts_and_own_buffer():
    """偏った記憶は想起の記憶だけを減らし、学習済みのバンク・診断 own・CE は変えない。"""
    data, spk, cfg, tr, g = _trained()
    before = [b["x"].clone() for b in tr["bank"]["buf"]]
    state = g.get_state()
    bal = evaluate_realtext(data, spk, tr, cfg, 8, (2,), g, episodes=5, detail=True)
    g.set_state(state)
    imb = evaluate_realtext(data, spk, tr, cfg, 8, (2,), g, episodes=5, mem_keep=[30, 8, 8], detail=True)
    assert all(torch.equal(a, b["x"]) for a, b in zip(before, tr["bank"]["buf"]))
    assert bal["native"] == imb["native"] and bal["ce_matrix"] == imb["ce_matrix"]
    res = imb["cues"]["2"]
    assert all(0.0 <= a <= 1.0 for a in res["acc"].values())
    assert res["lam_sd"]["active_replay_cmp"] >= 0.0
    assert abs(res["lam_sd"]["active_replay_cmp"] - res["lam_sd"]["active_replay_cmp_shuf"]) < 1e-6


def test_subsample_is_a_subset():
    from neurocortex.context_realtext import _subsample
    b = {"x": torch.arange(20)[:, None].repeat(1, 4), "y": torch.arange(20), "tag": torch.rand(20, 3),
         "seg": torch.arange(20)[:, None].repeat(1, 5)}
    s = _subsample(b, 7, torch.Generator().manual_seed(0))
    assert s["x"].shape == (7, 4) and len(set(s["y"].tolist())) == 7
    assert torch.equal(s["x"][:, 0], s["y"]) and torch.equal(s["seg"][:, 0], s["y"])


def _mem(acc_lang, lam_sd):
    cues = {}
    for n in ("1", "2", "4", "8", "16"):
        acc = {r: sum(v) / len(v) for r, v in acc_lang.items()}
        cues[n] = {"acc": acc, "acc_lang": acc_lang, "lam_sd": {"active_replay_cmp": lam_sd},
                   "purity": {r: 0.8 for r in REAL_RULES}, "lambda": {r: 0.5 for r in REAL_RULES}}
    return {"cues": cues, "native": {"own_buf": 1.0, "holdout": 1.5}}


def _run(seed, imb_cmp, bal_cmp=0.8):
    base = {r: [0.8, 0.8, 0.8] for r in REAL_RULES}
    imb = dict(base, active_replay_ca3=[0.9, 0.7, 0.7], active_replay_tcm=[0.8, 0.8, 0.8],
               active_replay_cmp=[imb_cmp] * 3, active_replay_cmp_shuf=[0.8, 0.8, 0.8])
    bal = dict(base, active_replay_cmp=[bal_cmp] * 3)
    return {"set": "R", "regime": "seq", "seed": seed, "langs": ["a", "b", "c"], "dominant": 0,
            "bal": _mem(bal, 0.1), "imb": _mem(imb, 0.2)}


def test_analyze_verdicts():
    """比較器が一貫して並べ替え・常に補完より良く、補完が少数の言語で害なら H9 はすべて合格。"""
    runs = [_run(s, 0.83 + 0.001 * s) for s in range(5)]
    v = ri.analyze(runs)
    assert all(v["合否"].values()), v["合否"]
    assert v["条件を作れた・エピソードごとの寄与"] and v["分布の広さが寄与を生んだ"] and v["偏った記憶ではゲートが要る"]
    # 比較器が並べ替えと同じなら H9-G・I は不合格、D は合格のまま
    runs = [_run(s, 0.8) for s in range(5)]
    v = ri.analyze(runs)
    assert v["合否"]["H9-D"] and not v["合否"]["H9-G"] and not v["合否"]["H9-I"]


def test_dominant_is_deterministic():
    assert ri.dominant_of(100, 4) == ri.dominant_of(100, 4)
    assert 0 <= ri.dominant_of(3, 6) < 6
