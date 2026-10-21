"""胴体とヘッドのずれの診断（12.6.253〜254節）のテスト。コーパスは取得せず、小さな合成の文字列で動かす。"""

import torch

from neurocortex.context_realtext import TransformerContextLM
from neurocortex.experiments import run_realtext_head_probe as hp
from neurocortex.realtext_forgetting import train_forgetting
from neurocortex.realtext_head_probe import components, linear_cka, probe_readout, readout_ce, train_and_probe
from test_realtext_forgetting import _cfg, _data


def _model(n_ctx=3, v=8, t=8):
    torch.manual_seed(0)
    return TransformerContextLM(v, n_ctx, t, 16, 1, 2, hidden=16).eval()


def test_hook_does_not_change_training():
    """スナップショットの hook は学習を変えない（同じシードで R 行列が一致）。start・end が各言語で呼ばれる。"""
    calls = []
    a = train_forgetting(_data(), 8, _cfg("replay"), [2, 0, 1], torch.Generator().manual_seed(5),
                         torch.Generator().manual_seed(6), hook=lambda e, p, m: calls.append((e, p)))
    b = train_forgetting(_data(), 8, _cfg("replay"), [2, 0, 1], torch.Generator().manual_seed(5),
                         torch.Generator().manual_seed(6))
    assert a["R_ce"] == b["R_ce"]
    assert calls == [("start", 0), ("end", 0), ("start", 1), ("end", 1), ("start", 2), ("end", 2)]


def test_probe_changes_only_readout_and_lowers_ce():
    """プローブは元のモデルを変えず、文脈 c の読み出しだけを学び直して CE を下げる（胴体は凍結）。"""
    m = _model()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    d = _data()[1]
    seg = d.test[torch.arange(20)[:, None] + torch.arange(9)[None]]
    ce0 = readout_ce(m, seg, 1)
    curve = probe_readout(m, d.train, seg, 1, 8, torch.Generator().manual_seed(0), steps=(5, 60), batch=16, lr=1e-2)
    assert len(curve) == 2 and curve[-1] < ce0
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())


def test_components_decomposition():
    """F = L + M、I は start から diag までに得た分のうち final に残る割合。"""
    c = components({"diag": 1.0, "final": 1.6}, {"start": 2.0, "diag": 0.9, "final": 1.2})
    assert abs(c["F"] - 0.6) < 1e-12 and abs(c["L"] - 0.3) < 1e-12 and abs(c["M"] - 0.3) < 1e-12
    assert abs(c["F"] - c["L"] - c["M"]) < 1e-12
    assert abs(c["I"] - 0.8 / 1.1) < 1e-12
    assert components({"diag": 1.0, "final": 1.0}, {"start": 0.9, "diag": 1.0, "final": 1.0})["I"] is None


def test_linear_cka():
    """線形 CKA は直交変換と拡大で 1、独立な乱数では小さい。"""
    g = torch.Generator().manual_seed(0)
    a = torch.randn(500, 8, generator=g)
    q, _ = torch.linalg.qr(torch.randn(8, 8, generator=g))
    assert abs(linear_cka(a, 3.0 * a @ q) - 1.0) < 1e-9
    assert linear_cka(a, torch.randn(500, 8, generator=g)) < 0.2


def test_train_and_probe_records():
    """各言語の3時点の CE・プローブの曲線・分解・CKA を記録し、学習は train_forgetting と同じ。"""
    cfg = _cfg("naive")
    out = train_and_probe(_data(), 8, cfg, [2, 0, 1], torch.Generator().manual_seed(5),
                          torch.Generator().manual_seed(6), torch.Generator().manual_seed(7), probe_segments=6)
    ref = train_forgetting(_data(), 8, cfg, [2, 0, 1], torch.Generator().manual_seed(5),
                           torch.Generator().manual_seed(6))
    assert out["R_ce"] == ref["R_ce"]
    langs = out["probe"]["langs"]
    assert [x["lang"] for x in langs] == [2, 0, 1]
    for x in langs:
        assert set(x["probe_curve"]) == {"start", "diag", "final"} and set(x["ce"]) == {"diag", "final"}
        assert len(x["probe_curve"]["final"]) == len(out["probe"]["steps"])
        assert abs(x["F"] - x["L"] - x["M"]) < 1e-9
        assert -1e-6 <= x["cka_diag_final"] <= 1 + 1e-6
    assert abs(langs[-1]["F"]) < 1e-9  # 最後の言語は diag = final


def _fake(cond, s, seed, f, m, i):
    lang = {"pos": 0, "F": f, "L": f - m, "M": m, "I": i, "gain_diag": 0.01, "gain_final": m, "learned": 0.5,
            "cka_diag_final": 0.9, "probe_curve": {"final": [1.0, 1.0, 1.0]}}
    last = lang | {"pos": 1, "F": 0.0, "M": 0.0, "L": 0.0}
    return {"cond": cond, "set": s, "seed": seed, "probe": {"langs": [lang, last]}}


def test_analyze_judgments():
    """判定の集計: 最後の言語を除いた平均、naive の M/F・I、replay との対の差。"""
    runs = []
    for s in ("R", "G"):
        for k in range(5):
            runs.append(_fake("naive", s, k, 0.5 + 0.01 * k, 0.2 + 0.01 * k, 0.6 + 0.01 * k))
            runs.append(_fake("replay", s, k, 0.05, 0.01, 0.95 + 0.001 * k))
    j = hp.analyze(runs)["判定"]
    assert j["V"]["合格"] and j["V2"]["合格"]
    assert j["H1"]["合格"] and 0.3 < j["H1"]["M/F"] < 0.5
    assert j["H2"]["合格"] and j["H3"]["合格"] and j["H4"]["合格"]
    assert hp.analyze(runs)["表"]["naive/R"]["n"] == 5


def test_jobs():
    assert len(hp.jobs_for("dev", ("R", "G"))) == 2 * 3 * 5
    assert {j[2] for j in hp.jobs_for("eval", ("R",))} == set(range(5))
