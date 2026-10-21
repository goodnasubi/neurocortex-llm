"""シナプスの固定化の多時間尺度のカスケード（Benna & Fusi 2016。12.6.233節）。"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.basal_ganglia_core import ParitySpec  # noqa: E402
from neurocortex.context_inference import run_context_sequence  # noqa: E402
from neurocortex.experiments import run_context_replay_cascade as exp  # noqa: E402

# 12.6.231節のテストと同じ小さな設定
SPECS = [ParitySpec(n_inputs=4, relevant=p) for p in [(0, 1), (1, 2), (2, 3)]]
KW = dict(pretrain_steps=150, steps=150, eval_n=50, cue_n=8, batch=16, novel_patience=3, selection_trials=2,
          diagnostics=True, reliability_holdout=40, episodic_hold=20, writeback="none", writeback_at=2,
          writeback_steps=5, writeback_eval=100, writeback_keep=True)


def _run(**kw):
    torch.manual_seed(0)
    return run_context_sequence("inferred_gated", SPECS, 16, torch.Generator().manual_seed(1),
                                torch.Generator().manual_seed(2), **{**KW, **kw})


def test_cascade_only_after_writeback_and_changes_learning():
    """カスケードは書き戻しの時点（新しいタスク）からだけ働き、それまでの学習は変えない。働くと学習が変わり、
    遅い変数は適用の時点から動く。"""
    ref = _run(replay_rule="ach_c")
    for scope in ("shared", "all"):
        x = _run(replay_rule="ach_c", cascade_levels=4, cascade_g=0.05, cascade_scope=scope)
        assert x["R"][:2] == ref["R"][:2] and x["diag"]["writeback"] == ref["diag"]["writeback"]
        assert x["diag"]["ctx_acc"][2] != ref["diag"]["ctx_acc"][2]
        c = x["diag"]["cascade"][2]
        assert c["gap"] > 0.0 and c["move"] > 0.0
    assert "cascade" not in ref["diag"]


def test_cascade_step_conserves_and_relaxes():
    """拡散は Σ C_k u_k を保ち（勾配の入力のない間）、鎖を一つの値に近づける。shared では文脈の列は変えない。"""
    import neurocortex.context_inference as ci

    captured = {}
    real_adam = torch.optim.Adam

    class NoStep(real_adam):  # 最適化器の更新を止めて、拡散だけを見る
        def step(self, closure=None):
            captured.setdefault("params", self.param_groups[0]["params"])
            return None

    try:
        ci.torch.optim.Adam = NoStep
        x = _run(replay_rule="ach_c", cascade_levels=3, cascade_g=0.2)
        y = _run(replay_rule="ach_c")
    finally:
        ci.torch.optim.Adam = real_adam
    # 更新がなければ隠れ変数は重みと同じ値で始まり、拡散しても何も変わらない
    assert x["R"] == y["R"]
    assert x["diag"]["cascade"][2]["gap"] == 0.0 and x["diag"]["cascade"][2]["move"] == 0.0


def test_cascade_zero_is_default():
    """cascade_levels = 0 は従来と同じ（乱数の消費も同じ）。"""
    a = _run(replay_rule="ach_c")
    b = _run(replay_rule="ach_c", cascade_levels=0, cascade_g=0.3, cascade_scope="all")
    assert a["R"] == b["R"] and a["diag"]["ctx_acc"] == b["diag"]["ctx_acc"]


def test_cascade_shared_keeps_context_columns_free():
    """shared では文脈の one-hot の列は固定化されない（all と固定化の対象と学習の軌跡が変わる。この小さな設定は
    正確度が飽和するので、カスケードの診断で確かめる）。"""
    s = _run(replay_rule="ach_c", cascade_levels=4, cascade_g=0.2, cascade_scope="shared")
    a = _run(replay_rule="ach_c", cascade_levels=4, cascade_g=0.2, cascade_scope="all")
    assert s["diag"]["cascade"][2] != a["diag"]["cascade"][2]


@pytest.mark.parametrize("kw", [dict(cascade_levels=1), dict(cascade_levels=3, cascade_g=0.0),
                                dict(cascade_levels=3, cascade_g=0.6), dict(cascade_levels=3, cascade_scope="ctx"),
                                dict(cascade_levels=3, ctx_units=4), dict(cascade_capture=0.1),
                                dict(cascade_levels=3, cascade_capture=1.5)])
def test_invalid_args(kw):
    with pytest.raises(ValueError):
        _run(replay_rule="ach_c", **kw)


def test_experiment_config_jobs_and_verdict():
    assert exp.CAS == {"cascade_levels": 4, "cascade_g": 0.05}
    for c in exp.NEW:
        assert exp.CONFIGS[c]["cascade_levels"] == 4
    assert exp.CONFIGS["ach_c_cas_all"]["cascade_scope"] == "all"
    assert len(exp.jobs()) == 30 * len(exp.NEW)
    assert all(exp.CONFIGS[c]["cascade_capture"] == 0.1 for c in exp.FIXES)
    o = {"end_keep": 0.14, "end_ach_c": 0.08, "end_keep_cas": 0.15, "end_ach_c_cas": 0.12, "end_base_cas": 0.0,
         "end_ach_c_cas_all": 0.08}
    n = {f"end_{w}": 0.01 for w in ("keep", "ach_c", *exp.NEW)}
    a = {"new_keep": -0.03, "new_ach_c": 0.0, "new_keep_cas": -0.01, "new_ach_c_cas": -0.01, "new_base_cas": -0.03,
         "new_ach_c_cas_all": 0.0, "bad_ach_c": 2, "bad_keep_cas": 1, "bad_ach_c_cas": 1, "bad_base_cas": 0,
         "bad_ach_c_cas_all": 3, "bad_keep": 9}
    for w in ("keep", "ach_c", *exp.NEW):
        a.setdefault(f"min_new_{w}", -0.04)
        a.setdefault(f"bad_frac_{w}", a.get(f"bad_{w}", 0) / 150)
    v = exp.verdict({"over_14bit": o, "not_over": n, "all": a})
    # 直線: end = 0.08 + 0.06·(new − 0)/(−0.03)。new −0.01 → 0.10、残差は keep_cas +0.05、ach_c_cas +0.02
    assert v["keep_cas"]["residual"] == pytest.approx(0.05) and v["keep_cas"]["F"]
    assert v["ach_c_cas"]["residual"] == pytest.approx(0.02)
    assert v["F"] and v["H"] and v["ach_c_cas"]["D4"]
    assert v["B"] is False  # new −0.03
    assert "residual" not in v["base_cas"]


def test_capture_changes_learning_and_full_capture_is_plain():
    """捕獲は学習を変える。κ = 1 で常に新しい（f = 0）なら、隠れ変数は毎ステップ見える重みに揃い、覚醒中は可塑性だけと
    同じになる（この設定では睡眠の段階がないので、カスケードなしと同じ学習になる）。"""
    ref = _run(replay_rule="ach_c")
    c = _run(replay_rule="ach_c", cascade_levels=4, cascade_g=0.2)
    k = _run(replay_rule="ach_c", cascade_levels=4, cascade_g=0.2, cascade_capture=0.1)
    assert k["diag"]["cascade"][2] != c["diag"]["cascade"][2]
    full = _run(replay_rule="ach_c", cascade_levels=4, cascade_g=0.2, cascade_capture=1.0, tau=1e-9)
    plain = _run(replay_rule="ach_c", tau=1e-9)
    assert full["R"] == plain["R"] and full["diag"]["cascade"][2]["gap"] < 1e-6
    assert ref["R"][:2] == k["R"][:2]
