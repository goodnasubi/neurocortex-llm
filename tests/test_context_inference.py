import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.basal_ganglia_core import ParitySpec  # noqa: E402
from neurocortex.context_inference import assignment_purity, run_context_sequence  # noqa: E402

SPECS = [ParitySpec(n_inputs=8, relevant=p) for p in [(0, 1), (2, 3), (4, 5)]]
KW = dict(pretrain_steps=4, steps=4, eval_n=20, cue_n=4, batch=8)


def _run(cond, **kw):
    torch.manual_seed(0)
    return run_context_sequence(cond, SPECS, 8, torch.Generator().manual_seed(1), torch.Generator().manual_seed(2),
                                **{**KW, **kw})


@pytest.mark.parametrize("cond", ["oracle", "none", "inferred", "random"])
def test_shapes_and_reproducibility(cond):
    a, b = _run(cond), _run(cond)
    assert a["R"] == b["R"] and len(a["R"]) == 3 and all(len(r) == 3 for r in a["R"])
    assert len(a["assign"]) == 12


def test_oracle_uses_true_task_as_context():
    assert [c for _, c in _run("oracle")["assign"]] == [0] * 4 + [1] * 4 + [2] * 4


def test_inferred_never_sees_task_identity():
    # inferred は τ を極端に大きくすると驚きが起きず、全タスクを文脈0で学習する（タスク ID を使っていない）
    r = _run("inferred", tau=1e9)
    assert {c for _, c in r["assign"]} == {0} and r["n_contexts"] == 1


def test_inferred_no_switch_when_never_established():
    # τ を極端に小さくすると、どの文脈も「確立済み」にならず切り替えも起きない
    r = _run("inferred", tau=-1.0)
    assert r["n_contexts"] == 1


def test_assignment_purity():
    p = assignment_purity([(0, 0), (0, 0), (1, 1), (1, 0)], 2)
    assert p["purity"] == pytest.approx(0.75) and p["distinct"] == 1.0


def test_cue_size_does_not_change_training():
    a = _run("inferred", cue_n=1, selection_trials=3)
    b = _run("inferred", cue_n=8, selection_trials=3)
    assert a["assign"] == b["assign"]
    assert 0.0 <= a["selection_acc"] <= 1.0
    assert _run("oracle", selection_trials=3)["selection_acc"] is None


def test_flip_spec_disagreement_rate():
    from neurocortex.context_inference import FlipSpec, make_batch
    g = torch.Generator().manual_seed(0)
    bits, base = make_batch(FlipSpec(8), 4000, g)
    _, flipped = make_batch(FlipSpec(8, flip=(2, 3)), 4000, torch.Generator().manual_seed(0))
    assert abs((base != flipped).float().mean().item() - 0.25) < 0.03
    ref_bits, ref = make_batch(ParitySpec(8, (0, 1)), 4000, torch.Generator().manual_seed(0))
    assert torch.equal(ref_bits, bits) and torch.equal(ref, base)


def test_transition_zero_matches_default_and_mixes_otherwise():
    a = _run("inferred")
    b = _run("inferred", transition_steps=0)
    assert a["R"] == b["R"] and a["assign"] == b["assign"]
    c = _run("oracle", transition_steps=3)
    assert len(c["R"]) == 3


def test_gated_variant_runs_and_default_unchanged():
    # inferred_gated を追加しても inferred の挙動は変わらない
    a = _run("inferred")
    assert a["R"] == _run("inferred")["R"]
    g = _run("inferred_gated", transition_steps=2)
    assert len(g["R"]) == 3 and 1 <= g["n_contexts"] <= 5


def test_gated_needs_persistent_novelty_to_allocate():
    # novel_patience を学習ステップ数より大きくすると、新しい文脈は作られない
    assert _run("inferred_gated", novel_patience=10_000)["n_contexts"] == 1


@pytest.mark.parametrize("cond", ["inferred", "inferred_gated", "oracle"])
def test_diagnostics_do_not_change_training(cond):
    # 12.6.116節: 診断は専用の乱数を使い、学習と評価の結果を変えない
    a = _run(cond, transition_steps=2)
    b = _run(cond, transition_steps=2, diagnostics=True)
    assert a["R"] == b["R"] and a["assign"] == b["assign"]
    d = b["diag"]
    assert len(d["ctx_acc"]) == len(d["chosen"]) == len(d["trained_src"]) == 3
    n_buf = sum(sum(v.values()) for v in d["buffer_src"].values())
    # ゲート付きは極小設定だと文脈が確立せずバッファが空のこともある
    assert n_buf > 0 or cond == "inferred_gated"


def test_store_consistent_only_affects_gated():
    # 12.6.116節: store_consistent は inferred_gated だけに効き、従来の inferred は変えない
    a = _run("inferred", transition_steps=2)
    b = _run("inferred", transition_steps=2, store_consistent=True)
    assert a["R"] == b["R"] and a["assign"] == b["assign"]
    g = _run("inferred_gated", transition_steps=2, store_consistent=True, diagnostics=True)
    assert len(g["R"]) == 3



def test_store_consistent_default_keeps_gated_behavior():
    # 既定（False）では 12.6.115節の inferred_gated と同じ
    kw = dict(steps=40, pretrain_steps=40, batch=16, tau=0.7, transition_steps=20)
    a = _run("inferred_gated", **kw)
    b = _run("inferred_gated", store_consistent=False, **kw)
    assert a["R"] == b["R"] and a["assign"] == b["assign"]


def test_native_contexts_and_buffer_metrics():
    # 12.6.116節: 作成が遅れて最頻の文脈が前のタスクの文脈でも、そのタスクで作られた文脈を本来の文脈とする
    from neurocortex.experiments.run_context_gated_long import (buffer_conflict, buffer_contamination,
                                                              native_contexts)
    assign = [(0, 0)] * 3 + [(1, 0)] * 3 + [(1, 1)] * 2 + [(2, 1)] * 2
    assert native_contexts(assign, 3) == [0, 1, 1]
    assert buffer_contamination({0: {0: 3}, 1: {0: 1, 1: 3}}, [0, 1]) == 1 / 7
    bits = torch.tensor([[1.0, 1.0, 1.0, -1.0], [1.0, -1.0, 1.0, 1.0]])
    buf = {"bits": bits, "labels": torch.tensor([0, 0]), "ctx": torch.tensor([1, 1]), "src": torch.tensor([0, 1])}
    specs = [ParitySpec(4, (0, 1)), ParitySpec(4, (2, 3))]
    # 文脈1の規則は b2 XOR b3: 1行目は 1（ラベル0と矛盾）、2行目は 0（整合）
    assert buffer_conflict(buf, [0, 1], specs) == 0.5


def test_reset_opt_on_new_default_and_scope():
    # 12.6.121節: 既定（False）は従来どおり。inferred・oracle では文脈を作らない/作り方が違うので影響しない
    kw = dict(steps=40, pretrain_steps=40, batch=16, tau=0.7, transition_steps=20, store_consistent=True)
    a = _run("inferred_gated", **kw)
    b = _run("inferred_gated", reset_opt_on_new=False, **kw)
    assert a["R"] == b["R"] and a["assign"] == b["assign"]
    assert _run("inferred", transition_steps=2)["R"] == _run("inferred", transition_steps=2, reset_opt_on_new=True)["R"]
    assert _run("oracle", transition_steps=2)["R"] == _run("oracle", transition_steps=2, reset_opt_on_new=True)["R"]


def test_reset_opt_on_new_resets_only_at_creation():
    # 新しい文脈が作られないなら、作り直しは起きず結果は同じ
    kw = dict(novel_patience=10_000, store_consistent=True, transition_steps=2)
    assert _run("inferred_gated", **kw)["R"] == _run("inferred_gated", reset_opt_on_new=True, **kw)["R"]


def test_trace_and_update_diagnostics_do_not_change_training():
    # 12.6.121節: trace は専用の乱数、更新ノルムは乱数を使わないので、R・割り当て・ctx_acc は変わらない
    kw = dict(steps=40, pretrain_steps=40, batch=16, tau=0.7, transition_steps=20, store_consistent=True)
    a = _run("inferred_gated", diagnostics=True, **kw)
    b = _run("inferred_gated", diagnostics=True, trace_every=5, **kw)
    assert a["R"] == b["R"] and a["assign"] == b["assign"] and a["diag"]["ctx_acc"] == b["diag"]["ctx_acc"]
    tr = b["diag"]["trace"]
    assert tr and all(len(e["acc"]) == 3 for e in tr) and {e["step"] % 5 for e in tr} == {0}
    assert len(b["diag"]["upd"]) > 0 and all(u[4] >= 0 for u in b["diag"]["upd"])


def test_diag_oracle_filter_only_with_transition():
    # 移行期間がなければ出どころは1タスクだけなので、反実仮想のフィルタは何もしない
    kw = dict(steps=40, pretrain_steps=40, batch=16, tau=0.7)
    assert _run("inferred_gated", **kw)["R"] == _run("inferred_gated", diag_oracle_filter="both", **kw)["R"]


def test_creation_dynamics():
    from neurocortex.experiments.run_context_gated_long import creation_dynamics
    assign = [(0, 0)] * 30 + [(1, 0)] * 20 + [(1, 1)] * 40
    trace = [{"task": 1, "step": s, "acc": [[0.9 if s < 20 else 0.7, 0.5]]} for s in range(0, 60, 10)]
    upd = [(1, s, 0, 0.01 if s < 20 else 0.5, 0.01 if s < 20 else 0.9) for s in range(60)]
    d = creation_dynamics(assign, trace, upd, [0, 1], window=10, after=20)
    assert abs(d["prev_drop_at_creation"] - 0.2) < 1e-9
    assert d["grad_pre"] == pytest.approx(0.01) and d["upd_post"] == pytest.approx(0.9)


def test_plasticity_cap_default_and_options():
    # 12.6.125節: 既定（plasticity_cap = 0、optimizer = "adam"）は従来どおり
    kw = dict(steps=40, pretrain_steps=40, batch=16, tau=0.7, transition_steps=20, store_consistent=True)
    a = _run("inferred_gated", **kw)
    b = _run("inferred_gated", plasticity_cap=0.0, plasticity_ref="raw", optimizer="adam", **kw)
    assert a["R"] == b["R"] and a["assign"] == b["assign"]
    assert _run("inferred")["R"] == _run("inferred", plasticity_cap=0.0)["R"]
    upd = lambda **o: [u[4] for u in _run("inferred_gated", diagnostics=True, **kw, **o)["diag"]["upd"]]  # noqa: E731
    base = upd()
    assert upd(plasticity_cap=1.0) != base and upd(optimizer="sgd") != base
    with pytest.raises(ValueError):
        _run("inferred_gated", optimizer="rmsprop")
    with pytest.raises(ValueError):
        _run("inferred_gated", plasticity_cap=1.5, plasticity_ref="x")


@pytest.mark.parametrize("optimizer", ["adam", "sgd"])
def test_plasticity_cap_bounds_updates(optimizer):
    # ρ = 1 で縮めた値を ū に使うと ū は増えないので、どの更新も最初の更新の大きさを超えない
    kw = dict(steps=40, pretrain_steps=40, batch=16, tau=0.7, transition_steps=20, store_consistent=True,
              diagnostics=True, optimizer=optimizer)
    upd = [u[4] for u in _run("inferred_gated", plasticity_cap=1.0, **kw)["diag"]["upd"]]
    assert max(upd[1:]) <= upd[0] * (1 + 1e-5)
    assert upd != [u[4] for u in _run("inferred_gated", **kw)["diag"]["upd"]]
