"""MoE 専門家の予測的プリフェッチ（12.6.116節）のテスト。"""

from __future__ import annotations

import numpy as np
import torch

from neurocortex.cerebellum import ForwardModel
from neurocortex.expert_prefetch import (CerebellumPredictor, ExpertDiskStore, FastCerebellumPredictor,
                                         FreqPredictor,
                                         LastPredictor, MarkovPredictor, RoutedCharLM, hit_rates,
                                         routing_trace, simulate_prefetch)


def _cycle_trace(n: int = 300, n_experts: int = 4, d: int = 6):
    choices = np.array([t % n_experts for t in range(n)], dtype=np.int64)
    hidden = np.random.default_rng(0).standard_normal((n, d)).astype(np.float32)
    return choices, hidden


def test_routing_trace_shapes():
    torch.manual_seed(0)
    model = RoutedCharLM(vocab_size=10, d=8, n_experts=4, d_ff=8)
    choices, hidden = routing_trace(model, torch.randint(0, 10, (50,)))
    assert choices.shape == (50,) and hidden.shape == (50, 8)
    assert choices.min() >= 0 and choices.max() < 4


def test_cerebellum_predictor_reuses_forward_model():
    p = CerebellumPredictor(n_experts=4, d_context=6, lr=0.2)
    assert all(type(m) is ForwardModel for m in p.models)


def test_simple_predictors():
    choices, hidden = _cycle_trace()
    # 周期列: 直前と同じは常に外れ、マルコフは学習後ほぼ全て当たる
    assert hit_rates(choices, hidden, LastPredictor(4))["top1"] == 0.0
    assert hit_rates(choices, hidden, MarkovPredictor(4))["top1"] > 0.95
    assert hit_rates(choices, hidden, FreqPredictor(4))["top1"] < 0.5


def test_cerebellum_learns_cycle():
    choices, hidden = _cycle_trace()
    r = hit_rates(choices, hidden, CerebellumPredictor(4, 6, lr=0.2, use_context=False))
    assert r["top1"] > 0.9 and r["top2"] >= r["top1"]


def test_prefetch_accounting(tmp_path):
    choices, hidden = _cycle_trace(n=40)
    weights = [np.full(16, e, dtype=np.float32) for e in range(4)]

    def run(pred, oracle=False, sub="a"):
        store = ExpertDiskStore(weights, tmp_path / sub, capacity=2, io_ms=1.0)
        return simulate_prefetch(choices, hidden, store, pred, k_pf=1, compute_ms=5.0, oracle=oracle)

    none = run(None, sub="none")
    assert none.stalls == 40 and none.prefetch_loads == 0 and none.wasted_loads == 0
    oracle = run(None, oracle=True, sub="oracle")
    # 先読みは計算時間内に終わるので、最初の1回以外は待たない。無駄 I/O もない
    assert oracle.stalls == 1 and oracle.wasted_loads == 0
    assert oracle.wall_s < none.wall_s
    # 一般の予測器でも、無駄 I/O は先読み回数を超えず、待ちは列長を超えない
    last = run(LastPredictor(4), sub="last")
    assert last.wasted_loads <= last.prefetch_loads and last.stalls <= 40


def test_fast_cerebellum_matches_forward_model_version():
    rng = np.random.default_rng(1)
    choices = rng.integers(0, 4, 200)
    hidden = rng.standard_normal((200, 6)).astype(np.float32)
    for use_context in (True, False):
        slow = CerebellumPredictor(4, 6, lr=0.2, use_context=use_context)
        fast = FastCerebellumPredictor(4, 6, lr=0.2, use_context=use_context)
        for t in range(199):
            e, h = int(choices[t]), hidden[t]
            assert slow.rank(e, h) == fast.rank(e, h)
            slow.observe(e, h, int(choices[t + 1]))
            fast.observe(e, h, int(choices[t + 1]))
        w = np.stack([m.weight.numpy() for m in slow.models])
        assert np.allclose(w, fast.weight, atol=1e-5)
        assert np.allclose([m.bias for m in slow.models], fast.bias, atol=1e-5)


def test_gate_blocks_low_confidence_prefetch(tmp_path):
    choices, hidden = _cycle_trace(n=30)
    weights = [np.zeros(4, dtype=np.float32) for _ in range(4)]
    store = ExpertDiskStore(weights, tmp_path, capacity=2, io_ms=0.1)
    # 頻度の確率は周期列で常に 1/4 程度なので、ゲート 0.5 では一度も先読みしない
    res = simulate_prefetch(choices, hidden, store, FreqPredictor(4), k_pf=1, compute_ms=0.1, gate=0.5)
    assert res.prefetch_loads == 0
    store2 = ExpertDiskStore(weights, tmp_path / "m", capacity=2, io_ms=0.1)
    res2 = simulate_prefetch(choices, hidden, store2, MarkovPredictor(4), k_pf=1, compute_ms=0.1, gate=0.5)
    assert res2.prefetch_loads > 20 and res2.wasted_loads == 0
