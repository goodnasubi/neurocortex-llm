"""予測器の並列化・複数の読み込み経路・大きな専門家（12.6.189節）のテスト。"""

from __future__ import annotations

import threading
import time

import numpy as np

from neurocortex.expert_prefetch import (FastCerebellumPredictor, MarkovPredictor, MultiPathExpertStore,
                                         expert_compute, simulate_prefetch_v2)


def _cycle_trace(n: int = 60, n_experts: int = 4, d: int = 6):
    choices = np.array([t % n_experts for t in range(n)], dtype=np.int64)
    hidden = np.random.default_rng(0).standard_normal((n, d)).astype(np.float32)
    return choices, hidden


def _weights(n: int = 4):
    return [np.full(16, e, dtype=np.float32) for e in range(n)]


def test_paths_limit_concurrent_reads(tmp_path):
    """経路の本数だけ同時に読める（1本なら直列、2本なら重なる）。"""
    for n_paths, expect_max in ((1, 1), (2, 2)):
        store = MultiPathExpertStore(_weights(), tmp_path / f"p{n_paths}", capacity=4, io_ms=20.0,
                                     n_paths=n_paths)
        active, peak = [0], [0]
        lock = threading.Lock()
        orig = store._read

        def counted(e, orig=orig):
            with lock:
                active[0] += 1
            out = orig(e)
            with lock:
                peak[0] = max(peak[0], active[0])
                active[0] -= 1
            return out

        store._read = counted
        t0 = time.perf_counter()
        th = [threading.Thread(target=store.try_prefetch, args=(e,)) for e in range(2)]
        [x.start() for x in th]
        [x.join() for x in th]
        elapsed = time.perf_counter() - t0
        assert store.prefetch_loads == 2
        if n_paths == 1:
            assert elapsed >= 0.038
        else:
            assert elapsed < 0.038


def test_use_waits_for_inflight_prefetch_without_double_read(tmp_path):
    store = MultiPathExpertStore(_weights(), tmp_path, capacity=2, io_ms=30.0, n_paths=2)
    th = threading.Thread(target=store.try_prefetch, args=(1,))
    th.start()
    time.sleep(0.005)
    w, stalled = store.use_tracked(1)
    th.join()
    assert stalled and float(w[0]) == 1.0
    assert store.prefetch_loads == 1 and store.demand_loads == 0  # 二重に読まない
    store.finish()
    assert store.wasted_loads == 0


def test_parallel_predictor_same_predictions_and_accounting(tmp_path):
    """予測器を並列にしても、同じ列に対する予測（学習後の重み）は直列と同じ。"""
    choices, hidden = _cycle_trace()
    weights = []
    for par in (False, True):
        pred = FastCerebellumPredictor(4, 6, lr=0.2)
        store = MultiPathExpertStore(_weights(), tmp_path / str(par), capacity=2, io_ms=0.5, n_paths=2)
        res = simulate_prefetch_v2(choices, hidden, store, pred, k_pf=1, compute_ms=2.0,
                                   parallel_predictor=par)
        assert res.wasted_loads <= res.prefetch_loads and res.stalls <= len(choices)
        assert res.demand_loads + res.prefetch_loads >= 4
        weights.append(pred.weight.copy())
    assert np.allclose(weights[0], weights[1])


def test_oracle_hides_io_when_compute_is_longer(tmp_path):
    choices, hidden = _cycle_trace(n=40)
    none = simulate_prefetch_v2(choices, hidden, MultiPathExpertStore(_weights(), tmp_path / "n", 2, 2.0, 1),
                                None, 1, 5.0)
    orc = simulate_prefetch_v2(choices, hidden, MultiPathExpertStore(_weights(), tmp_path / "o", 2, 2.0, 1),
                               None, 1, 5.0, oracle=True, parallel_predictor=True)
    assert none.stalls == 40 and none.prefetch_loads == 0
    assert orc.stalls <= 2 and orc.wasted_loads == 0 and orc.wall_s < none.wall_s


def test_gate_and_markov_in_v2(tmp_path):
    choices, hidden = _cycle_trace(n=40)
    store = MultiPathExpertStore(_weights(), tmp_path, 2, 0.1, 2)
    res = simulate_prefetch_v2(choices, hidden, store, MarkovPredictor(4), 1, 0.5, gate=0.5)
    assert res.prefetch_loads > 25 and res.wasted_loads == 0


def test_expert_compute_large_matmul_floor():
    w = np.zeros((2, 8, 16), dtype=np.float32)
    x = np.ones((3, 8), dtype=np.float32)
    t0 = time.perf_counter()
    expert_compute(w, x, 5.0)
    assert time.perf_counter() - t0 >= 0.0045
