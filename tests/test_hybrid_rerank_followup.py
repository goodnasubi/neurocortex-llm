"""ハイブリッド再ランクのゲート式合成と単体スコア評価（12.6.116〜117節）のテスト。"""

import numpy as np
import pytest
import torch

pytest.importorskip("faiss")

from neurocortex.experiments.run_hybrid_rerank_followup import (
    build_store, candidate_recall, labeled_queries, run_seed, select,
)

N, Q = 2000, 40


@pytest.fixture(scope="module")
def store_and_queries(tmp_path_factory):
    store = build_store(tmp_path_factory.mktemp("store"), N, seed=0, batch=500)
    q, true_idx = labeled_queries(store, Q, seed=0)
    return store, q, true_idx


def _top1(store, q, lam, combine="linear", margin=0.5):
    _, stats = store.read_with_hybrid_rerank(
        q, k_candidates=int(np.sqrt(N)), k_refine=30, lambda_inner=lam[0], lambda_density=lam[1],
        lambda_template=lam[2], use_gpu=False, combine=combine, gate_margin=margin)
    return stats.top1_index.numpy()


def test_gate_margin_zero_equals_inner(store_and_queries):
    store, q, _ = store_and_queries
    inner = _top1(store, q, (1.0, 0.0, 0.0))
    assert np.array_equal(_top1(store, q, (0.0, 1.0, 0.0), "gate", 0.0), inner)


def test_gate_without_secondary_weight_equals_inner(store_and_queries):
    store, q, _ = store_and_queries
    inner = _top1(store, q, (1.0, 0.0, 0.0))
    assert np.array_equal(_top1(store, q, (0.0, 0.0, 0.0), "gate", 5.0), inner)


def test_gate_huge_margin_equals_secondary_alone(store_and_queries):
    store, q, _ = store_and_queries
    density = _top1(store, q, (0.0, 1.0, 0.0))
    assert np.array_equal(_top1(store, q, (0.0, 1.0, 0.0), "gate", 1e9), density)


def test_gated_fraction_monotone_in_margin(store_and_queries):
    store, q, _ = store_and_queries
    fr = []
    for m in (0.0, 0.5, 1e9):
        _, _, bd = store.read_with_hybrid_rerank(
            q, k_candidates=int(np.sqrt(N)), k_refine=30, lambda_inner=0.0, lambda_density=1.0,
            lambda_template=0.0, use_gpu=False, return_score_breakdown=True, combine="gate", gate_margin=m)
        fr.append(bd["gated_fraction"])
    assert fr[0] == 0.0 and fr[0] <= fr[1] <= fr[2] == 1.0


def test_unknown_combine_rejected(store_and_queries):
    store, q, _ = store_and_queries
    with pytest.raises(ValueError):
        _top1(store, q, (1.0, 0.0, 0.0), "product")


def test_labels_and_recall(store_and_queries):
    store, q, true_idx = store_and_queries
    assert q.shape == (Q, 768) and len(np.unique(true_idx)) == Q
    store.read_with_pq_search(q[:1], k_candidates=4)  # PQ インデックスを用意
    rec = candidate_recall(store, q, true_idx, int(np.sqrt(N)))
    assert 0.0 <= rec["内積上位k_refineに正解"] <= rec["候補集合に正解"] <= 1.0
    # 再ランクは候補集合の外の正解を選べないので、候補集合の再現率が精度の上限
    acc = float((_top1(store, q, (0.0, 1.0, 0.0)) == true_idx).mean())
    assert acc <= rec["候補集合に正解"] + 1e-9


def test_run_seed_and_select(tmp_path):
    conds = {"内積単体": ((1.0, 0.0, 0.0), "linear", 0.0), "線形和 a": ((0.6, 0.3, 0.1), "linear", 0.0),
             "ゲート(密度, 幅0.5)": ((0.0, 1.0, 0.0), "gate", 0.5)}
    r = run_seed(1000, 0, 20, tmp_path, conds)
    assert set(r["top1"]) == set(conds) and "ゲート(密度, 幅0.5)" in r["ゲートが開いた割合"]
    assert list(tmp_path.iterdir()) == []  # ストアは削除される
    assert select([r], conds) == {"線形和": "線形和 a", "ゲート": "ゲート(密度, 幅0.5)"}
