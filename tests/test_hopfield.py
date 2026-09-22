"""海馬モジュール — パターン補完層（12.6.19節・ステップ6）の単体テスト。

最終的な想起精度ではなく、統制2（Hopfieldと1ステップ照合が数式・パラメータを
完全に共有していること）・統制6（反復回数が意図通りであること）という機構そのものを
機械的に検証する（`tests/test_path_audit.py`・`test_glia.py` と同型）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.hopfield import (  # noqa: E402
    MemorySpec,
    hopfield_retrieve,
    hopfield_step,
    knn_retrieve,
    make_cue,
    make_patterns,
    nearest_pattern_index,
)


# --- 記憶パターン・手がかり生成 --------------------------------------------------

def test_make_patterns_is_deterministic() -> None:
    spec = MemorySpec(dim=8, n_patterns=4)
    assert torch.equal(make_patterns(spec, seed=0), make_patterns(spec, seed=0))


def test_make_patterns_are_unit_norm() -> None:
    spec = MemorySpec(dim=16, n_patterns=6)
    K = make_patterns(spec, seed=1)
    assert torch.allclose(K.norm(dim=-1), torch.ones(6), atol=1e-5)


def test_make_cue_zero_corruption_returns_original() -> None:
    spec = MemorySpec(dim=10, n_patterns=5)
    K = make_patterns(spec, seed=0)
    g = torch.Generator().manual_seed(0)
    cue = make_cue(K, idx=2, corruption_rate=0.0, generator=g)
    assert torch.equal(cue, K[2])


def test_make_cue_masks_correct_number_of_dims() -> None:
    spec = MemorySpec(dim=20, n_patterns=5)
    K = make_patterns(spec, seed=0)
    g = torch.Generator().manual_seed(0)
    cue = make_cue(K, idx=0, corruption_rate=0.5, generator=g)
    n_zero = int((cue == 0.0).sum())
    assert n_zero == 10


def test_make_cue_full_corruption_is_all_zero() -> None:
    spec = MemorySpec(dim=12, n_patterns=5)
    K = make_patterns(spec, seed=0)
    g = torch.Generator().manual_seed(0)
    cue = make_cue(K, idx=0, corruption_rate=1.0, generator=g)
    assert torch.all(cue == 0.0)


# --- 更新式の共有（統制2） -------------------------------------------------------

def test_hopfield_retrieve_one_iter_matches_hopfield_step_directly() -> None:
    """統制2の核心: n_iters=1（対照群）は、単独のhopfield_step呼び出しと完全一致すること。"""
    spec = MemorySpec(dim=8, n_patterns=5, beta=4.0)
    K = make_patterns(spec, seed=0)
    xi = K[0].clone()
    xi[:2] = 0.0
    direct = hopfield_step(xi, K, spec.beta)
    via_retrieve, steps_used = hopfield_retrieve(xi, K, spec.beta, n_iters=1)
    assert torch.equal(direct, via_retrieve)
    assert steps_used == 1


def test_hopfield_step_weights_sum_to_one() -> None:
    spec = MemorySpec(dim=8, n_patterns=6, beta=2.0)
    K = make_patterns(spec, seed=0)
    xi = K[0]
    logits = spec.beta * (xi @ K.T)
    weights = torch.softmax(logits, dim=-1)
    assert weights.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_beta_zero_gives_uniform_average() -> None:
    """健全性: beta=0では全パターンへの重みが一様になり、出力は記憶の単純平均に一致すること。"""
    spec = MemorySpec(dim=6, n_patterns=4, beta=0.0)
    K = make_patterns(spec, seed=0)
    out = hopfield_step(K[0], K, beta=0.0)
    assert torch.allclose(out, K.mean(dim=0), atol=1e-5)


# --- 反復回数の監査（統制6） -----------------------------------------------------

def test_hopfield_retrieve_uses_exactly_n_iters() -> None:
    spec = MemorySpec(dim=10, n_patterns=5, beta=6.0)
    K = make_patterns(spec, seed=0)
    g = torch.Generator().manual_seed(0)
    cue = make_cue(K, idx=1, corruption_rate=0.5, generator=g)
    for n_iters in (1, 3, 8):
        _, steps_used = hopfield_retrieve(cue, K, spec.beta, n_iters=n_iters)
        assert steps_used == n_iters


def test_zero_iters_returns_input_unchanged() -> None:
    spec = MemorySpec(dim=8, n_patterns=4, beta=5.0)
    K = make_patterns(spec, seed=0)
    xi = K[0].clone()
    out, steps_used = hopfield_retrieve(xi, K, spec.beta, n_iters=0)
    assert torch.equal(out, xi)
    assert steps_used == 0


# --- 反復による収束の健全性 -------------------------------------------------------

def test_many_iterations_converge_to_a_fixed_point() -> None:
    """健全性チェック: 十分な反復回数の後は、1回追加適用してもほとんど動かなくなること
    （固定点に近づいていることの確認）。
    """
    spec = MemorySpec(dim=16, n_patterns=10, beta=6.0)
    K = make_patterns(spec, seed=0)
    g = torch.Generator().manual_seed(0)
    cue = make_cue(K, idx=3, corruption_rate=0.6, generator=g)

    xi_after_20, _ = hopfield_retrieve(cue, K, spec.beta, n_iters=20)
    delta_late = (hopfield_step(xi_after_20, K, spec.beta) - xi_after_20).norm()
    assert delta_late < 1e-3


def test_uncorrupted_cue_is_recovered_exactly_by_hopfield() -> None:
    """欠損率0（統制1の前提）なら、Hopfield条件は自分自身に最も近い記憶を返すこと。"""
    spec = MemorySpec(dim=24, n_patterns=8, beta=10.0)
    K = make_patterns(spec, seed=2)
    g = torch.Generator().manual_seed(0)
    cue = make_cue(K, idx=5, corruption_rate=0.0, generator=g)
    out, _ = hopfield_retrieve(cue, K, spec.beta, n_iters=8)
    assert nearest_pattern_index(out, K) == 5


# --- kNN対照群 --------------------------------------------------------------------

def test_knn_retrieve_returns_a_stored_pattern_exactly() -> None:
    spec = MemorySpec(dim=10, n_patterns=6)
    K = make_patterns(spec, seed=0)
    cue = K[3].clone()
    out = knn_retrieve(cue, K)
    assert torch.equal(out, K[3])


def test_knn_retrieve_finds_nearest_under_corruption() -> None:
    spec = MemorySpec(dim=20, n_patterns=5)
    K = make_patterns(spec, seed=1)
    g = torch.Generator().manual_seed(0)
    cue = make_cue(K, idx=2, corruption_rate=0.2, generator=g)
    out = knn_retrieve(cue, K)
    assert nearest_pattern_index(out, K) == 2


def test_nearest_pattern_index_identifies_exact_match() -> None:
    spec = MemorySpec(dim=12, n_patterns=7)
    K = make_patterns(spec, seed=0)
    assert nearest_pattern_index(K[4], K) == 4
