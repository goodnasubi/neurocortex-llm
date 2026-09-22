"""海馬モジュール — パターン補完層・連想メモリ（12.6.19節・ステップ6）。

12.3.2節が提案するパターン補完層の実装候補である「モダンHopfieldネットワーク」を
検証する。連続値モダンHopfieldの1回の更新式は softmax attention と数式的に同一
（`Ramsauer et al. 2020`）であり、素朴には「アテンション層を追加しただけ」と
区別がつかない（12.6.6節・12.6.11節・12.6.13節・12.6.17節と同型の罠）。

したがって本モジュールは、**同一の更新式・同一の重み `K`・同一の温度 `beta` を
共有し、反復回数だけが異なる**3条件を並べる。

  hopfield      … 複数回の反復更新で固定点（記憶パターン）に収束させる（本設計）
  single-step   … 同じ更新式を1回だけ適用する（対照群、統制5。注意機構と同型）
  knn（呼び出し側）… 反復もソフトマックスも持たない最近傍探索（対照群、既存手法）

`hopfield` と `single-step` は `hopfield_retrieve` の `n_iters` 引数だけが異なり、
パラメータ・数式は完全に共有する（`tests/test_hopfield.py` で機械的に監査する）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MemorySpec:
    dim: int = 32
    n_patterns: int = 16
    beta: float = 8.0


def make_patterns(spec: MemorySpec, seed: int) -> torch.Tensor:
    """記憶パターン集合 `K`（単位球面上のランダムベクトル、次元数が十分大きければほぼ直交）。"""
    g = torch.Generator().manual_seed(seed)
    K = torch.randn(spec.n_patterns, spec.dim, generator=g)
    return K / K.norm(dim=-1, keepdim=True)


def make_cue(patterns: torch.Tensor, idx: int, corruption_rate: float,
            generator: torch.Generator) -> torch.Tensor:
    """`patterns[idx]` の一部次元をランダムにゼロ化した手がかりを作る。

    `corruption_rate` はゼロ化する次元の割合（0.0=手がかり=記憶そのもの）。
    """
    dim = patterns.shape[1]
    cue = patterns[idx].clone()
    n_mask = int(round(corruption_rate * dim))
    if n_mask > 0:
        mask_idx = torch.randperm(dim, generator=generator)[:n_mask]
        cue[mask_idx] = 0.0
    return cue


def hopfield_step(xi: torch.Tensor, K: torch.Tensor, beta: float) -> torch.Tensor:
    """モダンHopfieldの1回の更新。`softmax(beta * xi @ K^T) @ K`（softmax attentionと同型）。"""
    logits = beta * (xi @ K.T)
    weights = torch.softmax(logits, dim=-1)
    return weights @ K


def hopfield_retrieve(xi: torch.Tensor, K: torch.Tensor, beta: float,
                      n_iters: int) -> tuple[torch.Tensor, int]:
    """`hopfield_step` を `n_iters` 回反復適用する。`n_iters=1` が統制5の対照群になる。

    戻り値は (最終状態, 実際に適用した反復回数)。後者はテストでの監査用。
    """
    steps_used = 0
    for _ in range(n_iters):
        xi = hopfield_step(xi, K, beta)
        steps_used += 1
    return xi, steps_used


def knn_retrieve(cue: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """対照群（既存手法）: L2距離最近傍の記憶パターンをそのまま返す。反復もソフトマックスもない。"""
    dists = (K - cue[None, :]).norm(dim=-1)
    return K[int(dists.argmin())]


def nearest_pattern_index(x: torch.Tensor, K: torch.Tensor) -> int:
    """出力 `x` が記憶パターン集合のうちどれに最も近いかを返す（想起の正誤判定に使う）。"""
    dists = (K - x[None, :]).norm(dim=-1)
    return int(dists.argmin())
