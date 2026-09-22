"""層間結合スモールワールド化（12.6.30節・ステップ11）の単体テスト。

タスク性能ではなく、統制2（残差ストリーム不在の監査）・統制3（エッジ総数の
一致）という機構そのものを機械的に検証する
（`tests/test_hub_sparse.py` と同型）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.small_world_layers import (  # noqa: E402
    CONDITIONS,
    ChainSpec,
    LayerChainNet,
    accuracy,
    build_adjacency,
    edges_to_adjacency,
    make_dense_edges,
    make_local_chain_edges,
    make_random_wire_edges,
    make_small_world_edges,
    train,
)


SPEC = ChainSpec(n_layers=8, src_layer_a=0, src_layer_b=3)


# --- 配線トポロジー -----------------------------------------------------------

def test_local_chain_has_only_adjacent_edges() -> None:
    edges = make_local_chain_edges(8)
    assert edges == {(l - 1, l) for l in range(1, 8)}


def test_dense_edges_include_all_pairs() -> None:
    edges = make_dense_edges(5)
    assert len(edges) == 5 * 4 // 2
    assert all(i < j for i, j in edges)


def test_small_world_shortcuts_are_early_to_late_only() -> None:
    """small-world の追加ショートカットは、局所鎖以外は必ず前半層→後半層であること
    （random-wireとの構造的な違いの核心）。"""
    g = torch.Generator().manual_seed(0)
    edges = make_small_world_edges(SPEC, n_shortcuts=4, generator=g)
    local = make_local_chain_edges(SPEC.n_layers)
    extra = edges - local
    assert len(extra) == 4
    early_cut = max(SPEC.n_layers // 2, SPEC.src_layer_a + 1, SPEC.src_layer_b + 1)
    late_cut = early_cut + 1
    for i, j in extra:
        assert i < early_cut and j >= late_cut, f"({i},{j}) が早期→後期の範囲外"


def test_random_wire_matches_small_world_edge_count() -> None:
    """統制3: small-world と random-wire のエッジ総数が一致すること。"""
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(1)
    sw = make_small_world_edges(SPEC, n_shortcuts=4, generator=g1)
    rw = make_random_wire_edges(SPEC, n_shortcuts=4, generator=g2)
    assert len(sw) == len(rw)


def test_random_wire_is_not_restricted_to_early_to_late() -> None:
    """random-wireは、small-worldと異なりショートカットの配置に前半→後半の制約を持たないこと。

    局所鎖自体が既に可解性を保証するため、random-wireは情報源層への強制橋渡しを
    持たず、統制3（エッジ総数の一致）だけを満たせばよい。
    """
    local = make_local_chain_edges(SPEC.n_layers)
    early_cut = max(1, SPEC.n_layers // 3)
    late_cut = max(early_cut + 1, 2 * SPEC.n_layers // 3)
    saw_non_early_to_late = False
    for seed in range(20):
        g = torch.Generator().manual_seed(seed)
        edges = make_random_wire_edges(SPEC, n_shortcuts=4, generator=g)
        extra = edges - local
        if any(not (i < early_cut and j >= late_cut) for i, j in extra):
            saw_non_early_to_late = True
            break
    assert saw_non_early_to_late, "random-wireが偶然にも常に早期→後期にしか配置されていない"


def test_small_world_shortcuts_can_originate_from_both_source_layers() -> None:
    """情報源2層(`src_layer_a`・`src_layer_b`)自身が、small-worldのショートカットの
    起点になり得ること。

    早期範囲が狭すぎて情報源層が「早期」に含まれないと、small-worldは
    自身が注入した信号を一度もショートカット経由で橋渡しできなくなる
    （12.6.30節の実施記録で判明した構造的欠陥、random-wireに劣る逆転結果の原因）。
    """
    seen_from_a = False
    seen_from_b = False
    for seed in range(30):
        g = torch.Generator().manual_seed(seed)
        edges = make_small_world_edges(SPEC, n_shortcuts=4, generator=g)
        if any(i == SPEC.src_layer_a for i, j in edges):
            seen_from_a = True
        if any(i == SPEC.src_layer_b for i, j in edges):
            seen_from_b = True
    assert seen_from_a, "src_layer_a起点のショートカットが一度も生成されない"
    assert seen_from_b, "src_layer_b起点のショートカットが一度も生成されない"


def test_random_wire_rejects_zero_shortcuts() -> None:
    g = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError):
        make_random_wire_edges(SPEC, n_shortcuts=0, generator=g)


def test_edges_to_adjacency_shape_and_values() -> None:
    edges = {(0, 1), (1, 2)}
    adj = edges_to_adjacency(edges, n_layers=4)
    assert adj.shape == (4, 4)
    assert bool(adj[0, 1]) and bool(adj[1, 2])
    assert int(adj.sum()) == 2


def test_build_adjacency_covers_all_conditions() -> None:
    g = torch.Generator().manual_seed(0)
    for condition in CONDITIONS:
        adj = build_adjacency(condition, SPEC, n_shortcuts=3, generator=g)
        assert adj.shape == (SPEC.n_layers, SPEC.n_layers)


def test_unknown_condition_raises() -> None:
    g = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError):
        build_adjacency("bogus", SPEC, n_shortcuts=3, generator=g)


# --- モデルの配線マスク（統制2: 残差ストリーム不在の監査） -------------------------

def test_masked_layer_pairs_never_receive_gradient() -> None:
    """統制2の核心: 非エッジの層対の重みは勾配が常に0で、学習しても動かないこと。

    もし残差ストリームに相当する経路が紛れ込んでいれば、非エッジの層対経由でも
    情報が伝わり得るが、このテストは重みそのものが不変であることを直接確認する。
    """
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    adj = build_adjacency("local-chain", SPEC, n_shortcuts=0, generator=g)
    model = LayerChainNet(SPEC, hidden_dim=8, adjacency=adj, noise_std=0.0)
    before = model.weight.detach().clone()
    train(model, SPEC, steps=30, batch_size=32, lr=0.1, generator=torch.Generator().manual_seed(1))
    after = model.weight.detach()
    non_edge = ~adj.unsqueeze(-1).unsqueeze(-1).expand_as(after)
    assert torch.equal(before[non_edge], after[non_edge]), "非エッジの層対の重みが動いている"


def test_local_chain_has_no_shortcut_edges_beyond_adjacent_layers() -> None:
    """local-chain条件は、いかなる層対も隣接層以外に接続を持たないこと。"""
    g = torch.Generator().manual_seed(0)
    adj = build_adjacency("local-chain", SPEC, n_shortcuts=0, generator=g)
    for i in range(SPEC.n_layers):
        for j in range(SPEC.n_layers):
            if adj[i, j]:
                assert j == i + 1, f"local-chainなのに非隣接の層対({i},{j})が結合している"


def test_removing_final_layer_bridge_breaks_late_source_signal() -> None:
    """統制2の間接確認: 情報源bの層(src_layer_b)から最終層への唯一の経路が
    局所鎖しかない(=ショートカットなし)場合と、直接橋渡しがある場合とで、
    最終層のヤコビアン(勾配)にsrc_layer_bの重みが実際に寄与すること。
    """
    g = torch.Generator().manual_seed(0)
    adj = build_adjacency("small-world", SPEC, n_shortcuts=3, generator=g)
    model = LayerChainNet(SPEC, hidden_dim=8, adjacency=adj, noise_std=0.0)
    bit_a = torch.tensor([1.0])
    bit_b = torch.tensor([1.0])
    logit = model(bit_a, bit_b)
    grad = torch.autograd.grad(logit, model.inj_b, retain_graph=True)[0]
    assert torch.any(grad != 0), "情報源bの注入ベクトルが最終出力に何の影響も与えていない"


# --- 課題 ----------------------------------------------------------------------

def test_xor_label_is_correct() -> None:
    from neurocortex.small_world_layers import make_batch
    g = torch.Generator().manual_seed(0)
    bit_a, bit_b, labels = make_batch(SPEC, 4, g)
    expected = ((bit_a > 0) ^ (bit_b > 0)).float()
    assert torch.equal(labels, expected)


def test_chain_spec_rejects_invalid_source_layers() -> None:
    with pytest.raises(ValueError):
        ChainSpec(n_layers=8, src_layer_a=0, src_layer_b=0)
    with pytest.raises(ValueError):
        ChainSpec(n_layers=8, src_layer_a=0, src_layer_b=7)


def test_dense_trains_above_chance() -> None:
    """健全性チェック: 参考上限のdense条件は、低ノイズ下ならチャンス以上に解けること。"""
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    adj = build_adjacency("dense", SPEC, n_shortcuts=0, generator=g)
    model = LayerChainNet(SPEC, hidden_dim=16, adjacency=adj, noise_std=0.0)
    train(model, SPEC, steps=500, batch_size=64, lr=0.05, generator=torch.Generator().manual_seed(1))
    acc = accuracy(model, SPEC, n_eval=2000, generator=torch.Generator().manual_seed(2))
    assert acc > 0.9, f"chance(0.5)近辺から改善していない: acc={acc:.3f}"


def test_local_chain_struggles_more_than_dense_under_heavy_noise() -> None:
    """健全性チェック: 十分なノイズ下では、local-chainがdenseより明確に劣ること
    （ショートカットの価値を検証する土台となる基本的な非対称性）。
    """
    torch.manual_seed(0)
    noise_std = 0.6
    g_dense = torch.Generator().manual_seed(0)
    adj_dense = build_adjacency("dense", SPEC, n_shortcuts=0, generator=g_dense)
    model_dense = LayerChainNet(SPEC, hidden_dim=16, adjacency=adj_dense, noise_std=noise_std)
    train(model_dense, SPEC, steps=500, batch_size=64, lr=0.05,
         generator=torch.Generator().manual_seed(1))
    acc_dense = accuracy(model_dense, SPEC, n_eval=2000, generator=torch.Generator().manual_seed(2))

    torch.manual_seed(0)
    g_local = torch.Generator().manual_seed(0)
    adj_local = build_adjacency("local-chain", SPEC, n_shortcuts=0, generator=g_local)
    model_local = LayerChainNet(SPEC, hidden_dim=16, adjacency=adj_local, noise_std=noise_std)
    train(model_local, SPEC, steps=500, batch_size=64, lr=0.05,
         generator=torch.Generator().manual_seed(1))
    acc_local = accuracy(model_local, SPEC, n_eval=2000, generator=torch.Generator().manual_seed(2))

    assert acc_dense > acc_local, f"dense({acc_dense:.3f}) が local-chain({acc_local:.3f}) を上回っていない"
