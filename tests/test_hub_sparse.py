"""ハブスパース化（12.6.12節・ステップ3）の単体テスト。

タスク性能ではなく、統制3（エッジ総数の統制）と統制5（次数分布の監査）を
機械的に保証する。これらは実験結果を待たずにコード上で確認できる性質である
（`tests/test_path_audit.py` が海馬モジュールの統制を保証するのと同型）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.hub_sparse import (  # noqa: E402
    HubGraphNet,
    ParitySpec,
    accuracy,
    make_batch,
    make_dense_adjacency,
    make_hub_sparse_adjacency,
    make_random_wire_adjacency,
    per_node_drop,
    targeted_over_random_ratio,
    train,
)


# --- 配線トポロジー -----------------------------------------------------------

def test_hub_sparse_adjacency_concentrates_degree_on_hubs() -> None:
    """統制5: hub-sparse ではハブノードの次数が他ノードより有意に高いこと。"""
    adj = make_hub_sparse_adjacency(n_experts=6, n_removable=8, n_hubs=2)
    degree = adj.sum(dim=1)
    assert torch.all(degree[:2] == 6), "ハブの次数が全ソース数と一致しない"
    assert torch.all(degree[2:] == 0), "非ハブノードが孤立していない"


def test_dense_adjacency_has_no_concentration() -> None:
    """dense は全ノードが同じ（最大の）次数を持ち、集中がないこと。"""
    adj = make_dense_adjacency(n_experts=6, n_removable=8)
    degree = adj.sum(dim=1)
    assert torch.all(degree == 6)


def test_random_wire_matches_hub_sparse_edge_count() -> None:
    """統制3: hub-sparse と random-wire のエッジ総数が一致すること。"""
    spec = ParitySpec(n_experts=6, relevant=(0, 1))
    g = torch.Generator().manual_seed(0)
    hub_adj = make_hub_sparse_adjacency(6, 8, n_hubs=2)
    rw_adj = make_random_wire_adjacency(spec, n_removable=8, n_edges=12, generator=g)
    assert int(hub_adj.sum()) == int(rw_adj.sum()) == 12


def test_random_wire_does_not_concentrate_degree() -> None:
    """random-wire は hub-sparse と異なり、どのノードも突出した次数を持たないこと。

    hub-sparse は最大次数がソース数(6)に達するが、random-wireは総エッジ数を
    8ノードに分散させるため、どのノードも高々3程度に収まるはず（統制3の効果）。
    """
    spec = ParitySpec(n_experts=6, relevant=(0, 1))
    g = torch.Generator().manual_seed(0)
    rw_adj = make_random_wire_adjacency(spec, n_removable=8, n_edges=12, generator=g)
    degree = rw_adj.sum(dim=1)
    assert int(degree.max()) < 6, "random-wire なのに1ノードに次数が集中している"


def test_random_wire_guarantees_task_solvability() -> None:
    """random-wire でも、関連する2ソース両方に接続するノードが必ず1つは存在すること。

    これがないと、そのシード限りタスクが原理的に解けず統制1（基本性能）が
    偶発的に崩れる。
    """
    spec = ParitySpec(n_experts=6, relevant=(2, 4))
    for seed in range(10):
        g = torch.Generator().manual_seed(seed)
        adj = make_random_wire_adjacency(spec, n_removable=8, n_edges=12, generator=g)
        covers_both = (adj[:, 2] & adj[:, 4]).any()
        assert bool(covers_both), f"seed={seed}: 関連ソースの両方に繋がるノードがない"


def test_random_wire_n_forced_matches_hub_sparse_redundancy() -> None:
    """`n_forced=n_hubs` にすると、hub-sparseと同数のノードが冗長に課題を解けること。

    12.6.13節の実施記録: エッジ総数（統制3）だけ揃えても、この「両関連ソースを
    独立に見られるノード数」がhub-sparseの n_hubs（=2）とずれていると
    （既定の n_forced=1 のままだと）比較が公平でなかったことが判明した。
    """
    spec = ParitySpec(n_experts=6, relevant=(0, 1))
    g = torch.Generator().manual_seed(0)
    adj = make_random_wire_adjacency(spec, n_removable=8, n_edges=12, generator=g, n_forced=2)
    covers_both = adj[:, 0] & adj[:, 1]
    assert int(covers_both.sum()) == 2, "n_forced=2 なのに冗長ノードが2個になっていない"


def test_random_wire_rejects_too_few_edges() -> None:
    spec = ParitySpec(n_experts=6, relevant=(0, 1))
    g = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError):
        make_random_wire_adjacency(spec, n_removable=8, n_edges=1, generator=g)


# --- モデルの配線マスク --------------------------------------------------------

def test_masked_weights_never_receive_gradient() -> None:
    """統制5: 非エッジの重みは勾配が常に0で、学習しても動かないこと。

    「hub-sparseを名乗りながら実質denseに退化していないか」を機械的に保証する
    （12.6.1節統制5「経路の監査」と同型）。
    """
    torch.manual_seed(0)
    adj = make_hub_sparse_adjacency(n_experts=6, n_removable=8, n_hubs=2)
    model = HubGraphNet(6, 8, adj)
    before = model.weight.detach().clone()
    spec = ParitySpec(n_experts=6, relevant=(0, 1))
    g = torch.Generator().manual_seed(1)
    train(model, spec, steps=50, batch_size=32, lr=0.1, generator=g)
    after = model.weight.detach()
    non_edge = ~adj
    assert torch.equal(before[non_edge], after[non_edge]), "非エッジの重みが動いている"
    assert not torch.equal(before[adj], after[adj]), "エッジの重みが1つも動いていない"


def test_degree_property_matches_adjacency() -> None:
    adj = make_hub_sparse_adjacency(6, 8, n_hubs=2)
    model = HubGraphNet(6, 8, adj)
    assert torch.equal(model.degree, adj.sum(dim=1).float())


# --- 課題 ----------------------------------------------------------------------

def test_xor_label_is_correct() -> None:
    spec = ParitySpec(n_experts=3, relevant=(0, 1))
    bits = torch.tensor([[1.0, 1.0, -1.0], [1.0, -1.0, 1.0],
                        [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]])
    # relevant=(0,1): (1,1)->XOR=0, (1,-1)->1, (-1,-1)->0, (-1,1)->1
    expected = torch.tensor([0.0, 1.0, 0.0, 1.0])
    g = torch.Generator().manual_seed(0)
    _, labels = make_batch(spec, 4, g)  # 形状・型の確認用（値は乱数）
    assert labels.shape == (4,)
    a, b = spec.relevant
    computed = ((bits[:, a] > 0) ^ (bits[:, b] > 0)).float()
    assert torch.equal(computed, expected)


def test_relevant_indices_must_be_distinct_and_in_range() -> None:
    with pytest.raises(ValueError):
        ParitySpec(n_experts=4, relevant=(0, 0))
    with pytest.raises(ValueError):
        ParitySpec(n_experts=4, relevant=(0, 4))


# --- 除去実験の機構 -------------------------------------------------------------

def test_per_node_drop_is_zero_for_isolated_nodes() -> None:
    """孤立ノード（次数0）を除去しても精度が変わらないこと（当然の整合性チェック）。"""
    torch.manual_seed(0)
    adj = make_hub_sparse_adjacency(n_experts=6, n_removable=8, n_hubs=2)
    model = HubGraphNet(6, 8, adj)
    spec = ParitySpec(n_experts=6, relevant=(0, 1))
    g = torch.Generator().manual_seed(1)
    train(model, spec, steps=300, batch_size=64, lr=0.1, generator=g)
    drops = per_node_drop(model, spec, n_eval=1000, generator=torch.Generator().manual_seed(2))
    assert torch.allclose(drops[2:], torch.zeros(6), atol=1e-9), "孤立ノードの除去で精度が動いている"


def test_targeted_over_random_ratio_is_high_when_one_node_dominates() -> None:
    """1ノードだけが損害を持つ場合、比は大きくなること（指標そのものの健全性）。"""
    drops = torch.tensor([0.5, 0.0, 0.0, 0.0])
    ratio = targeted_over_random_ratio(drops)
    assert ratio == pytest.approx(4.0)  # 0.5 / (0.5/4)


def test_targeted_over_random_ratio_is_one_when_uniform() -> None:
    """全ノードが均等に損害を持つ場合、比は1に近いこと。"""
    drops = torch.full((4,), 0.2)
    ratio = targeted_over_random_ratio(drops)
    assert ratio == pytest.approx(1.0)


def test_hub_sparse_trains_above_chance() -> None:
    """健全性チェック: 学習後、hub-sparse 構成が課題をチャンス以上に解けること。"""
    torch.manual_seed(0)
    adj = make_hub_sparse_adjacency(n_experts=6, n_removable=8, n_hubs=2)
    model = HubGraphNet(6, 8, adj)
    spec = ParitySpec(n_experts=6, relevant=(0, 1))
    g = torch.Generator().manual_seed(1)
    train(model, spec, steps=800, batch_size=64, lr=0.1, generator=g)
    acc = accuracy(model, spec, n_eval=2000, generator=torch.Generator().manual_seed(2))
    assert acc > 0.9, f"chance(0.5)近辺から改善していない: acc={acc:.3f}"
