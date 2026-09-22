"""皮質バックボーンのハブスパース化（12.6.12節・ステップ3）。

12.3.1節の当初案（MoEの専門家間通信をハブ経由に限定する）は、素朴に実装すると
「専門家間通信を初めて導入したことの効果」と「その通信を少数のハブに集中させた
ことの効果」が混ざる（12.6.12節「主張の設定」参照）。したがって本モジュールの
主張はタスク性能ではなく、**7.1節が実測したコネクトームの頑健性の非対称性**
（標的除去がランダム除去より著しく大きくネットワーク効率を落とす）に置く。

タスクは「少なくとも2つの情報源（source）を統合しないと解けない」合成課題
（2ソースのXOR。他のソースは無関係な囮）とし、3条件の配線トポロジーを比較する。

  hub-sparse   … 少数（`n_hubs`）の中継ノードだけが全ソースに接続し、残りの
                 中継ノードは孤立（次数0）。少数への集中を体現する構成
  dense        … 全中継ノードが全ソースに接続する（最大の冗長性、集中なし）
  random-wire  … hub-sparse と**エッジ総数を揃えた**上で、接続先を少数に
                 集中させずランダムに分散させる（統制3）。ただし課題が解ける
                 ことを保証するため、1つのノードだけは関連する2ソースの両方に
                 必ず接続する（それ以外はランダム）

除去実験は「各中継ノードを1つずつ取り除いたときの精度低下」を全ノードについて
測り、**標的除去 = 最大の低下を示したノード**、**ランダム除去 = 全ノードの
低下の平均**として定義する（次数によるproxyではなく、実際の損害を直接測る
より厳密な操作化）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 合成integration課題 ------------------------------------------------------

@dataclass
class ParitySpec:
    """2ソースXOR課題の仕様。`relevant`以外のソースは無関係な囮。"""

    n_experts: int = 6
    relevant: tuple[int, int] = (0, 1)

    def __post_init__(self) -> None:
        a, b = self.relevant
        if not (0 <= a < self.n_experts and 0 <= b < self.n_experts and a != b):
            raise ValueError("relevant は 0..n_experts-1 の異なる2添字")


def make_batch(spec: ParitySpec, batch_size: int,
               generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """(bits [B, n_experts] in {-1,+1}, labels [B] in {0,1}) を返す。"""
    bits = torch.randint(0, 2, (batch_size, spec.n_experts), generator=generator).float() * 2 - 1
    a, b = spec.relevant
    labels = ((bits[:, a] > 0) ^ (bits[:, b] > 0)).float()
    return bits, labels


# --- 配線トポロジー -----------------------------------------------------------

def make_hub_sparse_adjacency(n_experts: int, n_removable: int, n_hubs: int) -> torch.Tensor:
    """最初の `n_hubs` 個を全ソースに接続し、残りを孤立させる（bool [n_removable, n_experts]）。"""
    if not 1 <= n_hubs <= n_removable:
        raise ValueError("n_hubs は 1..n_removable の範囲")
    adj = torch.zeros(n_removable, n_experts, dtype=torch.bool)
    adj[:n_hubs] = True
    return adj


def make_dense_adjacency(n_experts: int, n_removable: int) -> torch.Tensor:
    """全中継ノードが全ソースに接続する（集中なし・最大冗長）。"""
    return torch.ones(n_removable, n_experts, dtype=torch.bool)


def make_random_wire_adjacency(spec: ParitySpec, n_removable: int, n_edges: int,
                               generator: torch.Generator, n_forced: int = 1) -> torch.Tensor:
    """hub-sparse とエッジ総数を揃えた、集中のないランダム配線（統制3）。

    課題が解けることを保証するため、ランダムに選んだ `n_forced` 個のノードは
    `spec.relevant` の両方に必ず接続する（残りのエッジは完全にランダム）。

    **`n_forced` は hub-sparse の `n_hubs` と揃えること。** 揃えないと、
    「関連する2ソースの両方を独立に見られるノードの数」という、この課題では
    次数の総数以上に頑健性を左右する冗長性の指標が条件間でずれてしまい、
    エッジ総数（統制3）を揃えただけでは公平な比較にならない
    （12.6.13節の実施記録で判明した交絡）。
    """
    a, b = spec.relevant
    n_experts = spec.n_experts
    if n_edges > n_removable * n_experts:
        raise ValueError("n_edges がノード×ソースの組み合わせ数を超えている")
    if n_edges < 2 * n_forced:
        raise ValueError("n_edges が n_forced 個のノードを両関連ソースに繋ぐのに足りない")
    if n_forced > n_removable:
        raise ValueError("n_forced が n_removable を超えている")

    adj = torch.zeros(n_removable, n_experts, dtype=torch.bool)
    forced_nodes = torch.randperm(n_removable, generator=generator)[:n_forced].tolist()
    for node in forced_nodes:
        adj[node, a] = True
        adj[node, b] = True
    remaining = n_edges - 2 * n_forced

    forced_pairs = {(node, a) for node in forced_nodes} | {(node, b) for node in forced_nodes}
    all_pairs = [(i, j) for i in range(n_removable) for j in range(n_experts)
                if (i, j) not in forced_pairs]
    perm = torch.randperm(len(all_pairs), generator=generator)[:remaining]
    for idx in perm.tolist():
        i, j = all_pairs[idx]
        adj[i, j] = True
    return adj


# --- モデル -------------------------------------------------------------------

class HubGraphNet(nn.Module):
    """固定配線トポロジーの上で動く1段の中継層＋読み出し。

    配線（`adjacency`）はアーキテクチャそのものであり学習しない。パラメータ
    `W` は形状を3条件で揃え（統制2）、`adjacency` によるマスクで非エッジを
    常に0にする。マスクされた要素は勾配が常に0になり更新されない
    （`tests/test_hub_sparse.py` の統制5監査で機械的に確認する）。
    """

    def __init__(self, n_experts: int, n_removable: int, adjacency: torch.Tensor) -> None:
        super().__init__()
        if adjacency.shape != (n_removable, n_experts):
            raise ValueError("adjacency の形状が (n_removable, n_experts) と一致しない")
        self.n_removable = n_removable
        self.register_buffer("mask", adjacency.float())
        self.weight = nn.Parameter(torch.randn(n_removable, n_experts) * 0.5)
        self.node_bias = nn.Parameter(torch.zeros(n_removable))
        self.readout_weight = nn.Parameter(torch.randn(n_removable) * 0.5)
        self.readout_bias = nn.Parameter(torch.zeros(1))

    @property
    def degree(self) -> torch.Tensor:
        """各中継ノードの次数（接続しているソース数）。[n_removable]"""
        return self.mask.sum(dim=1)

    def forward(self, bits: torch.Tensor, node_active: torch.Tensor | None = None) -> torch.Tensor:
        """bits [B, n_experts] → ロジット [B]。

        `node_active` [n_removable]（bool）を渡すと、Falseのノードの出力を
        0にする（統計除去実験用のレジオン）。省略時は全ノードが有効。
        """
        w = self.weight * self.mask
        h = torch.tanh(bits @ w.T + self.node_bias)  # [B, n_removable]
        if node_active is not None:
            h = h * node_active.to(h.dtype)
        return h @ self.readout_weight + self.readout_bias


# --- 学習・除去実験 ------------------------------------------------------------

def train(model: HubGraphNet, spec: ParitySpec, steps: int, batch_size: int,
          lr: float, generator: torch.Generator) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        bits, labels = make_batch(spec, batch_size, generator)
        logits = model(bits)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


@torch.no_grad()
def accuracy(model: HubGraphNet, spec: ParitySpec, n_eval: int,
             generator: torch.Generator, node_active: torch.Tensor | None = None) -> float:
    bits, labels = make_batch(spec, n_eval, generator)
    logits = model(bits, node_active=node_active)
    pred = (logits > 0).float()
    return float((pred == labels).float().mean())


@torch.no_grad()
def per_node_drop(model: HubGraphNet, spec: ParitySpec, n_eval: int,
                  generator: torch.Generator) -> torch.Tensor:
    """各中継ノードを1つずつ除去したときの精度低下 [n_removable]。

    同じ評価バッチを全ノードの除去に使い回すことで、バッチの当たり外れが
    ノード間の比較に紛れ込まないようにする（統制の一種）。
    """
    bits, labels = make_batch(spec, n_eval, generator)
    with torch.no_grad():
        base_pred = (model(bits) > 0).float()
        base_acc = float((base_pred == labels).float().mean())
        drops = torch.zeros(model.n_removable)
        for i in range(model.n_removable):
            active = torch.ones(model.n_removable, dtype=torch.bool)
            active[i] = False
            pred = (model(bits, node_active=active) > 0).float()
            acc = float((pred == labels).float().mean())
            drops[i] = base_acc - acc
    return drops


def targeted_over_random_ratio(drops: torch.Tensor, eps: float = 1e-6) -> float:
    """標的除去（最大の低下） ÷ ランダム除去（全ノード平均の低下）。"""
    return float(drops.max() / drops.mean().clamp_min(eps))
