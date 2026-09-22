"""皮質バックボーンの層間結合スモールワールド化（12.6.30節・ステップ11）。

標準的な残差ストリーム（`x <- x + f(x)`）を持つアーキテクチャは、どの層の出力も
以降の全層にO(1)ホップで到達する経路をすでに無料で持っている。したがって
明示的な層間ショートカットの価値を検証するには、**残差ストリームを持たず、
各ホップで情報が意図的に劣化する厳密な逐次連鎖**の上で比較しなければならない
（12.6.30節「素朴な等価性の罠」参照）。

課題は `hub_sparse.py` のXORパリティ課題を層の深さ方向に再構成したもの:
2つの情報源ビットを離れた2層に注入し、最終層でXOR結合できて初めて解ける。

  local-chain … 隣接層同士の結合のみ（統制1: 到達可能性の下限）
  small-world … 少数の長距離ショートカット。早い層から遅い層へ意図的に橋渡し
  random-wire … small-world とエッジ総数を揃えるが、橋渡しを意図せず
                ランダムに配置する（統制3。ただし課題を解けるようにするため
                情報源2層から最終層への強制エッジ2本を small-world と共有する）
  dense        … 全ての (i<j) 層対を結合する（参考上限、最大冗長）

全条件は同一の `LayerNet`（層対ごとに重み行列を持つが、`adjacency` でマスクした
エッジのみ有効）を共有し、配線トポロジーだけが異なる（統制2）。マスクされた
層対の重みは勾配が常に0のままであることを監査できる（`hub_sparse.py` の
統制5監査と同型）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 合成integration課題（層の深さ方向） ----------------------------------------

@dataclass
class ChainSpec:
    """層の深さ方向に再構成したXORパリティ課題の仕様。"""

    n_layers: int = 8
    src_layer_a: int = 0
    src_layer_b: int = 3

    def __post_init__(self) -> None:
        if not (0 <= self.src_layer_a < self.n_layers - 1 and
                0 <= self.src_layer_b < self.n_layers - 1 and
                self.src_layer_a != self.src_layer_b):
            raise ValueError("src_layer_a/b は 0..n_layers-2 の異なる値")


def make_batch(spec: ChainSpec, batch_size: int,
               generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(bit_a [B], bit_b [B] in {-1,+1}, labels [B] in {0,1}) を返す。"""
    bits = torch.randint(0, 2, (batch_size, 2), generator=generator).float() * 2 - 1
    bit_a, bit_b = bits[:, 0], bits[:, 1]
    labels = ((bit_a > 0) ^ (bit_b > 0)).float()
    return bit_a, bit_b, labels


# --- 配線トポロジー（層対の集合） ------------------------------------------------

def make_local_chain_edges(n_layers: int) -> set[tuple[int, int]]:
    """隣接層同士の結合のみ。"""
    return {(l - 1, l) for l in range(1, n_layers)}


def make_dense_edges(n_layers: int) -> set[tuple[int, int]]:
    """全ての層対 (i<j) を結合する（最大冗長・参考上限）。"""
    return {(i, j) for i in range(n_layers) for j in range(i + 1, n_layers)}


def make_small_world_edges(spec: ChainSpec, n_shortcuts: int,
                           generator: torch.Generator) -> set[tuple[int, int]]:
    """局所鎖 + 「早い層→遅い層」に限定した長距離ショートカット。

    局所鎖はどの条件にも共通の可解性の土台（情報は必ず最終層まで届くが、
    ホップごとに劣化する）であり、ショートカット自体には情報源層から最終層への
    強制橋渡しを含めない。small-world 固有の主張は、その少数のショートカットが
    **早期→後期に限定して配置される**ため、random-wire（無差別配置）より
    ホップ距離を系統的に縮める点にある。
    """
    if n_shortcuts < 1:
        raise ValueError("n_shortcuts は1以上が必要")
    n_layers = spec.n_layers
    edges = make_local_chain_edges(n_layers)

    # 前半/後半の中央値で分割する。情報源層(`src_layer_a`・`src_layer_b`)が
    # どちらも必ず「早期」側に入るようにすることが必須（早期範囲が狭すぎると
    # 中間の情報源層がショートカットの起点になれず、small-worldが自身の
    # 情報源信号を橋渡しできないという構造的欠陥になる。実際に
    # n_layers=16, src_layer_b=7 で早期範囲を1/3に限定した際にこれが起き、
    # random-wireに劣る結果を生んだ）。
    mid = n_layers // 2
    early_cut = max(mid, spec.src_layer_a + 1, spec.src_layer_b + 1)
    late_cut = early_cut + 1
    early = list(range(0, early_cut))
    late = list(range(late_cut, n_layers))
    candidates = [(i, j) for i in early for j in late if i < j and (i, j) not in edges]
    if n_shortcuts > len(candidates):
        raise ValueError("要求されたショートカット数に対して早期→後期の候補層対が足りない")
    perm = torch.randperm(len(candidates), generator=generator)[:n_shortcuts]
    for idx in perm.tolist():
        edges.add(candidates[idx])
    return edges


def make_random_wire_edges(spec: ChainSpec, n_shortcuts: int,
                           generator: torch.Generator) -> set[tuple[int, int]]:
    """局所鎖 + small-world と同数のショートカットを、早遅の区別なく全層対からランダムに配置する（統制3）。

    局所鎖自体がすでに最終層までの可解性を保証しているため、small-world と
    異なり情報源層への強制橋渡しは行わない。エッジ総数だけを揃え、配置の
    「早期→後期への集中」を持たないことが small-world との唯一の違いになる。
    """
    if n_shortcuts < 1:
        raise ValueError("n_shortcuts は1以上が必要")
    n_layers = spec.n_layers
    edges = make_local_chain_edges(n_layers)

    all_pairs = [(i, j) for i in range(n_layers) for j in range(i + 1, n_layers)
                if (i, j) not in edges]
    if n_shortcuts > len(all_pairs):
        raise ValueError("要求されたショートカット数に対して候補層対が足りない")
    perm = torch.randperm(len(all_pairs), generator=generator)[:n_shortcuts]
    for idx in perm.tolist():
        edges.add(all_pairs[idx])
    return edges


def edges_to_adjacency(edges: set[tuple[int, int]], n_layers: int) -> torch.Tensor:
    adj = torch.zeros(n_layers, n_layers, dtype=torch.bool)
    for i, j in edges:
        adj[i, j] = True
    return adj


# --- モデル ---------------------------------------------------------------------

class LayerChainNet(nn.Module):
    """固定配線トポロジー上で動く、残差ストリームを持たない逐次層ネット。

    層対 (f, l) ごとの重み行列 `weight[f, l]` は全て存在するが、`adjacency`
    でマスクした層対のみ forward で使われる（`hub_sparse.HubGraphNet` と同型）。
    各層の状態は「入ってくる全エッジを重み行列で線形変換して合算し、tanhと
    加法ノイズを通す」ことでのみ更新される。前層の状態をそのまま足し込む
    経路（残差ストリーム）は一切存在しない（統制2。`test_small_world_layers.py`
    の非エッジ勾配監査で機械的に確認する）。
    """

    def __init__(self, spec: ChainSpec, hidden_dim: int, adjacency: torch.Tensor,
                noise_std: float) -> None:
        super().__init__()
        n_layers = spec.n_layers
        if adjacency.shape != (n_layers, n_layers):
            raise ValueError("adjacency の形状が (n_layers, n_layers) と一致しない")
        self.spec = spec
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std
        self.register_buffer("mask", adjacency.float())
        w_std = 1.0 / (hidden_dim ** 0.5)
        self.weight = nn.Parameter(torch.randn(n_layers, n_layers, hidden_dim, hidden_dim) * w_std)
        self.bias = nn.Parameter(torch.zeros(n_layers, hidden_dim))
        self.inj_a = nn.Parameter(torch.randn(hidden_dim) * 0.7)
        self.inj_b = nn.Parameter(torch.randn(hidden_dim) * 0.7)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, bit_a: torch.Tensor, bit_b: torch.Tensor) -> torch.Tensor:
        """bit_a, bit_b [B] → ロジット [B]。

        入ってくるエッジが複数ある層（dense条件など）で予備活性化の分散が
        エッジ数に比例して爆発しtanhが飽和するのを防ぐため、エッジ数で平均する
        （エッジを1本増やすたびに単純加算せず、寄与を正規化する）。
        """
        n_layers = self.spec.n_layers
        batch = bit_a.shape[0]
        h: list[torch.Tensor] = []
        for l in range(n_layers):
            incoming = [f for f in range(l) if bool(self.mask[f, l])]
            pre = self.bias[l].unsqueeze(0).expand(batch, -1).clone()
            if incoming:
                acc = sum(h[f] @ (self.weight[f, l] * self.mask[f, l]) for f in incoming)
                pre = pre + acc / len(incoming)
            if l == self.spec.src_layer_a:
                pre = pre + bit_a.unsqueeze(-1) * self.inj_a
            if l == self.spec.src_layer_b:
                pre = pre + bit_b.unsqueeze(-1) * self.inj_b
            h_l = torch.tanh(pre)
            if self.noise_std > 0.0:
                h_l = h_l + torch.randn_like(h_l) * self.noise_std
            h.append(h_l)
        return self.head(h[-1]).squeeze(-1)


# --- 学習・評価 -------------------------------------------------------------------

def train(model: LayerChainNet, spec: ChainSpec, steps: int, batch_size: int,
          lr: float, generator: torch.Generator) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        bit_a, bit_b, labels = make_batch(spec, batch_size, generator)
        logits = model(bit_a, bit_b)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


@torch.no_grad()
def accuracy(model: LayerChainNet, spec: ChainSpec, n_eval: int,
            generator: torch.Generator) -> float:
    bit_a, bit_b, labels = make_batch(spec, n_eval, generator)
    logits = model(bit_a, bit_b)
    pred = (logits > 0).float()
    return float((pred == labels).float().mean())


CONDITIONS = ("local-chain", "small-world", "random-wire", "dense")


def build_adjacency(condition: str, spec: ChainSpec, n_shortcuts: int,
                    generator: torch.Generator) -> torch.Tensor:
    if condition == "local-chain":
        edges = make_local_chain_edges(spec.n_layers)
    elif condition == "small-world":
        edges = make_small_world_edges(spec, n_shortcuts, generator)
    elif condition == "random-wire":
        edges = make_random_wire_edges(spec, n_shortcuts, generator)
    elif condition == "dense":
        edges = make_dense_edges(spec.n_layers)
    else:
        raise ValueError(condition)
    return edges_to_adjacency(edges, spec.n_layers)
