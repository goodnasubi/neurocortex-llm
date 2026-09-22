"""基底核モジュール（12.6.10節・ステップ2）の方策層。

素朴な二重経路設計（ロジット = Go(a) − NoGo(a)）は、表現力の観点では単一の方策
ヘッドと等価である（任意の実数ロジットは Go=max(z,0), NoGo=max(-z,0) で表現できる
ため）。したがって「学習後の方策が良い」ことだけでは、二重経路構造が意味を持つ
ことを示せない（12.6.4節でHopfield型連想記憶がdictと区別できなかったのと同型の罠）。

12.6.10節の主張は最終方策の表現力ではなく、**正負の学習信号を別パラメータ・別
学習率で扱えることが生む学習ダイナミクス上の差**（破局的抑制の起きにくさ）に置く。

更新則は Frank の Opponent Actor Learning（OpAL）型に倣う。1試行ぶんのREINFORCE
勾配 ``dz = advantage * (onehot(a) - pi)`` を座標ごとに分けるのではなく、
**試行単位で丸ごと**その試行の優位度の符号によってGo側（正）かNoGo側（負）の
どちらかにだけ流す。これが6.2節「報酬後は直接路、罰後は間接路が強化される」の
生物学的な原型に対応する。

対照群（統制5）は単一経路のロジットを1本の（実際にはパラメータ数を揃えるための
2本の）線形層で持ちつつ、更新の学習率だけを優位度の符号で切り替える。二重経路の
優位性が「パラメータの分離」由来か「学習率の非対称性」だけで足りるのかを切り分ける。

すべての更新は `torch.no_grad()` の手動REINFORCE更新であり、`hippocampus.py` の
`fit_hebbian`/`fit_stdp` と同じ流儀（autogradを介さない局所的な更新則）を踏襲する。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _policy_gradient(logits: torch.Tensor, actions: torch.Tensor,
                      advantages: torch.Tensor) -> torch.Tensor:
    """1試行ぶんのREINFORCE勾配（ロジットに対する**上昇**方向）を返す。[B, n_actions]

    ``dz = advantage * (onehot(a) - pi(a))``。損失 ``-advantage*log pi(a)`` の
    ロジットに対する勾配は ``-dz`` なので、``W += lr * x^T @ dz`` が勾配上昇になる。
    """
    pi = F.softmax(logits, dim=-1)
    onehot = F.one_hot(actions, logits.shape[-1]).to(logits.dtype)
    return advantages[:, None] * (onehot - pi)


class SinglePathwayPolicy(nn.Module):
    """対照群: 単一経路の方策ヘッド。

    パラメータ数を `DualPathwayPolicy` と揃えるため、内部に2本の線形層 ``a``・``b``
    を持ち、出力は ``a(x) + b(x)``。両方を常に同じ更新則・同じ学習率（非対称なら
    符号ごとに同じ倍率）で同時に更新するため、Go/NoGoのような**選択的な経路分離は
    一切行わない**（統制2「パラメータ予算の統制」への対応）。

    ``asymmetric_lr=True`` のとき、優位度の符号で更新の学習率だけを切り替える
    （統制5）。ロジットを産む経路自体は1系統のまま変わらない点が二重経路との違い。
    """

    def __init__(self, d_model: int, n_actions: int, asymmetric_lr: bool = False) -> None:
        super().__init__()
        self.a = nn.Linear(d_model, n_actions)
        self.b = nn.Linear(d_model, n_actions)
        self.asymmetric_lr = asymmetric_lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a(x) + self.b(x)

    @torch.no_grad()
    def update(self, x: torch.Tensor, actions: torch.Tensor, advantages: torch.Tensor,
               lr: float, lr_neg: float | None = None) -> None:
        dz = _policy_gradient(self.forward(x), actions, advantages) / x.shape[0]
        if self.asymmetric_lr:
            if lr_neg is None:
                raise ValueError("asymmetric_lr=True には lr_neg が必要")
            scale = torch.where(advantages > 0, torch.full_like(advantages, lr),
                                 torch.full_like(advantages, lr_neg))
            dz = dz * scale[:, None]
        else:
            dz = dz * lr
        delta_w = dz.T @ x
        delta_b = dz.sum(0)
        # a と b は常に同時に同じだけ動く（経路分離なし。統制2）。
        self.a.weight += delta_w
        self.a.bias += delta_b
        self.b.weight += delta_w
        self.b.bias += delta_b


class DualPathwayPolicy(nn.Module):
    """提案群: 直接路（Go）／間接路（NoGo）を別パラメータとして持つ方策ヘッド。

    ロジット = Go(x) − NoGo(x)。優位度が正の試行はGoのみ、負の試行はNoGoのみを
    更新する（OpAL型ルーティング, モジュールdocstring参照）。他方の勾配は
    その試行からは一切流れない。
    """

    def __init__(self, d_model: int, n_actions: int) -> None:
        super().__init__()
        self.go = nn.Linear(d_model, n_actions)
        self.nogo = nn.Linear(d_model, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.go(x) - self.nogo(x)

    @torch.no_grad()
    def update(self, x: torch.Tensor, actions: torch.Tensor, advantages: torch.Tensor,
               lr_go: float, lr_nogo: float) -> None:
        # バッチ全体で正規化する（部分集合のサイズではなく）。こうしないと
        # 正負の混在比率によって実効学習率が変わってしまい、single と比較不能になる。
        dz = _policy_gradient(self.forward(x), actions, advantages) / x.shape[0]
        pos = advantages > 0
        if bool(pos.any()):
            xg, dzg = x[pos], dz[pos]
            self.go.weight += lr_go * (dzg.T @ xg)
            self.go.bias += lr_go * dzg.sum(0)
        neg = ~pos
        if bool(neg.any()):
            xn, dzn = x[neg], dz[neg]
            # z = Go - NoGo なので、z を dz 方向に動かすには NoGo を -dz 方向に動かす。
            self.nogo.weight += -lr_nogo * (dzn.T @ xn)
            self.nogo.bias += -lr_nogo * dzn.sum(0)


class Critic(nn.Module):
    """クリティック（価値ヘッド）。両群で共有し、方策の構成差だけを比較対象にする。"""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)

    def update(self, x: torch.Tensor, returns: torch.Tensor, lr: float = 0.1) -> float:
        self.zero_grad(set_to_none=True)
        loss = F.mse_loss(self.forward(x), returns)
        loss.backward()
        with torch.no_grad():
            for p in self.parameters():
                p -= lr * p.grad
        return float(loss.detach())
