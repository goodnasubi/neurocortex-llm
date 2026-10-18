"""MoE ルーティングの偏りを恒常性可塑性で防ぐ（12.6.127節）。

補助損失（勾配で「均等に使え」と教える工学的な項）の代わりに、専門家ごとの閾値を
勾配によらない積分制御で動かす。神経細胞の内因性可塑性（発火率の履歴に応じて閾値・興奮性を
調節する。Desai et al., 1999; Turrigiano, 2008; O'Leary et al., 2014）に対応する。

- 活動センサー: バッチ内の選択割合 f_e の移動平均 u_e ← (1−α) u_e + α f_e
- 閾値の更新: θ_e ← θ_e + η (E u_e − 1)、平均は毎回引き去る（ソフトマックスに効かない）
- ルータの対数 = W h + c − θ（選択とゲートの両方に効く。使われすぎた専門家は閾値が上がり選ばれにくくなる）

`expert_prefetch.py`・`hub_load.py` には手を入れない。ルータを `HomeostaticRouter` で包むので、
`hub_load.lesioned_forward` などは学習で得た閾値を含んだまま使える。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .expert_prefetch import RoutedCharLM


class HomeostaticRouter(nn.Module):
    """線形ルータ＋専門家ごとの閾値（勾配を流さないバッファ）。"""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.linear = linear
        self.register_buffer("threshold", torch.zeros(linear.out_features))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h) - self.threshold


class HomeostaticRoutedCharLM(RoutedCharLM):
    """`RoutedCharLM` のルータに恒常性の閾値を足したもの。初期化の乱数の消費は元と同じ。"""

    def __init__(self, vocab_size: int, d: int = 48, n_experts: int = 8, d_ff: int = 64,
                 eta: float = 0.01, alpha: float = 0.05) -> None:
        super().__init__(vocab_size, d=d, n_experts=n_experts, d_ff=d_ff)
        self.router = HomeostaticRouter(self.router)
        self.eta = eta
        self.alpha = alpha
        # 活動センサーの初期値は目標値（1/E）。学習前に偏りを仮定しない
        self.register_buffer("usage_ema", torch.full((n_experts,), 1.0 / n_experts))

    @torch.no_grad()
    def homeostatic_step(self, choice: torch.Tensor) -> None:
        """1学習ステップ分の選択 [B,T] から活動センサーと閾値を更新する（勾配によらない）。"""
        frac = torch.bincount(choice.reshape(-1), minlength=self.n_experts).float() / choice.numel()
        self.usage_ema.mul_(1 - self.alpha).add_(self.alpha * frac)
        th = self.router.threshold
        th.add_(self.eta * (self.n_experts * self.usage_ema - 1))
        th.sub_(th.mean())

    @torch.no_grad()
    def without_threshold(self) -> "_ThresholdOff":
        """閾値を一時的に 0 にする文脈（参考指標: 恒常性が重みに内面化されたか）。"""
        return _ThresholdOff(self)


class _ThresholdOff:
    def __init__(self, model: HomeostaticRoutedCharLM) -> None:
        self.model = model

    def __enter__(self) -> HomeostaticRoutedCharLM:
        self.saved = self.model.router.threshold.clone()
        self.model.router.threshold.zero_()
        return self.model

    def __exit__(self, *exc) -> None:
        self.model.router.threshold.copy_(self.saved)


def train_homeostatic_lm(model: HomeostaticRoutedCharLM, train_data: torch.Tensor, steps: int,
                         batch_size: int, seq_len: int, lr: float, generator: torch.Generator,
                         balance_coef: float = 0.0) -> float:
    """`expert_prefetch.train_routed_lm` と同じ手順（同じ乱数の消費）に、各ステップ後の閾値の更新を足す。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = float("nan")
    for _ in range(steps):
        idx = torch.randint(0, len(train_data) - seq_len - 1, (batch_size,), generator=generator)
        x = torch.stack([train_data[i:i + seq_len] for i in idx.tolist()])
        y = torch.stack([train_data[i + 1:i + seq_len + 1] for i in idx.tolist()])
        logits, balance, choice, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        ce = float(loss.detach())
        opt.zero_grad(set_to_none=True)
        (loss + balance_coef * balance).backward()
        opt.step()
        model.homeostatic_step(choice)
    return ce
