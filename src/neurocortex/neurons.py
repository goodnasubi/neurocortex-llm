"""スパイキングニューロン層（12.3.1節）と、その対照群である階段関数（12.9節）。

本プロジェクトの中核。2つの層はどちらも「非線形活性化」として同じ位置に差し込めるが、
位置（トークン）方向に状態を持つか否かだけが異なる。

- :class:`ALIFNeuron`   … 適応閾値LIF。膜電位・適応閾値を**トークン位置方向に持ち越す**。
                           位置間を結ぶ唯一の経路になる。出力は二値スパイク {0, 1}。
- :class:`StepActivation` … 12.9節で棄却された「整数スパイク数」方式。各位置の内部で
                           マイクロ時刻を n_steps 回まわし、発火率を返す。位置方向の
                           状態を一切持たないため、12.9節の検証通りスカラー→スカラーの
                           決定的な階段関数に厳密に退化する。識別テストの対照群。

どちらも位置方向の逐次ループを持つ（ALIF）か、マイクロ時刻の逐次ループを持つ（Step）。
12.3.1節の方針に従い、cumsum等による並列化やリセット項の除去は**行わない**。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .surrogate import spike


@dataclass
class NeuronConfig:
    """ニューロンのハイパーパラメータ。

    Attributes:
        beta: 膜電位の漏れ係数。保持時間の目安は 1/(1-beta)（12.6.1節の副次条件）。
        theta: 基準閾値。
        kappa: 発火による閾値上昇量（適応閾値）。0で適応なし。
        lam: 適応閾値の減衰係数。
        alpha: arctanサロゲート勾配の鋭さ。
        n_steps: StepActivation のマイクロ時刻数（ALIFでは未使用）。
    """

    beta: float = 0.9
    theta: float = 1.0
    kappa: float = 0.5
    lam: float = 0.9
    alpha: float = 2.0
    n_steps: int = 4


class ALIFNeuron(nn.Module):
    """適応閾値LIFニューロン。膜電位をトークン位置方向に持ち越す（12.3.1節）。

    漸化式（位置 t は系列中のトークン位置そのもの。マイクロ時刻ではない）::

        v_t  = beta * v_{t-1} + I_t
        s_t  = H(v_t - (theta + a_{t-1}))          # 二値スパイク
        v_t <- v_t - s_t * (theta + a_{t-1})       # ソフトリセット（減算型）
        a_t  = lam * a_{t-1} + kappa * s_t         # 適応閾値

    Args:
        cfg: ニューロン設定。
        carry_membrane: False にすると各位置で v, a を 0 に初期化し、位置方向の
            持ち越しだけを無効化する（12.6.1節 統制2「膜電位の因果的除去」）。
            重みは一切変えないので、同一モデル内の因果操作として使える。
    """

    def __init__(self, cfg: NeuronConfig, carry_membrane: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.carry_membrane = carry_membrane
        # 直近のフォワードでの発火率（副指標の監視用。勾配は持たない）
        self.last_firing_rate: float | None = None

    def extra_repr(self) -> str:
        c = self.cfg
        return (
            f"beta={c.beta}, theta={c.theta}, kappa={c.kappa}, "
            f"lam={c.lam}, alpha={c.alpha}, carry_membrane={self.carry_membrane}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x [B, T, H] の入力電流。Returns: [B, T, H] の二値スパイク。"""
        if x.dim() != 3:
            raise ValueError(f"ALIFNeuron は [B, T, H] を要求する（受領: {tuple(x.shape)}）")
        cfg = self.cfg
        b, t, h = x.shape
        v = x.new_zeros(b, h)
        a = x.new_zeros(b, h)
        outs = []
        # 位置方向の逐次ループ。これが設計上の必然（12.3.1節「実装上の代償」）。
        for i in range(t):
            if not self.carry_membrane:
                # 統制2: 持ち越しのみ遮断。前位置の v, a を捨てる。
                v = x.new_zeros(b, h)
                a = x.new_zeros(b, h)
            v = cfg.beta * v + x[:, i, :]
            thr = cfg.theta + a
            s = spike(v - thr, cfg.alpha)
            v = v - s * thr
            a = cfg.lam * a + cfg.kappa * s
            outs.append(s)
        out = torch.stack(outs, dim=1)
        with torch.no_grad():
            self.last_firing_rate = float(out.mean())
        return out


class StepActivation(nn.Module):
    """12.9節で棄却された「整数スパイク数」方式＝階段関数（識別テストの対照群）。

    各トークン位置の内部で、同一の定常入力 I を n_steps 回入れてマイクロ時刻を回し、
    発火回数の平均（発火率）を返す。位置方向の状態は一切持たない。12.9節の検証により
    この写像は入力ごとに独立な決定的階段関数に厳密に等価であることが示されている。

    ALIFNeuron と同じ漸化式・同じサロゲート勾配を使うので、両者の差は
    「状態が位置方向に持ち越されるか否か」だけに限定される。
    """

    def __init__(self, cfg: NeuronConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.last_firing_rate: float | None = None

    def extra_repr(self) -> str:
        c = self.cfg
        return f"beta={c.beta}, theta={c.theta}, kappa={c.kappa}, lam={c.lam}, n_steps={c.n_steps}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x [..., H]。Returns: 同形の発火率（{0, 1/n, ..., 1} のいずれか）。"""
        cfg = self.cfg
        v = torch.zeros_like(x)
        a = torch.zeros_like(x)
        acc = torch.zeros_like(x)
        # マイクロ時刻ループ。x は全ステップで同一（定常入力）。
        for _ in range(cfg.n_steps):
            v = cfg.beta * v + x
            thr = cfg.theta + a
            s = spike(v - thr, cfg.alpha)
            v = v - s * thr
            a = cfg.lam * a + cfg.kappa * s
            acc = acc + s
        out = acc / cfg.n_steps
        with torch.no_grad():
            self.last_firing_rate = float(out.mean())
        return out


def make_activation(kind: str, cfg: NeuronConfig, carry_membrane: bool = True) -> nn.Module:
    """kind in {"spiking", "step"} で活性化層を作る。"""
    if kind == "spiking":
        return ALIFNeuron(cfg, carry_membrane=carry_membrane)
    if kind == "step":
        return StepActivation(cfg)
    raise ValueError(f"未知の活性化種別: {kind}")
