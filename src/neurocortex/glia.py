"""グリア層 — BCM型ホメオスタシスによるロードバランシング（12.6.17節・ステップ5）。

12.3.5節の当初案（BCM則の滑動閾値をMoEの補助損失の代替にする）は、素朴には
「均等化できたかどうか」だけを見ると、BCMの特定の数式形（`θ_M ∝ ⟨y²⟩` という
滑動閾値の非線形性）に固有の効果なのか、単に「局所的な発火統計だけを見て
フィードバック制御する」ことでありさえすれば十分なのかを区別できない
（12.6.6節のSTDP対Hebb学習、12.6.11節のGo/NoGo対非対称学習率と同型の罠）。

したがって本モジュールは3種類の補正機構を並べる。

  aux-loss     … 標準的な補助損失＋勾配降下（グローバル・学習時のみ）
  proportional … 局所・勾配不要だが、BCMの滑動閾値を持たない素朴な比例制御
                 （対照群、統制5）
  bcm          … 8.2節のBCM則をゲイン補正 `correction_i` の更新則に読み替えたもの
                 （本設計）

`correction_i` は「その専門家が選ばれやすいか」を表すスカラーで、専門家の
ロジットに直接加算される。BCM・proportionalはいずれも**専門家iの`correction_i`
の更新が専門家iの利用率のみに依存し、他の専門家の情報を使わない**という
局所性を共有する（統制2。`tests/test_glia.py` で機械的に監査する）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LoadBalanceSpec:
    n_experts: int = 6
    batch_size: int = 64
    noise_scale: float = 1.0

    @property
    def target_rate(self) -> float:
        return 1.0 / self.n_experts


def make_bias(n_experts: int, seed: int, scale: float = 2.0) -> torch.Tensor:
    """専門家ごとの「地の偏り」。これが是正されなければ利用率は不均一になる。"""
    g = torch.Generator().manual_seed(seed)
    return scale * torch.randn(n_experts, generator=g)


def flip_bias(bias: torch.Tensor) -> torch.Tensor:
    """統制3: 偏りの向きを反転させる（どの専門家が過負荷かが入れ替わる）。"""
    return -bias


def route_batch(spec: LoadBalanceSpec, active_bias: torch.Tensor, correction: torch.Tensor,
                generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """1ステップぶんのルーティング。(usage [n_experts], soft_probs [B, n_experts]) を返す。

    `usage` はこのバッチでのハードルーティング（argmax）による実際の選択頻度
    （勾配は通さない）、`soft_probs` はaux-loss条件の微分可能な損失計算に使う
    softmax確率（`correction.requires_grad=True` のときだけ勾配が流れる）。
    意図的に `torch.no_grad()` で囲っていない点に注意 —
    `AuxLossCorrection.step()` がこの関数の出力を通じて `correction` に
    逆伝播できる必要があるため。
    """
    noise = spec.noise_scale * torch.randn(spec.batch_size, spec.n_experts, generator=generator)
    logits = active_bias[None, :] + correction[None, :] + noise
    with torch.no_grad():
        selected = logits.argmax(dim=-1)
        counts = torch.bincount(selected, minlength=spec.n_experts).float()
        usage = counts / spec.batch_size
    soft_probs = torch.softmax(logits, dim=-1)
    return usage, soft_probs


def imbalance(usage: torch.Tensor, target_rate: float) -> float:
    """均等からのズレ（二乗誤差の合計）。0が完全均等。"""
    return float(((usage - target_rate) ** 2).sum())


# --- 補正則 ---------------------------------------------------------------------

@torch.no_grad()
def bcm_update(correction: torch.Tensor, usage: torch.Tensor, msq_state: torch.Tensor,
               alpha: float, beta: float, lr: float) -> None:
    """BCM則（8.2節）をゲイン補正の更新に読み替える。

    `msq_state` は `⟨y²⟩` の指数移動平均（滑動閾値の元になる、専門家ごとの状態）。
    過活動（`usage > θ`）の専門家は `correction` を下げる方向（抑制＝LTD相当）、
    低活動（`usage < θ`）の専門家は上げる方向（促進＝LTP相当）に働く。
    """
    msq_state.mul_(1 - beta).add_(beta * usage**2)
    theta = alpha * msq_state
    phi = usage * (usage - theta)
    correction -= lr * phi


@torch.no_grad()
def proportional_update(correction: torch.Tensor, usage: torch.Tensor,
                        target_rate: float, lr: float) -> None:
    """対照群（統制5）: 目標利用率との差にだけ比例する、素朴なホメオスタシス則。

    BCMと同じく局所的・勾配不要だが、滑動閾値の非線形性を持たない。
    """
    correction += lr * (target_rate - usage)


class AuxLossCorrection:
    """標準的な補助損失によるロードバランシング（対照群、既存手法）。

    `correction` は `nn.Parameter` 相当として勾配で更新する。`freeze()` 後は
    `step()` が何もしなくなり、**推論時のみのフェーズでは適応できない**ことを
    コード上で保証する（主張(a)の直接的な操作化）。
    """

    def __init__(self, n_experts: int, lr: float) -> None:
        self.correction = torch.zeros(n_experts, requires_grad=True)
        self.opt = torch.optim.SGD([self.correction], lr=lr)
        self._frozen = False

    def freeze(self) -> None:
        self._frozen = True

    def step(self, soft_probs: torch.Tensor, target_rate: float) -> float:
        """soft_probsに対する分散正則化損失で1ステップ更新する。凍結中は何もしない。"""
        if self._frozen:
            return 0.0
        mean_p = soft_probs.mean(dim=0)
        loss = ((mean_p - target_rate) ** 2).sum()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return float(loss.detach())
