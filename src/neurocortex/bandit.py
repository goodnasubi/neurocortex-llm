"""基底核モジュール（12.6.10節・ステップ2）の2フェーズ・バンディット課題。

腕は「良腕」（成功時に正報酬を疎に返す。うち腕0だけ報酬確率が高い「チャンピオン」）
と「無関係腕」に分かれる。

  フェーズ1: 良腕にのみ疎な正報酬を与え、腕0（チャンピオン）が良腕どうしの中で
             明確に選好されるまで学習する
  フェーズ2: 無関係腕にのみ負報酬を**集中投入**する（腕は方策からではなく無関係腕
             の中から強制的にサンプリングする）。良腕には一切触れない

測定対象は「良腕全体の選択確率」ではなく**良腕どうしの相対的な選好**
（`champion_share = pi[0] / sum(pi[良腕])`）である。前者は無関係腕の確率が
下がるだけで softmax の正規化を通じて機械的に上昇してしまい、経路の設計差とは
無関係な天井効果の主因になることが分かっている（12.6.10節実施記録参照）。
後者は良腕どうしの比を見るため、この正規化アーチファクトの影響を受けない。

12.6.10節の設計どおり状態は持たない。方策への入力は定数ベクトル1本のみで、
実質的にロジットそのものがパラメータになる。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BanditSpec:
    """良腕のうち腕0だけ報酬確率を高くし「チャンピオン」にする（下記参照）。"""

    n_good: int = 4
    n_bad: int = 4
    good_reward_prob: float = 0.1
    champion_reward_prob: float = 0.3
    bad_penalty_prob: float = 1.0

    @property
    def n_arms(self) -> int:
        return self.n_good + self.n_bad

    @property
    def good_arms(self) -> range:
        return range(0, self.n_good)

    @property
    def bad_arms(self) -> range:
        return range(self.n_good, self.n_good + self.n_bad)

    @property
    def reward_probs(self) -> torch.Tensor:
        """腕ごとの報酬確率。無関係腕は0、良腕は`good_reward_prob`、腕0だけ

        `champion_reward_prob`。**総良腕確率（good_prob）は、無関係腕のロジットを
        下げるだけで softmax の正規化を通じて機械的に押し上がってしまい、経路の
        設計差とは無関係な"天井効果"の主因になる**（12.6.10節実施記録参照）。
        腕0を明確なチャンピオンにしておけば、良腕どうしの**相対的な選好**
        （腕0の取り分 `champion_share = pi[0] / sum(pi[good])`）を主指標にでき、
        これは無関係腕の確率が動いても正規化だけでは動かない（良腕間の相対比を見るため）。
        """
        p = torch.zeros(self.n_arms)
        p[: self.n_good] = self.good_reward_prob
        p[0] = self.champion_reward_prob
        return p


def dummy_context(batch_size: int, device: torch.device | None = None) -> torch.Tensor:
    """状態なしバンディットの入力。方策への入力は定数1本のみ。"""
    return torch.ones(batch_size, 1, device=device)


def sample_phase1(spec: BanditSpec, actions: torch.Tensor,
                   generator: torch.Generator) -> torch.Tensor:
    """フェーズ1の報酬。腕ごとの確率は `spec.reward_probs`（腕0だけ高い）。"""
    probs = spec.reward_probs[actions]
    hit = torch.rand(actions.shape, generator=generator) < probs
    return hit.to(torch.float32)


def sample_phase2_actions(spec: BanditSpec, batch_size: int,
                          generator: torch.Generator) -> torch.Tensor:
    """フェーズ2で強制的に引く腕。無関係腕のみから一様にサンプリングする。

    方策からサンプリングしないのは、「無関係腕への失敗信号を集中投入する」という
    12.6.10節の課題設計を、方策がその時点で無関係腕をどれだけ選びやすいかに
    依存させないためである。
    """
    return spec.n_good + torch.randint(0, spec.n_bad, (batch_size,), generator=generator)


def sample_phase2_rewards(spec: BanditSpec, actions: torch.Tensor,
                          generator: torch.Generator) -> torch.Tensor:
    """フェーズ2の報酬。無関係腕は確率 `bad_penalty_prob` で-1、それ以外は0。"""
    hit = torch.rand(actions.shape, generator=generator) < spec.bad_penalty_prob
    return torch.where(hit, -torch.ones_like(actions, dtype=torch.float32),
                        torch.zeros_like(actions, dtype=torch.float32))
