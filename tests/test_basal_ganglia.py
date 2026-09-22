"""基底核モジュール（12.6.10節・ステップ2）の単体テスト。

出力（最終的に良い方策に収束すること）ではなく、**設計上の主張の機構そのもの**を
検証する。12.6.10節の主張は「二重経路の更新則が優位度の符号で経路を分離すること」
であり、これは実験結果を待たずにコード上で保証できる性質である
（`tests/test_path_audit.py` が海馬モジュールの統制5を機械的に保証するのと同型）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.bandit import (  # noqa: E402
    BanditSpec,
    dummy_context,
    sample_phase1,
    sample_phase2_actions,
    sample_phase2_rewards,
)
from neurocortex.basal_ganglia import (  # noqa: E402
    Critic,
    DualPathwayPolicy,
    SinglePathwayPolicy,
)


# --- 方策ヘッドの機構 ---------------------------------------------------------

def test_dual_pathway_positive_advantage_only_touches_go() -> None:
    """優位度がすべて正の試行では、NoGoのパラメータが1ビットも変わらないこと。

    これが「正の学習信号はGo側だけに流れる」という設計上の主張そのものであり、
    最終方策の良し悪しとは独立に、コード上で保証されるべき性質である。
    """
    torch.manual_seed(0)
    policy = DualPathwayPolicy(d_model=1, n_actions=4)
    nogo_before = [p.detach().clone() for p in policy.nogo.parameters()]
    x = dummy_context(8)
    actions = torch.randint(0, 4, (8,))
    advantages = torch.rand(8) + 0.1  # 全て正
    policy.update(x, actions, advantages, lr_go=0.1, lr_nogo=0.1)
    for p, q in zip(policy.nogo.parameters(), nogo_before):
        assert torch.equal(p, q), "正の優位度なのにNoGoが更新されている"
    go_after = list(policy.go.parameters())
    assert any(p.abs().sum() > 0 for p in go_after), "Goが更新されていない"


def test_dual_pathway_negative_advantage_only_touches_nogo() -> None:
    """優位度がすべて負の試行では、Goのパラメータが1ビットも変わらないこと。"""
    torch.manual_seed(0)
    policy = DualPathwayPolicy(d_model=1, n_actions=4)
    go_before = [p.detach().clone() for p in policy.go.parameters()]
    x = dummy_context(8)
    actions = torch.randint(0, 4, (8,))
    advantages = -(torch.rand(8) + 0.1)  # 全て負
    policy.update(x, actions, advantages, lr_go=0.1, lr_nogo=0.1)
    for p, q in zip(policy.go.parameters(), go_before):
        assert torch.equal(p, q), "負の優位度なのにGoが更新されている"


def test_dual_pathway_mixed_batch_routes_each_trial_independently() -> None:
    """正負が混在するバッチでは、両方のパラメータが動くこと（片方に丸められない）。"""
    torch.manual_seed(0)
    policy = DualPathwayPolicy(d_model=1, n_actions=4)
    go_before = policy.go.weight.detach().clone()
    nogo_before = policy.nogo.weight.detach().clone()
    x = dummy_context(16)
    actions = torch.randint(0, 4, (16,))
    advantages = torch.cat([torch.rand(8) + 0.1, -(torch.rand(8) + 0.1)])
    policy.update(x, actions, advantages, lr_go=0.1, lr_nogo=0.1)
    assert not torch.equal(policy.go.weight, go_before)
    assert not torch.equal(policy.nogo.weight, nogo_before)


def test_single_pathway_has_no_selective_routing() -> None:
    """対照群: 正のみ・負のみのどちらのバッチでも、a・bの両方が常に同時に動くこと。

    二重経路と違い、経路の選択的な分離が一切ないことを保証する（統制2の前提）。
    """
    torch.manual_seed(0)
    for advantages in (torch.rand(8) + 0.1, -(torch.rand(8) + 0.1)):
        policy = SinglePathwayPolicy(d_model=1, n_actions=4)
        a_before = policy.a.weight.detach().clone()
        b_before = policy.b.weight.detach().clone()
        x = dummy_context(8)
        actions = torch.randint(0, 4, (8,))
        policy.update(x, actions, advantages, lr=0.1)
        assert not torch.equal(policy.a.weight, a_before)
        assert not torch.equal(policy.b.weight, b_before)
        # a と b は常に同一量だけ動く（分離なし）。
        assert torch.allclose(policy.a.weight - a_before, policy.b.weight - b_before)


def test_single_and_dual_have_matched_parameter_count() -> None:
    """統制2: パラメータ予算を揃える。single は a・b の2本、dual は go・nogo の2本。"""
    single = SinglePathwayPolicy(d_model=3, n_actions=5)
    dual = DualPathwayPolicy(d_model=3, n_actions=5)
    n_single = sum(p.numel() for p in single.parameters())
    n_dual = sum(p.numel() for p in dual.parameters())
    assert n_single == n_dual


def test_asymmetric_single_pathway_scales_by_advantage_sign() -> None:
    """統制5: asymmetric_lr は優位度の符号でどちらの学習率を使うかだけを切り替える

    （経路そのものは1系統のまま分離しない）。全試行が正の優位度のとき lr_neg は
    一切参照されず、全試行が負の優位度のとき lr は一切参照されないことを確認する。
    """
    torch.manual_seed(0)
    x = dummy_context(4)
    actions = torch.tensor([0, 1, 2, 3])

    def weight_after(advantages: torch.Tensor, lr: float, lr_neg: float) -> torch.Tensor:
        torch.manual_seed(1)  # 3ケースとも同じ初期化から始める
        policy = SinglePathwayPolicy(d_model=1, n_actions=4, asymmetric_lr=True)
        policy.update(x, actions, advantages, lr=lr, lr_neg=lr_neg)
        return policy.a.weight.detach().clone()

    before = weight_after(torch.zeros(4), lr=0.0, lr_neg=0.0)

    pos_adv = torch.ones(4)
    # lr_neg を変えても、全試行が正なら結果は変わらないはず（参照されない）。
    w_pos_a = weight_after(pos_adv, lr=0.2, lr_neg=0.0)
    w_pos_b = weight_after(pos_adv, lr=0.2, lr_neg=99.0)
    assert torch.allclose(w_pos_a, w_pos_b), "正の優位度なのに lr_neg が結果に影響している"
    assert not torch.allclose(w_pos_a, before), "lr=0.2 なのに更新が起きていない"

    neg_adv = -torch.ones(4)
    # lr を変えても、全試行が負なら結果は変わらないはず（参照されない）。
    w_neg_a = weight_after(neg_adv, lr=0.0, lr_neg=0.2)
    w_neg_b = weight_after(neg_adv, lr=99.0, lr_neg=0.2)
    assert torch.allclose(w_neg_a, w_neg_b), "負の優位度なのに lr が結果に影響している"
    assert not torch.allclose(w_neg_a, before), "lr_neg=0.2 なのに更新が起きていない"


def test_asymmetric_lr_requires_lr_neg() -> None:
    policy = SinglePathwayPolicy(d_model=1, n_actions=2, asymmetric_lr=True)
    with pytest.raises(ValueError):
        policy.update(dummy_context(2), torch.tensor([0, 1]), torch.ones(2), lr=0.1)


# --- 学習の健全性（機構が方向として正しいこと） -------------------------------

def test_policy_gradient_increases_prob_of_rewarded_action() -> None:
    """正の優位度を与え続けると、その行動の選択確率が上がること（符号の健全性）。"""
    torch.manual_seed(0)
    for policy, kwargs in (
        (SinglePathwayPolicy(1, 4), dict(lr=0.2)),
        (DualPathwayPolicy(1, 4), dict(lr_go=0.2, lr_nogo=0.2)),
    ):
        x = dummy_context(1)
        p0 = torch.softmax(policy(x), dim=-1)[0, 0].item()
        for _ in range(50):
            policy.update(dummy_context(8), torch.zeros(8, dtype=torch.long),
                          torch.ones(8), **kwargs)
        p1 = torch.softmax(policy(x), dim=-1)[0, 0].item()
        assert p1 > p0, f"{type(policy).__name__}: 正の優位度で選択確率が上がらない"


def test_critic_reduces_prediction_error() -> None:
    torch.manual_seed(0)
    critic = Critic(d_model=1)
    x = dummy_context(16)
    returns = torch.ones(16)
    loss0 = critic.update(x, returns, lr=0.1)
    for _ in range(30):
        loss = critic.update(x, returns, lr=0.1)
    assert loss < loss0


# --- バンディット課題 --------------------------------------------------------

def test_phase1_rewards_only_from_good_arms() -> None:
    spec = BanditSpec(n_good=3, n_bad=5, good_reward_prob=1.0, champion_reward_prob=1.0)
    g = torch.Generator().manual_seed(0)
    actions = torch.cat([torch.arange(3), torch.arange(3, 8)])  # 全腕を一度ずつ
    rewards = sample_phase1(spec, actions, g)
    assert torch.all(rewards[:3] == 1.0), "良腕なのに報酬が出ていない（good_reward_prob=1.0）"
    assert torch.all(rewards[3:] == 0.0), "無関係腕からフェーズ1の報酬が出ている"


def test_phase2_actions_never_include_good_arms() -> None:
    """フェーズ2は無関係腕からのみ強制サンプリングする（良腕には一切触れない）。"""
    spec = BanditSpec(n_good=3, n_bad=5)
    g = torch.Generator().manual_seed(0)
    actions = sample_phase2_actions(spec, 1000, g)
    assert torch.all(actions >= spec.n_good)
    assert torch.all(actions < spec.n_arms)


def test_phase2_rewards_are_nonpositive() -> None:
    spec = BanditSpec(n_good=3, n_bad=5, bad_penalty_prob=1.0)
    g = torch.Generator().manual_seed(0)
    actions = sample_phase2_actions(spec, 100, g)
    rewards = sample_phase2_rewards(spec, actions, g)
    assert torch.all(rewards == -1.0), "bad_penalty_prob=1.0 なのに罰が出ていない試行がある"


def test_reward_probs_gives_arm0_the_champion_rate() -> None:
    """腕0だけ`champion_reward_prob`、他の良腕は`good_reward_prob`、無関係腕は0。"""
    spec = BanditSpec(n_good=4, n_bad=4, good_reward_prob=0.1, champion_reward_prob=0.3)
    p = spec.reward_probs
    assert p[0].item() == pytest.approx(0.3)
    assert torch.allclose(p[1:4], torch.full((3,), 0.1))
    assert torch.all(p[4:] == 0.0)
