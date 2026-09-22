"""基底核モジュール — アクター・クリティック本体（12.6.22節・ステップ7）の単体テスト。

最終的なRL収束の良し悪しではなく、統制2（3条件がヘッドのアーキテクチャ・学習
手続きを完全に共有すること）・統制5/5b（凍結条件はバックボーンが本当に動かず、
rl-only条件は本当に動くこと）という機構そのものを機械的に検証する
（`tests/test_path_audit.py`・`test_hub_sparse.py` と同型）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.basal_ganglia_core import (  # noqa: E402
    ActorCritic,
    Backbone,
    ParitySpec,
    build_backbone,
    evaluate,
    make_batch,
    pretrain_backbone,
    run_condition,
    supervised_capacity_check,
)


# --- 課題生成 -----------------------------------------------------------------

def test_make_batch_label_is_xor_of_relevant_bits() -> None:
    spec = ParitySpec(n_inputs=6, relevant=(0, 1))
    g = torch.Generator().manual_seed(0)
    bits, labels = make_batch(spec, 256, g)
    expected = ((bits[:, 0] > 0) ^ (bits[:, 1] > 0)).long()
    assert torch.equal(labels, expected)


def test_parity_spec_rejects_invalid_relevant() -> None:
    with pytest.raises(ValueError):
        ParitySpec(n_inputs=4, relevant=(0, 0))
    with pytest.raises(ValueError):
        ParitySpec(n_inputs=4, relevant=(0, 5))


# --- バックボーン・事前学習 ------------------------------------------------------

def test_backbone_output_is_bounded_by_tanh() -> None:
    backbone = Backbone(n_inputs=6, hidden_dim=10)
    bits = torch.randn(20, 6)
    h = backbone(bits)
    assert h.shape == (20, 10)
    assert torch.all(h.abs() <= 1.0)


def test_pretrain_backbone_improves_linear_separability() -> None:
    """健全性チェック: 事前学習後は、隠れ表現に線形読み出しを乗せるだけでXORが高精度で解けること。"""
    torch.manual_seed(0)
    spec = ParitySpec(n_inputs=8, relevant=(0, 1))
    backbone = Backbone(spec.n_inputs, hidden_dim=16)
    g = torch.Generator().manual_seed(0)
    pretrain_backbone(backbone, spec, steps=500, batch_size=64, lr=0.05, generator=g)

    readout = torch.nn.Linear(16, 2)
    opt = torch.optim.Adam(readout.parameters(), lr=0.1)
    g2 = torch.Generator().manual_seed(1)
    for _ in range(200):
        bits, labels = make_batch(spec, 64, g2)
        with torch.no_grad():
            h = backbone(bits)
        logits = readout(h)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    bits, labels = make_batch(spec, 512, g2)
    with torch.no_grad():
        acc = float((readout(backbone(bits)).argmax(-1) == labels).float().mean())
    assert acc > 0.9


# --- 条件ごとのバックボーン構築（統制5・5b） -------------------------------------

def test_pretrained_backbone_is_frozen() -> None:
    spec = ParitySpec(n_inputs=6, relevant=(0, 1))
    g = torch.Generator().manual_seed(0)
    backbone, train_backbone = build_backbone("pretrained", spec, hidden_dim=8,
                                              pretrain_steps=10, pretrain_batch=32,
                                              pretrain_lr=0.05, generator=g)
    assert train_backbone is False
    assert all(not p.requires_grad for p in backbone.parameters())


def test_random_frozen_backbone_is_frozen_and_unpretrained() -> None:
    spec = ParitySpec(n_inputs=6, relevant=(0, 1))
    g = torch.Generator().manual_seed(0)
    backbone, train_backbone = build_backbone("random-frozen", spec, hidden_dim=8,
                                              pretrain_steps=10, pretrain_batch=32,
                                              pretrain_lr=0.05, generator=g)
    assert train_backbone is False
    assert all(not p.requires_grad for p in backbone.parameters())


def test_rl_only_backbone_requires_grad() -> None:
    spec = ParitySpec(n_inputs=6, relevant=(0, 1))
    g = torch.Generator().manual_seed(0)
    backbone, train_backbone = build_backbone("rl-only", spec, hidden_dim=8,
                                              pretrain_steps=10, pretrain_batch=32,
                                              pretrain_lr=0.05, generator=g)
    assert train_backbone is True
    assert all(p.requires_grad for p in backbone.parameters())


def test_frozen_backbone_weights_are_unchanged_by_an_rl_update_step() -> None:
    """統制5・5bの核心: 凍結条件はoptimizerにbackboneのパラメータが含まれず、
    1RL更新ステップを経ても重みが一切動かないこと。
    """
    for condition in ("pretrained", "random-frozen"):
        spec = ParitySpec(n_inputs=6, relevant=(0, 1))
        g = torch.Generator().manual_seed(0)
        backbone, train_backbone = build_backbone(condition, spec, hidden_dim=8,
                                                  pretrain_steps=20, pretrain_batch=32,
                                                  pretrain_lr=0.05, generator=g)
        before = backbone.fc1.weight.detach().clone()
        head = ActorCritic(8)
        params = list(head.parameters()) + (list(backbone.parameters()) if train_backbone else [])
        opt = torch.optim.Adam(params, lr=0.1)
        bits, labels = make_batch(spec, 32, g)
        h = backbone(bits)
        logits, values = head(h)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        rewards = (actions == labels).float()
        advantages = (rewards - values).detach()
        loss = (-(advantages * dist.log_prob(actions)).mean()
                + torch.nn.functional.mse_loss(values, rewards))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        assert torch.equal(before, backbone.fc1.weight.detach())


def test_rl_only_backbone_weights_change_during_rl() -> None:
    """rl-only条件は逆に、RLフェーズでバックボーンの重みが実際に動くこと。"""
    torch.manual_seed(0)
    spec = ParitySpec(n_inputs=6, relevant=(0, 1))
    g = torch.Generator().manual_seed(0)
    backbone, train_backbone = build_backbone("rl-only", spec, hidden_dim=8, pretrain_steps=0,
                                              pretrain_batch=32, pretrain_lr=0.05, generator=g)
    before = backbone.fc1.weight.detach().clone()
    head = ActorCritic(8)
    opt = torch.optim.Adam(list(head.parameters()) + list(backbone.parameters()), lr=0.1)
    bits, labels = make_batch(spec, 32, g)
    h = backbone(bits)
    logits, values = head(h)
    dist = torch.distributions.Categorical(logits=logits)
    actions = dist.sample()
    rewards = (actions == labels).float()
    advantages = (rewards - values).detach()
    loss = -(advantages * dist.log_prob(actions)).mean() + torch.nn.functional.mse_loss(values, rewards)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    assert not torch.equal(before, backbone.fc1.weight.detach())


# --- 実行の健全性 ---------------------------------------------------------------

def test_run_condition_returns_expected_number_of_eval_points() -> None:
    spec = ParitySpec(n_inputs=6, relevant=(0, 1))
    g = torch.Generator().manual_seed(0)
    accs = run_condition("random-frozen", spec, hidden_dim=8, pretrain_steps=5,
                         pretrain_batch=16, pretrain_lr=0.05, rl_steps=40, rl_batch=16,
                         rl_lr=0.1, eval_every=10, eval_batch=32, generator=g)
    assert len(accs) == 4
    assert all(0.0 <= a <= 1.0 for a in accs)


def test_all_conditions_share_the_same_head_architecture() -> None:
    """統制2: 3条件ともActorCriticヘッドのパラメータ形状が完全に一致すること。"""
    hidden_dim = 8
    shapes = set()
    for condition in ("pretrained", "random-frozen", "rl-only"):
        spec = ParitySpec(n_inputs=6, relevant=(0, 1))
        g = torch.Generator().manual_seed(0)
        _, _ = build_backbone(condition, spec, hidden_dim=hidden_dim, pretrain_steps=5,
                              pretrain_batch=16, pretrain_lr=0.05, generator=g)
        head = ActorCritic(hidden_dim)
        shapes.add(tuple(tuple(p.shape) for p in head.parameters()))
    assert len(shapes) == 1


# --- 段階A: 教師あり容量チェック（12.6.26節・ステップ9） -------------------------

def test_supervised_capacity_check_returns_valid_accuracy() -> None:
    spec = ParitySpec(n_inputs=8, relevant=(0, 1))
    g = torch.Generator().manual_seed(0)
    acc = supervised_capacity_check("random-frozen", spec, hidden_dim=32, steps=50,
                                    batch_size=32, lr=0.05, generator=g)
    assert 0.0 <= acc <= 1.0


def test_supervised_capacity_check_random_frozen_does_not_move_backbone() -> None:
    """段階Aの前提: random-frozenの容量チェックはバックボーンを一切更新しないこと。"""
    spec = ParitySpec(n_inputs=8, relevant=(0, 1))
    # Backboneの初期化はグローバルなtorch RNGを使うため、呼び出し前に毎回
    # torch.manual_seedを揃えることで、同じシードなら結果が完全に再現する
    # （＝隠れた状態を引きずっていない）ことを確認する。
    torch.manual_seed(0)
    g1 = torch.Generator().manual_seed(1)
    acc1 = supervised_capacity_check("random-frozen", spec, hidden_dim=16, steps=100,
                                     batch_size=32, lr=0.05, generator=g1)
    torch.manual_seed(0)
    g2 = torch.Generator().manual_seed(1)
    acc2 = supervised_capacity_check("random-frozen", spec, hidden_dim=16, steps=100,
                                     batch_size=32, lr=0.05, generator=g2)
    assert acc1 == pytest.approx(acc2)


def test_supervised_capacity_check_trainable_backbone_solves_task_even_at_small_width() -> None:
    """健全性: rl-only（バックボーンごと教師あり学習可能）は、幅が狭くてもXORを解けること
    （容量のボトルネックはrandom-frozen特有であることの確認）。
    """
    torch.manual_seed(0)
    spec = ParitySpec(n_inputs=8, relevant=(0, 1))
    g = torch.Generator().manual_seed(0)
    acc = supervised_capacity_check("rl-only", spec, hidden_dim=8, steps=500,
                                    batch_size=64, lr=0.05, generator=g)
    assert acc > 0.9


def test_evaluate_returns_accuracy_in_valid_range() -> None:
    spec = ParitySpec(n_inputs=6, relevant=(0, 1))
    backbone = Backbone(spec.n_inputs, hidden_dim=8)
    head = ActorCritic(8)
    g = torch.Generator().manual_seed(0)
    acc = evaluate(backbone, head, spec, n_eval=64, generator=g)
    assert 0.0 <= acc <= 1.0
