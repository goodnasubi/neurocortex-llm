"""マルチ学習則の学習安定性 — 目的関数が対立するシナリオでの再検証（12.6.38節・ステップ15）。

ステップ14（12.6.36〜37節）は、海馬（教師あり分類の勾配降下）と基底核
（REINFORCEの方策勾配）が**同じ目的関数**を共有する場合には干渉が観測されない
ことを確認したが、「目的関数自体が対立するシナリオ（海馬が過去タスクの表現を
保持しようとし、基底核が現在の報酬のために表現を書き換えようとする）は未検証」
と明記していた。

本モジュールは、タスクA（海馬が教師あり学習で習得し、リプレイバッファに保存する）
を学習した後、同じバックボーンでタスクB（基底核がREINFORCEで学習する）を学習する
逐次シナリオで、海馬のリプレイ勾配（タスクAを忘れないよう引き戻す力）と基底核の
RL勾配（タスクBに向けて押し出す力）という、文字通り対立する2つの力を同時に
バックボーンへ流した場合に、性能が劣化するかを見る。

  shared-conflict     … バックボーン共有、リプレイ勾配とRL勾配を同時に流してタスクBを学習する（本設計）
  independent         … 海馬用・基底核用に別々のバックボーンを用意する（統制1、タスクA保持率の参照点）
  rl-only-no-replay   … バックボーン共有だがリプレイ勾配なし、RL勾配のみでタスクBを学習する
                          （統制2、ステップ10のnaive-finetune相当。タスクB性能の参照点）
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basal_ganglia_core import ActorCritic, Backbone, ParitySpec, make_batch

CONDITIONS = ("shared-conflict", "independent", "rl-only-no-replay")


def train_task_a(backbone: Backbone, hippo_head: nn.Linear, spec_a: ParitySpec, steps: int,
                 batch_size: int, lr: float, generator: torch.Generator) -> None:
    """タスクAを教師あり学習で収束させる（`hippocampus_replay.train_task`と同型）。"""
    opt = torch.optim.Adam(list(backbone.parameters()) + list(hippo_head.parameters()), lr=lr)
    for _ in range(steps):
        bits, labels = make_batch(spec_a, batch_size, generator)
        loss = F.cross_entropy(hippo_head(backbone(bits)), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def make_replay_buffer(spec_a: ParitySpec, buffer_size: int,
                       generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """タスクA収束時点で、タスクAの生サンプルを固定サイズだけ保存する
    （`hippocampus_replay.make_replay_buffer`と同型）。
    """
    return make_batch(spec_a, buffer_size, generator)


def _replay_loss(backbone: Backbone, hippo_head: nn.Linear,
                 buffer: tuple[torch.Tensor, torch.Tensor], batch_size: int,
                 generator: torch.Generator) -> torch.Tensor:
    buf_bits, buf_labels = buffer
    idx = torch.randint(buf_bits.shape[0], (batch_size,), generator=generator)
    logits = hippo_head(backbone(buf_bits[idx]))
    return F.cross_entropy(logits, buf_labels[idx])


def _rl_loss(backbone: Backbone, rl_head: ActorCritic, spec_b: ParitySpec, batch_size: int,
            generator: torch.Generator) -> torch.Tensor:
    bits, labels = make_batch(spec_b, batch_size, generator)
    h = backbone(bits)
    logits, values = rl_head(h)
    dist = torch.distributions.Categorical(logits=logits)
    actions = dist.sample()
    rewards = (actions == labels).float()
    advantages = (rewards - values).detach()
    actor_loss = -(advantages * dist.log_prob(actions)).mean()
    critic_loss = F.mse_loss(values, rewards)
    return actor_loss + critic_loss


@torch.no_grad()
def evaluate_hippocampus(backbone: Backbone, hippo_head: nn.Linear, spec: ParitySpec, n_eval: int,
                         generator: torch.Generator) -> float:
    bits, labels = make_batch(spec, n_eval, generator)
    pred = hippo_head(backbone(bits)).argmax(dim=-1)
    return float((pred == labels).float().mean())


@torch.no_grad()
def evaluate_rl(backbone: Backbone, rl_head: ActorCritic, spec: ParitySpec, n_eval: int,
               generator: torch.Generator) -> float:
    bits, labels = make_batch(spec, n_eval, generator)
    logits, _ = rl_head(backbone(bits))
    pred = logits.argmax(dim=-1)
    return float((pred == labels).float().mean())


@dataclass
class RunResult:
    task_a_retention: float  # タスクA保持率（海馬側、タスクB学習後に測る）
    task_b_acc: float        # タスクB性能（基底核RL側）


def run_condition(condition: str, spec_a: ParitySpec, spec_b: ParitySpec, hidden_dim: int,
                  buffer_size: int, pretrain_steps: int, pretrain_batch: int, pretrain_lr: float,
                  b_steps: int, b_batch: int, b_lr: float, eval_n: int,
                  generator: torch.Generator) -> RunResult:
    """1条件・1シードぶんの実行。"""
    if condition not in CONDITIONS:
        raise ValueError(condition)

    n_inputs = spec_a.n_inputs

    if condition == "independent":
        # 海馬用バックボーンはタスクAだけを学習し、以降タスクBには一切触れない
        # （タスクA保持率の理想的な参照点）。
        backbone_h = Backbone(n_inputs, hidden_dim)
        hippo_head = nn.Linear(hidden_dim, 2)
        train_task_a(backbone_h, hippo_head, spec_a, pretrain_steps, pretrain_batch, pretrain_lr,
                    generator)

        backbone_rl = Backbone(n_inputs, hidden_dim)
        rl_head = ActorCritic(hidden_dim)
        opt_rl = torch.optim.Adam(list(backbone_rl.parameters()) + list(rl_head.parameters()),
                                  lr=b_lr)
        for _ in range(b_steps):
            loss = _rl_loss(backbone_rl, rl_head, spec_b, b_batch, generator)
            opt_rl.zero_grad(set_to_none=True)
            loss.backward()
            opt_rl.step()

        task_a_retention = evaluate_hippocampus(backbone_h, hippo_head, spec_a, eval_n, generator)
        task_b_acc = evaluate_rl(backbone_rl, rl_head, spec_b, eval_n, generator)
        return RunResult(task_a_retention=task_a_retention, task_b_acc=task_b_acc)

    # shared-conflict / rl-only-no-replay: 共通のバックボーンでタスクAを事前学習してから
    # タスクBの学習に入る。
    backbone = Backbone(n_inputs, hidden_dim)
    hippo_head = nn.Linear(hidden_dim, 2)
    train_task_a(backbone, hippo_head, spec_a, pretrain_steps, pretrain_batch, pretrain_lr,
                generator)
    buffer = make_replay_buffer(spec_a, buffer_size, generator)

    rl_head = ActorCritic(hidden_dim)
    if condition == "shared-conflict":
        params = list(backbone.parameters()) + list(hippo_head.parameters()) + list(rl_head.parameters())
    else:  # rl-only-no-replay
        params = list(backbone.parameters()) + list(rl_head.parameters())
    opt = torch.optim.Adam(params, lr=b_lr)

    for _ in range(b_steps):
        rl_loss = _rl_loss(backbone, rl_head, spec_b, b_batch, generator)
        if condition == "shared-conflict":
            replay_loss = _replay_loss(backbone, hippo_head, buffer, b_batch, generator)
            loss = replay_loss + rl_loss
        else:
            loss = rl_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    task_a_retention = evaluate_hippocampus(backbone, hippo_head, spec_a, eval_n, generator)
    task_b_acc = evaluate_rl(backbone, rl_head, spec_b, eval_n, generator)
    return RunResult(task_a_retention=task_a_retention, task_b_acc=task_b_acc)
