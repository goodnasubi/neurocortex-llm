"""基底核モジュール — アクター・クリティック本体（12.6.22節・ステップ7）。

12.3.3節の二重経路化（Go/NoGo分離）は既にステップ2（12.6.10〜11節）で検証済み
（否定的結果）。本モジュールは12.3.3節の残り、すなわち「アクター・クリティックが
皮質バックボーンの**共有・事前学習済み表現**の上に乗ること自体に意味があるか」
だけを対象にする（12.6.22節「対象の絞り込み」参照）。

素朴には「皮質の表現の上でRLをすれば学習が速い」という結果は、単なる一般的な
転移学習の効果（訓練済み特徴量の上のRLは効率的）を確認しただけになりかねない。
さらにステップ1（12.6.5節: 学習させた分離層も固定ランダム射影と同点）と同型の
罠——「訓練済みであること」ではなく「次元の合った固定表現でありさえすれば十分」
という可能性——もある。したがって3条件を並べる。

  pretrained     … 教師ありパリティ分類で事前学習し、RL開始時に凍結した
                   バックボーンの上でヘッドだけをRL学習する（本設計）
  random-frozen  … 初期化のまま凍結した（事前学習していない）同一アーキテクチャの
                   バックボーンの上でヘッドだけをRL学習する（対照群、統制5）
  rl-only        … 事前学習を経ず、RLの報酬信号だけでバックボーンごと
                   ゼロから学習する（対照群、統制5b）

3条件ともヘッド（アクター・クリティック）のアーキテクチャ・学習手続き
（REINFORCE+ベースライン、Adam、学習率、バッチサイズ）は完全に共有する（統制2）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

CONDITIONS = ("pretrained", "random-frozen", "rl-only")


@dataclass
class ParitySpec:
    """2入力XORパリティ課題の仕様。`relevant`以外の次元は無関係な囮。"""

    n_inputs: int = 8
    relevant: tuple[int, int] = (0, 1)

    def __post_init__(self) -> None:
        a, b = self.relevant
        if not (0 <= a < self.n_inputs and 0 <= b < self.n_inputs and a != b):
            raise ValueError("relevant は 0..n_inputs-1 の異なる2添字")


def make_batch(spec: ParitySpec, batch_size: int,
               generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """(bits [B, n_inputs] in {-1,+1}, labels [B] in {0,1}) を返す。"""
    bits = torch.randint(0, 2, (batch_size, spec.n_inputs), generator=generator).float() * 2 - 1
    a, b = spec.relevant
    labels = ((bits[:, a] > 0) ^ (bits[:, b] > 0)).long()
    return bits, labels


class Backbone(nn.Module):
    """「皮質バックボーン」役の1層MLP（入力→隠れ表現）。"""

    def __init__(self, n_inputs: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, hidden_dim)

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc1(bits))


def pretrain_backbone(backbone: Backbone, spec: ParitySpec, steps: int, batch_size: int,
                      lr: float, generator: torch.Generator) -> None:
    """教師ありパリティ分類でbackboneを事前学習する（一時的な読み出しヘッドを併用、破棄する）。"""
    head = nn.Linear(backbone.fc1.out_features, 2)
    opt = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=lr)
    for _ in range(steps):
        bits, labels = make_batch(spec, batch_size, generator)
        logits = head(backbone(bits))
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


class ActorCritic(nn.Module):
    """アクター（方策ロジット）・クリティック（価値）の線形ヘッド。3条件で共有する構成。"""

    def __init__(self, hidden_dim: int, n_actions: int = 2) -> None:
        super().__init__()
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(h), self.critic(h).squeeze(-1)


def build_backbone(condition: str, spec: ParitySpec, hidden_dim: int, pretrain_steps: int,
                   pretrain_batch: int, pretrain_lr: float,
                   generator: torch.Generator) -> tuple[Backbone, bool]:
    """条件に応じてbackboneを構築する。戻り値: (backbone, train_backbone_during_rl)。"""
    backbone = Backbone(spec.n_inputs, hidden_dim)
    if condition == "pretrained":
        pretrain_backbone(backbone, spec, pretrain_steps, pretrain_batch, pretrain_lr, generator)
        for p in backbone.parameters():
            p.requires_grad_(False)
        return backbone, False
    if condition == "random-frozen":
        for p in backbone.parameters():
            p.requires_grad_(False)
        return backbone, False
    if condition == "rl-only":
        return backbone, True
    raise ValueError(condition)


@torch.no_grad()
def evaluate(backbone: Backbone, head: ActorCritic, spec: ParitySpec, n_eval: int,
            generator: torch.Generator) -> float:
    bits, labels = make_batch(spec, n_eval, generator)
    logits, _ = head(backbone(bits))
    pred = logits.argmax(dim=-1)
    return float((pred == labels).float().mean())


def supervised_capacity_check(condition: str, spec: ParitySpec, hidden_dim: int, steps: int,
                              batch_size: int, lr: float, generator: torch.Generator) -> float:
    """RLを使わず教師あり勾配降下で、`condition`のバックボーン構成が到達できる正解率の
    上限を測る（12.6.26節・ステップ9の段階A）。

    `random-frozen`はバックボーンを凍結したまま線形ヘッドだけを学習し、それ以外の
    条件はバックボーンごと学習する（capacity上限の確認という段階Aの目的上、
    `pretrained`・`rl-only`はバックボーンが可塑的なので原理的にボトルネックに
    ならない。ボトルネックの有無を確認する対象は実質`random-frozen`のみ）。
    """
    backbone = Backbone(spec.n_inputs, hidden_dim)
    if condition == "random-frozen":
        for p in backbone.parameters():
            p.requires_grad_(False)
    head = nn.Linear(hidden_dim, 2)
    params = list(head.parameters())
    if condition != "random-frozen":
        params += list(backbone.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        bits, labels = make_batch(spec, batch_size, generator)
        logits = head(backbone(bits))
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    bits, labels = make_batch(spec, 2000, generator)
    with torch.no_grad():
        acc = float((head(backbone(bits)).argmax(-1) == labels).float().mean())
    return acc


def run_condition(condition: str, spec: ParitySpec, hidden_dim: int, pretrain_steps: int,
                  pretrain_batch: int, pretrain_lr: float, rl_steps: int, rl_batch: int,
                  rl_lr: float, eval_every: int, eval_batch: int,
                  generator: torch.Generator) -> list[float]:
    """1条件・1シードぶんの実行。`eval_every`ステップごとの評価正解率のリストを返す。"""
    backbone, train_backbone = build_backbone(condition, spec, hidden_dim, pretrain_steps,
                                              pretrain_batch, pretrain_lr, generator)
    head = ActorCritic(hidden_dim)
    params = list(head.parameters()) + (list(backbone.parameters()) if train_backbone else [])
    opt = torch.optim.Adam(params, lr=rl_lr)

    accs = []
    for step in range(rl_steps):
        bits, labels = make_batch(spec, rl_batch, generator)
        h = backbone(bits)
        logits, values = head(h)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        rewards = (actions == labels).float()
        advantages = (rewards - values).detach()
        actor_loss = -(advantages * dist.log_prob(actions)).mean()
        critic_loss = F.mse_loss(values, rewards)
        loss = actor_loss + critic_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step + 1) % eval_every == 0:
            accs.append(evaluate(backbone, head, spec, eval_batch, generator))
    return accs
