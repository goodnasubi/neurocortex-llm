"""マルチ学習則の学習安定性 — 海馬×基底核の勾配干渉（12.6.36節・ステップ14）。

12.7節の懸念「海馬からのリプレイ勾配とRLの方策勾配が競合する」を検証する。
海馬ヘッド（教師あり分類、勾配降下）と基底核ヘッド（REINFORCE、方策勾配）が
同じ皮質バックボーン（`basal_ganglia_core.Backbone`）を共有して同時学習した
場合、学習信号が干渉して性能が劣化するかを見る。

  shared-both              … 1つのバックボーンを、海馬・基底核両方の勾配で同時学習する（本設計）
  independent-both         … 海馬用・基底核用に別々のバックボーンで独立に学習する（統制1）
  shared-hippocampus-only  … バックボーン共有のまま海馬勾配のみで学習する（統制2a、健全性確認）
  shared-rl-only           … バックボーン共有のまま基底核勾配のみで学習する（統制2b、健全性確認）

全条件で同一の`ParitySpec`（教師あり分類・RLとも同じ課題）を使う（統制3）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .basal_ganglia_core import ActorCritic, Backbone, ParitySpec, make_batch

CONDITIONS = ("shared-both", "independent-both", "shared-hippocampus-only", "shared-rl-only")


def _rl_step_losses(backbone: Backbone, rl_head: ActorCritic, spec: ParitySpec, batch_size: int,
                    generator: torch.Generator) -> torch.Tensor:
    """基底核のREINFORCE+ベースライン損失（`basal_ganglia_core.run_condition`と同じ定義）。"""
    bits, labels = make_batch(spec, batch_size, generator)
    h = backbone(bits)
    logits, values = rl_head(h)
    dist = torch.distributions.Categorical(logits=logits)
    actions = dist.sample()
    rewards = (actions == labels).float()
    advantages = (rewards - values).detach()
    actor_loss = -(advantages * dist.log_prob(actions)).mean()
    critic_loss = F.mse_loss(values, rewards)
    return actor_loss + critic_loss


def _ce_step_loss(backbone: Backbone, hippo_head: torch.nn.Linear, spec: ParitySpec,
                  batch_size: int, generator: torch.Generator) -> torch.Tensor:
    bits, labels = make_batch(spec, batch_size, generator)
    logits = hippo_head(backbone(bits))
    return F.cross_entropy(logits, labels)


def _cosine_similarity(grad_a: tuple[torch.Tensor, ...], grad_b: tuple[torch.Tensor, ...]) -> float:
    flat_a = torch.cat([g.reshape(-1) for g in grad_a])
    flat_b = torch.cat([g.reshape(-1) for g in grad_b])
    return float(F.cosine_similarity(flat_a, flat_b, dim=0))


@torch.no_grad()
def evaluate_hippocampus(backbone: Backbone, hippo_head: torch.nn.Linear, spec: ParitySpec,
                         n_eval: int, generator: torch.Generator) -> float:
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
    hippo_acc: float | None  # 海馬ヘッドの分類正解率（学習していない条件ではNone）
    rl_acc: float | None     # 基底核ヘッドのRL正解率（学習していない条件ではNone）
    cosine_similarities: list[float]  # shared-bothでのみ非空。各ステップの勾配コサイン類似度


def run_condition(condition: str, spec: ParitySpec, hidden_dim: int, steps: int, batch_size: int,
                  lr: float, eval_n: int, generator: torch.Generator) -> RunResult:
    """1条件・1シードぶんの実行。"""
    if condition not in CONDITIONS:
        raise ValueError(condition)

    n_inputs = spec.n_inputs
    cosine_similarities: list[float] = []

    if condition == "shared-both":
        backbone = Backbone(n_inputs, hidden_dim)
        hippo_head = torch.nn.Linear(hidden_dim, 2)
        rl_head = ActorCritic(hidden_dim)
        params = list(backbone.parameters()) + list(hippo_head.parameters()) + list(rl_head.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(steps):
            ce_loss = _ce_step_loss(backbone, hippo_head, spec, batch_size, generator)
            rl_loss = _rl_step_losses(backbone, rl_head, spec, batch_size, generator)

            grad_ce = torch.autograd.grad(ce_loss, list(backbone.parameters()), retain_graph=True)
            grad_rl = torch.autograd.grad(rl_loss, list(backbone.parameters()), retain_graph=True)
            cosine_similarities.append(_cosine_similarity(grad_ce, grad_rl))

            loss = ce_loss + rl_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        hippo_acc = evaluate_hippocampus(backbone, hippo_head, spec, eval_n, generator)
        rl_acc = evaluate_rl(backbone, rl_head, spec, eval_n, generator)

    elif condition == "independent-both":
        backbone_h = Backbone(n_inputs, hidden_dim)
        hippo_head = torch.nn.Linear(hidden_dim, 2)
        opt_h = torch.optim.Adam(list(backbone_h.parameters()) + list(hippo_head.parameters()), lr=lr)

        backbone_rl = Backbone(n_inputs, hidden_dim)
        rl_head = ActorCritic(hidden_dim)
        opt_rl = torch.optim.Adam(list(backbone_rl.parameters()) + list(rl_head.parameters()), lr=lr)

        for _ in range(steps):
            ce_loss = _ce_step_loss(backbone_h, hippo_head, spec, batch_size, generator)
            opt_h.zero_grad(set_to_none=True)
            ce_loss.backward()
            opt_h.step()

            rl_loss = _rl_step_losses(backbone_rl, rl_head, spec, batch_size, generator)
            opt_rl.zero_grad(set_to_none=True)
            rl_loss.backward()
            opt_rl.step()

        hippo_acc = evaluate_hippocampus(backbone_h, hippo_head, spec, eval_n, generator)
        rl_acc = evaluate_rl(backbone_rl, rl_head, spec, eval_n, generator)

    elif condition == "shared-hippocampus-only":
        backbone = Backbone(n_inputs, hidden_dim)
        hippo_head = torch.nn.Linear(hidden_dim, 2)
        opt = torch.optim.Adam(list(backbone.parameters()) + list(hippo_head.parameters()), lr=lr)
        for _ in range(steps):
            ce_loss = _ce_step_loss(backbone, hippo_head, spec, batch_size, generator)
            opt.zero_grad(set_to_none=True)
            ce_loss.backward()
            opt.step()

        hippo_acc = evaluate_hippocampus(backbone, hippo_head, spec, eval_n, generator)
        rl_acc = None

    elif condition == "shared-rl-only":
        backbone = Backbone(n_inputs, hidden_dim)
        rl_head = ActorCritic(hidden_dim)
        opt = torch.optim.Adam(list(backbone.parameters()) + list(rl_head.parameters()), lr=lr)
        for _ in range(steps):
            rl_loss = _rl_step_losses(backbone, rl_head, spec, batch_size, generator)
            opt.zero_grad(set_to_none=True)
            rl_loss.backward()
            opt.step()

        hippo_acc = None
        rl_acc = evaluate_rl(backbone, rl_head, spec, eval_n, generator)

    return RunResult(hippo_acc=hippo_acc, rl_acc=rl_acc, cosine_similarities=cosine_similarities)
