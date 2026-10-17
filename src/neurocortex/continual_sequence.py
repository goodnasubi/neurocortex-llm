"""タスク列での順次学習（12.6.101節）。ステップ10の2タスク版（hippocampus_replay.py）を T タスクに一般化し、
各タスクの学習後に全タスクの正確度を測って正確度行列 R を返す。

  naive-finetune … 保護機構なし
  replay         … 各タスク収束時のサンプルをタスクごとに buffer_size 個ずつ蓄積し、以後の学習で混ぜて再生する
  ewc            … 各タスク収束時のフィッシャー情報量と重みを蓄積し、全タスク分の罰則を足す

学習の乱数（generator）はステップ10と同じ順に消費し、評価には別の乱数（eval_generator）を使う。
T = 2 では学習の過程がステップ10と完全に一致する。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .basal_ganglia_core import ParitySpec, make_batch
from .hippocampus_replay import (Model, compute_fisher_diagonal, evaluate, make_model,
                                 make_replay_buffer, train_task, train_task_b_replay)


def _train_ewc_multi(model: Model, spec: ParitySpec, anchors: list[tuple[list[torch.Tensor], list[torch.Tensor]]],
                     ewc_lambda: float, steps: int, batch_size: int, lr: float,
                     generator: torch.Generator) -> None:
    params = model.parameters()
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        bits, labels = make_batch(spec, batch_size, generator)
        loss = F.cross_entropy(model(bits), labels)
        penalty = sum((f * (p - op) ** 2).sum()
                      for old, fisher in anchors for f, p, op in zip(fisher, params, old))
        opt.zero_grad(set_to_none=True)
        (loss + ewc_lambda * penalty).backward()
        opt.step()


def run_sequence(condition: str, specs: list[ParitySpec], hidden_dim: int, generator: torch.Generator,
                 eval_generator: torch.Generator, pretrain_steps: int = 500, pretrain_batch: int = 64,
                 pretrain_lr: float = 0.05, pretrain_weight_decay: float = 0.01, buffer_size: int = 50,
                 fisher_batches: int = 20, fisher_batch_size: int = 64, ewc_lambda: float = 1000.0,
                 steps: int = 500, batch: int = 64, lr: float = 0.05, eval_n: int = 2000) -> list[list[float]]:
    """正確度行列 R（R[i][j] = タスク i 学習後のタスク j の正確度）を返す。"""
    if condition not in ("naive-finetune", "replay", "ewc"):
        raise ValueError(condition)
    n_inputs = specs[0].n_inputs
    model = make_model(n_inputs, hidden_dim)
    buffers: list[tuple[torch.Tensor, torch.Tensor]] = []
    anchors: list[tuple[list[torch.Tensor], list[torch.Tensor]]] = []
    R: list[list[float]] = []
    for k, spec in enumerate(specs):
        if k == 0:
            train_task(model, spec, pretrain_steps, pretrain_batch, pretrain_lr, generator,
                       weight_decay=pretrain_weight_decay)
        else:
            prev = specs[k - 1]
            if condition == "naive-finetune":
                train_task(model, spec, steps, batch, lr, generator)
            elif condition == "replay":
                buffers.append(make_replay_buffer(prev, buffer_size, generator))
                buf = (torch.cat([b[0] for b in buffers]), torch.cat([b[1] for b in buffers]))
                train_task_b_replay(model, spec, buf, steps, batch, lr, generator)
            else:
                old = [p.detach().clone() for p in model.parameters()]
                fisher = compute_fisher_diagonal(model, prev, fisher_batches, fisher_batch_size, generator)
                anchors.append((old, fisher))
                _train_ewc_multi(model, spec, anchors, ewc_lambda, steps, batch, lr, generator)
        R.append([evaluate(model, s, eval_n, eval_generator) for s in specs])
    return R
