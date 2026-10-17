"""タスク列での順次学習（12.6.101節）。ステップ10の2タスク版（hippocampus_replay.py）を T タスクに一般化し、
各タスクの学習後に全タスクの正確度を測って正確度行列 R を返す。

  naive-finetune … 保護機構なし
  replay         … 各タスク収束時のサンプルをタスクごとに buffer_size 個ずつ蓄積し、以後の学習で混ぜて再生する
  ewc            … 各タスク収束時のフィッシャー情報量と重みを蓄積し、全タスク分の罰則を足す

`task_id_input=True` では入力にタスク ID の one-hot を連結する（ステップ12・13と同じ入力形式, 12.6.104節）。

学習の乱数（generator）はステップ10と同じ順に消費し、評価には別の乱数（eval_generator）を使う。
`task_id_input=False` かつ T = 2 では、学習の過程がステップ10と完全に一致する。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .basal_ganglia_core import ParitySpec, make_batch
from .hippocampus_replay import Model, make_model


def run_sequence(condition: str, specs: list[ParitySpec], hidden_dim: int, generator: torch.Generator,
                 eval_generator: torch.Generator, pretrain_steps: int = 500, pretrain_batch: int = 64,
                 pretrain_lr: float = 0.05, pretrain_weight_decay: float = 0.01, buffer_size: int = 50,
                 fisher_batches: int = 20, fisher_batch_size: int = 64, ewc_lambda: float = 1000.0,
                 steps: int = 500, batch: int = 64, lr: float = 0.05, eval_n: int = 2000,
                 task_id_input: bool = False) -> list[list[float]]:
    """正確度行列 R（R[i][j] = タスク i 学習後のタスク j の正確度）を返す。"""
    if condition not in ("naive-finetune", "replay", "ewc"):
        raise ValueError(condition)
    T = len(specs)
    n_inputs = specs[0].n_inputs

    def batch_of(k: int, n: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        bits, labels = make_batch(specs[k], n, g)
        if task_id_input:
            onehot = torch.zeros(n, T)
            onehot[:, k] = 1.0
            bits = torch.cat([bits, onehot], dim=1)
        return bits, labels

    model: Model = make_model(n_inputs + (T if task_id_input else 0), hidden_dim)

    def fit(k: int, n_steps: int, bsz: int, step_lr: float, weight_decay: float = 0.0,
            buffer: tuple[torch.Tensor, torch.Tensor] | None = None, anchors=None) -> None:
        params = model.parameters()
        opt = torch.optim.Adam(params, lr=step_lr, weight_decay=weight_decay)
        for _ in range(n_steps):
            bits, labels = batch_of(k, bsz, generator)
            if buffer is not None:
                idx = torch.randint(buffer[0].shape[0], (bsz,), generator=generator)
                bits = torch.cat([bits, buffer[0][idx]], dim=0)
                labels = torch.cat([labels, buffer[1][idx]], dim=0)
            loss = F.cross_entropy(model(bits), labels)
            if anchors:
                loss = loss + ewc_lambda * sum((f * (p - op) ** 2).sum()
                                               for old, fisher in anchors for f, p, op in zip(fisher, params, old))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    def fisher_of(k: int) -> list[torch.Tensor]:
        params = model.parameters()
        fisher = [torch.zeros_like(p) for p in params]
        for _ in range(fisher_batches):
            bits, _ = batch_of(k, fisher_batch_size, generator)
            log_probs = F.log_softmax(model(bits), dim=-1)
            with torch.no_grad():
                sampled = torch.multinomial(log_probs.exp(), 1).squeeze(-1)
            grads = torch.autograd.grad(F.nll_loss(log_probs, sampled), params)
            for f, gr in zip(fisher, grads):
                f += gr.detach() ** 2
        return [f / fisher_batches for f in fisher]

    @torch.no_grad()
    def accuracy(k: int) -> float:
        bits, labels = batch_of(k, eval_n, eval_generator)
        return float((model(bits).argmax(dim=-1) == labels).float().mean())

    buffers: list[tuple[torch.Tensor, torch.Tensor]] = []
    anchors: list[tuple[list[torch.Tensor], list[torch.Tensor]]] = []
    R: list[list[float]] = []
    for k in range(T):
        if k == 0:
            fit(0, pretrain_steps, pretrain_batch, pretrain_lr, weight_decay=pretrain_weight_decay)
        elif condition == "naive-finetune":
            fit(k, steps, batch, lr)
        elif condition == "replay":
            buffers.append(batch_of(k - 1, buffer_size, generator))
            fit(k, steps, batch, lr, buffer=(torch.cat([b[0] for b in buffers]), torch.cat([b[1] for b in buffers])))
        else:
            old = [p.detach().clone() for p in model.parameters()]
            anchors.append((old, fisher_of(k - 1)))
            fit(k, steps, batch, lr, anchors=anchors)
        R.append([accuracy(j) for j in range(T)])
    return R
