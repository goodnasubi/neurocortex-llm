"""海馬モジュール — リプレイによる皮質への固定化（12.6.28節・ステップ10）。

「過去の経験を蓄積し、新タスクの学習に混ぜて再生することで破局的忘却を防ぐ」
（rehearsal / experience replay）は継続学習の分野で広く知られた汎用技法であり、
「海馬」という枠組み固有の主張ではない。したがって主張は「リプレイが忘却を
防ぐか」ではなく、**生サンプルを保持するリプレイが、個々のサンプルを一切
保持せず旧タスクへのドリフトだけを罰する重み正則化（EWC型）と比べて固有の
優位性を持つか**に置く（12.6.28節「素朴な等価性の罠」参照）。

  naive-finetune（統制1の前提確認） … 保護機構なしでタスクBだけを学習する
  replay（本設計）                  … タスクA収束時に保存した生サンプルを、
                                      タスクB学習中の各バッチに混ぜる
  ewc（対照群、統制5の要）          … 生サンプルを持たず、タスクA収束時点の
                                      対角フィッシャー情報量で重みのドリフト
                                      を罰する
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basal_ganglia_core import Backbone, ParitySpec, make_batch

CONDITIONS = ("naive-finetune", "replay", "ewc")


@dataclass
class Model:
    """バックボーン＋線形ヘッド。全条件で同一のアーキテクチャを共有する（統制2）。"""

    backbone: Backbone
    head: nn.Linear

    def __call__(self, bits: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(bits))

    def parameters(self) -> list[nn.Parameter]:
        return list(self.backbone.parameters()) + list(self.head.parameters())


def make_model(n_inputs: int, hidden_dim: int) -> Model:
    return Model(Backbone(n_inputs, hidden_dim), nn.Linear(hidden_dim, 2))


def train_task(model: Model, spec: ParitySpec, steps: int, batch_size: int, lr: float,
               generator: torch.Generator, weight_decay: float = 0.0) -> None:
    """通常の教師あり学習でタスクを収束させる（タスクAの事前学習、およびnaive-finetuneの
    タスクB学習に使う）。

    `weight_decay`はタスクAの事前学習にのみ使う。CEで学習した分類器はロジットが
    際限なく大きくなり続け（マージン最大化方向に発散し）、勾配がほぼ0まで
    小さくなる。これは対角フィッシャー情報量（勾配の二乗の期待値）を無意味な
    ほぼ0の値に潰してしまい、EWC条件の正則化強度をどれだけ上げても効果が
    出なくなる（実測で確認済み、12.6.29節）。適度な重み減衰でロジットの発散を
    抑え、フィッシャー情報量を意味のある大きさに保つ。
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        bits, labels = make_batch(spec, batch_size, generator)
        loss = F.cross_entropy(model(bits), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def make_replay_buffer(spec: ParitySpec, buffer_size: int,
                       generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """タスクA収束時点で、タスクAの生サンプルを固定サイズだけ保存する。"""
    return make_batch(spec, buffer_size, generator)


def compute_fisher_diagonal(model: Model, spec: ParitySpec, n_batches: int, batch_size: int,
                            generator: torch.Generator) -> list[torch.Tensor]:
    """タスクA収束時点の対角フィッシャー情報量（重要度）を近似する。

    真のラベルに対する損失の勾配ではなく、**モデル自身の予測分布からサンプルした
    ラベル**に対する対数尤度の勾配の二乗を使う（標準的なEWCの定義）。収束済みの
    モデルは真のラベルに対する損失の勾配がほぼ0になり、それをそのままフィッシャー
    情報量として使うと常にほぼ0になってしまう（収束後は「間違えないから重要度も
    測れない」という无意味な近似になる）ため、この置き換えが必要になる。

    個々のサンプルは保持せず、複数バッチにわたる勾配の二乗の平均だけを保持する
    （EWC条件はこの1つのベクトル群以外、タスクAのデータを一切参照しない）。
    """
    params = model.parameters()
    fisher = [torch.zeros_like(p) for p in params]
    for _ in range(n_batches):
        bits, _ = make_batch(spec, batch_size, generator)
        logits = model(bits)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            sampled = torch.multinomial(log_probs.exp(), 1).squeeze(-1)
        loss = F.nll_loss(log_probs, sampled)
        grads = torch.autograd.grad(loss, params)
        for f, g in zip(fisher, grads):
            f += g.detach() ** 2
    return [f / n_batches for f in fisher]


def train_task_b_replay(model: Model, spec_b: ParitySpec, buffer: tuple[torch.Tensor, torch.Tensor],
                        steps: int, batch_size: int, lr: float,
                        generator: torch.Generator) -> None:
    """replay条件: 各ステップでタスクBの新サンプルとバッファからの再生サンプルを混ぜて学習する。"""
    buf_bits, buf_labels = buffer
    n_buffer = buf_bits.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        bits_b, labels_b = make_batch(spec_b, batch_size, generator)
        idx = torch.randint(n_buffer, (batch_size,), generator=generator)
        bits = torch.cat([bits_b, buf_bits[idx]], dim=0)
        labels = torch.cat([labels_b, buf_labels[idx]], dim=0)
        loss = F.cross_entropy(model(bits), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def train_task_b_ewc(model: Model, spec_b: ParitySpec, old_params: list[torch.Tensor],
                     fisher: list[torch.Tensor], ewc_lambda: float, steps: int, batch_size: int,
                     lr: float, generator: torch.Generator) -> None:
    """ewc条件: タスクB損失に、タスクA収束時の重みからのドリフトを罰する正則化項を加える。

    生サンプルは一切参照しない。`old_params`・`fisher`はタスクA収束時に1回だけ
    計算した固定値であり、タスクB学習中は更新しない。
    """
    params = model.parameters()
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        bits, labels = make_batch(spec_b, batch_size, generator)
        loss = F.cross_entropy(model(bits), labels)
        penalty = sum((f * (p - op) ** 2).sum() for f, p, op in zip(fisher, params, old_params))
        total = loss + ewc_lambda * penalty
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()


@torch.no_grad()
def evaluate(model: Model, spec: ParitySpec, n_eval: int, generator: torch.Generator) -> float:
    bits, labels = make_batch(spec, n_eval, generator)
    pred = model(bits).argmax(dim=-1)
    return float((pred == labels).float().mean())


def run_condition(condition: str, n_inputs: int, hidden_dim: int, spec_a: ParitySpec,
                  spec_b: ParitySpec, pretrain_steps: int, pretrain_batch: int,
                  pretrain_lr: float, buffer_size: int, fisher_batches: int,
                  fisher_batch_size: int, ewc_lambda: float, b_steps: int, b_batch: int,
                  b_lr: float, eval_n: int, generator: torch.Generator,
                  pretrain_weight_decay: float = 0.01) -> dict:
    """1条件・1シードぶんの実行。戻り値: {"acc_a": タスクA性能, "acc_b": タスクB性能}
    （いずれもタスクB学習後の値）。
    """
    model = make_model(n_inputs, hidden_dim)
    train_task(model, spec_a, pretrain_steps, pretrain_batch, pretrain_lr, generator,
              weight_decay=pretrain_weight_decay)

    if condition == "naive-finetune":
        train_task(model, spec_b, b_steps, b_batch, b_lr, generator)
    elif condition == "replay":
        buffer = make_replay_buffer(spec_a, buffer_size, generator)
        train_task_b_replay(model, spec_b, buffer, b_steps, b_batch, b_lr, generator)
    elif condition == "ewc":
        old_params = [p.detach().clone() for p in model.parameters()]
        fisher = compute_fisher_diagonal(model, spec_a, fisher_batches, fisher_batch_size, generator)
        train_task_b_ewc(model, spec_b, old_params, fisher, ewc_lambda, b_steps, b_batch, b_lr,
                         generator)
    else:
        raise ValueError(condition)

    acc_a = evaluate(model, spec_a, eval_n, generator)
    acc_b = evaluate(model, spec_b, eval_n, generator)
    return {"acc_a": acc_a, "acc_b": acc_b}
