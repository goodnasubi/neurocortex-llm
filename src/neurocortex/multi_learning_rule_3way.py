"""マルチ学習則の学習安定性 — 小脳モジュールを加えた3モジュール構成の再検証（12.6.44節・ステップ18）。

ステップ14（`multi_learning_rule.py`）は海馬（教師あり勾配降下）×基底核（REINFORCE
方策勾配）の2モジュールに限れば干渉が観測されないことを確認したが、「小脳や
STDPを含む4モジュール全体の組み合わせは未検証」と明記していた。本モジュールは
そのうち小脳（第3の教師あり回帰ヘッド）を加えた3モジュール構成を対象にする。

小脳ヘッドは、海馬・基底核と同一のビット列入力から「1の個数（ハミング重み）」を
連続値として回帰する（教師あり、MSE損失、勾配降下）。入力分布を完全に共有する
ことで、小脳自身の入力分布の違いが交絡変数にならないようにする。

条件（7つ）:
  independent-all     … 3モジュールとも独立バックボーンで学習する（統制1）
  shared-all           … 1つのバックボーンを3モジュール全ての勾配で同時学習する（本設計）
  shared-hippo-only    … バックボーン共有のまま海馬勾配のみで学習する（健全性確認）
  shared-rl-only       … バックボーン共有のまま基底核勾配のみで学習する（健全性確認）
  shared-cereb-only    … バックボーン共有のまま小脳勾配のみで学習する（健全性確認）
  shared-hippo-rl      … 海馬・基底核のみ共有（ペア、ステップ14のshared-both相当）
  shared-hippo-cereb   … 海馬・小脳のみ共有（ペア）
  shared-rl-cereb      … 基底核・小脳のみ共有（ペア）

全条件で同一の`ParitySpec`を使う（統制3）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basal_ganglia_core import ActorCritic, Backbone, ParitySpec, make_batch

CONDITIONS = (
    "independent-all",
    "shared-all",
    "shared-hippo-only",
    "shared-rl-only",
    "shared-cereb-only",
    "shared-hippo-rl",
    "shared-hippo-cereb",
    "shared-rl-cereb",
)

_MODULES = ("hippo", "rl", "cereb")

_PAIR_CONDITIONS = {
    "shared-hippo-rl": ("hippo", "rl"),
    "shared-hippo-cereb": ("hippo", "cereb"),
    "shared-rl-cereb": ("rl", "cereb"),
}

_SOLO_CONDITIONS = {
    "shared-hippo-only": ("hippo",),
    "shared-rl-only": ("rl",),
    "shared-cereb-only": ("cereb",),
}


def _hamming_weight_target(bits: torch.Tensor) -> torch.Tensor:
    """{-1,+1}のビット列から「1の個数」（0..n_inputs）を返す。"""
    return ((bits > 0).float()).sum(dim=-1)


def _ce_step_loss(backbone: Backbone, hippo_head: nn.Linear, spec: ParitySpec, batch_size: int,
                  generator: torch.Generator) -> torch.Tensor:
    bits, labels = make_batch(spec, batch_size, generator)
    logits = hippo_head(backbone(bits))
    return F.cross_entropy(logits, labels)


def _rl_step_loss(backbone: Backbone, rl_head: ActorCritic, spec: ParitySpec, batch_size: int,
                  generator: torch.Generator) -> torch.Tensor:
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


def _cereb_step_loss(backbone: Backbone, cereb_head: nn.Linear, spec: ParitySpec, batch_size: int,
                     generator: torch.Generator) -> torch.Tensor:
    bits, _ = make_batch(spec, batch_size, generator)
    pred = cereb_head(backbone(bits)).squeeze(-1)
    target = _hamming_weight_target(bits)
    return F.mse_loss(pred, target)


def _cosine_similarity(grad_a: tuple[torch.Tensor, ...], grad_b: tuple[torch.Tensor, ...]) -> float:
    flat_a = torch.cat([g.reshape(-1) for g in grad_a])
    flat_b = torch.cat([g.reshape(-1) for g in grad_b])
    return float(F.cosine_similarity(flat_a, flat_b, dim=0))


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


@torch.no_grad()
def evaluate_cerebellum(backbone: Backbone, cereb_head: nn.Linear, spec: ParitySpec, n_eval: int,
                        generator: torch.Generator) -> float:
    """ハミング重み回帰のMSE誤差を返す（小さいほど良い）。"""
    bits, _ = make_batch(spec, n_eval, generator)
    pred = cereb_head(backbone(bits)).squeeze(-1)
    target = _hamming_weight_target(bits)
    return float(F.mse_loss(pred, target))


@dataclass
class RunResult:
    hippo_acc: float | None       # 海馬ヘッドの分類正解率（学習していない条件ではNone）
    rl_acc: float | None          # 基底核ヘッドのRL正解率（学習していない条件ではNone）
    cereb_mse: float | None       # 小脳ヘッドのハミング重み回帰MSE（学習していない条件ではNone）
    cosine_similarities: dict[str, list[float]]  # ペア名 -> 各ステップのコサイン類似度（sharedが2つ以上の条件でのみ非空）


def _active_modules(condition: str) -> tuple[str, ...]:
    if condition == "independent-all":
        return _MODULES
    if condition == "shared-all":
        return _MODULES
    if condition in _SOLO_CONDITIONS:
        return _SOLO_CONDITIONS[condition]
    if condition in _PAIR_CONDITIONS:
        return _PAIR_CONDITIONS[condition]
    raise ValueError(condition)


def run_condition(condition: str, spec: ParitySpec, hidden_dim: int, steps: int, batch_size: int,
                  lr: float, eval_n: int, generator: torch.Generator) -> RunResult:
    """1条件・1シードぶんの実行。"""
    if condition not in CONDITIONS:
        raise ValueError(condition)

    n_inputs = spec.n_inputs

    if condition == "independent-all":
        backbones = {m: Backbone(n_inputs, hidden_dim) for m in _MODULES}
        hippo_head = nn.Linear(hidden_dim, 2)
        rl_head = ActorCritic(hidden_dim)
        cereb_head = nn.Linear(hidden_dim, 1)
        opt_hippo = torch.optim.Adam(list(backbones["hippo"].parameters()) + list(hippo_head.parameters()), lr=lr)
        opt_rl = torch.optim.Adam(list(backbones["rl"].parameters()) + list(rl_head.parameters()), lr=lr)
        opt_cereb = torch.optim.Adam(list(backbones["cereb"].parameters()) + list(cereb_head.parameters()), lr=lr)

        for _ in range(steps):
            ce_loss = _ce_step_loss(backbones["hippo"], hippo_head, spec, batch_size, generator)
            opt_hippo.zero_grad(set_to_none=True)
            ce_loss.backward()
            opt_hippo.step()

            rl_loss = _rl_step_loss(backbones["rl"], rl_head, spec, batch_size, generator)
            opt_rl.zero_grad(set_to_none=True)
            rl_loss.backward()
            opt_rl.step()

            cereb_loss = _cereb_step_loss(backbones["cereb"], cereb_head, spec, batch_size, generator)
            opt_cereb.zero_grad(set_to_none=True)
            cereb_loss.backward()
            opt_cereb.step()

        return RunResult(
            hippo_acc=evaluate_hippocampus(backbones["hippo"], hippo_head, spec, eval_n, generator),
            rl_acc=evaluate_rl(backbones["rl"], rl_head, spec, eval_n, generator),
            cereb_mse=evaluate_cerebellum(backbones["cereb"], cereb_head, spec, eval_n, generator),
            cosine_similarities={},
        )

    active = _active_modules(condition)
    backbone = Backbone(n_inputs, hidden_dim)
    hippo_head = nn.Linear(hidden_dim, 2) if "hippo" in active else None
    rl_head = ActorCritic(hidden_dim) if "rl" in active else None
    cereb_head = nn.Linear(hidden_dim, 1) if "cereb" in active else None

    params = list(backbone.parameters())
    if hippo_head is not None:
        params += list(hippo_head.parameters())
    if rl_head is not None:
        params += list(rl_head.parameters())
    if cereb_head is not None:
        params += list(cereb_head.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    pair_names = [f"{a}x{b}" for i, a in enumerate(active) for b in active[i + 1:]]
    cosine_similarities: dict[str, list[float]] = {name: [] for name in pair_names}

    for _ in range(steps):
        losses = {}
        grads = {}
        if hippo_head is not None:
            losses["hippo"] = _ce_step_loss(backbone, hippo_head, spec, batch_size, generator)
        if rl_head is not None:
            losses["rl"] = _rl_step_loss(backbone, rl_head, spec, batch_size, generator)
        if cereb_head is not None:
            losses["cereb"] = _cereb_step_loss(backbone, cereb_head, spec, batch_size, generator)

        if len(active) >= 2:
            for name in active:
                grads[name] = torch.autograd.grad(losses[name], list(backbone.parameters()),
                                                   retain_graph=True)
            for i, a in enumerate(active):
                for b in active[i + 1:]:
                    cosine_similarities[f"{a}x{b}"].append(_cosine_similarity(grads[a], grads[b]))

        total_loss = sum(losses.values())
        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        opt.step()

    return RunResult(
        hippo_acc=evaluate_hippocampus(backbone, hippo_head, spec, eval_n, generator) if hippo_head is not None else None,
        rl_acc=evaluate_rl(backbone, rl_head, spec, eval_n, generator) if rl_head is not None else None,
        cereb_mse=evaluate_cerebellum(backbone, cereb_head, spec, eval_n, generator) if cereb_head is not None else None,
        cosine_similarities=cosine_similarities,
    )
