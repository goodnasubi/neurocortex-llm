"""マルチ学習則の学習安定性 — STDPを統合した4モジュール構成（12.6.92節・ステップ42）。

ステップ18（`multi_learning_rule_3way.py`）は3モジュール構成（海馬+基底核+小脳）での
干渉テストを行ったが、STDPは未統合であった。ステップ26-28でSTDP単体の数値安定性が
確認されたため、本ステップでは4モジュール構成（海馬←STDP+教師あり勾配降下、基底核←
方策勾配、小脳←教師あり回帰）での干渉テストを実施する。

STDP は海馬モジュールのパターン分離層に適用される（hippocampus.fit_stdp の
パラメータ: a_plus=0.1, a_minus=0.1, tau=0.9）。

条件（16個）:
  - 基本8条件（ステップ18と同じ）:
    independent-all, shared-all, shared-hippo-only, shared-rl-only,
    shared-cereb-only, shared-hippo-rl, shared-hippo-cereb, shared-rl-cereb

  - 各条件に STDP on/off フラグを追加:
    <base-condition>-stdp-on, <base-condition>-stdp-off

全条件で同一の`ParitySpec`を使う（統制3）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basal_ganglia_core import ActorCritic, Backbone, ParitySpec, make_batch
from .hippocampus import PatternSeparator, fit_stdp


_BASE_CONDITIONS = (
    "independent-all",
    "shared-all",
    "shared-hippo-only",
    "shared-rl-only",
    "shared-cereb-only",
    "shared-hippo-rl",
    "shared-hippo-cereb",
    "shared-rl-cereb",
)

# 16条件: 各基本条件に -stdp-on, -stdp-off を追加
CONDITIONS = tuple(
    f"{base}-stdp-{flag}"
    for base in _BASE_CONDITIONS
    for flag in ("on", "off")
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
    """{-1,+1}のビット列から「1の個数」を返す。"""
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
    bits, _ = make_batch(spec, n_eval, generator)
    pred = cereb_head(backbone(bits)).squeeze(-1)
    target = _hamming_weight_target(bits)
    return float(F.mse_loss(pred, target))


@dataclass
class RunResult:
    """1条件・1シードの実行結果。"""
    hippo_acc: float | None       # 海馬ヘッドの分類正解率（学習していない条件ではNone）
    rl_acc: float | None          # 基底核ヘッドのRL正解率（学習していない条件ではNone）
    cereb_mse: float | None       # 小脳ヘッドのハミング重み回帰MSE（学習していない条件ではNone）
    cosine_similarities: dict[str, list[float]]  # ペア名 -> 各ステップのコサイン類似度
    nan_count: int                # NaN発生回数
    stdp_enabled: bool            # STDPが有効か
    stdp_weight_update_norms: list[float] = field(default_factory=list)  # epochごとの生の更新量ノルム


def _has_stdp(condition: str) -> bool:
    """条件文字列から STDP フラグを抽出。"""
    _, stdp_flag = _extract_base_and_stdp_flag(condition)
    return stdp_flag


def _extract_base_and_stdp_flag(condition: str) -> tuple[str, bool]:
    """条件文字列から基本条件とSTDPフラグを抽出。

    例: "shared-all-stdp-on" → ("shared-all", True)
    """
    for base in _BASE_CONDITIONS:
        if condition.startswith(base + "-stdp-"):
            flag_str = condition[len(base) + 6:]  # "-stdp-" の後ろ
            if flag_str == "on":
                return base, True
            elif flag_str == "off":
                return base, False
    raise ValueError(f"Invalid condition: {condition}")


def _active_modules(condition: str) -> tuple[str, ...]:
    base, _ = _extract_base_and_stdp_flag(condition)
    if base == "independent-all":
        return _MODULES
    if base == "shared-all":
        return _MODULES
    if base in _SOLO_CONDITIONS:
        return _SOLO_CONDITIONS[base]
    if base in _PAIR_CONDITIONS:
        return _PAIR_CONDITIONS[base]
    raise ValueError(base)


def run_condition(condition: str, spec: ParitySpec, hidden_dim: int, steps: int, batch_size: int,
                  lr: float, eval_n: int, generator: torch.Generator) -> RunResult:
    """1条件・1シードぶんの実行（STDPの有無を含む）。"""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")

    base, stdp_enabled = _extract_base_and_stdp_flag(condition)
    n_inputs = spec.n_inputs
    stdp_weight_update_norms: list[float] = []
    nan_count = 0

    # STDP用の生の更新量ノルム計算コールバック
    def stdp_raw_update_callback(ep: int, raw_norm: float) -> None:
        stdp_weight_update_norms.append(raw_norm)

    if base == "independent-all":
        # 3つの独立したバックボーン
        backbones = {m: Backbone(n_inputs, hidden_dim) for m in _MODULES}
        hippo_head = nn.Linear(hidden_dim, 2)
        rl_head = ActorCritic(hidden_dim)
        cereb_head = nn.Linear(hidden_dim, 1)

        if stdp_enabled:
            sep = PatternSeparator(n_inputs, n_units=128, k=10, mode="sdr", seed=0)
        else:
            sep = None

        opt_hippo = torch.optim.Adam(list(backbones["hippo"].parameters()) + list(hippo_head.parameters()), lr=lr)
        opt_rl = torch.optim.Adam(list(backbones["rl"].parameters()) + list(rl_head.parameters()), lr=lr)
        opt_cereb = torch.optim.Adam(list(backbones["cereb"].parameters()) + list(cereb_head.parameters()), lr=lr)

        for step_i in range(steps):
            # 海馬（通常勾配）
            ce_loss = _ce_step_loss(backbones["hippo"], hippo_head, spec, batch_size, generator)
            opt_hippo.zero_grad(set_to_none=True)
            ce_loss.backward()
            opt_hippo.step()

            if torch.isnan(ce_loss):
                nan_count += 1

            # STDP（独立時はパターン分離層に適用）
            if stdp_enabled and step_i % 5 == 0:
                bits = make_batch(spec, min(256, batch_size), generator)[0]
                spikes = (bits > 0).float().unsqueeze(0).expand(10, -1, -1)
                fit_stdp(sep, spikes, epochs=1, a_plus=0.1, a_minus=0.1, tau=0.9,
                        raw_update_callback=stdp_raw_update_callback, generator=generator)

            # 基底核
            rl_loss = _rl_step_loss(backbones["rl"], rl_head, spec, batch_size, generator)
            opt_rl.zero_grad(set_to_none=True)
            rl_loss.backward()
            opt_rl.step()

            if torch.isnan(rl_loss):
                nan_count += 1

            # 小脳
            cereb_loss = _cereb_step_loss(backbones["cereb"], cereb_head, spec, batch_size, generator)
            opt_cereb.zero_grad(set_to_none=True)
            cereb_loss.backward()
            opt_cereb.step()

            if torch.isnan(cereb_loss):
                nan_count += 1

        return RunResult(
            hippo_acc=evaluate_hippocampus(backbones["hippo"], hippo_head, spec, eval_n, generator),
            rl_acc=evaluate_rl(backbones["rl"], rl_head, spec, eval_n, generator),
            cereb_mse=evaluate_cerebellum(backbones["cereb"], cereb_head, spec, eval_n, generator),
            cosine_similarities={},
            nan_count=nan_count,
            stdp_enabled=stdp_enabled,
            stdp_weight_update_norms=stdp_weight_update_norms,
        )

    # 共有バックボーン構成
    active = _active_modules(condition)
    backbone = Backbone(n_inputs, hidden_dim)
    hippo_head = nn.Linear(hidden_dim, 2) if "hippo" in active else None
    rl_head = ActorCritic(hidden_dim) if "rl" in active else None
    cereb_head = nn.Linear(hidden_dim, 1) if "cereb" in active else None

    if stdp_enabled:
        sep = PatternSeparator(n_inputs, n_units=128, k=10, mode="sdr", seed=0)
    else:
        sep = None

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

    for step_i in range(steps):
        losses = {}
        grads = {}

        if "hippo" in active:
            losses["hippo"] = _ce_step_loss(backbone, hippo_head, spec, batch_size, generator)
            if torch.isnan(losses["hippo"]):
                nan_count += 1

        if "rl" in active:
            losses["rl"] = _rl_step_loss(backbone, rl_head, spec, batch_size, generator)
            if torch.isnan(losses["rl"]):
                nan_count += 1

        if "cereb" in active:
            losses["cereb"] = _cereb_step_loss(backbone, cereb_head, spec, batch_size, generator)
            if torch.isnan(losses["cereb"]):
                nan_count += 1

        # コサイン類似度計算（2つ以上のモジュールがactiveな場合）
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

        # STDP（共有時はパターン分離層に適用）
        if stdp_enabled and step_i % 5 == 0:
            bits = make_batch(spec, min(256, batch_size), generator)[0]
            spikes = (bits > 0).float().unsqueeze(0).expand(10, -1, -1)
            fit_stdp(sep, spikes, epochs=1, a_plus=0.1, a_minus=0.1, tau=0.9,
                    raw_update_callback=stdp_raw_update_callback, generator=generator)

    return RunResult(
        hippo_acc=evaluate_hippocampus(backbone, hippo_head, spec, eval_n, generator) if hippo_head else None,
        rl_acc=evaluate_rl(backbone, rl_head, spec, eval_n, generator) if rl_head else None,
        cereb_mse=evaluate_cerebellum(backbone, cereb_head, spec, eval_n, generator) if cereb_head else None,
        cosine_similarities=cosine_similarities,
        nan_count=nan_count,
        stdp_enabled=stdp_enabled,
        stdp_weight_update_norms=stdp_weight_update_norms,
    )
