"""恒常性 MoE の規模の変化（12.6.137節）。専門家の数と MoE 層の数を変えて、12.6.124・128・134節の指標を測る部品。

`expert_prefetch.py`・`hub_load.py`・`homeostatic_moe.py` には手を入れない。ここに置く `ScaledRoutedCharLM` は
`RoutedCharLM` を L 層に積んだもの（GRU 隠れ状態 h_0 → h_l = h_{l-1} + g_l y_l(h_{l-1}) → 出力層）で、
n_layers=1・閾値 0 では `RoutedCharLM` と同じ乱数の消費・同じ出力になる（`tests/test_moe_scale.py`）。

- 学習条件: 補助損失（層ごとの Switch 型の負荷分散損失の和）／ゲート込みの恒常性（12.6.127節）／
  選択だけに効かせる恒常性（12.6.133節）。閾値・活動センサーは層ごと。
- 除去: 層ごとに無効な専門家の bool マスク [L,E]。出力の除去（選択を変えず出力を 0）と再ルーティング。
- ゲートの大きさを除いた頑健さ（12.6.137節）: 除去した専門家が担っていた「ゲートの質量」
  （そのトークンのゲートの和 ÷ 全トークン数×層数）あたりの悪化と、ゲートを 1 に固定した評価。
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hub_load import gini, max_over_mean, normalized_entropy


class ScaledRoutedCharLM(nn.Module):
    """L 層のトップ1 MoE 文字 LM。`kind` は "plain"（閾値なし）・"gate"（ゲート込み）・"select"（選択専用）。"""

    def __init__(self, vocab_size: int, d: int = 48, n_experts: int = 8, d_ff: int = 64, n_layers: int = 1,
                 kind: str = "plain", eta: float = 0.0, alpha: float = 0.05) -> None:
        super().__init__()
        if kind not in ("plain", "gate", "select"):
            raise ValueError(kind)
        self.n_experts, self.n_layers, self.kind = n_experts, n_layers, kind
        self.eta, self.alpha = eta, alpha
        # 1層目は RoutedCharLM と同じ順で作る（embed, rnn, router, w1, w2, head）。2層目以降はその後
        self.embed = nn.Embedding(vocab_size, d)
        self.rnn = nn.GRU(d, d, batch_first=True)
        routers, w1, w2 = [nn.Linear(d, n_experts)], [nn.Parameter(torch.randn(n_experts, d, d_ff) * d ** -0.5)], []
        w2.append(nn.Parameter(torch.randn(n_experts, d_ff, d) * d_ff ** -0.5))
        self.head = nn.Linear(d, vocab_size)
        for _ in range(n_layers - 1):
            routers.append(nn.Linear(d, n_experts))
            w1.append(nn.Parameter(torch.randn(n_experts, d, d_ff) * d ** -0.5))
            w2.append(nn.Parameter(torch.randn(n_experts, d_ff, d) * d_ff ** -0.5))
        self.routers = nn.ModuleList(routers)
        self.w1 = nn.ParameterList(w1)
        self.w2 = nn.ParameterList(w2)
        self.register_buffer("threshold", torch.zeros(n_layers, n_experts))
        self.register_buffer("usage_ema", torch.full((n_layers, n_experts), 1.0 / n_experts))

    def run(self, x: torch.Tensor, disabled: torch.Tensor | None = None, mode: str = "zero",
            gate_one: bool = False, use_threshold: bool = True) -> dict:
        """順伝播。disabled は bool [L,E]。返り値: logits・choice [L,B,T]・gate [L,B,T]・balance（層の和）。"""
        if mode not in ("zero", "reroute"):
            raise ValueError(mode)
        h, _ = self.rnn(self.embed(x))
        choices, gates, balance = [], [], h.new_zeros(())
        for l in range(self.n_layers):
            raw = self.routers[l](h)
            th = self.threshold[l] if use_threshold else torch.zeros_like(self.threshold[l])
            if self.kind == "gate":
                raw = raw - th
            if disabled is not None and mode == "reroute":
                raw = raw.masked_fill(disabled[l].to(torch.bool), float("-inf"))
            probs = F.softmax(raw, dim=-1)
            choice = (raw - th).argmax(-1) if self.kind == "select" else probs.argmax(-1)
            gate = probs.gather(-1, choice.unsqueeze(-1))
            y = torch.einsum("btd,btdf->btf", h, self.w1[l][choice]).relu()
            y = torch.einsum("btf,btfd->btd", y, self.w2[l][choice])
            if disabled is not None and mode == "zero":
                y = y * (~disabled[l].to(torch.bool))[choice].unsqueeze(-1).to(y.dtype)
            h = h + (torch.ones_like(gate) if gate_one else gate) * y
            frac = F.one_hot(choice, self.n_experts).float().mean(dim=(0, 1))
            balance = balance + self.n_experts * (frac * probs.mean(dim=(0, 1))).sum()
            choices.append(choice)
            gates.append(gate.squeeze(-1))
        return {"logits": self.head(h), "choice": torch.stack(choices), "gate": torch.stack(gates),
                "balance": balance}

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = self.run(x)
        return r["logits"], r["balance"], r["choice"]

    @torch.no_grad()
    def homeostatic_step(self, choice: torch.Tensor) -> None:
        """`HomeostaticRoutedCharLM.homeostatic_step` と同じ更新を層ごとに行う（choice [L,B,T]）。"""
        if self.kind == "plain":
            return
        for l in range(self.n_layers):
            frac = torch.bincount(choice[l].reshape(-1), minlength=self.n_experts).float() / choice[l].numel()
            self.usage_ema[l].mul_(1 - self.alpha).add_(self.alpha * frac)
            self.threshold[l].add_(self.eta * (self.n_experts * self.usage_ema[l] - 1))
            self.threshold[l].sub_(self.threshold[l].mean())


def train_scaled(model: ScaledRoutedCharLM, train_data: torch.Tensor, steps: int, batch_size: int, seq_len: int,
                 lr: float, generator: torch.Generator, balance_coef: float = 0.0) -> float:
    """`expert_prefetch.train_routed_lm`・`homeostatic_moe.train_homeostatic_lm` と同じ手順・乱数の消費。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = float("nan")
    for _ in range(steps):
        idx = torch.randint(0, len(train_data) - seq_len - 1, (batch_size,), generator=generator)
        x = torch.stack([train_data[i:i + seq_len] for i in idx.tolist()])
        y = torch.stack([train_data[i + 1:i + seq_len + 1] for i in idx.tolist()])
        logits, balance, choice = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        ce = float(loss.detach())
        opt.zero_grad(set_to_none=True)
        (loss + balance_coef * balance).backward()
        opt.step()
        model.homeostatic_step(choice)
    return ce


@torch.no_grad()
def eval_ce(model: ScaledRoutedCharLM, x: torch.Tensor, y: torch.Tensor, **kw) -> float:
    logits = model.run(x, **kw)["logits"]
    return float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)))


@torch.no_grad()
def load_stats(model: ScaledRoutedCharLM, x: torch.Tensor, use_threshold: bool = True) -> tuple[np.ndarray, dict]:
    """層ごとの負荷 [L,E] と、層の平均のジニ・最大負荷比・エントロピー・使用専門家数（層の平均）。"""
    r = model.run(x, use_threshold=use_threshold)
    c = r["choice"].numpy()
    load = np.stack([np.bincount(c[l].ravel(), minlength=model.n_experts) for l in range(model.n_layers)])
    load = load.astype(np.float64)
    out = {"gini": float(np.mean([gini(v) for v in load])),
           "max_over_mean": float(np.mean([max_over_mean(v) for v in load])),
           "entropy": float(np.mean([normalized_entropy(v) for v in load])),
           "n_used": float(np.mean([(v > 0).sum() for v in load])),
           "all_used": bool((load > 0).all())}
    if model.kind == "select":
        probs_arg = model.run(x, use_threshold=False)["choice"]
        out["shifted_frac"] = float((r["choice"] != probs_arg).float().mean())
    else:
        out["shifted_frac"] = 0.0
    out["gate_mean"] = float(r["gate"].mean())
    return load, out


def hub_masks(load: np.ndarray, k: int, n_random: int, rng: np.random.Generator) -> tuple[torch.Tensor, list]:
    """各層の負荷上位 k 個をハブとするマスク [L,E] と、各層の非ハブから k 個ずつ選ぶランダム除去のマスクの列。

    組合せが n_random 以下なら全通り（1層では 12.6.124節と同じ全 C(E−k, k) 通り）、多ければ無作為に n_random 通り。
    """
    n_layers, n_experts = load.shape
    hub = torch.zeros(n_layers, n_experts, dtype=torch.bool)
    non_hubs = []
    for l in range(n_layers):
        order = sorted(range(n_experts), key=lambda e: (-load[l, e], e))
        hub[l, order[:k]] = True
        non_hubs.append(order[k:])
    per_layer = [list(itertools.combinations(nh, k)) for nh in non_hubs]
    total = int(np.prod([len(p) for p in per_layer]))
    if total <= n_random:
        combos = list(itertools.product(*per_layer))
    else:
        combos = [tuple(tuple(sorted(rng.choice(nh, size=k, replace=False).tolist())) for nh in non_hubs)
                  for _ in range(n_random)]
    masks = []
    for combo in combos:
        m = torch.zeros(n_layers, n_experts, dtype=torch.bool)
        for l, c in enumerate(combo):
            m[l, list(c)] = True
        masks.append(m)
    return hub, masks


@torch.no_grad()
def lesion_study(model: ScaledRoutedCharLM, x: torch.Tensor, y: torch.Tensor, load: np.ndarray, k: int,
                 mode: str = "zero", n_random: int = 30, seed: int = 0, gate_one: bool = False) -> dict:
    """ハブ（各層の負荷上位 k 個）の同時除去と非ハブ同数の除去を比べる。`hub_load.lesion_study` の多層版。

    トークン割合あたりの悪化に加えて、ゲートの質量あたりの悪化（12.6.137節）を返す。ゲートの質量 =
    除去した専門家に送られたトークンのゲートの和 ÷（トークン数 × 層数）。gate_one=True ではゲートを 1 に
    固定した順伝播で除去前後の CE を測る（ゲートの質量はトークン割合と同じになる）。
    """
    hub, masks = hub_masks(load, k, n_random, np.random.default_rng(seed))
    r0 = model.run(x, gate_one=gate_one)
    choice, gate = r0["choice"], (torch.ones_like(r0["gate"]) if gate_one else r0["gate"])
    base = eval_ce(model, x, y, gate_one=gate_one)
    n = choice[0].numel() * model.n_layers

    def masses(m):
        sel = torch.stack([m[l][choice[l]] for l in range(model.n_layers)])
        return float(sel.float().sum()) / n, float((gate * sel).sum()) / n

    d_target = eval_ce(model, x, y, disabled=hub, mode=mode, gate_one=gate_one) - base
    d_random = [eval_ce(model, x, y, disabled=m, mode=mode, gate_one=gate_one) - base for m in masks]
    hub_frac, hub_gate = masses(hub)
    rand = [masses(m) for m in masks]
    per_load = [d / f for d, (f, _) in zip(d_random, rand) if f > 0]
    per_gate = [d / g for d, (_, g) in zip(d_random, rand) if g > 0]
    return {"base_ce": base, "d_target": d_target, "d_random_mean": float(np.mean(d_random)),
            "n_random": len(masks), "hub_load_frac": hub_frac, "hub_gate_mass": hub_gate,
            "random_load_frac_mean": float(np.mean([f for f, _ in rand])),
            "random_gate_mass_mean": float(np.mean([g for _, g in rand])),
            "d_target_per_load": d_target / hub_frac if hub_frac > 0 else float("nan"),
            "d_random_per_load": float(np.mean(per_load)) if per_load else float("nan"),
            "d_target_per_gate": d_target / hub_gate if hub_gate > 0 else float("nan"),
            "d_random_per_gate": float(np.mean(per_gate)) if per_gate else float("nan")}
