"""ハブスパース MoE のハブ専門家への負荷集中（12.6.123節）。

12.7節「スケールの検証不足」のリスク（ハブ専門家がボトルネック化する）を、
`expert_prefetch.RoutedCharLM`（トップ1ルーティングの最小 MoE 文字 LM）で小規模に測る部品。
`expert_prefetch.py` には手を入れず、除去付きの順伝播はここで同じ計算を書き直す
（除去なしで元の `forward` と一致することは `tests/test_hub_load.py` で確かめる）。

- 負荷の偏り: `gini`・`normalized_entropy`・`max_over_mean`
- 専門家間の機能的結合: 隣接トークン間の遷移グラフ（`transition_matrix`・`functional_degree`）
- 除去: 出力の除去（`mode="zero"`、ルーティングは変えず専門家の出力だけ 0）と
  再ルーティング（`mode="reroute"`、無効化した専門家をルータの候補から外す）
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn.functional as F

from .expert_prefetch import RoutedCharLM


# --- 負荷の偏り ---------------------------------------------------------------

def gini(counts) -> float:
    """非負の負荷のジニ係数。一様で 0、1個に全て集中すると (E-1)/E。"""
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    i = np.arange(1, n + 1)
    return float((2 * i - n - 1) @ x / (n * x.sum()))


def normalized_entropy(counts) -> float:
    """負荷分布のエントロピー ÷ log E（一様で 1）。"""
    p = np.asarray(counts, dtype=np.float64)
    p = p / p.sum()
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(len(p)))


def max_over_mean(counts) -> float:
    c = np.asarray(counts, dtype=np.float64)
    return float(c.max() / c.mean())


def spearman(a, b) -> float:
    """Spearman の順位相関（同順位は平均順位）。"""
    def ranks(v):
        v = np.asarray(v, dtype=np.float64)
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v))
        r[order] = np.arange(len(v), dtype=np.float64)
        for val in np.unique(v):
            m = v == val
            r[m] = r[m].mean()
        return r
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# --- 除去付きの順伝播 ---------------------------------------------------------

def lesioned_forward(model: RoutedCharLM, x: torch.Tensor, disabled: torch.Tensor | None = None,
                     mode: str = "zero") -> tuple[torch.Tensor, torch.Tensor]:
    """x [B,T] → (logits [B,T,V], 選択 [B,T])。`disabled` は bool [E]（True が無効）。

    mode="zero": ルーティングはそのまま、無効な専門家に送られたトークンでは専門家の出力を 0 にする。
    mode="reroute": 無効な専門家のルータ出力を -inf にして残りから選び直す（確率も残りで正規化）。
    """
    if mode not in ("zero", "reroute"):
        raise ValueError(mode)
    h, _ = model.rnn(model.embed(x))
    logits_r = model.router(h)
    if disabled is not None and mode == "reroute":
        logits_r = logits_r.masked_fill(disabled.to(torch.bool), float("-inf"))
    probs = F.softmax(logits_r, dim=-1)
    choice = probs.argmax(-1)
    gate = probs.gather(-1, choice.unsqueeze(-1))
    y = torch.einsum("btd,btdf->btf", h, model.w1[choice]).relu()
    y = torch.einsum("btf,btfd->btd", y, model.w2[choice])
    if disabled is not None and mode == "zero":
        keep = (~disabled.to(torch.bool))[choice].unsqueeze(-1).to(y.dtype)
        y = y * keep
    return model.head(h + gate * y), choice


def make_windows(data: torch.Tensor, seq_len: int, n_windows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """検証列から重ならない窓 (x [N,T], y [N,T]) を先頭から切り出す。"""
    n = min(n_windows, (len(data) - 1) // seq_len)
    if n < 1:
        raise ValueError("data が短すぎる")
    x = torch.stack([data[i * seq_len:(i + 1) * seq_len] for i in range(n)])
    y = torch.stack([data[i * seq_len + 1:(i + 1) * seq_len + 1] for i in range(n)])
    return x, y


@torch.no_grad()
def eval_ce(model: RoutedCharLM, x: torch.Tensor, y: torch.Tensor, disabled: torch.Tensor | None = None,
            mode: str = "zero") -> float:
    logits, _ = lesioned_forward(model, x, disabled, mode)
    return float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)))


@torch.no_grad()
def expert_choices(model: RoutedCharLM, x: torch.Tensor) -> torch.Tensor:
    return lesioned_forward(model, x)[1]


# --- 専門家間の機能的結合（遷移グラフ） --------------------------------------

def transition_matrix(choices: torch.Tensor, n_experts: int) -> np.ndarray:
    """各窓の中で隣接するトークンの専門家遷移 e_t → e_{t+1} の頻度 [E,E]（窓をまたがない）。"""
    c = choices.numpy()
    m = np.zeros((n_experts, n_experts))
    np.add.at(m, (c[:, :-1].ravel(), c[:, 1:].ravel()), 1)
    return m


def functional_degree(trans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(結合相手の数, 結合強度)。自己遷移を除いた入出力の遷移 trans+trans.T を使う。

    結合相手の数は、相手との遷移が全遷移の 1/E² 以上ある相手の数（一様な遷移なら全員と結ばれる）。
    結合強度は自己遷移を除いた入出力の遷移の合計 ÷ 全遷移。
    """
    e = trans.shape[0]
    sym = trans + trans.T
    np.fill_diagonal(sym, 0)
    total = trans.sum()
    degree = (sym >= total / e ** 2).sum(axis=1)
    strength = sym.sum(axis=1) / max(total, 1.0)
    return degree.astype(np.int64), strength


# --- 標的除去とランダム除去 ---------------------------------------------------

def lesion_study(model: RoutedCharLM, x: torch.Tensor, y: torch.Tensor, load: np.ndarray, k: int,
                 mode: str = "zero") -> dict:
    """負荷上位 k 個（ハブ）の同時除去と、非ハブから k 個の全組合せの除去の平均を比べる。

    同じ評価窓を全ての除去に使い回す。負荷が同じ専門家は添字の小さい方を上位とする（決定的）。
    """
    n_experts = model.n_experts
    base = eval_ce(model, x, y)
    order = sorted(range(n_experts), key=lambda e: (-load[e], e))
    hubs, non_hubs = order[:k], order[k:]

    def mask(idx):
        m = torch.zeros(n_experts, dtype=torch.bool)
        m[list(idx)] = True
        return m

    frac = np.asarray(load, dtype=np.float64) / max(float(np.sum(load)), 1.0)
    combos = list(itertools.combinations(non_hubs, k))
    d_target = eval_ce(model, x, y, mask(hubs), mode) - base
    d_random = [eval_ce(model, x, y, mask(c), mode) - base for c in combos]
    single = [eval_ce(model, x, y, mask([e]), mode) - base for e in range(n_experts)]
    # 参考（12.6.124節の修正1）: 除去した専門家が担っていたトークン割合あたりの悪化。
    # 負荷 0 の組合せ（使われていない専門家だけの除去）は割合あたりが定義できないので除く。
    hub_frac = float(frac[hubs].sum())
    per_load_random = [d / frac[list(c)].sum() for d, c in zip(d_random, combos) if frac[list(c)].sum() > 0]
    return {"base_ce": base, "hubs": hubs, "d_target": d_target,
            "d_random_mean": float(np.mean(d_random)), "d_random_max": float(np.max(d_random)),
            "d_single": single, "spearman_load_single": spearman(load, single),
            "hub_load_frac": hub_frac,
            "random_load_frac_mean": float(np.mean([frac[list(c)].sum() for c in combos])),
            "d_target_per_load": d_target / hub_frac if hub_frac > 0 else float("nan"),
            "d_random_per_load": float(np.mean(per_load_random)) if per_load_random else float("nan"),
            "n_random_with_load": len(per_load_random)}
