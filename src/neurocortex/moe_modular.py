"""領域ごとに分かれた土台の上で、専門家だけが領域をまたげる MoE 文字 LM（12.6.201節）。

皮質は領域ごとのモジュール（局所結合が密、長距離結合が疎）からなり、領域をまたぐ情報は rich club のハブを
経由する（van den Heuvel & Sporns, 2011; Meunier et al., 2010; Sporns & Betzel, 2016）。12.6.200節では
統合（a1 の概念と b1 の言語の結合）が全専門家の共有する GRU の土台の中で済んでいたとみられたので、土台を
言語（領域）ごとに分ける。

- 領域: 入力の文字の言語（`moe_integration.LANGS` の子音・母音で決まる）。区切り（空白・"|"・改行）は直前の
  領域の文字の領域に属する（因果的。`moe_integration.integration_corpus` の注釈と同じ割り当て）。
- モジュール m は自分の埋め込み・GRU・出力層を持つ。GRU の入力は、領域 m の位置では自分の埋め込み、
  それ以外では学習される「不在」のベクトル（他の領域の文字は届かない）。モジュール間の直接の結合はない。
- 専門家（と経路選択）だけが全モジュールの状態（連結 [h_1, …, h_K]）を読み、出力をいまの領域のモジュールの
  状態に足す: h' = h_r + g·y_e([h_1..h_K])。予測はいまの領域の出力層 head_r(h')。
  したがって、領域をまたぐ情報は専門家を通るしかない（専門家の出力は再帰に戻らない）。
- cut_remote: 専門家の入力のうち、いまの領域以外のモジュールの部分を 0 にする（経路選択は変えない）。
  専門家ごとの bool [E] で、一部の専門家（ハブなど）の領域をまたぐ読み取りだけを切れる。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe_homeo_band import BandHomeostaticLM
from .moe_integration import LANGS


def region_table(stoi: dict[str, int], langs=LANGS) -> list[int]:
    """語彙の番号 → 領域（言語）の番号。区切りなどどの言語の子音でもない文字は −1。

    母音は族で共有なので、言語は子音で決める。語は CVCV で子音から始まるので、母音の位置は直前の子音の
    領域に属させる（−1 として直前の領域を引き継ぐ）。
    """
    tab = [-1] * len(stoi)
    for m, (cons, _) in enumerate(langs):
        for ch in cons:
            if ch in stoi:
                tab[stoi[ch]] = m
    return tab


def sticky_regions(x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """[B,T] の各位置の領域。表で −1 の文字は直前の位置の領域を引き継ぐ（窓の先頭で未定なら 0）。"""
    r = table[x]
    out = torch.empty_like(r)
    prev = torch.zeros(x.shape[0], dtype=r.dtype, device=x.device)
    for t in range(x.shape[1]):
        prev = torch.where(r[:, t] >= 0, r[:, t], prev)
        out[:, t] = prev
    return out


class ModularRoutedLM(BandHomeostaticLM):
    """領域ごとのモジュール（埋め込み・GRU・出力層）と、全モジュールを読む専門家（1層・トップ1）。

    恒常性（選択専用の閾値・目標域）・補助損失・除去の扱いは `BandHomeostaticLM` と同じ。n_layers は 1 のみ。
    d は使わない（`run_moe_scale.run_one` との互換のため受ける）。モジュールの幅は d_mod。
    """

    def __init__(self, vocab_size: int, d: int = 48, n_experts: int = 8, d_ff: int = 64, n_layers: int = 1,
                 kind: str = "select", eta: float = 0.0, alpha: float = 0.05, r_lo: float = 1.0,
                 r_hi: float = 1.0, r_floor: float = 0.01, region_of_token: list[int] | None = None,
                 d_mod: int = 24) -> None:
        nn.Module.__init__(self)
        if kind not in ("plain", "gate", "select"):
            raise ValueError(kind)
        if n_layers != 1:
            raise ValueError(n_layers)
        if not 0 < r_lo <= r_hi:
            raise ValueError((r_lo, r_hi))
        if region_of_token is None or len(region_of_token) != vocab_size:
            raise ValueError("region_of_token")
        self.n_experts, self.n_layers, self.kind = n_experts, 1, kind
        self.eta, self.alpha = eta, alpha
        self.r_lo, self.r_hi, self.r_floor = r_lo, r_hi, r_floor
        self.n_modules = max(region_of_token) + 1
        self.d_mod = d_mod
        K = self.n_modules
        self.register_buffer("region_tab", torch.as_tensor(region_of_token, dtype=torch.long))
        self.embeds = nn.ModuleList(nn.Embedding(vocab_size, d_mod) for _ in range(K))
        self.absent = nn.Parameter(torch.zeros(K, d_mod))
        self.rnns = nn.ModuleList(nn.GRU(d_mod, d_mod, batch_first=True) for _ in range(K))
        self.routers = nn.ModuleList([nn.Linear(K * d_mod, n_experts)])
        self.w1 = nn.ParameterList([nn.Parameter(torch.randn(n_experts, K * d_mod, d_ff) * (K * d_mod) ** -0.5)])
        self.w2 = nn.ParameterList([nn.Parameter(torch.randn(n_experts, d_ff, d_mod) * d_ff ** -0.5)])
        self.head_w = nn.Parameter(torch.randn(K, d_mod, vocab_size) * d_mod ** -0.5)
        self.head_b = nn.Parameter(torch.zeros(K, vocab_size))
        self.register_buffer("threshold", torch.zeros(1, n_experts))
        self.register_buffer("usage_ema", torch.full((1, n_experts), 1.0 / n_experts))

    def module_states(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(領域 [B,T], モジュールの状態 [B,T,K,d_mod])。モジュール m には領域 m の位置の文字だけが届く。"""
        reg = sticky_regions(x, self.region_tab)
        hs = []
        for m in range(self.n_modules):
            here = (reg == m).unsqueeze(-1)
            inp = torch.where(here, self.embeds[m](x), self.absent[m].expand(*x.shape, -1))
            hs.append(self.rnns[m](inp)[0])
        return reg, torch.stack(hs, dim=2)

    def run(self, x: torch.Tensor, disabled: torch.Tensor | None = None, mode: str = "zero",
            gate_one: bool = False, use_threshold: bool = True, cut_remote=None) -> dict:
        """`ScaledRoutedCharLM.run` と同じ返り値（L=1）。cut_remote は True（全専門家）か bool [E]。"""
        if mode not in ("zero", "reroute"):
            raise ValueError(mode)
        reg, H = self.module_states(x)
        B, T, K, D = H.shape
        own = F.one_hot(reg, K).to(H.dtype).unsqueeze(-1)  # [B,T,K,1]
        h = (H * own).sum(2)  # いまの領域のモジュールの状態
        z = H.reshape(B, T, K * D)
        raw = raw0 = self.routers[0](z)
        th = self.threshold[0] if use_threshold else torch.zeros_like(self.threshold[0])
        if self.kind == "gate":
            raw = raw - th
        if disabled is not None and mode == "reroute":
            raw = raw.masked_fill(disabled[0].to(torch.bool), float("-inf"))
        probs = F.softmax(raw, dim=-1)
        choice = (raw - th).argmax(-1) if self.kind == "select" else probs.argmax(-1)
        gate = probs.gather(-1, choice.unsqueeze(-1))
        z_in = z
        if cut_remote is not None:
            cut = torch.ones(self.n_experts, dtype=torch.bool) if cut_remote is True else cut_remote.to(torch.bool)
            keep = torch.where(cut[choice].unsqueeze(-1).unsqueeze(-1), own, torch.ones_like(own))
            z_in = (H * keep).reshape(B, T, K * D)
        y = torch.einsum("btd,btdf->btf", z_in, self.w1[0][choice]).relu()
        y = torch.einsum("btf,btfd->btd", y, self.w2[0][choice])
        if disabled is not None and mode == "zero":
            y = y * (~disabled[0].to(torch.bool))[choice].unsqueeze(-1).to(y.dtype)
        gy = (torch.ones_like(gate) if gate_one else gate) * y
        h = h + gy
        logits = torch.einsum("btd,btdv->btv", h, self.head_w[reg]) + self.head_b[reg]
        frac = F.one_hot(choice, self.n_experts).float().mean(dim=(0, 1))
        balance = self.n_experts * (frac * probs.mean(dim=(0, 1))).sum()
        return {"logits": logits, "choice": choice.unsqueeze(0), "gate": gate.squeeze(-1).unsqueeze(0),
                "contrib": gy.norm(dim=-1).unsqueeze(0), "router": raw0.unsqueeze(0), "balance": balance,
                "region": reg}
