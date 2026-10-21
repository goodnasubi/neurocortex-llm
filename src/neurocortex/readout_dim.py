"""読み出しの出力に効く次元 K を操作する継続学習（12.6.251〜252節）。

12.6.248節では、共有の隠れ層を広げると古いタスクの表現の動きは増えたが、その大部分は2クラスの読み出しの零空間
（W − 1 次元）に入り、マージンを動かす割合は増えなかった。これが「零空間の次元が広いこと」（冗長な符号、
Rule et al. 2020）による受け身の薄まりなら、幅 W を変えずに古いタスクの読み出しの出力に効く次元 K を増やすと、
出力に効く部分空間に入る動きの割合 f_pot は K/W に比例して増えるはずである（運動野の output-potent / output-null
空間、Kaufman et al. 2014）。

課題: 入力は ±1 の n ビット。古いタスクは K 個の出力ビットの符号で、ビット k は異なる入力の組の XOR
（ビット 0 はどの K でも同じ組）。新しいタスクは別の組の XOR を1ビット。共有の隠れ層（fc1 + tanh、幅 W）の上に
古いタスクと新しいタスクの読み出しを別々に置く（頭を分けた継続学習）。古いタスクを学習した後、新しいタスクを
  naive  … 新しいタスクだけで学習する（古い読み出しは変わらない）
  replay … 古いタスクを学習した直後に保存した生サンプル（海馬の痕跡）を毎ステップ 32 個混ぜて学習する
  replay_lo … 同じリプレイを毎ステップ 8 個（弱いリプレイ）
で学習し、古いタスクの同じ探査入力の隠れ層の活動の動き Δh を、直前の古い読み出しの K 本の重みの張る部分空間と
その零空間（W − K 次元）に分ける（drift_stats）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

RULES = ("naive", "replay", "replay_lo")
REPLAY_BATCH = {"replay": 32, "replay_lo": 8}


@dataclass(frozen=True)
class CodeSpec:
    """古いタスクの出力ビットの組（old_pairs[:K] を使う）と新しいタスクの組。"""

    n_inputs: int
    old_pairs: tuple[tuple[int, int], ...]
    new_pair: tuple[int, int]


def draw_spec(n_inputs: int, k_max: int, generator: torch.Generator) -> CodeSpec:
    """重ならない入力の組を k_max + 1 個引く（最初の k_max 個が古いタスク、最後が新しいタスク）。"""
    pairs = [(a, b) for a in range(n_inputs) for b in range(a + 1, n_inputs)]
    if k_max + 1 > len(pairs):
        raise ValueError("組が足りない")
    idx = torch.randperm(len(pairs), generator=generator)[: k_max + 1].tolist()
    chosen = [pairs[i] for i in idx]
    return CodeSpec(n_inputs, tuple(chosen[:k_max]), chosen[k_max])


def xor_labels(bits: torch.Tensor, pairs) -> torch.Tensor:
    """[B, len(pairs)] の 0/1 の正解。"""
    return torch.stack([((bits[:, a] > 0) ^ (bits[:, b] > 0)) for a, b in pairs], dim=1).float()


def sample_bits(n: int, n_inputs: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randint(0, 2, (n, n_inputs), generator=generator).float() * 2 - 1


def drift_stats(h0: torch.Tensor, h1: torch.Tensor, head0: tuple, head1: tuple, y: torch.Tensor) -> dict:
    """古いタスクの表現の動き Δh = h1 − h0（同じ探査入力 [N, W]）を、直前の古い読み出しの重み D0 [K, W] の行空間
    （出力に効く K 次元）と零空間に分ける。head0・head1 は直前・最後の古い読み出し (weight [K, W], bias [K])、
    y は探査入力の古いタスクの正解 [N, K]。ビットごとの値はビットの平均にする。
      d_tot    … mean ||Δh||²
      f_pot    … mean ||P Δh||² / d_tot（P は D0 の行空間への射影。等方な乱雑な方向なら K/W）
      e_iso    … W · f_pot / K（等方な方向に対する濃縮。< 1 なら零空間に寄る）
      f_pot1   … ビット 0 の方向だけの割合（どの K でも同じ関数のビット）
      pot_rel  … ビットごとの rms(d0_k·Δh) / rms(m0_k)（m0_k = d0_k·h0 + c0_k）の平均
      harm_rep … ビットごとの mean(s_k · d0_k·Δh) / mean|m0_k| の平均（s = 2y − 1。負なら古いタスクを損なう向き）
      harm_ro  … ビットごとの mean(s_k · ((d1_k − d0_k)·h1 + Δc_k)) / mean|m0_k| の平均（読み出しの変化の寄与）
      comp     … −Σ(表現の項)(読み出しの項) / Σ(表現の項)²（全ビット合わせて。読み出しが打ち消す割合）
      acc0/acc1 … 探査入力でのビットごとの正確度の平均（直前・最後）"""
    w0, b0 = head0
    w1, b1 = head1
    dh = h1 - h0
    n, width = dh.shape
    k = w0.shape[0]
    d_tot = float((dh ** 2).sum(-1).mean())
    q, _ = torch.linalg.qr(w0.T)  # [W, K] の正規直交基底（K ≤ W）
    f_pot = float(((dh @ q) ** 2).sum(-1).mean()) / max(d_tot, 1e-12)
    u = w0[0] / w0[0].norm().clamp_min(1e-12)
    m0 = h0 @ w0.T + b0
    rep = dh @ w0.T
    ro = h1 @ (w1 - w0).T + (b1 - b0)
    s = 2.0 * y - 1.0
    am0 = m0.abs().mean(0).clamp_min(1e-12)
    return {"d_tot": d_tot, "f_pot": f_pot, "e_iso": width * f_pot / k,
            "f_pot1": float(((dh @ u) ** 2).mean()) / max(d_tot, 1e-12),
            "pot_rel": float(((rep ** 2).mean(0).sqrt() / (m0 ** 2).mean(0).sqrt().clamp_min(1e-12)).mean()),
            "harm_rep": float(((s * rep).mean(0) / am0).mean()), "harm_ro": float(((s * ro).mean(0) / am0).mean()),
            "comp": float(-(rep * ro).sum() / (rep ** 2).sum().clamp_min(1e-12)),
            "acc0": float(((m0 > 0).float() == y).float().mean()),
            "acc1": float(((h1 @ w1.T + b1 > 0).float() == y).float().mean())}


class Net(nn.Module):
    def __init__(self, n_inputs: int, width: int, k: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, width)
        self.old = nn.Linear(width, k)
        self.new = nn.Linear(width, 1)

    def hidden(self, bits: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc1(bits))


def _acc(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ビットごとの正確度 [K]。"""
    return ((logits > 0).float() == y).float().mean(0)


def run(spec: CodeSpec, k: int, width: int, rule: str, seed: int, pretrain_steps: int = 1500, steps: int = 500,
        batch: int = 64, lr: float = 0.01, weight_decay: float = 1e-3, buffer_size: int = 50,
        probe_n: int = 256, eval_n: int = 2000) -> dict:
    """1回の学習。乱数は seed から作り、K・W・規則によらず古いタスクの入力列・新しいタスクの入力列・探査入力・
    評価入力は同じにする（K が違っても入力は同じで、正解の列の数だけが違う）。"""
    if rule not in RULES:
        raise ValueError(rule)
    pairs = spec.old_pairs[:k]
    g_init = torch.Generator().manual_seed(seed * 1000 + width)
    net = Net(spec.n_inputs, width, k)
    with torch.no_grad():  # 初期値は K によらず同じ（old の行は k_max 行から最初の K 行）
        for p, src in ((net.fc1.weight, (width, spec.n_inputs)), (net.fc1.bias, (width,))):
            p.copy_((torch.rand(src, generator=g_init) * 2 - 1) / spec.n_inputs ** 0.5)
        k_max = len(spec.old_pairs)
        ow = (torch.rand((k_max, width), generator=g_init) * 2 - 1) / width ** 0.5
        ob = (torch.rand((k_max,), generator=g_init) * 2 - 1) / width ** 0.5
        net.old.weight.copy_(ow[:k])
        net.old.bias.copy_(ob[:k])
        net.new.weight.copy_((torch.rand((1, width), generator=g_init) * 2 - 1) / width ** 0.5)
        net.new.bias.copy_((torch.rand((1,), generator=g_init) * 2 - 1) / width ** 0.5)
    g_old = torch.Generator().manual_seed(seed * 1000 + 1)
    g_new = torch.Generator().manual_seed(seed * 1000 + 2)
    g_eval = torch.Generator().manual_seed(seed * 1000 + 3)
    probe = sample_bits(probe_n, spec.n_inputs, g_eval)
    ev = sample_bits(eval_n, spec.n_inputs, g_eval)
    ev_old, ev_new = xor_labels(ev, pairs), xor_labels(ev, (spec.new_pair,))
    y_probe = xor_labels(probe, pairs)

    opt = torch.optim.Adam(list(net.fc1.parameters()) + list(net.old.parameters()), lr=lr, weight_decay=weight_decay)
    for _ in range(pretrain_steps):
        x = sample_bits(batch, spec.n_inputs, g_old)
        loss = F.binary_cross_entropy_with_logits(net.old(net.hidden(x)), xor_labels(x, pairs), reduction="none")
        loss = loss.sum(1).mean()  # ビットごとの損失の和（ビットあたりの学習の強さを K によらず同じにする）
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    buf = sample_bits(buffer_size, spec.n_inputs, g_old)
    buf_y = xor_labels(buf, pairs)
    with torch.no_grad():
        h0 = net.hidden(probe)
        head0 = (net.old.weight.clone(), net.old.bias.clone())
        pre = _acc(net.old(net.hidden(ev)), ev_old)

    params = list(net.fc1.parameters()) + list(net.new.parameters())
    replay_batch = REPLAY_BATCH.get(rule, 0)
    if replay_batch:
        params += list(net.old.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    g_rep = torch.Generator().manual_seed(seed * 1000 + 4)
    for _ in range(steps):
        x = sample_bits(batch, spec.n_inputs, g_new)
        loss = F.binary_cross_entropy_with_logits(net.new(net.hidden(x)), xor_labels(x, (spec.new_pair,)))
        if replay_batch:
            i = torch.randint(0, buffer_size, (replay_batch,), generator=g_rep)
            lr_loss = F.binary_cross_entropy_with_logits(net.old(net.hidden(buf[i])), buf_y[i], reduction="none")
            loss = loss + lr_loss.sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        h1 = net.hidden(probe)
        head1 = (net.old.weight.clone(), net.old.bias.clone())
        end = _acc(net.old(net.hidden(ev)), ev_old)
        new_acc = float(_acc(net.new(net.hidden(ev)), ev_new).mean())
        st = drift_stats(h0, h1, head0, head1, y_probe)
    return {**st, "pre": float(pre.mean()), "end": float(end.mean()), "old_change": float((end - pre).mean()),
            "pre_b0": float(pre[0]), "change_b0": float(end[0] - pre[0]), "new": new_acc}
