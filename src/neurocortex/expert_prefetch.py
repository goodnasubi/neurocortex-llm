"""MoE 専門家の予測的プリフェッチと小脳順モデルの結合（12.8.2節、12.6.116節）。

12.8.5節のリスク「予測的中率と実効速度向上を別々に計測すべき」に従い、
(a) 次ステップの専門家の予測的中率と、(b) ディスク読み込みを模した遅延の下での
実効速度（壁時計時間・待ち回数・無駄 I/O）を別々に測るための部品を置く。

- `RoutedCharLM`: トップ1ルーティングの最小 MoE 文字 LM（ルーティング列を作るためだけのもの）
- 予測器: `LastPredictor`・`FreqPredictor`・`MarkovPredictor`・`CerebellumPredictor`
  （`CerebellumPredictor` は `cerebellum.ForwardModel` を専門家ごとにそのまま使う）
- `ExpertDiskStore`・`simulate_prefetch`: 専門家の重みをファイルに置き、固定遅延付きで
  読み込む常駐管理（LRU）と、計算と並行する先読みの模擬
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cerebellum import ForwardModel


# --- ルーティング列を作る最小 MoE 文字 LM ------------------------------------

class RoutedCharLM(nn.Module):
    """GRU 隠れ状態 → ルータ（トップ1）→ 専門家 MLP → 次文字の予測。"""

    def __init__(self, vocab_size: int, d: int = 48, n_experts: int = 8, d_ff: int = 64) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.embed = nn.Embedding(vocab_size, d)
        self.rnn = nn.GRU(d, d, batch_first=True)
        self.router = nn.Linear(d, n_experts)
        self.w1 = nn.Parameter(torch.randn(n_experts, d, d_ff) * d ** -0.5)
        self.w2 = nn.Parameter(torch.randn(n_experts, d_ff, d) * d_ff ** -0.5)
        self.head = nn.Linear(d, vocab_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """x [B, T] → (logits [B,T,V], 負荷分散損失, 選択 [B,T], 隠れ状態 [B,T,d])。"""
        h, _ = self.rnn(self.embed(x))
        probs = F.softmax(self.router(h), dim=-1)  # [B,T,E]
        choice = probs.argmax(-1)
        gate = probs.gather(-1, choice.unsqueeze(-1))  # [B,T,1]
        w1 = self.w1[choice]  # [B,T,d,d_ff]
        w2 = self.w2[choice]
        y = torch.einsum("btd,btdf->btf", h, w1).relu()
        y = torch.einsum("btf,btfd->btd", y, w2)
        out = h + gate * y
        # Switch Transformer 型の負荷分散損失: E * Σ_e (選択割合_e × 平均確率_e)
        frac = F.one_hot(choice, self.n_experts).float().mean(dim=(0, 1))
        balance = self.n_experts * (frac * probs.mean(dim=(0, 1))).sum()
        return self.head(out), balance, choice, h


def train_routed_lm(model: RoutedCharLM, train_data: torch.Tensor, steps: int, batch_size: int,
                    seq_len: int, lr: float, balance_coef: float, generator: torch.Generator) -> float:
    """学習して最後の CE を返す。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = float("nan")
    for _ in range(steps):
        idx = torch.randint(0, len(train_data) - seq_len - 1, (batch_size,), generator=generator)
        x = torch.stack([train_data[i:i + seq_len] for i in idx.tolist()])
        y = torch.stack([train_data[i + 1:i + seq_len + 1] for i in idx.tolist()])
        logits, balance, _, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        ce = float(loss.detach())
        opt.zero_grad(set_to_none=True)
        (loss + balance_coef * balance).backward()
        opt.step()
    return ce


@torch.no_grad()
def routing_trace(model: RoutedCharLM, data: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """1本の列 data [T] に対する専門家選択列 [T] と隠れ状態 [T, d]。"""
    _, _, choice, h = model(data.unsqueeze(0))
    return choice[0].numpy().astype(np.int64), h[0].numpy().astype(np.float32)


# --- 予測器（オンライン・因果的） ---------------------------------------------
# 共通の約束: rank(t) は時刻 t までの情報（e_t, h_t）で e_{t+1} の順位（専門家添字の
# 並び、先頭が最有力）を返す。observe(t, e_next) で e_{t+1} を知らされて更新する。

class _Base:
    def __init__(self, n_experts: int) -> None:
        self.n_experts = n_experts
        self.counts = np.zeros(n_experts)

    def _by_freq(self) -> list[int]:
        # 同数は添字の小さい順（決定的）
        return sorted(range(self.n_experts), key=lambda e: (-self.counts[e], e))

    def observe_count(self, e: int) -> None:
        self.counts[e] += 1


class LastPredictor(_Base):
    """直前と同じ専門家。2位以降は頻度順。"""

    def rank(self, e_t: int, h_t: np.ndarray) -> list[int]:
        return [e_t] + [e for e in self._by_freq() if e != e_t]

    def observe(self, e_t: int, h_t: np.ndarray, e_next: int) -> None:
        self.observe_count(e_next)


class FreqPredictor(_Base):
    """これまでの頻度が最大の専門家。"""

    def rank(self, e_t: int, h_t: np.ndarray) -> list[int]:
        return self._by_freq()

    def prob(self, e_t: int, h_t: np.ndarray) -> np.ndarray:
        return self.counts / max(self.counts.sum(), 1.0)

    def observe(self, e_t: int, h_t: np.ndarray, e_next: int) -> None:
        self.observe_count(e_next)


class MarkovPredictor(_Base):
    """1次マルコフ遷移の頻度（ステップ34の予測器と同型）。未観測の遷移は頻度順で補う。"""

    def __init__(self, n_experts: int) -> None:
        super().__init__(n_experts)
        self.trans = np.zeros((n_experts, n_experts))

    def rank(self, e_t: int, h_t: np.ndarray) -> list[int]:
        row = self.trans[e_t]
        return sorted(range(self.n_experts), key=lambda e: (-row[e], -self.counts[e], e))

    def prob(self, e_t: int, h_t: np.ndarray) -> np.ndarray:
        """遷移の経験確率 P(e_{t+1} | e_t)。未観測の行は 0（先読みしない）。"""
        row = self.trans[e_t]
        return row / max(row.sum(), 1.0)

    def observe(self, e_t: int, h_t: np.ndarray, e_next: int) -> None:
        self.trans[e_t, e_next] += 1
        self.observe_count(e_next)


class CerebellumPredictor:
    """専門家ごとの `ForwardModel`（小脳の順モデル）で「次にその専門家が選ばれるか」を予測する。

    入力は文脈 h_t（苔状線維入力に相当）と現在の選択 onehot(e_t)（遠心性コピーに相当）を
    連結して単位ノルムに正規化したもの。教師は 0/1、誤差（登上線維に相当）でデルタ則更新する。
    `ForwardModel` 本体には手を入れず、列全体で同じインスタンスを使い続ける（持続性）。
    `use_context=False` は onehot(e_t) だけを入力にする切除条件。
    """

    def __init__(self, n_experts: int, d_context: int, lr: float, use_context: bool = True) -> None:
        self.n_experts = n_experts
        self.use_context = use_context
        self.lr = lr
        dim = (d_context if use_context else 0) + n_experts
        self.models = [ForwardModel(dim) for _ in range(n_experts)]

    def _features(self, e_t: int, h_t: np.ndarray) -> torch.Tensor:
        onehot = torch.zeros(self.n_experts)
        onehot[e_t] = 1.0
        if self.use_context:
            ctx = torch.from_numpy(h_t)
            ctx = ctx / ctx.norm().clamp_min(1e-6)
            x = torch.cat([ctx, onehot])
        else:
            x = onehot
        return x / x.norm()

    def rank(self, e_t: int, h_t: np.ndarray) -> list[int]:
        x = self._features(e_t, h_t)
        scores = [float(m.predict(x)) for m in self.models]
        return sorted(range(self.n_experts), key=lambda e: (-scores[e], e))

    def observe(self, e_t: int, h_t: np.ndarray, e_next: int) -> None:
        x = self._features(e_t, h_t)
        for e, m in enumerate(self.models):
            m.update(x, 1.0 if e == e_next else 0.0, self.lr)


class FastCerebellumPredictor:
    """`CerebellumPredictor` と数学的に同一の計算を、専門家ぶんの `ForwardModel` を1つの行列に
    まとめて行う実装（速度計測用）。`ForwardModel` を1個ずつ呼ぶと torch の小さな演算の
    呼び出し負荷が1ステップ約 0.65 ms になり、計算そのものではなく実装の負荷を測ってしまうため
    （12.6.116節の修正2）。一致は `tests/test_expert_prefetch.py` で確かめる。
    """

    def __init__(self, n_experts: int, d_context: int, lr: float, use_context: bool = True) -> None:
        self.n_experts = n_experts
        self.use_context = use_context
        self.lr = lr
        dim = (d_context if use_context else 0) + n_experts
        self.weight = np.zeros((n_experts, dim), dtype=np.float32)  # 各行が1個の ForwardModel.weight
        self.bias = np.zeros(n_experts, dtype=np.float64)  # ForwardModel.bias（Python の float）
        self._x_t: int | None = None
        self._x: np.ndarray | None = None

    def _features(self, e_t: int, h_t: np.ndarray) -> np.ndarray:
        onehot = np.zeros(self.n_experts, dtype=np.float32)
        onehot[e_t] = 1.0
        if self.use_context:
            ctx = h_t / max(float(np.linalg.norm(h_t)), 1e-6)
            x = np.concatenate([ctx.astype(np.float32), onehot])
        else:
            x = onehot
        return x / np.float32(np.linalg.norm(x))

    def _scores(self, x: np.ndarray) -> np.ndarray:
        return (self.weight @ x).astype(np.float64) + self.bias

    def rank(self, e_t: int, h_t: np.ndarray) -> list[int]:
        x = self._features(e_t, h_t)
        return np.lexsort((np.arange(self.n_experts), -self._scores(x))).tolist()

    def prob(self, e_t: int, h_t: np.ndarray) -> np.ndarray:
        """0/1 の教師への回帰出力。選ばれる確率の推定値として扱う。"""
        return self._scores(self._features(e_t, h_t))

    def observe(self, e_t: int, h_t: np.ndarray, e_next: int) -> None:
        x = self._features(e_t, h_t)
        y = np.zeros(self.n_experts)
        y[e_next] = 1.0
        err = y - self._scores(x)
        self.weight += (self.lr * err).astype(np.float32)[:, None] * x[None, :]
        self.bias += self.lr * err


def make_predictor(name: str, n_experts: int, d_context: int, cb_lr: float):
    if name == "last":
        return LastPredictor(n_experts)
    if name == "freq":
        return FreqPredictor(n_experts)
    if name == "markov":
        return MarkovPredictor(n_experts)
    if name == "cerebellum":
        return FastCerebellumPredictor(n_experts, d_context, cb_lr, use_context=True)
    if name == "cerebellum_onehot":
        return FastCerebellumPredictor(n_experts, d_context, cb_lr, use_context=False)
    raise ValueError(name)


def hit_rates(choices: np.ndarray, hidden: np.ndarray, predictor, ks: tuple[int, ...] = (1, 2)) -> dict:
    """(a) 予測的中率。各 t で e_{t+1} が順位の上位 k に入った割合。"""
    hits = {k: 0 for k in ks}
    n = len(choices) - 1
    for t in range(n):
        r = predictor.rank(int(choices[t]), hidden[t])
        e_next = int(choices[t + 1])
        for k in ks:
            hits[k] += e_next in r[:k]
        predictor.observe(int(choices[t]), hidden[t], e_next)
    return {f"top{k}": hits[k] / n for k in ks}


# --- (b) ディスク読み込みを模した常駐管理と先読み ------------------------------

class ExpertDiskStore:
    """専門家の重みを1個ずつ .npy ファイルに置き、固定遅延付きで読む。常駐は LRU で `capacity` 個。"""

    def __init__(self, weights: list[np.ndarray], persist_dir: str | Path, capacity: int,
                 io_ms: float) -> None:
        self.dir = Path(persist_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self.io_s = io_ms / 1000.0
        self.nbytes = []
        for e, w in enumerate(weights):
            np.save(self.dir / f"expert_{e}.npy", w)
            self.nbytes.append(int(w.nbytes))
        self.resident: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self.prefetched_unused: set[int] = set()  # 先読みしたがまだ使われていない専門家
        self.lock = threading.Lock()
        self.io_lock = threading.Lock()  # ディスクは1本の読み込み経路（先読みと同期読み込みが帯域を取り合う）
        self.demand_loads = 0
        self.prefetch_loads = 0
        self.wasted_loads = 0
        self.wasted_bytes = 0

    def _read(self, e: int) -> np.ndarray:
        with self.io_lock:
            time.sleep(self.io_s)  # ディスクの遅延を模す
            return np.load(self.dir / f"expert_{e}.npy")

    def _insert(self, e: int, w: np.ndarray, prefetched: bool) -> None:
        with self.lock:
            self.resident[e] = w
            self.resident.move_to_end(e)
            if prefetched:
                self.prefetched_unused.add(e)
            while len(self.resident) > self.capacity:
                old, _ = self.resident.popitem(last=False)
                if old in self.prefetched_unused:
                    self.prefetched_unused.discard(old)
                    self.wasted_loads += 1
                    self.wasted_bytes += self.nbytes[old]

    def is_resident(self, e: int) -> bool:
        with self.lock:
            return e in self.resident

    def prefetch_load(self, e: int) -> None:
        w = self._read(e)
        self.prefetch_loads += 1
        self._insert(e, w, prefetched=True)

    def use(self, e: int) -> np.ndarray:
        """常駐していれば使い、なければ同期読み込み（待ち）。"""
        with self.lock:
            if e in self.resident:
                self.resident.move_to_end(e)
                self.prefetched_unused.discard(e)
                return self.resident[e]
        w = self._read(e)
        self.demand_loads += 1
        self._insert(e, w, prefetched=False)
        with self.lock:
            self.prefetched_unused.discard(e)
        return w

    def finish(self) -> None:
        """列の終わりまで使われなかった先読みも無駄 I/O に数える。"""
        with self.lock:
            for e in self.prefetched_unused:
                self.wasted_loads += 1
                self.wasted_bytes += self.nbytes[e]
            self.prefetched_unused.clear()


@dataclass
class PrefetchResult:
    wall_s: float
    stalls: int  # 必要な専門家が常駐しておらず待った回数（先読み中の待ちを含む）
    stall_s: float  # 待ちの合計時間
    demand_loads: int
    prefetch_loads: int
    wasted_loads: int
    wasted_bytes: int
    predictor_s: float


def simulate_prefetch(choices: np.ndarray, hidden: np.ndarray, store: ExpertDiskStore, predictor,
                      k_pf: int, compute_ms: float, oracle: bool = False,
                      gate: float | None = None) -> PrefetchResult:
    """(b) 実効速度。predictor=None かつ oracle=False がプリフェッチなし。

    各ステップ t: 専門家 e_t を使う（常駐していなければ待つ）→ 予測器で e_{t+1} の順位を出し、
    上位 k_pf 個のうち非常駐のものを別スレッドで読み始める → 計算（compute_ms）→ 予測器を更新。
    予測器の計算時間は壁時計時間に含まれる。読み込みは先読み・同期読み込みとも1本の経路を
    取り合う（帯域の並列性は仮定しない）ので、外れた先読みは後の同期読み込みを遅らせうる。
    `gate` を与えると、予測器の `prob` が `gate` 以上の候補だけを先読みする（確信度ゲート）。
    """
    compute_s = compute_ms / 1000.0
    pending: dict[int, Future] = {}
    stalls = 0
    stall_s = 0.0
    predictor_s = 0.0
    n = len(choices)
    with ThreadPoolExecutor(max_workers=1) as pool:
        t0 = time.perf_counter()
        for t in range(n):
            e = int(choices[t])
            s0 = time.perf_counter()
            fut = pending.pop(e, None)
            if fut is not None and not fut.done():
                stalls += 1
                fut.result()
            elif fut is None and not store.is_resident(e):
                stalls += 1
            w = store.use(e)
            stall_s += time.perf_counter() - s0
            if t + 1 < n and (predictor is not None or oracle):
                p0 = time.perf_counter()
                if oracle:
                    cands = [int(choices[t + 1])]
                elif gate is not None:
                    pr = predictor.prob(e, hidden[t])
                    order = np.lexsort((np.arange(len(pr)), -pr))[:k_pf]
                    cands = [int(c) for c in order if pr[c] >= gate]
                else:
                    cands = predictor.rank(e, hidden[t])
                predictor_s += time.perf_counter() - p0
                for c in cands[:k_pf]:
                    if c == e or c in pending or store.is_resident(c):
                        continue  # 「直前と同じ」の予測は常駐済みなので何も読まない
                    pending[c] = pool.submit(store.prefetch_load, c)
            # 計算（実際の行列積＋固定時間）。先読みはこの間に進む。
            c0 = time.perf_counter()
            _ = float(np.tanh(w[:4] @ w[:4].T).sum())
            rest = compute_s - (time.perf_counter() - c0)
            if rest > 0:
                time.sleep(rest)
            if predictor is not None and t + 1 < n:
                p0 = time.perf_counter()
                predictor.observe(e, hidden[t], int(choices[t + 1]))
                predictor_s += time.perf_counter() - p0
            # 終わった先読みの記録を片付ける
            for c in [c for c, f in pending.items() if f.done()]:
                pending.pop(c)
        for f in pending.values():
            f.result()
        wall = time.perf_counter() - t0
    store.finish()
    return PrefetchResult(wall, stalls, stall_s, store.demand_loads, store.prefetch_loads,
                          store.wasted_loads, store.wasted_bytes, predictor_s)
