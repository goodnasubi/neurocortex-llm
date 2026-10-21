"""小脳の順モデルを先読みの帰結（遅れて届く登上線維の誤差）で学習させる（12.6.211節）。

12.6.119節の予測器は「次の時刻の専門家」の正解ラベルで学ぶ。本モジュールの `OutcomeCerebellum` は
同じ入力・重みの形・デルタ則で、教師だけを先読みの帰結に替える:

- 先読みした c: 読み込みが終わってから使われた（間に合った）→ y = 1、読み込み中に使われて待った → y = y_late、
  使われないまま追い出された → y = 0。帰結が確定した時点で、発行の時刻の入力 x（適格性のタグ）に対して更新する。
- 先読みしなかった対象の c: 窓 W の中で使われたら y = 1、窓の終わりまで使われなければ y = 0。
- 常駐・読み込み中・直前と同じ候補は行動の余地がないのでタグを残さない。

誤差 δ = y − v_c(x) が登上線維の信号（δ < 0 が LTD、δ > 0 が LTP）にあたる（Ito 2001; Raymond & Medina 2018）。
`shuffle=True` は帰結を長さ `shuffle_buf` のバッファで並べ替えた対照（値の分布と更新の回数は同じ、
帰結と状態の対応だけを壊す）。既存の `expert_prefetch*.py` は変えない。
"""

from __future__ import annotations

import random
import threading
from collections import Counter, OrderedDict

import numpy as np

from .expert_prefetch_delayed import CFTaggedStore


class OutcomeCerebellum:
    """先読みの帰結で学ぶ小脳の順モデル。`simulate_prefetch_v2` の `predictor` と `adaptive_gate` の
    両方に同じインスタンスを渡す（`observe` は何もしない）。"""

    def __init__(self, n_experts: int, d_context: int, lr: float, theta: float, window: int = 2,
                 y_late: float = 0.5, shuffle: bool = False, shuffle_buf: int = 64, seed: int = 0) -> None:
        if window < 1:
            raise ValueError("window は 1 以上")
        self.n_experts = n_experts
        self.lr = lr
        self.theta = theta
        self.window = window
        self.y_late = y_late
        self.shuffle = shuffle
        self.shuffle_buf = shuffle_buf
        self.rng = random.Random(seed)
        self.buf: list[float] = []
        dim = d_context + n_experts
        self.weight = np.zeros((n_experts, dim), dtype=np.float64)
        self.bias = np.zeros(n_experts, dtype=np.float64)
        self.lock = threading.RLock()
        self.t = 0
        self._last_x: np.ndarray | None = None
        self.elig: dict[int, list[tuple[np.ndarray, int]]] = {}  # 見送った候補 → [(x, 残りの窓)]
        self.pending: dict[int, tuple[np.ndarray, int]] = {}  # 先読みを決めた候補 → (x, 決めた時刻)
        self.issued: dict[int, tuple[np.ndarray, int]] = {}  # 読み込みが始まった先読み → (x, 決めた時刻)
        self.counts: Counter = Counter()
        self.delay_used: Counter = Counter()

    # --- 予測器の面 -------------------------------------------------------------
    def _features(self, e_t: int, h_t: np.ndarray) -> np.ndarray:
        """12.6.119節の `FastCerebellumPredictor._features` と同じ入力（文脈と onehot の正規化した連結）。"""
        onehot = np.zeros(self.n_experts)
        onehot[e_t] = 1.0
        ctx = h_t / max(float(np.linalg.norm(h_t)), 1e-6)
        x = np.concatenate([ctx.astype(np.float64), onehot])
        return x / np.linalg.norm(x)

    def prob(self, e_t: int, h_t: np.ndarray) -> np.ndarray:
        """各候補を先読みする価値の推定 v_c(x_t)。直後の `decide` のために x_t を覚える。"""
        x = self._features(e_t, h_t)
        with self.lock:
            self._last_x = x
            return self.weight @ x + self.bias

    def rank(self, e_t: int, h_t: np.ndarray) -> list[int]:
        v = self.prob(e_t, h_t)
        return np.lexsort((np.arange(self.n_experts), -v)).tolist()

    def observe(self, e_t: int, h_t: np.ndarray, e_next: int) -> None:
        """正解ラベルは使わない（教師は帰結だけ）。"""

    # --- 学習（登上線維の誤差） ------------------------------------------------------
    def _learn(self, c: int, x: np.ndarray, y: float, kind: str) -> None:
        self.counts[kind] += 1
        if self.shuffle:
            self.buf.append(y)
            if len(self.buf) < self.shuffle_buf:
                return
            y = self.buf.pop(self.rng.randrange(len(self.buf)))
        err = y - (float(self.weight[c] @ x) + self.bias[c])
        self.weight[c] += self.lr * err * x
        self.bias[c] += self.lr * err

    # --- ゲートの面 -------------------------------------------------------------
    def decide(self, prob: np.ndarray, e_t: int, busy) -> int | None:
        """対象の候補（直前と同じでなく、常駐でも読み込み中でもない）にタグを残し、
        v の最も大きい対象を v ≥ θ なら先読みする。busy はゲートのロックの外で調べる
        （ストアはロックを持ったままゲートを呼ぶので、逆順に取らない）。"""
        order = [int(c) for c in np.lexsort((np.arange(len(prob)), -prob))]
        targets = [c for c in order if c != e_t and not busy(c)]
        with self.lock:
            x = self._last_x
            pick = None
            for c in targets:
                if pick is None and prob[c] >= self.theta:
                    pick = c
                    self.pending[c] = (x, self.t)
                else:
                    self.elig.setdefault(c, []).append((x, self.window))
            if pick is not None:
                self.counts["issued"] += 1
            return pick

    def feedback(self, e: int) -> None:
        """時刻ごとに使われた専門家 e で見送りのタグを解決し（y = 1）、窓を進める（窓の外は y = 0）。"""
        with self.lock:
            self.t += 1
            for x, _ in self.elig.pop(e, []):
                self._learn(e, x, 1.0, "skip_hit")
            for c in list(self.elig):
                keep = []
                for x, r in self.elig[c]:
                    if r - 1 <= 0:
                        self._learn(c, x, 0.0, "skip_miss")
                    else:
                        keep.append((x, r - 1))
                if keep:
                    self.elig[c] = keep
                else:
                    del self.elig[c]

    def on_prefetch_start(self, c: int) -> None:
        """ストアが c の先読みの読み込みを始めた（経路を取った）ときに呼ぶ。決めた時の印をその読み込みに付ける
        （決めた後に別の経路で読まれて先読みが空振りした印は、読み込みに付かない）。"""
        with self.lock:
            tag = self.pending.pop(c, None)
            if tag is not None:
                self.issued[c] = tag
                self.counts["started"] += 1

    def on_prefetch_used(self, c: int, late: bool = False) -> None:
        with self.lock:
            tag = self.issued.pop(c, None)
            if tag is None:
                return
            x, t0 = tag
            self.delay_used[self.t - t0] += 1
            self._learn(c, x, self.y_late if late else 1.0, "used_late" if late else "used")

    def on_prefetch_wasted(self, c: int) -> None:
        with self.lock:
            tag = self.issued.pop(c, None)
            if tag is None:
                return
            self._learn(c, tag[0], 0.0, "wasted")

    def reset_pending(self) -> None:
        """暖機の終わりに未解決のタグを捨てる（測定はストアが空から始まる）。"""
        with self.lock:
            self.elig.clear()
            self.pending.clear()
            self.issued.clear()

    def stats(self) -> dict:
        with self.lock:
            return {"counts": dict(self.counts), "delay_used": dict(sorted(self.delay_used.items()))}


class EligibleTopGate:
    """正解ラベルの予測器に `OutcomeCerebellum` と同じ選び方（常駐でない最上位）で固定閾値を掛けるゲート。
    学習はしない（`feedback` は何もしない）。選び方の違いだけで交換曲線が動く可能性を除く対照。"""

    def __init__(self, theta: float) -> None:
        self.theta = theta

    def decide(self, prob: np.ndarray, e_t: int, busy) -> int | None:
        for c in np.lexsort((np.arange(len(prob)), -prob)):
            c = int(c)
            if c == e_t or busy(c):
                continue
            return c if prob[c] >= self.theta else None
        return None

    def feedback(self, e: int) -> None:
        pass


class OutcomeTaggedStore(CFTaggedStore):
    """`CFTaggedStore` に、先読みした専門家が読み込み中に使われて待ったか（間に合わなかったか）を
    `gate.on_prefetch_used(c, late)` で知らせる処理を加えたもの。帰結はストアのロックを持ったまま知らせる
    （ロックを放した隙に同じ専門家の次の先読みの印が付き、帰結が別の印に当たるのを防ぐ）。"""

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
                    if self.gate is not None:
                        self.gate.on_prefetch_wasted(old)

    def try_prefetch(self, e: int) -> bool:
        ev = self._claim(e)
        if ev is None:
            return False
        if self.gate is not None and hasattr(self.gate, "on_prefetch_start"):
            self.gate.on_prefetch_start(e)
        self._load_claimed(e, ev, prefetched=True)
        return True

    def use_tracked(self, e: int) -> tuple[np.ndarray, bool]:
        stalled = False
        while True:
            with self.lock:
                if e in self.resident:
                    self.resident.move_to_end(e)
                    if e in self.prefetched_unused:
                        self.prefetched_unused.discard(e)
                        if self.gate is not None:
                            self.gate.on_prefetch_used(e, late=stalled)
                    return self.resident[e], stalled
                ev = self.inflight.get(e)
            stalled = True
            if ev is not None:
                ev.wait()
                continue
            ev = self._claim(e)
            if ev is None:
                continue
            w = self._load_claimed(e, ev, prefetched=False)
            with self.lock:
                self.prefetched_unused.discard(e)
            return w, stalled


def warm_outcome(model: OutcomeCerebellum, choices: np.ndarray, hidden: np.ndarray, capacity: int) -> None:
    """速度を測る前の列で `OutcomeCerebellum` を学習させる。I/O の時間は使わず、常駐（容量 `capacity` の
    LRU）・先読みの印・追い出しを `OutcomeTaggedStore` と同じ規則で模す（待ちがないので帰結は 1 か 0）。"""
    resident: "OrderedDict[int, None]" = OrderedDict()
    pf: set[int] = set()

    def insert(e: int, prefetched: bool) -> None:
        resident[e] = None
        resident.move_to_end(e)
        if prefetched:
            pf.add(e)
        while len(resident) > capacity:
            old, _ = resident.popitem(last=False)
            if old in pf:
                pf.discard(old)
                model.on_prefetch_wasted(old)

    n = len(choices)
    for t in range(n):
        e = int(choices[t])
        if e in resident:
            resident.move_to_end(e)
            if e in pf:
                pf.discard(e)
                model.on_prefetch_used(e, late=False)
        else:
            insert(e, prefetched=False)
        model.feedback(e)
        if t + 1 >= n:
            break
        c = model.decide(model.prob(e, hidden[t]), e, lambda x: x in resident)
        if c is not None:
            model.on_prefetch_start(c)
            insert(c, prefetched=True)
    model.reset_pending()
