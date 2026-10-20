"""遅れて届く登上線維の誤差と、遅れに合わせた適格性の窓による先読みの閾値の適応（12.6.209節）。

12.6.193節の `AdaptiveThresholdGate` は、先読みの正誤を「次の時刻にその専門家が使われたか」で
判定した。本モジュールは、ストアが数える無駄 I/O と同じ定義で誤差を作る:

- 先読みした専門家が使われないまま追い出された時点で誤差（登上線維、LTD: θ_c += eta_up）
- 追い出される前に使われた時点で誤差なしの帰結（LTP: θ_c −= eta_down）
- 先読みを見送った候補には窓 W ステップの適格性を残し、窓の中で使われたら LTP

帰結が届くまでの時間は常駐の動きで決まり、次の時刻とは限らない。W はその遅れに合わせる
（Suvrathan, Payne & Raymond 2016 の、タイミング則がフィードバックの遅れに合うことの対応物）。
既存の `expert_prefetch.py` は変えず、`simulate_prefetch_v2` の `adaptive_gate` にそのまま渡せる形にする。
"""

from __future__ import annotations

import threading
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

from .expert_prefetch import MultiPathExpertStore


class DelayedCFGate:
    """先読みの帰結（使われた・使われずに追い出された）が届いた時点で閾値を更新するゲート。

    `decide`・`feedback` は `AdaptiveThresholdGate` と同じ形（予測器のスレッドから呼ばれる）。
    `on_prefetch_used`・`on_prefetch_wasted` はストア（主の流れ・先読みのスレッド）から呼ばれる。
    """

    def __init__(self, n_experts: int, eta_up: float, eta_down: float, theta0: float = 0.5,
                 window: int = 2, lo: float = 0.0, hi: float = 1.5) -> None:
        if window < 1:
            raise ValueError("window は 1 以上")
        self.theta = np.full(n_experts, theta0, dtype=np.float64)
        self.eta_up = eta_up
        self.eta_down = eta_down
        self.window = window
        self.lo, self.hi = lo, hi
        self.lock = threading.Lock()
        self.t = 0  # feedback の呼ばれた回数（＝時刻）
        self.elig: dict[int, int] = {}  # 見送った候補 → 残りの窓
        self.issued_at: dict[int, int] = {}  # 先読みした候補 → 発行の時刻（帰結の遅れを測る）
        self.n_issued = self.n_used = self.n_wasted = self.n_skipped_hit = 0
        self.delay_used: Counter = Counter()
        self.delay_wasted: Counter = Counter()

    def _clip(self, c: int) -> None:
        self.theta[c] = min(max(self.theta[c], self.lo), self.hi)

    def decide(self, prob: np.ndarray, e_t: int, busy) -> int | None:
        """先読みする専門家（なければ None）。busy(c) は常駐か読み込み中か。"""
        c = int(np.lexsort((np.arange(len(prob)), -prob))[0])
        if c == e_t or busy(c):
            return None
        with self.lock:
            if prob[c] >= self.theta[c]:
                self.elig.pop(c, None)  # 先読みしたら見送りの適格性は消す（二重に数えない）
                self.issued_at[c] = self.t
                self.n_issued += 1
                return c
            self.elig[c] = self.window
            return None

    def feedback(self, e: int) -> None:
        """時刻ごとに使われた専門家 e を受け、見送りの適格性を判定して窓を1つ進める。"""
        with self.lock:
            self.t += 1
            if self.elig.pop(e, None) is not None:
                self.n_skipped_hit += 1
                self.theta[e] -= self.eta_down
                self._clip(e)
            for c in list(self.elig):
                self.elig[c] -= 1
                if self.elig[c] <= 0:
                    del self.elig[c]

    def on_prefetch_used(self, c: int) -> None:
        with self.lock:
            t0 = self.issued_at.pop(c, None)
            if t0 is None:
                return
            self.n_used += 1
            self.delay_used[self.t - t0] += 1
            self.theta[c] -= self.eta_down
            self._clip(c)

    def on_prefetch_wasted(self, c: int) -> None:
        with self.lock:
            t0 = self.issued_at.pop(c, None)
            if t0 is None:
                return
            self.n_wasted += 1
            self.delay_wasted[self.t - t0] += 1
            self.theta[c] += self.eta_up
            self._clip(c)

    def reset_pending(self) -> None:
        """暖機の終わりに、帰結の届いていない印と適格性を捨てる（測定はストアが空から始まる）。"""
        with self.lock:
            self.elig.clear()
            self.issued_at.clear()

    def stats(self) -> dict:
        with self.lock:
            return {"theta": self.theta.tolist(), "issued": self.n_issued, "used": self.n_used,
                    "wasted": self.n_wasted, "skipped_hit": self.n_skipped_hit,
                    "delay_used": dict(sorted(self.delay_used.items())),
                    "delay_wasted": dict(sorted(self.delay_wasted.items()))}


class CFTaggedStore(MultiPathExpertStore):
    """先読みした専門家が使われた・使われずに追い出されたときに `DelayedCFGate` を呼ぶストア。

    `gate=None` なら `MultiPathExpertStore` と同じ。列の終わりまで使われなかった先読みは
    `finish` で無駄 I/O に数えるが、ゲートは呼ばない（測定の後の更新は結果に影響しない）。
    """

    def __init__(self, weights: list[np.ndarray], persist_dir: str | Path, capacity: int,
                 io_ms: float, n_paths: int = 1, gate: DelayedCFGate | None = None) -> None:
        super().__init__(weights, persist_dir, capacity, io_ms, n_paths)
        self.gate = gate

    def _insert(self, e: int, w: np.ndarray, prefetched: bool) -> None:
        evicted = []
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
                    evicted.append(old)
        if self.gate is not None:
            for old in evicted:
                self.gate.on_prefetch_wasted(old)

    def use_tracked(self, e: int) -> tuple[np.ndarray, bool]:
        stalled = False
        while True:
            with self.lock:
                if e in self.resident:
                    self.resident.move_to_end(e)
                    was_pf = e in self.prefetched_unused
                    self.prefetched_unused.discard(e)
                    w = self.resident[e]
                    ev = None
                else:
                    was_pf = False
                    w = None
                    ev = self.inflight.get(e)
            if w is not None:
                if was_pf and self.gate is not None:
                    self.gate.on_prefetch_used(e)
                return w, stalled
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


def warm_delayed(predictor, choices: np.ndarray, hidden: np.ndarray, gate: DelayedCFGate | None,
                 capacity: int) -> None:
    """速度を測る前の列で予測器と `DelayedCFGate` を学習させる。I/O の時間は使わず、常駐（容量
    `capacity` の LRU）・先読みの印・追い出しを `CFTaggedStore` と同じ規則で模す。"""
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
                if gate is not None:
                    gate.on_prefetch_wasted(old)

    n = len(choices)
    for t in range(n):
        e = int(choices[t])
        if e in resident:
            resident.move_to_end(e)
            if e in pf:
                pf.discard(e)
                if gate is not None:
                    gate.on_prefetch_used(e)
        else:
            insert(e, prefetched=False)
        if t + 1 >= n:
            if gate is not None:
                gate.feedback(e)
            break
        if gate is not None:
            gate.feedback(e)
            c = gate.decide(predictor.prob(e, hidden[t]), e, lambda x: x in resident)
            if c is not None:
                insert(c, prefetched=True)
        predictor.observe(e, hidden[t], int(choices[t + 1]))
    if gate is not None:
        gate.reset_pending()


def pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """(無駄 I/O, 短縮率) の点から、他のどの点にも両方で負けない点を無駄 I/O の順に返す。"""
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    front: list[tuple[float, float]] = []
    for w, r in pts:
        if not front or r > front[-1][1]:
            front.append((w, r))
    return front


def curve_residual(points: list[tuple[float, float]], w: float, r: float) -> float:
    """点 (w, r) の、パレートの前線を線形につないだ曲線に対する短縮率の残差（右端の先は端の値）。"""
    front = pareto_front(points)
    xs = [p[0] for p in front]
    ys = [p[1] for p in front]
    return r - float(np.interp(w, xs, ys))
