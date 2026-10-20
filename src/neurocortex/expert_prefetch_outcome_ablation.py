"""先読みの帰結で学ぶ小脳の順モデル（12.6.211節）の利得の源の切り分け（12.6.217節）。

`OutcomeCerebellum` と同じ入力・重みの形・デルタ則・選び方（常駐でない最上位を v ≥ θ で先読み）のまま、
教師または可塑性だけを替えた対照:

- `SupervisedEligibleCerebellum`: 教師あり対照。各時刻、行動の余地のある候補（直前と同じでなく、常駐でも
  読み込み中でもない c）すべてに「c が次の W ステップ以内に使われるか」の正解（未来の列を見る、特権的な
  教師）を、その時刻に即時に与える。遅れて届く誤差・適格性のタグ・先読みの実際の帰結は使わない。
- `OneSidedOutcomeCerebellum`: 誤差 δ = y − v の片側だけで学ぶ。`plasticity="ltd"` は δ < 0 だけ
  （初期値 v = 1、増強された状態から抑圧だけで学ぶ。Marr–Albus の LTD のみの学習）、
  `plasticity="ltp"` は δ > 0 だけ（初期値 v = 0）。
- 帰結の段階づけを除く対照は `OutcomeCerebellum(y_late=1.0)`（待っても使われれば 1）。

既存の `expert_prefetch*.py` は変えない。
"""

from __future__ import annotations

import numpy as np

from .expert_prefetch_outcome import OutcomeCerebellum


class SupervisedEligibleCerebellum(OutcomeCerebellum):
    """教師あり対照: 行動の余地のある候補すべてに、次の W ステップ以内に使われるかの正解を即時に与える。

    `set_trace(choices)` で、これから `decide` を呼ぶ列（時刻 0 から順に1回ずつ）を渡す。
    `simulate_prefetch_v2`・`warm_outcome` はどちらも時刻 0..n−2 で1回ずつ順に `decide` を呼ぶ。
    更新は先読みを決めた後に行う（その時刻の決定は正解を見ない）。"""

    def __init__(self, *args, **kw) -> None:
        if kw.get("shuffle"):
            raise ValueError("教師あり対照に並べ替えはない")
        super().__init__(*args, **kw)
        self.trace: np.ndarray | None = None
        self.k = 0

    def set_trace(self, choices: np.ndarray) -> None:
        with self.lock:
            self.trace = np.asarray(choices)
            self.k = 0

    def decide(self, prob: np.ndarray, e_t: int, busy) -> int | None:
        order = [int(c) for c in np.lexsort((np.arange(len(prob)), -prob))]
        targets = [c for c in order if c != e_t and not busy(c)]
        with self.lock:
            if self.trace is None:
                raise RuntimeError("set_trace を先に呼ぶ")
            t = self.k
            self.k += 1
            if t + 1 >= len(self.trace) or int(self.trace[t]) != e_t:
                raise RuntimeError("decide の呼び出しが列と合わない")
            x = self._last_x
            pick = next((c for c in targets if prob[c] >= self.theta), None)
            future = {int(v) for v in self.trace[t + 1: t + 1 + self.window]}
            for c in targets:
                hit = c in future
                self._learn(c, x, 1.0 if hit else 0.0, "sup_pos" if hit else "sup_neg")
            if pick is not None:
                self.counts["issued"] += 1
            return pick

    def feedback(self, e: int) -> None:
        with self.lock:
            self.t += 1

    def on_prefetch_start(self, c: int) -> None:
        with self.lock:
            self.counts["started"] += 1

    def on_prefetch_used(self, c: int, late: bool = False) -> None:
        with self.lock:
            self.counts["obs_used_late" if late else "obs_used"] += 1

    def on_prefetch_wasted(self, c: int) -> None:
        with self.lock:
            self.counts["obs_wasted"] += 1


class OneSidedOutcomeCerebellum(OutcomeCerebellum):
    """帰結の誤差の片側だけで学ぶ対照（`plasticity` が "ltd" か "ltp"）。帰結・タグ・窓は `OutcomeCerebellum`
    と同じ。`v0` は全候補の初期の価値（bias）。既定は LTD のみで 1、LTP のみで 0。"""

    def __init__(self, *args, plasticity: str = "ltd", v0: float | None = None, **kw) -> None:
        if plasticity not in ("ltd", "ltp"):
            raise ValueError("plasticity は ltd か ltp")
        if kw.get("shuffle"):
            raise ValueError("片側の対照に並べ替えはない")
        super().__init__(*args, **kw)
        self.plasticity = plasticity
        self.bias[:] = (1.0 if plasticity == "ltd" else 0.0) if v0 is None else v0

    def _learn(self, c: int, x: np.ndarray, y: float, kind: str) -> None:
        self.counts[kind] += 1
        err = y - (float(self.weight[c] @ x) + self.bias[c])
        if (self.plasticity == "ltd" and err > 0) or (self.plasticity == "ltp" and err < 0):
            self.counts["blocked"] += 1
            return
        self.weight[c] += self.lr * err * x
        self.bias[c] += self.lr * err
