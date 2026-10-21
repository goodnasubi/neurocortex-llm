"""継続学習の破局的忘却の評価指標（12.6.101節、12.7節「評価指標の欠如」）。

正確度行列 R[i][j]（タスク i の学習後に測ったタスク j の正確度）から、チャンス正規化した
ACC・BWT・F・LA を計算する。課題やモジュールに依存しない。

  ACC … 最終時点の全タスク平均正確度
  BWT … 後方転移（Lopez-Paz & Ranzato 2017）。負なら忘却
  F   … 忘却量（Chaudhry et al. 2018）。過去の最高値からの低下の平均
  LA  … 新規学習能力。各タスクを学習した直後の正確度の平均
"""

from __future__ import annotations

from collections.abc import Sequence


def normalize(acc: float, chance: float) -> float:
    if not 0.0 <= chance < 1.0:
        raise ValueError("chance は [0, 1) の値")
    return (acc - chance) / (1.0 - chance)


def forgetting_metrics(R: Sequence[Sequence[float]], chance: float = 0.0) -> dict[str, float]:
    """R は T×T（R[i][j] はタスク i 学習後のタスク j の正確度）。T ≥ 2。"""
    T = len(R)
    if T < 2 or any(len(row) != T for row in R):
        raise ValueError("R は T×T（T ≥ 2）")
    Rn = [[normalize(a, chance) for a in row] for row in R]
    last = Rn[-1]
    acc = sum(last) / T
    bwt = sum(last[j] - Rn[j][j] for j in range(T - 1)) / (T - 1)
    f = sum(max(Rn[i][j] for i in range(j, T - 1)) - last[j] for j in range(T - 1)) / (T - 1)
    la = sum(Rn[j][j] for j in range(T)) / T
    return {"ACC": acc, "BWT": bwt, "F": f, "LA": la}
