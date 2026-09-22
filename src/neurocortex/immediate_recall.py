"""評価指標 — 海馬モジュールの即時想起率（12.6.46節・ステップ19）。

`hippocampus.HippocampalMemory`に事実（キー用皮質表現・値のペア）を書き込んだ
直後、同じキーで読み出した結果を評価する。2つの量を対で報告する。

  即時想起率     … 既知キーに対する読み出しが「成功」する割合。成功の条件は
                   (a) 連想ストアの最良一致が正しいスロットを指すこと、かつ
                   (b) ゲートが実際に開いている（新規性なしと正しく判定）こと
  誤想起率       … 未知キー（未書き込み）に対して、ゲートが誤って開く割合（偽陽性）

`PatternSeparator`の3モード（sdr・dense・identity、ステップ1の統制）を横並びで
比較できるようにする。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .hippocampus import HippocampalMemory

GATE_OPEN_THRESHOLD = 0.5


def sample_facts(n: int, d_model: int, value_dim: int,
                 generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """標準正規分布からランダムな(キー用皮質表現, 値)のペアをn件生成する。"""
    keys = torch.randn(n, d_model, generator=generator)
    values = torch.randn(n, value_dim, generator=generator)
    return keys, values


def sample_unknown_keys(n: int, d_model: int, generator: torch.Generator) -> torch.Tensor:
    """未書き込みのランダムなキー用皮質表現をn件生成する。"""
    return torch.randn(n, d_model, generator=generator)


@dataclass
class RecallResult:
    recall_rate: float             # 既知キーに対する成功率
    false_positive_rate: float     # 未知キーに対するゲート誤開放率
    chance_recall_rate: float      # 参考値: 完全ランダムな最良一致が正解スロットを指す確率 (1/n_facts)


@torch.no_grad()
def evaluate_immediate_recall(memory: HippocampalMemory, known_keys: torch.Tensor,
                              unknown_keys: torch.Tensor) -> RecallResult:
    """`memory`に書き込み済みの`known_keys`と、未書き込みの`unknown_keys`を評価する。

    書き込み自体は呼び出し側の責任（`memory.write`済みであることを前提にする）。
    """
    n_facts = known_keys.shape[0]

    known_query = memory.separator(known_keys)
    m_known, stats_known = memory.store.read(known_query)
    g_known = torch.sigmoid(memory.gate_slope * (stats_known.max_score - memory.gate_bias))
    correct_slot = stats_known.top1_index == torch.arange(n_facts)
    gate_open = g_known > GATE_OPEN_THRESHOLD
    success = correct_slot & gate_open
    recall_rate = float(success.float().mean())

    unknown_query = memory.separator(unknown_keys)
    _, stats_unknown = memory.store.read(unknown_query)
    g_unknown = torch.sigmoid(memory.gate_slope * (stats_unknown.max_score - memory.gate_bias))
    false_positive_rate = float((g_unknown > GATE_OPEN_THRESHOLD).float().mean())

    return RecallResult(
        recall_rate=recall_rate,
        false_positive_rate=false_positive_rate,
        chance_recall_rate=1.0 / n_facts,
    )


def run_condition(mode: str, d_model: int, value_dim: int, n_units: int, k: int, n_facts: int,
                  n_unknown: int, gate_slope: float, gate_bias: float,
                  generator: torch.Generator, seed: int) -> RecallResult:
    """1条件（分離層モード）・1シードぶんの実行: 書き込み→即時読み出し評価。"""
    memory = HippocampalMemory(d_model=d_model, value_dim=value_dim, n_units=n_units, k=k,
                               mode=mode, gate_slope=gate_slope, gate_bias=gate_bias, seed=seed)
    known_keys, known_values = sample_facts(n_facts, d_model, value_dim, generator)
    memory.write(known_keys, known_values)
    unknown_keys = sample_unknown_keys(n_unknown, d_model, generator)
    return evaluate_immediate_recall(memory, known_keys, unknown_keys)
