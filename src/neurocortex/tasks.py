"""識別テスト用の合成課題（12.6.1節）。

テストA: 遅延想起課題 `[cue] [filler x N] [query]`
テストB: 順序判別課題（記号A, Bのどちらが後に出たか）
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RecallSpec:
    """遅延想起課題の仕様。

    遅延 N は **cue位置から解答位置までの距離**と定義する。N=0 は cue 自身の位置で
    答える課題（12.6.1節 統制1の「遅延ゼロ」）であり、位置間の記憶を一切必要としない。
    N>=1 では cue の N-1 個あとまで filler が続き、最終位置に query 記号が置かれる。

    語彙の割り当て:
        0 .. n_cue-1                  : cue記号（そのまま正解ラベルになる）
        n_cue .. n_cue+n_filler-1     : filler記号（cueと互いに素）
        n_cue+n_filler                : query記号
        n_cue+n_filler+1              : pad記号（系列長をそろえるための先頭詰め）
    """

    n_cue: int = 8
    n_filler: int = 8
    max_delay: int = 16

    @property
    def query_id(self) -> int:
        return self.n_cue + self.n_filler

    @property
    def pad_id(self) -> int:
        return self.n_cue + self.n_filler + 1

    @property
    def vocab_size(self) -> int:
        return self.n_cue + self.n_filler + 2

    @property
    def seq_len(self) -> int:
        # [pad...] [cue] [filler x (N-1)] [query]  （N=0 のときは最終位置が cue）
        return self.max_delay + 1

    @property
    def chance(self) -> float:
        return 1.0 / self.n_cue


def make_recall_batch(
    spec: RecallSpec,
    batch_size: int,
    generator: torch.Generator,
    delay: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """遅延想起課題のバッチを作る。

    Args:
        spec: 課題仕様。
        batch_size: バッチサイズ。
        generator: 乱数生成器（シード制御のため必須）。
        delay: None ならサンプルごとに 0..max_delay の一様乱数（固定タイミングの
            学習を防ぐ, 12.6.1節）。整数を与えるとその遅延に固定する。

    Returns:
        (tokens [B, T], labels [B])。query は必ず最終位置 T-1 に来る。
    """
    t = spec.seq_len
    if delay is None:
        delays = torch.randint(0, spec.max_delay + 1, (batch_size,), generator=generator)
    else:
        if not 0 <= delay <= spec.max_delay:
            raise ValueError(f"delay は 0..{spec.max_delay} の範囲（受領: {delay}）")
        delays = torch.full((batch_size,), delay, dtype=torch.long)

    tokens = torch.full((batch_size, t), spec.pad_id, dtype=torch.long)
    labels = torch.randint(0, spec.n_cue, (batch_size,), generator=generator)
    fillers = torch.randint(
        spec.n_cue, spec.n_cue + spec.n_filler, (batch_size, spec.max_delay), generator=generator
    )
    for b in range(batch_size):
        d = int(delays[b])
        cue_pos = t - 1 - d  # 解答位置（最終位置）から d 個前
        tokens[b, cue_pos] = labels[b]
        if d >= 1:
            tokens[b, -1] = spec.query_id
            if d >= 2:
                tokens[b, cue_pos + 1 : t - 1] = fillers[b, : d - 1]
    return tokens, labels


@dataclass
class OrderSpec:
    """テストB: 順序判別課題の仕様。記号A, Bが1回ずつ出る。"""

    n_filler: int = 8
    seq_len: int = 16

    @property
    def a_id(self) -> int:
        return self.n_filler

    @property
    def b_id(self) -> int:
        return self.n_filler + 1

    @property
    def vocab_size(self) -> int:
        return self.n_filler + 2

    @property
    def chance(self) -> float:
        return 0.5


def make_order_batch(
    spec: OrderSpec, batch_size: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """順序判別課題のバッチ。ラベル1はAがBより後、0はBがAより後。"""
    tokens = torch.randint(0, spec.n_filler, (batch_size, spec.seq_len), generator=generator)
    labels = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        pos = torch.randperm(spec.seq_len, generator=generator)[:2]
        p, q = int(pos[0]), int(pos[1])
        first, second = (p, q) if p < q else (q, p)
        a_later = bool(torch.randint(0, 2, (1,), generator=generator))
        if a_later:
            tokens[b, first] = spec.b_id
            tokens[b, second] = spec.a_id
            labels[b] = 1
        else:
            tokens[b, first] = spec.a_id
            tokens[b, second] = spec.b_id
            labels[b] = 0
    return tokens, labels
