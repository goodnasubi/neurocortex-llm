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


@dataclass
class FactSpec:
    """ステップ1（海馬モジュール, 12.6.4節）の三つ組課題の仕様。

    系列は `[接頭] [接尾] [関係] [目的語]` の4トークン。書き込み相では全体を、
    想起相では先頭3トークンだけを通し、最終位置（関係トークンの位置）の残差
    ストリームをキーにする。

    **主語の類似度を制御できることが設計の要点**である（12.6.4節）。主語は
    (接頭, 接尾) の対で表され、`n_prefix` が小さいほど多くの主語が接頭トークンを
    共有するため、埋め込み空間で互いに似る。パターン分離層の有無で差が出るはずの
    領域を、この軸で狙い撃ちする。

    目的語は主語と独立に毎回サンプリングされるため、**皮質バックボーンが事前に
    その対応を知ることは原理的にありえない**。これにより統制2（因果的除去）が
    構成上保証される。

    語彙の割り当て:
        0 .. n_prefix-1                        : 主語の接頭トークン
        n_prefix .. n_prefix+n_suffix-1        : 主語の接尾トークン
        n_prefix+n_suffix                      : 関係トークン
        その次から n_object 個                  : 目的語
    """

    n_prefix: int = 8
    n_suffix: int = 4096
    n_object: int = 64

    @property
    def relation_id(self) -> int:
        return self.n_prefix + self.n_suffix

    @property
    def object_offset(self) -> int:
        return self.relation_id + 1

    @property
    def vocab_size(self) -> int:
        return self.object_offset + self.n_object

    @property
    def seq_len(self) -> int:
        return 4

    @property
    def n_subject(self) -> int:
        return self.n_prefix * self.n_suffix

    @property
    def chance(self) -> float:
        return 1.0 / self.n_object


def make_fact_batch(
    spec: FactSpec,
    n_facts: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """互いに異なる主語を持つ三つ組を n_facts 件作る。

    Returns:
        (prompts [N, 3], objects [N])。prompts は `[接頭] [接尾] [関係]`、
        objects は語彙IDでの目的語（正解ラベル）。
    """
    if n_facts > spec.n_subject:
        raise ValueError(f"主語が足りない（{spec.n_subject} 通りに対し {n_facts} 件）")
    # 主語は重複させない。重複を許すと「同じキーに別の値」という別種の失敗が
    # 混ざり、パターン分離の効果と区別がつかなくなる。
    flat = torch.randperm(spec.n_subject, generator=generator)[:n_facts]
    prefix = flat // spec.n_suffix
    suffix = flat % spec.n_suffix + spec.n_prefix
    relation = torch.full((n_facts,), spec.relation_id, dtype=torch.long)
    objects = torch.randint(0, spec.n_object, (n_facts,), generator=generator) + spec.object_offset
    return torch.stack([prefix, suffix, relation], dim=1), objects
