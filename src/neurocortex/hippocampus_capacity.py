"""海馬モジュール — リプレイバッファの容量制約下での刈り込み戦略（12.6.32節・ステップ12）。

ステップ10（`hippocampus_replay.py`）はタスクA→Bの2タスク限定で、バッファは
「タスクA収束時点の生サンプルを1回だけ固定サイズで保存する」という単純な
構造であり、容量制約そのものは検証していない。

本モジュールはタスク1..Nを順に学習させ、バッファ容量が全タスク分を
下回る状況（溢れが必ず発生する）を作った上で、溢れた際に何を残すかの
戦略を比較する。

  unbounded                       … 容量無制限（統制1の前提確認、参考上限）
  fifo                            … 古いタスクのサンプルから機械的に破棄する（統制1、最弱ベースライン）
  uniform-compress                … 重要度を見ず、全タスクを均等にタスクあたりのサンプル数で間引く（統制3）
  importance-weighted             … 対角フィッシャー情報量（ステップ10のEWC実装を踏襲）に基づき、
                                     重要度に応じて各タスクの保持サンプル数を配分する（ステップ12の本設計。
                                     フィッシャー崩壊により逆転して負けた — 12.6.33節参照）
  importance-weighted-fisher-decay … 上と同じフィッシャー情報量だが、`weight_decay`でロジット発散を
                                     抑え、フィッシャー崩壊を防いだ版（ステップ13の本設計a、12.6.34節）
  importance-weighted-loss-based   … フィッシャー情報量を使わず、タスク収束直後の予測損失を重要度
                                     スコアとする版（ステップ13の本設計b、12.6.34節）

##### 実装時に発覚した問題（統制1の前提が崩れていた）

当初、各タスクの入力ビット列に**タスクIDを一切含めず**、5タスク分の
異なるXOR写像（タスク`t`の関連ビット対のパリティ）を単一の共有出力ヘッドに
同時に満たそうとしていた。しかし各タスクの入力分布（全次元が一様乱数）は
どのタスクも区別がつかないため、共有ヘッドは「今どのタスクのサンプルか」を
原理的に判別できず、5つの（一般には互いに矛盾する）ラベル関数を1つの
固定関数で同時に満たすことは数学的に不可能だった（統制1のunbounded参考上限
ですらH1≈0.54とチャンス水準に留まり、動作点探索の最初の実行で発覚）。
これはステップ10が2タスクで「部分的」な結果だったのと同じ原因（タスク数を
増やしたことで破綻が露呈した）。継続学習研究でいう「タスクIDを与えない
class-incremental設定」の限界であり、本ステップが検証したいバッファ容量・
刈り込み戦略の効果とは無関係な交絡である。したがって、各サンプルの入力に
**タスクIDのone-hotベクトル**を明示的に付与する（継続学習研究でいう
task-incremental設定）よう修正した。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .basal_ganglia_core import ParitySpec, make_batch
from .hippocampus_replay import Model, make_model

CONDITIONS = ("unbounded", "fifo", "uniform-compress", "importance-weighted",
             "importance-weighted-fisher-decay", "importance-weighted-loss-based")
IMPORTANCE_WEIGHTED_CONDITIONS = (
    "importance-weighted", "importance-weighted-fisher-decay", "importance-weighted-loss-based",
)

Segment = tuple[torch.Tensor, torch.Tensor]
Buffer = "OrderedDict[int, Segment]"


def make_task_specs(n_tasks: int, n_inputs: int) -> list[ParitySpec]:
    """タスク1..nを、互いに素なビット対を関連次元とするParitySpecの列として作る。"""
    if n_inputs < 2 * n_tasks:
        raise ValueError("n_inputs はタスク数の2倍以上が必要（各タスクが互いに素なビット対を持つため）")
    return [ParitySpec(n_inputs=n_inputs, relevant=(2 * t, 2 * t + 1)) for t in range(n_tasks)]


def model_input_dim(n_inputs: int, n_tasks: int) -> int:
    """モデルの入力次元 = 元のビット数 + タスクIDのone-hot次元。"""
    return n_inputs + n_tasks


def _task_batch(spec: ParitySpec, task_idx: int, n_tasks: int, batch_size: int,
                generator: torch.Generator) -> Segment:
    """タスクIDのone-hotを付与した (bits, labels) を返す。"""
    bits, labels = make_batch(spec, batch_size, generator)
    onehot = F.one_hot(torch.full((batch_size,), task_idx, dtype=torch.long), num_classes=n_tasks)
    aug = torch.cat([bits, onehot.float()], dim=-1)
    return aug, labels


def snapshot_task(spec: ParitySpec, task_idx: int, n_tasks: int, size: int,
                  generator: torch.Generator) -> Segment:
    """タスク収束時点で、そのタスクの生サンプル（タスクID付き）を固定サイズだけ切り出す。"""
    return _task_batch(spec, task_idx, n_tasks, size, generator)


def _fisher_diagonal_for_task(model: Model, spec: ParitySpec, task_idx: int, n_tasks: int,
                              n_batches: int, batch_size: int,
                              generator: torch.Generator) -> list[torch.Tensor]:
    """タスク収束時点の対角フィッシャー情報量（モデル自身の予測分布からサンプルした
    ラベルに対する対数尤度の勾配の二乗、標準的なEWCの定義。12.6.29節の実装を踏襲）。
    """
    params = model.parameters()
    fisher = [torch.zeros_like(p) for p in params]
    for _ in range(n_batches):
        bits, _ = _task_batch(spec, task_idx, n_tasks, batch_size, generator)
        logits = model(bits)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            sampled = torch.multinomial(log_probs.exp(), 1).squeeze(-1)
        loss = F.nll_loss(log_probs, sampled)
        grads = torch.autograd.grad(loss, params)
        for f, g in zip(fisher, grads):
            f += g.detach() ** 2
    return [f / n_batches for f in fisher]


def task_importance(model: Model, spec: ParitySpec, task_idx: int, n_tasks: int, n_batches: int,
                    batch_size: int, generator: torch.Generator) -> float:
    """タスク収束時点の重要度スカラー値（対角フィッシャー情報量の全パラメータ平均）。
    収束済みタスクではほぼ0に潰れる（12.6.33節参照）。
    """
    fisher = _fisher_diagonal_for_task(model, spec, task_idx, n_tasks, n_batches, batch_size,
                                       generator)
    total = sum(f.sum().item() for f in fisher)
    count = sum(f.numel() for f in fisher)
    return total / count


@torch.no_grad()
def task_importance_loss_based(model: Model, spec: ParitySpec, task_idx: int, n_tasks: int,
                               n_batches: int, batch_size: int,
                               generator: torch.Generator) -> float:
    """タスク収束時点の重要度スカラー値（予測損失の平均、フィッシャー情報量を使わない代替指標）。

    損失は収束済みでも0に潰れきらない（交差エントロピーは正解確率が1に近づくほど
    小さくなるが、対角フィッシャー情報量のように勾配の二乗として0に急減しない）。
    損失が大きい＝現時点でうまく再現できていない＝保護の優先度が高い、とみなす。
    """
    total_loss = 0.0
    for _ in range(n_batches):
        bits, labels = _task_batch(spec, task_idx, n_tasks, batch_size, generator)
        loss = F.cross_entropy(model(bits), labels)
        total_loss += float(loss)
    return total_loss / n_batches


def _segment_len(seg: Segment) -> int:
    return int(seg[0].shape[0])


def _subsample(seg: Segment, n: int, generator: torch.Generator) -> Segment:
    """segからn件をランダムに（重複なし）間引く。n >= len(seg)ならそのまま返す。"""
    bits, labels = seg
    total = bits.shape[0]
    if n >= total:
        return seg
    if n <= 0:
        return bits[:0], labels[:0]
    idx = torch.randperm(total, generator=generator)[:n]
    return bits[idx], labels[idx]


def evict_fifo(buffer: Buffer, capacity: int) -> Buffer:
    """挿入順（タスク順）でサンプル単位のFIFO。古いタスクのセグメントから切り詰める。"""
    total = sum(_segment_len(seg) for seg in buffer.values())
    if total <= capacity:
        return OrderedDict(buffer)
    to_drop = total - capacity
    result: Buffer = OrderedDict()
    for task_idx, seg in buffer.items():
        seg_len = _segment_len(seg)
        if to_drop <= 0:
            result[task_idx] = seg
            continue
        if to_drop >= seg_len:
            to_drop -= seg_len
            continue
        bits, labels = seg
        result[task_idx] = (bits[to_drop:], labels[to_drop:])
        to_drop = 0
    return result


def evict_uniform(buffer: Buffer, capacity: int, generator: torch.Generator) -> Buffer:
    """重要度を見ず、保持中の全タスクを均等なサンプル数まで間引く。"""
    n_tasks = len(buffer)
    if n_tasks == 0:
        return OrderedDict(buffer)
    total = sum(_segment_len(seg) for seg in buffer.values())
    if total <= capacity:
        return OrderedDict(buffer)
    per_task = capacity // n_tasks
    return OrderedDict((t, _subsample(seg, per_task, generator)) for t, seg in buffer.items())


def evict_importance_weighted(buffer: Buffer, importances: dict[int, float], capacity: int,
                              generator: torch.Generator) -> Buffer:
    """重要度に比例して保持サンプル数を配分する。"""
    n_tasks = len(buffer)
    if n_tasks == 0:
        return OrderedDict(buffer)
    total = sum(_segment_len(seg) for seg in buffer.values())
    if total <= capacity:
        return OrderedDict(buffer)
    weights = [max(importances[t], 1e-12) for t in buffer]
    weight_sum = sum(weights)
    result: Buffer = OrderedDict()
    allocated = 0
    tasks = list(buffer.keys())
    for i, t in enumerate(tasks):
        if i == len(tasks) - 1:
            n = capacity - allocated  # 端数は最後のタスクに寄せて容量を使い切る
        else:
            n = int(capacity * weights[i] / weight_sum)
        n = max(n, 0)
        result[t] = _subsample(buffer[t], n, generator)
        allocated += _segment_len(result[t])
    return result


def train_with_replay(model: Model, spec: ParitySpec, task_idx: int, n_tasks: int, buffer: Buffer,
                      steps: int, batch_size: int, lr: float, generator: torch.Generator,
                      weight_decay: float = 0.0) -> None:
    """新タスクのデータに、バッファ中の全保持タスクのサンプルを混ぜて学習する。
    バッファが空（最初のタスク）の場合は新タスクのデータのみで学習する。

    `weight_decay`は`importance-weighted-fisher-decay`条件のみで使う。ステップ10の
    EWC事前学習と同じ理由（交差エントロピーで収束するとロジットが発散し、対角
    フィッシャー情報量がほぼ0に潰れる — 12.6.29節・12.6.33節）で、ロジットの発散を
    抑えてフィッシャー情報量を意味のある大きさに保つための対策。
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    segments = [seg for seg in buffer.values() if _segment_len(seg) > 0]
    for _ in range(steps):
        bits_new, labels_new = _task_batch(spec, task_idx, n_tasks, batch_size, generator)
        bits_list, labels_list = [bits_new], [labels_new]
        for buf_bits, buf_labels in segments:
            n_buf = buf_bits.shape[0]
            idx = torch.randint(n_buf, (batch_size,), generator=generator)
            bits_list.append(buf_bits[idx])
            labels_list.append(buf_labels[idx])
        bits = torch.cat(bits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        loss = F.cross_entropy(model(bits), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


@torch.no_grad()
def evaluate(model: Model, spec: ParitySpec, task_idx: int, n_tasks: int, n_eval: int,
            generator: torch.Generator) -> float:
    bits, labels = _task_batch(spec, task_idx, n_tasks, n_eval, generator)
    pred = model(bits).argmax(dim=-1)
    return float((pred == labels).float().mean())


@dataclass
class RunResult:
    per_task_acc: list[float]  # 最終モデルでの、各旧タスク（最終タスクを除く）の正解率
    mean_old_task_acc: float   # H1: 旧タスク全体の平均正解率
    oldest_task_acc: float     # H2: 最古タスク（タスク1）の正解率


def run_condition(condition: str, task_specs: list[ParitySpec], hidden_dim: int, capacity: int,
                  snapshot_size: int, task_steps: int, batch_size: int, lr: float,
                  fisher_batches: int, fisher_batch_size: int, eval_n: int,
                  generator: torch.Generator, fisher_decay_weight_decay: float = 0.01) -> RunResult:
    """1条件・1シードぶんの実行。タスク1..Nを順に学習し、最終モデルで旧タスク
    （タスク1..N-1）の保持率を測定する。

    `fisher_decay_weight_decay`は`importance-weighted-fisher-decay`条件でのみ使う
    （12.6.34節、ステップ13の本設計a）。
    """
    if condition not in CONDITIONS:
        raise ValueError(condition)

    n_tasks = len(task_specs)
    n_inputs = task_specs[0].n_inputs
    model = make_model(model_input_dim(n_inputs, n_tasks), hidden_dim)
    buffer: Buffer = OrderedDict()
    importances: dict[int, float] = {}

    weight_decay = fisher_decay_weight_decay if condition == "importance-weighted-fisher-decay" else 0.0

    for t, spec in enumerate(task_specs):
        train_with_replay(model, spec, t, n_tasks, buffer, task_steps, batch_size, lr, generator,
                          weight_decay=weight_decay)

        if condition in ("importance-weighted", "importance-weighted-fisher-decay"):
            importances[t] = task_importance(model, spec, t, n_tasks, fisher_batches,
                                              fisher_batch_size, generator)
        elif condition == "importance-weighted-loss-based":
            importances[t] = task_importance_loss_based(model, spec, t, n_tasks, fisher_batches,
                                                         fisher_batch_size, generator)

        buffer[t] = snapshot_task(spec, t, n_tasks, snapshot_size, generator)

        if condition == "unbounded":
            pass
        elif condition == "fifo":
            buffer = evict_fifo(buffer, capacity)
        elif condition == "uniform-compress":
            buffer = evict_uniform(buffer, capacity, generator)
        elif condition in IMPORTANCE_WEIGHTED_CONDITIONS:
            buffer = evict_importance_weighted(buffer, importances, capacity, generator)
            importances = {t: v for t, v in importances.items() if t in buffer}

    old_specs = task_specs[:-1]
    per_task_acc = [evaluate(model, spec, t, n_tasks, eval_n, generator)
                    for t, spec in enumerate(old_specs)]
    mean_old = sum(per_task_acc) / len(per_task_acc)
    oldest = per_task_acc[0]
    return RunResult(per_task_acc=per_task_acc, mean_old_task_acc=mean_old, oldest_task_acc=oldest)
