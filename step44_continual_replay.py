#!/usr/bin/env python3
"""ステップ44.10: A+replay（条件 R）。海馬の外部エピソードメモリの保存・再生を加えた継続学習。

step44_continual_forgetting（44.9）と同じ課題・評価で、フェーズ A のモデルに外部エピソードメモリを加える。
- 保存: 各タスクの学習の終わりに、そのタスクの学習断片から M 本を一様に非復元抽出して保存（入力の単語 ID 列）。
- 再生: 2 番目以降のタスクの各更新で、メモリ全体から r 本を復元抽出し、新しいタスクのバッチに連結して通常の損失で学ぶ。
抽出の乱数は全体の乱数と独立。最初のタスクではメモリが空なので A と同じ計算になる。

使い方: python step44_continual_replay.py --seeds 100 --out-dir results/step44_continual_replay/dev
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import step44_continual_forgetting as cf  # noqa: E402
from colab_step44_full_scale_experiment import BrainInspiredLLMFullScale, FullScaleTrainer  # noqa: E402
from neurocortex.run_env import env_info  # noqa: E402


class EpisodicMemory:
    """海馬の外部エピソードメモリ（CLS 理論の速い保存と再生）。タスクごとに入力断片を M 本保存する。"""

    def __init__(self, per_task: int):
        self.per_task = per_task
        self.slots: list[torch.Tensor] = []  # 各タスクの (M, T) の単語 ID
        self.task_of: list[int] = []

    def __len__(self):
        return sum(len(s) for s in self.slots)

    def store(self, dataset, generator: torch.Generator) -> torch.Tensor:
        """dataset（WordChunkDataset）から per_task 本を一様に非復元抽出して保存。保存した断片の番号を返す。"""
        n = len(dataset)
        idx = torch.randperm(n, generator=generator)[:min(self.per_task, n)]
        self.slots.append(dataset.chunks[idx].clone())
        self.task_of.append(len(self.slots) - 1)
        return idx

    def sample(self, r: int, generator: torch.Generator) -> torch.Tensor:
        """メモリ全体から r 本を一様に復元抽出する（タスクごとの保存数が等しいので過去タスク間で均等）。"""
        allm = torch.cat(self.slots)
        return allm[torch.randint(0, len(allm), (r,), generator=generator)]


def concat_replay(batch: dict, replay: torch.Tensor) -> dict:
    x = torch.cat([batch["input_ids"], replay])
    return {"input_ids": x, "labels": x, "attention_mask": torch.ones_like(x)}


def run_continual_replay(seed: int, data: dict, args, log=print, base_phase: str = "A") -> dict:
    """条件 R: base_phase（既定 A）のモデルに外部エピソードメモリの保存・再生を加えて K タスクを順に学ぶ。

    run_continual（44.9）と同じ初期化・データ順・評価で、違いはメモリの保存と再生だけ。replay_size=0 なら A と同一。
    replay_source="current"（44.11 の計算量対照 C1）では、メモリの代わりに現在のタスクの学習断片から r 本を
    復元抽出して連結する（更新回数・断片数・抽出の乱数は R と同一。メモリは使わず保存もしない）。
    """
    source = getattr(args, "replay_source", "memory")
    label = getattr(args, "label", None) or "R"
    if source not in ("memory", "current"):
        raise ValueError(f"replay_source: {source}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = cf.make_config(base_phase, args)
    model = BrainInspiredLLMFullScale(cfg, vocab_size=data["vocab_size"], hidden_size=cfg.hidden_size,
                                      num_layers=cfg.num_layers, nhead=cfg.nhead)
    trainer = FullScaleTrainer(model, cfg)
    memory = EpisodicMemory(args.memory_per_task)
    order = cf.task_order(data["langs"], seed)
    T = len(order)
    acc = [[0.0] * T for _ in range(T)]
    ce = [[0.0] * T for _ in range(T)]
    losses, n_replayed, stored, t0 = [], [], [], time.time()
    for i, lang in enumerate(order):
        trainer.optimizer.zero_grad()
        task_losses, nrep = [], 0
        g_rep = torch.Generator().manual_seed(seed * 1000 + 700 + i)
        train = data["tasks"][lang]["train"]
        for epoch in range(args.epochs):
            g = torch.Generator().manual_seed(seed * 1000 + i * 10 + epoch)  # 44.9 と同じ並び
            for batch in DataLoader(train, batch_size=cfg.batch_size, shuffle=True, generator=g):
                if args.replay_size > 0 and i > 0 and source == "current":
                    extra = train.chunks[torch.randint(0, len(train), (args.replay_size,), generator=g_rep)]
                    batch = concat_replay(batch, extra)
                    nrep += args.replay_size
                elif len(memory) and args.replay_size > 0:
                    batch = concat_replay(batch, memory.sample(args.replay_size, g_rep))
                    nrep += args.replay_size
                task_losses.append(trainer.train_step(batch))
        losses.append(float(np.mean(task_losses[-20:])))
        n_replayed.append(nrep)
        if source == "memory":
            idx = memory.store(train, torch.Generator().manual_seed(seed * 1000 + 500 + i))
            stored.append(idx.tolist())
        for j, lj in enumerate(order):
            ce[i][j], acc[i][j] = cf.eval_task(model, data["tasks"][lj]["eval"])
        log(f"[{label} seed{seed}] task{i} {lang} done {time.time() - t0:.0f}s mem {len(memory)} replayed {nrep} "
            f"diag ce {ce[i][i]:.3f} acc {acc[i][i]:.3f} | row ce " + " ".join(f"{c:.2f}" for c in ce[i]))
    params = sum(p.numel() for p in model.parameters())
    return {"phase": label, "base_phase": base_phase, "replay_source": source, "seed": seed, "order": order,
            "acc": acc, "ce": ce,
            "metrics": cf.summarize(acc, ce), "final_train_loss_per_task": losses, "n_params": params,
            "n_params_backbone": sum(p.numel() for n, p in model.named_parameters()
                                     if not n.startswith(("hippocampus", "basal_ganglia", "cerebellum"))),
            "memory_per_task": args.memory_per_task, "replay_size": args.replay_size,
            "n_replayed_per_task": n_replayed, "stored_chunk_idx": stored,
            "train_seconds": time.time() - t0, "env": env_info()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--memory-per-task", type=int, default=64)
    ap.add_argument("--replay-size", type=int, default=16)
    ap.add_argument("--replay-source", choices=("memory", "current"), default="memory",
                    help="memory: 外部エピソードメモリから再生（R）。current: 現在のタスクの断片を追加（44.11 の C1）")
    ap.add_argument("--label", default="R", help="結果の条件名（ファイル名 phase<label>_seed<seed>.json）")
    ap.add_argument("--europarl-dir", default=str(ROOT / "data" / "europarl_raw"))
    ap.add_argument("--langs", nargs="+", default=list(cf.LANGS))
    ap.add_argument("--per-lang-vocab", type=int, default=2500)
    ap.add_argument("--n-train", type=int, default=250_000)
    ap.add_argument("--n-eval", type=int, default=25_000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--max-seq-length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = cf.build_tasks(Path(args.europarl_dir), tuple(args.langs), args.per_lang_vocab, args.n_train, args.n_eval,
                          args.max_seq_length)
    print(f"vocab {data['vocab_size']}", flush=True)
    for seed in args.seeds:
        path = out / f"phase{args.label}_seed{seed}.json"
        if path.exists():
            continue
        res = run_continual_replay(seed, data, args, log=lambda s: print(s, flush=True))
        res["config"] = {k: v for k, v in vars(args).items()}
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(f"saved {path} metrics {res['metrics']}", flush=True)


if __name__ == "__main__":
    main()
