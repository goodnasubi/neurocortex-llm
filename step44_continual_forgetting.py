#!/usr/bin/env python3
"""ステップ44.9: 継続学習の忘却指標での、脳型モジュール付き（A/B/C）と vanilla（V）の比較。

step44 のモデル（colab_step44_full_scale_experiment の BrainInspiredLLMFullScale・FullScaleTrainer）を
そのまま使い、Europarl の K 言語を順に学ぶ（語彙は共有、リプレイ・EWC なしの naive）。
各タスク学習後に全タスクの保持断片（学習に使わない日の議事録）で backbone lm_head の CE と top-1 正確度を測り、
正確度行列 R と CE 行列から ACC・BWT・F・LA（continual_metrics）と CE 版の忘却を出す。

使い方: python step44_continual_forgetting.py --phases V A B C --seeds 100 101 --out-dir results/step44_continual_forgetting/dev
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from colab_step44_full_scale_experiment import (  # noqa: E402
    BrainInspiredLLMFullScale, FullScaleTrainer, Step444Config, WordChunkDataset)
from neurocortex.context_realtext import read_europarl  # noqa: E402
from neurocortex.continual_metrics import forgetting_metrics  # noqa: E402
from neurocortex.run_env import env_info  # noqa: E402

logger = logging.getLogger("step44_continual")

LANGS = ("english", "german", "french", "finnish")
UNK = 0


# ---------------------------------------------------------------- データ

def words_of(docs: list[str]) -> list[str]:
    """議事録（小文字・空白正規化済み）の単語列。"""
    return [w for d in docs for w in d.split()]


def build_shared_vocab(train_words: dict[str, list[str]], per_lang: int) -> list[str]:
    """各言語の学習語の頻度上位 per_lang 語の和集合。番号 0 は <unk>。言語の並びに依存しないよう名前順に並べる。"""
    vocab = {"<unk>"}
    for lang in sorted(train_words):
        vocab.update(w for w, _ in Counter(train_words[lang]).most_common(per_lang))
    return ["<unk>"] + sorted(vocab - {"<unk>"})


def encode_words(words: list[str], stoi: dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi.get(w, UNK) for w in words], dtype=torch.long)


def build_tasks(root: Path, langs=LANGS, per_lang_vocab: int = 2500, n_train: int = 250_000, n_eval: int = 25_000,
                seq_len: int = 128, train_days: int = 7) -> dict:
    """言語ごとに、学習用（最初の train_days 日）と保持評価用（残りの日）の単語 ID を作る。

    学習用は先頭 n_train 語、評価用は残りの日の先頭 n_eval 語（seq_len の倍数に切る）。全条件・全 seed で同一。
    """
    raw = {}
    for lang in langs:
        docs = read_europarl(Path(root), lang)
        raw[lang] = (words_of(docs[:train_days]), words_of(docs[train_days:]))
    vocab = build_shared_vocab({l: raw[l][0] for l in langs}, per_lang_vocab)
    stoi = {w: i for i, w in enumerate(vocab)}
    tasks = {}
    for lang in langs:
        tr, ev = raw[lang]
        if len(tr) < n_train or len(ev) < n_eval:
            raise ValueError(f"{lang}: 語数が不足（train {len(tr)} < {n_train} または eval {len(ev)} < {n_eval}）")
        train_ids = encode_words(tr[:n_train], stoi)
        eval_ids = encode_words(ev[:n_eval], stoi)
        tasks[lang] = {"train": WordChunkDataset(train_ids, seq_len), "eval": WordChunkDataset(eval_ids, seq_len),
                       "unk_rate_eval": float((eval_ids == UNK).float().mean())}
    return {"vocab_size": len(vocab), "tasks": tasks, "langs": list(langs)}


def task_order(langs: list[str], seed: int) -> list[str]:
    """seed ごとに言語の並びを巡回させる（順序効果を seed 間で平均する。同一 seed では全条件で同じ並び）。"""
    k = seed % len(langs)
    return list(langs[k:]) + list(langs[:k])


# ---------------------------------------------------------------- 評価

@torch.no_grad()
def eval_task(model, dataset, batch_size: int = 64) -> tuple[float, float]:
    """backbone lm_head の次トークン CE（nats）と top-1 正確度。全フェーズ共通。"""
    model.eval()
    ce_sum, correct, n = 0.0, 0, 0
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        x = batch["input_ids"]
        logits = model(input_ids=x, labels=None, attention_mask=batch["attention_mask"])["logits"][:, :-1]
        tgt = x[:, 1:]
        ce_sum += torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                                    reduction="sum").item()
        correct += int((logits.argmax(-1) == tgt).sum().item())
        n += tgt.numel()
    return ce_sum / n, correct / n


def ce_forgetting(CE: list[list[float]]) -> dict[str, float]:
    """CE 行列（CE[i][j] = タスク i 学習後のタスク j の CE）からの忘却量（大きいほど忘却）。

    F_ce   … 過去の最小 CE（i = j..T-2）からの最終 CE の増加の平均（Chaudhry 流。F の CE 版）
    dCE    … 学習直後（対角）からの最終 CE の増加の平均（BWT の符号を反転した CE 版）
    ACC_ce … 最終時点の全タスク平均 CE（小さいほどよい）
    LA_ce  … 各タスクの学習直後の CE の平均（小さいほどよい。可塑性）
    """
    T = len(CE)
    last = CE[-1]
    return {
        "F_ce": sum(last[j] - min(CE[i][j] for i in range(j, T - 1)) for j in range(T - 1)) / (T - 1),
        "dCE": sum(last[j] - CE[j][j] for j in range(T - 1)) / (T - 1),
        "ACC_ce": sum(last) / T,
        "LA_ce": sum(CE[j][j] for j in range(T)) / T,
    }


def summarize(acc: list[list[float]], ce: list[list[float]]) -> dict[str, float]:
    m = {f"{k}_acc": v for k, v in forgetting_metrics(acc, chance=0.0).items()}  # 語彙が大きいので chance ≈ 0 とみなす
    m.update(ce_forgetting(ce))
    return m


# ---------------------------------------------------------------- 学習

def make_config(phase: str, args) -> Step444Config:
    return Step444Config(phase=phase, device="cpu", fp16=False, max_seq_length=args.max_seq_length,
                         hidden_size=args.hidden_size, num_layers=args.num_layers, nhead=args.nhead,
                         batch_size=args.batch_size, learning_rate=args.lr,
                         gradient_accumulation_steps=args.grad_accum)


def run_continual(phase: str, seed: int, data: dict, args, log=print) -> dict:
    """1 つの (フェーズ, seed) で K タスクを順に学び、各タスク後に全タスクを評価する。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = make_config(phase, args)
    model = BrainInspiredLLMFullScale(cfg, vocab_size=data["vocab_size"], hidden_size=cfg.hidden_size,
                                      num_layers=cfg.num_layers, nhead=cfg.nhead)
    trainer = FullScaleTrainer(model, cfg)  # オプティマイザはタスクをまたいで 1 つ（リセットしない）
    order = task_order(data["langs"], seed)
    T = len(order)
    acc = [[0.0] * T for _ in range(T)]
    ce = [[0.0] * T for _ in range(T)]
    losses, t0 = [], time.time()
    for i, lang in enumerate(order):
        trainer.optimizer.zero_grad()
        task_losses = []
        for epoch in range(args.epochs):
            g = torch.Generator().manual_seed(seed * 1000 + i * 10 + epoch)  # フェーズに依らず同じ並び
            for batch in DataLoader(data["tasks"][lang]["train"], batch_size=cfg.batch_size, shuffle=True,
                                    generator=g):
                task_losses.append(trainer.train_step(batch))
        losses.append(float(np.mean(task_losses[-20:])))
        for j, lj in enumerate(order):
            ce[i][j], acc[i][j] = eval_task(model, data["tasks"][lj]["eval"])
        log(f"[{phase} seed{seed}] task{i} {lang} done {time.time() - t0:.0f}s "
            f"diag ce {ce[i][i]:.3f} acc {acc[i][i]:.3f} | row ce " + " ".join(f"{c:.2f}" for c in ce[i]))
    params = sum(p.numel() for p in model.parameters())
    return {"phase": phase, "seed": seed, "order": order, "acc": acc, "ce": ce, "metrics": summarize(acc, ce),
            "final_train_loss_per_task": losses, "n_params": params,
            "n_params_backbone": sum(p.numel() for n, p in model.named_parameters()
                                     if not n.startswith(("hippocampus", "basal_ganglia", "cerebellum"))),
            "train_seconds": time.time() - t0, "env": env_info()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", nargs="+", default=["V", "A", "B", "C"], choices=["A", "B", "C", "V"])
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--europarl-dir", default=str(ROOT / "data" / "europarl_raw"))
    ap.add_argument("--langs", nargs="+", default=list(LANGS))
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.threads:
        torch.set_num_threads(args.threads)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = build_tasks(Path(args.europarl_dir), tuple(args.langs), args.per_lang_vocab, args.n_train, args.n_eval,
                       args.max_seq_length)
    print(f"vocab {data['vocab_size']} unk_rate_eval "
          + str({l: round(data['tasks'][l]['unk_rate_eval'], 3) for l in data['langs']}), flush=True)
    for seed in args.seeds:
        for phase in args.phases:
            path = out / f"phase{phase}_seed{seed}.json"
            if path.exists():
                continue
            res = run_continual(phase, seed, data, args, log=lambda s: print(s, flush=True))
            res["config"] = {k: v for k, v in vars(args).items()}
            path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
            print(f"saved {path} metrics {res['metrics']}", flush=True)


if __name__ == "__main__":
    main()
