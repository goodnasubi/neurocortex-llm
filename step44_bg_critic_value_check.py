#!/usr/bin/env python3
"""基底核 critic の価値学習の検証: V_t と実現割引リターン G_t = Σ γ^k r_{t+k}（r = −CE, 系列末で打ち切り）の一致度。"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from step44_optimized_validation import (
    BrainInspiredLLMPhase, CorpusDataset, OptimizedTrainer, OptimizedValidationConfig, load_corpus,
)

MODES = ["legacy", "td", "td_sg"]
OUT = Path("results/step44_bg_critic_ablation/value_check.json")


def value_fit(model, loader, gamma):
    model.eval()
    vs, gs = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["input_ids"]
            bg = model.basal_ganglia
            h = model.backbone[1](model.backbone[0](x),
                                  src_mask=torch.nn.Transformer.generate_square_subsequent_mask(x.size(1)),
                                  is_causal=True)
            v = bg.critic(h)[:, :-1, 0]
            logits = bg.actor(h)[:, :-1]
            r = -torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), reduction="none"
            ).view(v.shape)
            g = torch.zeros_like(r)
            acc = torch.zeros(r.size(0))
            for t in range(r.size(1) - 1, -1, -1):
                acc = r[:, t] + gamma * acc
                g[:, t] = acc
            vs.append(v); gs.append(g)
    V, G = torch.cat(vs).numpy(), torch.cat(gs).numpy()
    # 位置ごとの平均を除いた相関: 打ち切り由来の位置依存性を除き、内容依存の価値予測を測る
    Vc, Gc = (V - V.mean(0)).ravel(), (G - G.mean(0)).ravel()
    v, g = V.ravel(), G.ravel()
    return {
        "pearson_r_position_centered": float(np.corrcoef(Vc, Gc)[0, 1]),
        "pearson_r": float(np.corrcoef(v, g)[0, 1]),
        "explained_variance": float(1 - np.var(g - v) / np.var(g)),
        "mean_V": float(v.mean()), "mean_G": float(g.mean()),
    }


def run(mode, seed):
    cfg = OptimizedValidationConfig(phase="B", bg_critic_mode=mode)
    torch.manual_seed(seed); np.random.seed(seed)
    corpus = load_corpus(seed=0)
    cfg.vocab_size = corpus.vocab_size
    train = DataLoader(CorpusDataset(corpus.train, cfg.num_train_samples, cfg.seq_length, seed=seed),
                       batch_size=cfg.batch_size, shuffle=True)
    val = DataLoader(CorpusDataset(corpus.val, cfg.num_val_samples, cfg.seq_length, seed=seed + 1000),
                     batch_size=cfg.batch_size)
    model = BrainInspiredLLMPhase(cfg)
    trainer = OptimizedTrainer(model, cfg)
    before = value_fit(model, val, cfg.bg_gamma)
    for _ in range(cfg.num_epochs):
        trainer.train_epoch(train)
    return {"before": before, "after": value_fit(model, val, cfg.bg_gamma)}


def main():
    out = {}
    for mode in MODES:
        out[mode] = [run(mode, s) for s in range(3)]
        a = out[mode]
        print(f"{mode:7s} r: {np.mean([x['before']['pearson_r'] for x in a]):+.3f} -> "
              f"{np.mean([x['after']['pearson_r'] for x in a]):+.3f} | EV: "
              f"{np.mean([x['after']['explained_variance'] for x in a]):+.3f} | pos-centered r: "
              f"{np.mean([x['before']['pearson_r_position_centered'] for x in a]):+.3f} -> "
              f"{np.mean([x['after']['pearson_r_position_centered'] for x in a]):+.3f} | "
              f"mean V {np.mean([x['after']['mean_V'] for x in a]):+.3f} vs G {np.mean([x['after']['mean_G'] for x in a]):+.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
