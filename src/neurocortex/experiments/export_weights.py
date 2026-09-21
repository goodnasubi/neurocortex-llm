"""学習済みLMの重みを .npz へ書き出す（12.11.2節の環境間受け渡し方針）。

WSL2側(torch 2.x)で学習し、Jetson側(torch 1.12)で読むため、torch.save の形式に
依存しない numpy 形式を使う。将来のRust実装からも同じファイルを読める。

使い方:
    python -m neurocortex.experiments.export_weights --steps 1000 --out results/weights
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..data import load_corpus
from ..lm import DenseBaseline, SpikingLM, count_params
from ..neurons import NeuronConfig


def train(model, corpus, args, tag):
    torch.manual_seed(args.seed)
    g = torch.Generator(); g.manual_seed(args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    t0 = time.time()
    for step in range(args.steps):
        model.train()
        x, y = corpus.batch("train", args.batch_size, args.seq_len, g)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % 200 == 0:
            print(f"  [{tag}] step {step+1}/{args.steps} loss={float(loss):.4f}", flush=True)
    return {"tag": tag, "params": count_params(model), "final_loss": float(loss),
            "train_sec": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="docs/brain-structure-research.md")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--beta", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default="results/weights")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    corpus = load_corpus(args.data)
    cfg = NeuronConfig(beta=args.beta)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    meta = {"args": vars(args), "vocab_size": corpus.vocab_size, "runs": []}

    builders = {
        "spiking": lambda: SpikingLM(corpus.vocab_size, d_model=args.d_model,
                                     n_layers=args.n_layers, n_heads=args.n_heads, cfg=cfg),
        "dense": lambda: DenseBaseline(corpus.vocab_size, d_model=args.d_model,
                                       n_layers=args.n_layers, n_heads=args.n_heads),
    }
    for tag, build in builders.items():
        torch.manual_seed(args.seed)
        model = build()
        info = train(model, corpus, args, tag)
        arrays = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        np.savez(out / f"{tag}.npz", **arrays)
        info["npz"] = str(out / f"{tag}.npz")
        info["n_tensors"] = len(arrays)
        meta["runs"].append(info)
        print(f"  [{tag}] 保存: {info['npz']} ({len(arrays)} テンソル)", flush=True)

    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("完了:", out / "meta.json")


if __name__ == "__main__":
    main()
