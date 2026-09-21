"""言語モデル副指標（12.6節）の実験ランナー。

確認する項目:
  - 同規模の密なTransformerとの言語モデリング性能（検証PPL）の比較
  - 発火率の分布（0%や100%に張り付いていないこと）
  - 小バッチへの過学習が可能なこと
  - 学習が発散しないこと

使い方:
    python -m neurocortex.experiments.run_lm --steps 1500 --seq-len 64
    python -m neurocortex.experiments.run_lm --data path/to/corpus.txt
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ..data import load_corpus
from ..lm import DenseBaseline, SpikingLM, count_params
from ..neurons import NeuronConfig


def _loss(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


@torch.no_grad()
def eval_ppl(model, corpus, batch_size: int, seq_len: int, n_batches: int, seed: int) -> float:
    model.eval()
    g = torch.Generator()
    g.manual_seed(seed + 555)
    total = 0.0
    for _ in range(n_batches):
        x, y = corpus.batch("val", batch_size, seq_len, g)
        total += float(_loss(model, x, y))
    return math.exp(total / n_batches)


def train_lm(model, corpus, args, seed: int, tag: str) -> dict:
    """1モデルを学習して学習曲線・PPL・発火率を返す。"""
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    curve: list[dict] = []
    diverged = False
    t0 = time.time()
    for step in range(args.steps):
        model.train()
        x, y = corpus.batch("train", args.batch_size, args.seq_len, g)
        loss = _loss(model, x, y)
        if not torch.isfinite(loss):
            diverged = True
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % max(1, args.steps // 10) == 0:
            ppl = eval_ppl(model, corpus, args.batch_size, args.seq_len, 4, seed)
            curve.append({"step": step + 1, "train_loss": float(loss), "val_ppl": ppl,
                          "grad_norm": float(gnorm)})
            print(f"  [{tag}] step {step + 1}/{args.steps} loss={float(loss):.4f} "
                  f"val_ppl={ppl:.3f} gnorm={float(gnorm):.2f}", flush=True)
    out = {
        "params": count_params(model),
        "diverged": diverged,
        "curve": curve,
        "val_ppl": eval_ppl(model, corpus, args.batch_size, args.seq_len, 16, seed)
        if not diverged else None,
        "seconds": time.time() - t0,
    }
    if hasattr(model, "firing_rates"):
        out["firing_rates"] = model.firing_rates()
    return out


def overfit_check(model, corpus, args, seed: int, tag: str) -> dict:
    """小バッチ（8系列）を繰り返し与えて過学習できるかを見る健全性チェック。"""
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    x, y = corpus.batch("train", 8, args.seq_len, g)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    first = last = None
    for step in range(args.overfit_steps):
        loss = _loss(model, x, y)
        if step == 0:
            first = float(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = float(loss)
    print(f"  [{tag}] overfit {first:.4f} -> {last:.4f}", flush=True)
    return {"initial_loss": first, "final_loss": last, "final_ppl": math.exp(last)}


def main() -> None:
    p = argparse.ArgumentParser(description="言語モデル副指標（12.6節）")
    p.add_argument("--data", type=str, default=None, help="テキストファイル。省略時は合成コーパス")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--overfit-steps", type=int, default=300)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--out", type=str, default="results/lm.json")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    corpus = load_corpus(args.data, seed=args.seed)
    ncfg = NeuronConfig(beta=args.beta)
    common = dict(vocab_size=corpus.vocab_size, d_model=args.d_model,
                  n_layers=args.n_layers, n_heads=args.n_heads, max_len=args.seq_len)

    def build_spiking(activation: str = "spiking"):
        return SpikingLM(cfg=ncfg, activation=activation, **common)

    results = {
        "args": vars(args),
        "corpus": {"vocab_size": corpus.vocab_size, "train_chars": int(corpus.train.numel()),
                   "val_chars": int(corpus.val.numel()), "source": args.data or "synthetic"},
        "runs": {},
        "overfit": {},
    }
    builders = {
        "spiking": build_spiking,
        "step": lambda: build_spiking("step"),
        "dense": lambda: DenseBaseline(**common),
    }
    for tag, build in builders.items():
        print(f"== {tag} ==", flush=True)
        results["runs"][tag] = train_lm(build(), corpus, args, args.seed, tag)
        results["overfit"][tag] = overfit_check(build(), corpus, args, args.seed, tag)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: {"params": v["params"], "val_ppl": v["val_ppl"],
                          "diverged": v["diverged"],
                          "firing_rates": v.get("firing_rates")}
                      for k, v in results["runs"].items()}, indent=2, ensure_ascii=False))
    print(f"保存: {out}")


if __name__ == "__main__":
    main()
