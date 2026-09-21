"""遅延 N と漏れ係数 β の掃引（12.6.1節の副次条件・失敗時の打ち切り基準）。

目的:
  1. N ∈ {4, 8, 16, 32, 64} の精度曲線を取り、β から予測される保持時間
     （おおよそ 1/(1-β)）と整合するかを確認する。整合すればメカニズムの
     理解が正しいことの追加証拠になる。
  2. 12.6.1節「失敗時の扱い」が定める β の5点掃引をここで実施する。

使い方:
    python -m neurocortex.experiments.run_delay_sweep --seeds 3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..neurons import NeuronConfig
from ..tasks import RecallSpec
from .identification import TrainConfig, evaluate, train_recall


def main() -> None:
    p = argparse.ArgumentParser(description="遅延Nとβの掃引")
    p.add_argument("--delays", type=str, default="4,8,16,32,64")
    p.add_argument("--betas", type=str, default="0.5,0.7,0.9,0.95,0.99")
    p.add_argument("--activations", type=str, default="spiking")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--out", type=str, default="results/delay_sweep.json")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    delays = [int(x) for x in args.delays.split(",")]
    betas = [float(x) for x in args.betas.split(",")]
    acts = args.activations.split(",")
    rows: list[dict] = []

    for act in acts:
        for beta in betas:
            for delay in delays:
                spec = RecallSpec(n_cue=8, n_filler=8, max_delay=delay)
                ncfg = NeuronConfig(beta=beta)
                tcfg = TrainConfig(steps=args.steps, lr=args.lr, batch_size=args.batch_size,
                                   d_model=args.d_model, n_layers=args.n_layers)
                accs = []
                t0 = time.time()
                for seed in range(args.seeds):
                    model, _ = train_recall(spec, ncfg, tcfg, act, seed)
                    accs.append(evaluate(model, spec, tcfg, seed, delay=delay))
                row = {
                    "activation": act, "beta": beta, "delay": delay,
                    "retention_estimate": 1.0 / (1.0 - beta),
                    "acc_per_seed": accs,
                    "mean_acc": sum(accs) / len(accs),
                    "chance": spec.chance,
                }
                rows.append(row)
                print(f"[sweep] {act} beta={beta:<5g} N={delay:<3d} "
                      f"mean_acc={row['mean_acc']:.4f} "
                      f"(保持時間の目安 {row['retention_estimate']:.1f}) "
                      f"({time.time() - t0:.0f}s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"保存: {out}")


if __name__ == "__main__":
    main()
