"""テストB（12.6.1節・確認用、限定的）: 実構成での順序判別。

単層の因果的線形アテンションを持つ構成で、記号A, Bのどちらが後に出たかを答える。
アテンション状態は順序不変なので、膜電位を持つ版だけが解けるはずである。
12.6.1節の明記の通り、主張の重みはテストAに置き、本テストは補助扱いとする。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ..neurons import NeuronConfig
from ..order_model import OrderModel
from ..tasks import OrderSpec, make_order_batch


def run_one(spec: OrderSpec, ncfg: NeuronConfig, activation: str, seed: int,
            steps: int, lr: float, batch_size: int, d_model: int) -> tuple[OrderModel, float]:
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed + 10_000)
    model = OrderModel(spec.vocab_size, d_model=d_model, cfg=ncfg, activation=activation)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        model.train()
        x, y = make_order_batch(spec, batch_size, g)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model, evaluate(model, spec, seed)


@torch.no_grad()
def evaluate(model: OrderModel, spec: OrderSpec, seed: int, n_batches: int = 8,
             batch_size: int = 256) -> float:
    model.eval()
    g = torch.Generator()
    g.manual_seed(seed + 999_983)
    correct = total = 0
    for _ in range(n_batches):
        x, y = make_order_batch(spec, batch_size, g)
        correct += int((model(x).argmax(-1) == y).sum())
        total += y.numel()
    return correct / total


def main() -> None:
    p = argparse.ArgumentParser(description="テストB: 順序判別（単層限定）")
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--beta", type=float, default=0.95)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--out", type=str, default="results/order_testB.json")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    spec = OrderSpec(seq_len=args.seq_len)
    ncfg = NeuronConfig(beta=args.beta)
    res: dict = {"args": vars(args), "chance": spec.chance, "runs": {}, "ablation": {}}
    for act in ("spiking", "step"):
        accs, abl = [], []
        for seed in range(args.seeds):
            t0 = time.time()
            model, acc = run_one(spec, ncfg, act, seed, args.steps, args.lr,
                                 args.batch_size, args.d_model)
            accs.append(acc)
            if act == "spiking":
                model.set_carry_membrane(False)
                abl.append(evaluate(model, spec, seed))
                model.set_carry_membrane(True)
            print(f"[testB] {act:8s} seed={seed} acc={acc:.4f}"
                  + (f" (膜電位ゼロ化: {abl[-1]:.4f})" if abl else "")
                  + f" ({time.time() - t0:.0f}s)", flush=True)
        res["runs"][act] = {"acc_per_seed": accs, "mean_acc": sum(accs) / len(accs)}
        if abl:
            res["ablation"][act] = {"acc_per_seed": abl, "mean_acc": sum(abl) / len(abl)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"保存: {out}")


if __name__ == "__main__":
    main()
