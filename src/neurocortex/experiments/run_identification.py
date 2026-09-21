"""識別テスト（12.6.1節 テストA）の本実験ランナー。

実施内容:
  主実験  : 遅延 0..N をランダムに振って学習し、N で評価。スパイキング版 vs 階段関数版。
  統制1    : 対照群（階段関数）が N=0 の課題は解けることを確認（容量の統制）。
  統制2    : スパイキング版の学習済みモデルの膜電位の持ち越しだけを推論時にゼロにする。
  統制3    : 学習ステップ数・学習率グリッド・シードを両群で完全に共有（本スクリプトの構造で担保）。
  統制4    : 既定5シード。
  統制5    : tests/test_path_audit.py で機械的に保証（本スクリプトの対象外）。

使い方:
    python -m neurocortex.experiments.run_identification --delay 16 --seeds 5
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from ..neurons import NeuronConfig
from ..tasks import RecallSpec
from .identification import TrainConfig, evaluate, train_recall


def _sigma(p: float, n: int) -> float:
    """二項分布の標準偏差（チャンスレベル+2σ の判定に使う）。"""
    return math.sqrt(p * (1.0 - p) / n)


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    spec = RecallSpec(n_cue=args.n_cue, n_filler=args.n_filler, max_delay=args.delay)
    ncfg = NeuronConfig(beta=args.beta)
    seeds = list(range(args.seeds))
    lrs = [float(x) for x in args.lrs.split(",")]
    n_eval = args.eval_batches * args.eval_batch_size
    chance = spec.chance
    threshold = chance + 2.0 * _sigma(chance, n_eval)

    results: dict = {
        "spec": {"n_cue": spec.n_cue, "n_filler": spec.n_filler, "delay": args.delay,
                 "vocab_size": spec.vocab_size, "seq_len": spec.seq_len},
        "neuron": {"beta": ncfg.beta, "theta": ncfg.theta, "kappa": ncfg.kappa,
                   "lam": ncfg.lam, "alpha": ncfg.alpha, "n_steps": ncfg.n_steps},
        "chance": chance,
        "chance_plus_2sigma": threshold,
        "n_eval_samples": n_eval,
        "lr_grid": lrs,
        "seeds": seeds,
        "main": {},
        "control1_zero_delay": {},
        "control2_membrane_ablation": {},
    }

    tcfg_base = dict(steps=args.steps, batch_size=args.batch_size, d_model=args.d_model,
                     n_layers=args.n_layers, eval_batches=args.eval_batches,
                     eval_batch_size=args.eval_batch_size)

    # ---- 主実験（統制3・4: 両群でまったく同じ (lr, seed) の組を回す） ----
    trained: dict[tuple[str, float, int], object] = {}
    for act in ("spiking", "step"):
        per_lr: dict[str, dict] = {}
        for lr in lrs:
            accs, rates, losses = [], [], []
            for seed in seeds:
                tcfg = TrainConfig(lr=lr, **tcfg_base)
                t0 = time.time()
                model, log = train_recall(spec, ncfg, tcfg, act, seed)
                acc = evaluate(model, spec, tcfg, seed, delay=args.delay)
                accs.append(acc)
                rates.append(log["firing_rates"])
                losses.append(log["final_loss"])
                trained[(act, lr, seed)] = model
                print(f"[main] {act:8s} lr={lr:<7g} seed={seed} acc@{args.delay}={acc:.4f} "
                      f"loss={log['final_loss']:.4f} fr={[round(r, 3) for r in log['firing_rates']]} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            per_lr[f"{lr:g}"] = {
                "acc_per_seed": accs,
                "mean_acc": sum(accs) / len(accs),
                "min_acc": min(accs),
                "final_loss_per_seed": losses,
                "firing_rates_per_seed": rates,
            }
        best_lr = max(per_lr, key=lambda k: per_lr[k]["mean_acc"])
        results["main"][act] = {"per_lr": per_lr, "best_lr": best_lr, **per_lr[best_lr]}

    # ---- 統制1: 対照群が N=0 を解けるか（容量の統制） ----
    for act in ("spiking", "step"):
        accs = []
        for seed in seeds:
            tcfg = TrainConfig(lr=float(results["main"][act]["best_lr"]), **tcfg_base)
            model, _ = train_recall(spec, ncfg, tcfg, act, seed, train_delay=0)
            acc = evaluate(model, spec, tcfg, seed, delay=0)
            accs.append(acc)
            print(f"[ctrl1] {act:8s} seed={seed} acc@0={acc:.4f}", flush=True)
        results["control1_zero_delay"][act] = {
            "lr": results["main"][act]["best_lr"],
            "acc_per_seed": accs,
            "mean_acc": sum(accs) / len(accs),
            "min_acc": min(accs),
        }

    # ---- 統制2: スパイキング版の膜電位の持ち越しだけを推論時にゼロにする ----
    best_lr_sp = float(results["main"]["spiking"]["best_lr"])
    accs_on, accs_off = [], []
    for seed in seeds:
        model = trained[("spiking", best_lr_sp, seed)]
        tcfg = TrainConfig(lr=best_lr_sp, **tcfg_base)
        model.set_carry_membrane(True)  # type: ignore[attr-defined]
        a_on = evaluate(model, spec, tcfg, seed, delay=args.delay)  # type: ignore[arg-type]
        model.set_carry_membrane(False)  # type: ignore[attr-defined]
        a_off = evaluate(model, spec, tcfg, seed, delay=args.delay)  # type: ignore[arg-type]
        model.set_carry_membrane(True)  # type: ignore[attr-defined]
        accs_on.append(a_on)
        accs_off.append(a_off)
        print(f"[ctrl2] seed={seed} carry=ON {a_on:.4f} -> OFF {a_off:.4f}", flush=True)
    results["control2_membrane_ablation"] = {
        "lr": best_lr_sp,
        "acc_carry_on": accs_on,
        "acc_carry_off": accs_off,
        "mean_off": sum(accs_off) / len(accs_off),
        "max_off": max(accs_off),
    }

    # ---- 合格判定（12.6.1節の合格条件） ----
    sp = results["main"]["spiking"]
    st = results["main"]["step"]
    verdict = {
        "spiking_all_seeds_ge_90": all(a >= 0.90 for a in sp["acc_per_seed"]),
        "control_all_seeds_within_chance_2sigma": all(a <= threshold for a in st["acc_per_seed"]),
        "control1_step_solves_zero_delay": results["control1_zero_delay"]["step"]["min_acc"] >= 0.90,
        "control2_ablation_drops_to_chance": all(
            a <= threshold for a in results["control2_membrane_ablation"]["acc_carry_off"]
        ),
    }
    verdict["passed"] = all(verdict.values())
    results["verdict"] = verdict
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="識別テスト（12.6.1節 テストA）")
    p.add_argument("--delay", type=int, default=16, help="評価する遅延 N")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lrs", type=str, default="1e-3,3e-3,1e-2")
    p.add_argument("--beta", type=float, default=0.95)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-cue", type=int, default=8)
    p.add_argument("--n-filler", type=int, default=8)
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--out", type=str, default="results/identification_N{delay}.json")
    args = p.parse_args()

    res = run(args)
    out = Path(args.out.format(delay=args.delay))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res["verdict"], indent=2, ensure_ascii=False))
    print(f"保存: {out}")


if __name__ == "__main__":
    main()
