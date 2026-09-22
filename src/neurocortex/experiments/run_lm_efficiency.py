"""ステップ17（スパイクベース設計に伴う学習効率上の代償の実測, 12.6.42節）の主実験。

(a) 収束速度: `spiking`・`dense`を複数シードで固定ステップ数学習し、目標PPLへの
    到達ステップ数・最終PPLを比較する。
(b) 学習率感度: 複数の学習率を振り、発散頻度を比較する。

`lm.py`の`SpikingLM`・`DenseBaseline`と、`run_lm.py`の`train_lm`をそのまま使う。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_lm_efficiency
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from ..data import load_corpus
from ..lm import DenseBaseline, SpikingLM
from ..neurons import NeuronConfig
from .run_lm import train_lm


@dataclass
class Args:
    """`train_lm`が参照する属性だけを持つ軽量な設定オブジェクト。"""

    steps: int
    seq_len: int
    batch_size: int
    lr: float


def steps_to_reach_ppl(curve: list[dict], target_ppl: float) -> int | None:
    """学習曲線（ステップごとの`val_ppl`の記録）から、目標PPLに初めて到達したステップ数を返す。
    到達しなければ`None`。
    """
    for point in curve:
        if point["val_ppl"] <= target_ppl:
            return point["step"]
    return None


def build_model(tag: str, vocab_size: int, d_model: int, n_layers: int, n_heads: int,
                seq_len: int, ncfg: NeuronConfig) -> torch.nn.Module:
    if tag == "spiking":
        return SpikingLM(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
                         n_heads=n_heads, max_len=seq_len, cfg=ncfg, activation="spiking")
    if tag == "dense":
        return DenseBaseline(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
                             n_heads=n_heads, max_len=seq_len)
    raise ValueError(tag)


def run_convergence_comparison(corpus, d_model: int, n_layers: int, n_heads: int, seq_len: int,
                               batch_size: int, lr: float, steps: int, seeds: int, ncfg: NeuronConfig,
                               target_ppl: float) -> dict:
    """(a) 収束速度の比較。"""
    args = Args(steps=steps, seq_len=seq_len, batch_size=batch_size, lr=lr)
    results: dict[str, dict] = {}
    for tag in ("spiking", "dense"):
        steps_to_target: list[int | None] = []
        final_ppls: list[float | None] = []
        for seed in range(seeds):
            model = build_model(tag, corpus.vocab_size, d_model, n_layers, n_heads, seq_len, ncfg)
            out = train_lm(model, corpus, args, seed, tag)
            steps_to_target.append(steps_to_reach_ppl(out["curve"], target_ppl))
            final_ppls.append(out["val_ppl"])
        results[tag] = {
            "steps_to_target_ppl": steps_to_target,
            "final_val_ppl": final_ppls,
            "reached_target_fraction": sum(s is not None for s in steps_to_target) / seeds,
        }
    return results


def run_lr_sensitivity(corpus, d_model: int, n_layers: int, n_heads: int, seq_len: int,
                       batch_size: int, steps: int, seeds: int, lrs: list[float],
                       ncfg: NeuronConfig) -> dict:
    """(b) 学習率感度の比較。"""
    results: dict[str, dict] = {}
    for tag in ("spiking", "dense"):
        per_lr: dict[float, dict] = {}
        for lr in lrs:
            args = Args(steps=steps, seq_len=seq_len, batch_size=batch_size, lr=lr)
            diverged_flags = []
            for seed in range(seeds):
                model = build_model(tag, corpus.vocab_size, d_model, n_layers, n_heads, seq_len,
                                    ncfg)
                out = train_lm(model, corpus, args, seed, f"{tag}@lr={lr}")
                diverged_flags.append(out["diverged"])
            per_lr[lr] = {"diverged_fraction": sum(diverged_flags) / seeds}
        results[tag] = per_lr
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2e-3, help="収束速度比較で使う固定学習率")
    ap.add_argument("--target-ppl", type=float, default=3.0)
    ap.add_argument("--lr-sweep-steps", type=int, default=300)
    ap.add_argument("--lrs", type=float, nargs="+",
                    default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    ap.add_argument("--beta", type=float, default=0.95)
    ap.add_argument("--out", type=Path,
                    default=Path("results/lm_efficiency/spiking_vs_dense_efficiency.json"))
    args = ap.parse_args()

    t0 = time.time()
    corpus = load_corpus(None, seed=0)
    ncfg = NeuronConfig(beta=args.beta)

    convergence = run_convergence_comparison(
        corpus, args.d_model, args.n_layers, args.n_heads, args.seq_len, args.batch_size,
        args.lr, args.steps, args.seeds, ncfg, args.target_ppl)

    lr_sensitivity = run_lr_sensitivity(
        corpus, args.d_model, args.n_layers, args.n_heads, args.seq_len, args.batch_size,
        args.lr_sweep_steps, args.seeds, args.lrs, ncfg)

    def _mean_steps_to_target(entry: dict) -> float | None:
        """未到達(None)を含む場合は`steps`（打ち切り上限）として扱い、保守的に平均する。"""
        values = [s if s is not None else args.steps for s in entry["steps_to_target_ppl"]]
        return sum(values) / len(values)

    spiking_mean_steps = _mean_steps_to_target(convergence["spiking"])
    dense_mean_steps = _mean_steps_to_target(convergence["dense"])
    spiking_mean_ppl = sum(convergence["spiking"]["final_val_ppl"]) / args.seeds
    dense_mean_ppl = sum(convergence["dense"]["final_val_ppl"]) / args.seeds
    # 目標到達率が同等でも、到達ステップ数または最終PPLのどちらかで明確に劣れば代償ありとみなす。
    claim_a_ok = bool(spiking_mean_steps > dense_mean_steps * 1.1 or
                      spiking_mean_ppl > dense_mean_ppl * 1.1)

    spiking_div = {lr: v["diverged_fraction"] for lr, v in lr_sensitivity["spiking"].items()}
    dense_div = {lr: v["diverged_fraction"] for lr, v in lr_sensitivity["dense"].items()}
    claim_b_ok = bool(sum(spiking_div.values()) > sum(dense_div.values()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "ステップ17 スパイクベース設計に伴う学習効率上の代償の実測: "
                "SpikingLM対DenseBaselineの(a)収束速度・(b)学習率感度の比較（12.6.42節）",
        "条件": vars(args) | {"out": str(args.out)},
        "convergence": convergence,
        "lr_sensitivity": lr_sensitivity,
        "convergence_summary": {
            "spiking_mean_steps_to_target": spiking_mean_steps,
            "dense_mean_steps_to_target": dense_mean_steps,
            "spiking_mean_final_ppl": spiking_mean_ppl,
            "dense_mean_final_ppl": dense_mean_ppl,
        },
        "claim_a_ok（収束速度で代償が見られたか）": claim_a_ok,
        "claim_b_ok（学習率感度で代償が見られたか）": claim_b_ok,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n書き出し: {args.out}")
    print("-- (a) 収束速度 --")
    for tag, r in convergence.items():
        print(f"  {tag}: 目標到達率={r['reached_target_fraction']:.2f}  "
              f"steps_to_target={r['steps_to_target_ppl']}  final_ppl={r['final_val_ppl']}")
    print("-- (b) 学習率感度（発散率） --")
    for tag, per_lr in lr_sensitivity.items():
        row = "  ".join(f"lr={lr}:{v['diverged_fraction']:.2f}" for lr, v in per_lr.items())
        print(f"  {tag}: {row}")
    print(f"\nclaim_a_ok: {claim_a_ok}  claim_b_ok: {claim_b_ok}")


if __name__ == "__main__":
    main()
