#!/usr/bin/env python3
"""ステップ44 基底核 critic 比較: legacy / none / td / td_rpe（docs/decisions/2026-09-26-step44-bg-critic-fix.md）。"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from step44_optimized_validation import OptimizedValidationConfig, run_phase_validation

MODES = ["legacy", "none", "td", "td_rpe", "td_sg", "td_rpe_sg"]


def final_ppls(phase, mode, seeds, pos_encoding=False):
    cfg = OptimizedValidationConfig(phase=phase, bg_critic_mode=mode, pos_encoding=pos_encoding)
    runs = [run_phase_validation(cfg, s) for s in seeds]
    return [r["val_losses"][-1] for r in runs], [r["train_losses"][-1] for r in runs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-encoding", action="store_true", help="位置埋め込みあり（ステップ44.7）")
    args = ap.parse_args()
    pe = args.pos_encoding
    OUT = Path("results/step44_bg_critic_ablation_pos" if pe else "results/step44_bg_critic_ablation")
    logging.basicConfig(level=logging.WARNING)
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = range(OptimizedValidationConfig().num_seeds)

    ppl_a, _ = final_ppls("A", "legacy", seeds, pe)
    summary = {"A": {"ppls": ppl_a, "mean": float(np.mean(ppl_a)), "std": float(np.std(ppl_a))}}
    print(f"A: {np.mean(ppl_a):.4f} ± {np.std(ppl_a):.4f}")

    for mode in MODES:
        summary[mode] = {}
        for phase in ["B", "C"]:
            ppls, train = final_ppls(phase, mode, seeds, pe)
            m = float(np.mean(ppls))
            summary[mode][phase] = {
                "ppls": ppls, "mean": m, "std": float(np.std(ppls)),
                "final_train_loss": [float(t) for t in train],
                "vs_A_pct": (m / summary["A"]["mean"] - 1) * 100,
            }
            print(f"{mode:7s} {phase}: {m:.4f} ± {np.std(ppls):.4f} "
                  f"(A比 {summary[mode][phase]['vs_A_pct']:+.2f}%, train_loss {np.mean(train):+.4f})")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
