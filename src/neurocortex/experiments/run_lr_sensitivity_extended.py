"""ステップ25（学習率0.1台でのスパイキング発散再検証, 12.6.58節）の実験スクリプト。

`run_lm_efficiency.py`の`run_lr_sensitivity`関数（コード変更なし）をそのまま使い、
ステップ17の既存6学習率（1e-4〜3e-2）に0.1・0.3を加えた8点で`SpikingLM`・
`DenseBaseline`の`diverged_fraction`を比較する。

`run_lm_efficiency.main()`は本実験に不要な(a)収束速度比較（既定1000ステップ×5シード×2モデル）も
常に実行してしまうため、時間短縮のため`run_lr_sensitivity`のみを直接呼び出す薄いラッパーとした
（`run_lr_sensitivity`関数自体は一切変更していない）。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_lr_sensitivity_extended
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..data import load_corpus
from ..neurons import NeuronConfig
from .run_lm_efficiency import run_lr_sensitivity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3,
                    help="設計文書の基準は5だが、実行時間短縮のため既定を3に調整"
                         "（詳細はdocs/brain-structure-research.md 12.6.59節参照）")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr-sweep-steps", type=int, default=300)
    ap.add_argument("--lrs", type=float, nargs="+",
                    default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3])
    ap.add_argument("--beta", type=float, default=0.95)
    ap.add_argument("--out", type=Path,
                    default=Path("results/lm_efficiency/lr_sensitivity_extended.json"))
    args = ap.parse_args()

    t0 = time.time()
    corpus = load_corpus(None, seed=0)
    ncfg = NeuronConfig(beta=args.beta)

    lr_sensitivity = run_lr_sensitivity(
        corpus, args.d_model, args.n_layers, args.n_heads, args.seq_len, args.batch_size,
        args.lr_sweep_steps, args.seeds, args.lrs, ncfg)

    existing_lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    spiking = {lr: v["diverged_fraction"] for lr, v in lr_sensitivity["spiking"].items()}
    dense = {lr: v["diverged_fraction"] for lr, v in lr_sensitivity["dense"].items()}

    control_diffs = {lr: abs(spiking[lr] - dense[lr]) for lr in existing_lrs}
    control_b_ok = all(d <= 0.2 for d in control_diffs.values())

    high_lrs = [lr for lr in args.lrs if lr not in existing_lrs]
    claim_a_diffs = {lr: spiking[lr] - dense[lr] for lr in high_lrs}
    claim_a_ok = any(d >= 0.4 for d in claim_a_diffs.values())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "ステップ25 学習率0.1台でのスパイキング発散再検証（12.6.58〜59節）: "
                "SpikingLM対DenseBaselineのdiverged_fractionを1e-4〜0.3の8学習率で比較",
        "条件": vars(args) | {"out": str(args.out)},
        "lr_sensitivity": lr_sensitivity,
        "control_diffs（既存6点の|spiking-dense|）": control_diffs,
        "claim_a_diffs（拡張2点のspiking-dense）": claim_a_diffs,
        "control_b_ok（統制/主張b: 既存6点で差が全点0.2以内）": control_b_ok,
        "claim_a_ok（主張a: 0.1または0.3のいずれかで差が0.4以上）": claim_a_ok,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n書き出し: {args.out}")
    for tag, per_lr in lr_sensitivity.items():
        row = "  ".join(f"lr={lr}:{v['diverged_fraction']:.2f}" for lr, v in per_lr.items())
        print(f"  {tag}: {row}")
    print(f"\ncontrol_b_ok: {control_b_ok}  claim_a_ok: {claim_a_ok}")


if __name__ == "__main__":
    main()
