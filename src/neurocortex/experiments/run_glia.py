"""ステップ5（グリア層のBCM型ロードバランシング, 12.6.17節）の主実験。

3条件（aux-loss / proportional / bcm）で専門家選択のロードバランシングを行い、
訓練フェーズ（勾配・局所更新とも有効）の後、入力の偏りを反転させ（統制3）、
以降は**aux-lossの補正だけを凍結**（推論時のみのフェーズを模す）して再収束を
比較する。主張(a): aux-lossは凍結後に適応できず不均衡が残る。主張(b): BCMは
proportionalより偏り反転後の再収束が速い・安定している。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_glia --seeds 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..glia import (
    AuxLossCorrection,
    LoadBalanceSpec,
    bcm_update,
    flip_bias,
    imbalance,
    make_bias,
    proportional_update,
    route_batch,
)

CONDITIONS = ("aux-loss", "proportional", "bcm")


def run_one(condition: str, spec: LoadBalanceSpec, args: argparse.Namespace, seed: int) -> list[dict]:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 40_000)
    bias_v1 = make_bias(spec.n_experts, seed=seed, scale=args.bias_scale)
    bias_v2 = flip_bias(bias_v1)

    if condition == "aux-loss":
        aux = AuxLossCorrection(spec.n_experts, lr=args.lr_aux)
        correction = aux.correction
    else:
        aux = None
        correction = torch.zeros(spec.n_experts)
    msq_state = torch.zeros(spec.n_experts)

    rows = []
    for step in range(1, args.n_steps + 1):
        active_bias = bias_v1 if step <= args.flip_step else bias_v2
        if step == args.flip_step + 1 and condition == "aux-loss":
            aux.freeze()  # 統制3: 反転直後から aux-loss だけ「推論時のみ」に入る

        usage, soft_probs = route_batch(spec, active_bias, correction, g)

        if condition == "aux-loss":
            aux.step(soft_probs, spec.target_rate)
        elif condition == "proportional":
            proportional_update(correction, usage, spec.target_rate, args.lr_local)
        elif condition == "bcm":
            bcm_update(correction, usage, msq_state, args.alpha, args.beta, args.lr_local)

        rows.append({
            "step": step,
            "phase": "pre-flip" if step <= args.flip_step else "post-flip",
            "imbalance": imbalance(usage, spec.target_rate),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--n-experts", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--noise-scale", type=float, default=1.0)
    ap.add_argument("--bias-scale", type=float, default=2.0)
    ap.add_argument("--n-steps", type=int, default=400)
    ap.add_argument("--flip-step", type=int, default=200)
    ap.add_argument("--lr-aux", type=float, default=0.5)
    ap.add_argument("--lr-local", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=1.0, help="BCM: theta = alpha * <y^2>")
    ap.add_argument("--beta", type=float, default=0.1, help="BCM: <y^2>の移動平均の減衰率")
    ap.add_argument("--converge-window", type=int, default=20,
                    help="再収束判定・後期集計に使う直近ステップ数")
    ap.add_argument("--converge-threshold", type=float, default=0.01)
    ap.add_argument("--out", type=Path, default=Path("results/glia/bcm_vs_proportional.json"))
    args = ap.parse_args()

    spec = LoadBalanceSpec(n_experts=args.n_experts, batch_size=args.batch_size,
                           noise_scale=args.noise_scale)

    t0 = time.time()
    all_rows: list[dict] = []
    for condition in args.conditions:
        for seed in range(args.seeds):
            for row in run_one(condition, spec, args, seed):
                all_rows.append(dict(row, condition=condition, seed=seed))

    def steps_to_recover(cond_seed_rows: list[dict]) -> int:
        """flip後、直近window内の平均imbalanceがthresholdを初めて下回るまでのステップ数。"""
        post = [r for r in cond_seed_rows if r["phase"] == "post-flip"]
        w = args.converge_window
        for i in range(w, len(post) + 1):
            window = [r["imbalance"] for r in post[i - w:i]]
            if sum(window) / w < args.converge_threshold:
                return post[i - 1]["step"] - args.flip_step
        return args.n_steps - args.flip_step  # 収束しなかった

    summary: dict[str, dict] = {}
    for condition in args.conditions:
        cond_rows = [r for r in all_rows if r["condition"] == condition]
        pre_final = [r["imbalance"] for r in cond_rows
                    if r["phase"] == "pre-flip" and r["step"] > args.flip_step - args.converge_window]
        post_final = [r["imbalance"] for r in cond_rows
                     if r["phase"] == "post-flip" and r["step"] > args.n_steps - args.converge_window]
        recover_steps = []
        for seed in range(args.seeds):
            seed_rows = [r for r in cond_rows if r["seed"] == seed]
            recover_steps.append(steps_to_recover(seed_rows))
        recover_t = torch.tensor(recover_steps, dtype=torch.float32)
        summary[condition] = {
            "pre_flip_final_imbalance_mean": float(torch.tensor(pre_final).mean()),
            "post_flip_final_imbalance_mean": float(torch.tensor(post_final).mean()),
            "recover_steps_mean": float(recover_t.mean()),
            "recover_steps_std": float(recover_t.std(unbiased=False)),
            "recover_steps_per_seed": recover_steps,
        }

    # 統制1: 反転前（pre-flip）の最終的な均等化性能が3条件で同等であること。
    pre_means = [summary[c]["pre_flip_final_imbalance_mean"] for c in args.conditions]
    control1_ok = (max(pre_means) - min(pre_means)) < 0.02
    if not control1_ok:
        print(f"警告: 統制1（学習時性能の統制）が崩れています。"
              f" pre_flip_final_imbalance の条件間差: {max(pre_means) - min(pre_means):.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "ステップ5 グリア層: 偏り反転後、aux-lossのみ凍結した状態での"
                "再収束比較（主張a）とBCM対proportionalの再収束速度比較（主張b）"
                "（12.6.17節）",
        "条件": vars(args) | {"out": str(args.out)},
        "control1_ok": control1_ok,
        "summary": summary,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n書き出し: {args.out}")
    for condition in args.conditions:
        s = summary[condition]
        print(f"{condition:>14}  pre_flip={s['pre_flip_final_imbalance_mean']:.4f}"
              f"  post_flip={s['post_flip_final_imbalance_mean']:.4f}"
              f"  recover_steps={s['recover_steps_mean']:.1f}±{s['recover_steps_std']:.1f}")


if __name__ == "__main__":
    main()
