"""ステップ14（マルチ学習則の学習安定性 — 海馬×基底核の勾配干渉, 12.6.36節）の主実験。

段階1: independent-both条件のRL側の学習曲線が十分安定していることを軽く確認する
       （実施前メモで予告した事前チェック — RL自体の不安定さと干渉を混同しないため）。
段階2: 4条件（shared-both / independent-both / shared-hippocampus-only / shared-rl-only）
       を比較する本実験を行う。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_multi_learning_rule
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..basal_ganglia_core import ParitySpec
from ..multi_learning_rule import CONDITIONS, run_condition


def probe_rl_stability(spec: ParitySpec, hidden_dim: int, steps: int, batch_size: int, lr: float,
                       eval_n: int, seeds: int) -> dict:
    """段階1: independent-bothのRL側単体の学習が安定して解けることを確認する。"""
    accs = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        r = run_condition("independent-both", spec, hidden_dim, steps, batch_size, lr, eval_n, g)
        accs.append(r.rl_acc)
    return {"mean": sum(accs) / len(accs), "accs": accs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-inputs", type=int, default=8)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--control2-threshold", type=float, default=0.1,
                    help="単一信号条件がindependent-bothよりこの差を超えて劣れば統制2が崩れたとみなす")
    ap.add_argument("--out", type=Path,
                    default=Path("results/multi_learning_rule/gradient_interference.json"))
    args = ap.parse_args()

    t0 = time.time()
    spec = ParitySpec(n_inputs=args.n_inputs, relevant=(0, 1))

    probe = probe_rl_stability(spec, args.hidden_dim, steps=400, batch_size=64, lr=args.lr,
                               eval_n=500, seeds=3)
    if probe["mean"] < 0.9:
        print(f"警告: 段階1のRL単体安定性チェックが弱い（mean_acc={probe['mean']:.4f}）。"
              "本実験の解釈に注意。")

    all_results: dict[str, dict] = {c: {"hippo_acc": [], "rl_acc": [], "cosine_mean": []}
                                    for c in CONDITIONS}
    for condition in CONDITIONS:
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            g = torch.Generator().manual_seed(seed + 40_000)
            r = run_condition(condition, spec, args.hidden_dim, args.steps, args.batch_size,
                              args.lr, args.n_eval, g)
            if r.hippo_acc is not None:
                all_results[condition]["hippo_acc"].append(r.hippo_acc)
            if r.rl_acc is not None:
                all_results[condition]["rl_acc"].append(r.rl_acc)
            if r.cosine_similarities:
                all_results[condition]["cosine_mean"].append(
                    sum(r.cosine_similarities) / len(r.cosine_similarities))

    def _stats(values: list[float]) -> dict | None:
        if not values:
            return None
        t = torch.tensor(values)
        return {"mean": float(t.mean()), "std": float(t.std(unbiased=False)), "per_seed": t.tolist()}

    summary: dict[str, dict] = {}
    for condition in CONDITIONS:
        r = all_results[condition]
        summary[condition] = {
            "hippo_acc": _stats(r["hippo_acc"]),
            "rl_acc": _stats(r["rl_acc"]),
            "cosine_similarity_mean_per_seed": _stats(r["cosine_mean"]),
        }

    # 統制2: 単一信号条件がindependent-bothの対応するタスクと遜色ないこと。
    indep_hippo = summary["independent-both"]["hippo_acc"]["mean"]
    indep_rl = summary["independent-both"]["rl_acc"]["mean"]
    shared_h_only_hippo = summary["shared-hippocampus-only"]["hippo_acc"]["mean"]
    shared_rl_only_rl = summary["shared-rl-only"]["rl_acc"]["mean"]
    control2_ok = bool(
        shared_h_only_hippo > indep_hippo - args.control2_threshold and
        shared_rl_only_rl > indep_rl - args.control2_threshold
    )
    if not control2_ok:
        print("警告: 統制2（バックボーン共有という構造自体の健全性）が崩れています。")

    # 主張（強い形）: shared-bothが、独立条件・単一信号条件のいずれよりも両タスクで明確に劣ること。
    shared_hippo = summary["shared-both"]["hippo_acc"]["mean"]
    shared_rl = summary["shared-both"]["rl_acc"]["mean"]
    claim_strong_ok = bool(
        shared_hippo < indep_hippo - args.control2_threshold and
        shared_hippo < shared_h_only_hippo - args.control2_threshold and
        shared_rl < indep_rl - args.control2_threshold and
        shared_rl < shared_rl_only_rl - args.control2_threshold
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "ステップ14 マルチ学習則の学習安定性（海馬×基底核の勾配干渉）: "
                "shared-both（本設計）対 independent-both（統制1）対 "
                "shared-hippocampus-only・shared-rl-only（統制2）の、"
                "同一XORパリティ課題での両タスク最終性能とバックボーン勾配の"
                "コサイン類似度比較（12.6.36節）",
        "条件": vars(args) | {"out": str(args.out)},
        "rl_stability_probe": probe,
        "control2_ok": control2_ok,
        "claim_strong_ok": claim_strong_ok,
        "summary": summary,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n書き出し: {args.out}")
    print(f"-- 段階1（RL単体安定性プローブ、3シード） -- mean_acc={probe['mean']:.4f}")
    print("-- 本実験 --")
    for condition in CONDITIONS:
        s = summary[condition]
        h = s["hippo_acc"]
        r = s["rl_acc"]
        c = s["cosine_similarity_mean_per_seed"]
        h_str = f"{h['mean']:.3f}±{h['std']:.3f}" if h else "N/A"
        r_str = f"{r['mean']:.3f}±{r['std']:.3f}" if r else "N/A"
        c_str = f"{c['mean']:.3f}±{c['std']:.3f}" if c else "N/A"
        print(f"{condition:>24}  hippo_acc={h_str:>14}  rl_acc={r_str:>14}  cos_sim={c_str}")
    print(f"\n統制2 (control2_ok): {control2_ok}")
    print(f"主張（強い形, claim_strong_ok）: {claim_strong_ok}")


if __name__ == "__main__":
    main()
