"""ステップ3（皮質バックボーンのハブスパース化, 12.6.12節）の主実験。

3条件（hub-sparse / dense / random-wire）で2ソースXOR課題を学習させ、
中継ノードを1つずつ除去したときの精度低下から「標的除去 ÷ ランダム除去」の
比を比較する。主張は hub-sparse だけがこの比で他の2条件より有意に大きい値を
示すこと（7.1節のコネクトームの頑健性非対称性の再現）。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_hub_sparse --seeds 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..hub_sparse import (
    HubGraphNet,
    ParitySpec,
    accuracy,
    make_dense_adjacency,
    make_hub_sparse_adjacency,
    make_random_wire_adjacency,
    per_node_drop,
    targeted_over_random_ratio,
    train,
)

CONDITIONS = ("hub-sparse", "dense", "random-wire")


def build_model(condition: str, spec: ParitySpec, n_removable: int, n_hubs: int,
                generator: torch.Generator) -> HubGraphNet:
    if condition == "hub-sparse":
        adj = make_hub_sparse_adjacency(spec.n_experts, n_removable, n_hubs)
    elif condition == "dense":
        adj = make_dense_adjacency(spec.n_experts, n_removable)
    elif condition == "random-wire":
        n_edges = n_hubs * spec.n_experts  # hub-sparse とエッジ総数を揃える（統制3）
        # n_forced=n_hubs: 「関連する2ソースを両方見られるノード数」も揃える
        # （hub-sparseはn_hubs個のハブが全員それを満たす）。揃えないと冗長性が
        # 条件間でずれ、エッジ総数だけでは公平な比較にならない（12.6.13節）。
        adj = make_random_wire_adjacency(spec, n_removable, n_edges, generator, n_forced=n_hubs)
    else:
        raise ValueError(condition)
    return HubGraphNet(spec.n_experts, n_removable, adj)


def run_one(condition: str, spec: ParitySpec, args: argparse.Namespace, seed: int) -> dict:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 20_000)
    model = build_model(condition, spec, args.n_removable, args.n_hubs, g)

    train(model, spec, args.steps, args.batch_size, args.lr, g)
    base_acc = accuracy(model, spec, args.n_eval, g)
    drops = per_node_drop(model, spec, args.n_eval, g)
    ratio = targeted_over_random_ratio(drops)

    return {
        "condition": condition,
        "seed": seed,
        "base_acc": base_acc,
        "targeted_drop": float(drops.max()),
        "random_drop": float(drops.mean()),
        "ratio": ratio,
        "degree": model.degree.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--n-experts", type=int, default=6)
    ap.add_argument("--relevant", type=int, nargs=2, default=[0, 1])
    ap.add_argument("--n-removable", type=int, default=8)
    ap.add_argument("--n-hubs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--base-acc-min", type=float, default=0.9,
                    help="統制1: 除去なしの正答率がこれを下回ったら警告する")
    ap.add_argument("--out", type=Path,
                    default=Path("results/hub_sparse/targeted_vs_random.json"))
    args = ap.parse_args()

    spec = ParitySpec(n_experts=args.n_experts, relevant=tuple(args.relevant))

    t0 = time.time()
    rows: list[dict] = []
    for condition in args.conditions:
        for seed in range(args.seeds):
            row = run_one(condition, spec, args, seed)
            rows.append(row)
            print(f"{condition:>12} seed={seed}: base_acc={row['base_acc']:.4f}"
                  f" targeted_drop={row['targeted_drop']:.4f}"
                  f" random_drop={row['random_drop']:.4f} ratio={row['ratio']:.2f}", flush=True)

    summary: dict[str, dict] = {}
    for condition in args.conditions:
        cond_rows = [r for r in rows if r["condition"] == condition]
        base = torch.tensor([r["base_acc"] for r in cond_rows])
        ratios = torch.tensor([r["ratio"] for r in cond_rows])
        summary[condition] = {
            "base_acc_mean": float(base.mean()),
            "base_acc_min": float(base.min()),
            "ratio_mean": float(ratios.mean()),
            "ratio_std": float(ratios.std(unbiased=False)),
            "ratio_min": float(ratios.min()),
            "ratio_max": float(ratios.max()),
        }

    # 統制1: 除去なしの基本性能が条件間で同等であること（3条件とも base_acc_min を上回る）。
    control1_ok = all(summary[c]["base_acc_min"] >= args.base_acc_min for c in args.conditions)
    if not control1_ok:
        print(f"警告: 統制1（基本性能の統制）が崩れています。base_acc_min={args.base_acc_min} を"
              " 下回った条件があります。")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "ステップ3 皮質バックボーンのハブスパース化: 標的除去とランダム除去の"
                "比の比較（12.6.12節）",
        "条件": vars(args) | {"out": str(args.out)},
        "control1_ok": control1_ok,
        "summary": summary,
        "rows": rows,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n書き出し: {args.out}")
    for condition in args.conditions:
        s = summary[condition]
        print(f"{condition:>12}  ratio={s['ratio_mean']:.2f}±{s['ratio_std']:.2f}"
              f"  (base_acc={s['base_acc_mean']:.4f})")


if __name__ == "__main__":
    main()
