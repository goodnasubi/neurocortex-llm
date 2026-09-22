"""ステップ11（皮質バックボーン — 層間結合のスモールワールド化, 12.6.30節）の主実験。

段階1: `dense`条件でノイズ強度を段階的に強め、可解性が保たれる範囲を確認する
       （実施前メモで予告した「難易度の動作点」探索の軽量版）。
段階2: 段階1で選んだノイズ強度のもとで、4条件（local-chain / small-world /
       random-wire / dense）を比較する主実験を行う。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_small_world_layers
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..small_world_layers import CONDITIONS, ChainSpec, LayerChainNet, accuracy, build_adjacency, train


def probe_noise_levels(spec: ChainSpec, hidden_dim: int, noise_levels: list[float],
                       steps: int, batch_size: int, lr: float, n_eval: int,
                       seeds: int) -> dict:
    """段階1: dense条件（参考上限）が各ノイズ強度で解けるかを確認する。"""
    results = {}
    for noise_std in noise_levels:
        accs = []
        for seed in range(seeds):
            torch.manual_seed(seed)
            g = torch.Generator().manual_seed(seed)
            adj = build_adjacency("dense", spec, n_shortcuts=0, generator=g)
            model = LayerChainNet(spec, hidden_dim, adj, noise_std)
            train(model, spec, steps, batch_size, lr, g)
            accs.append(accuracy(model, spec, n_eval, torch.Generator().manual_seed(seed + 1)))
        results[noise_std] = {"mean": sum(accs) / len(accs), "accs": accs}
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10,
                    help="small-world対random-wireの差が僅差だったため、既定の5から増やした")
    ap.add_argument("--n-layers", type=int, default=16)
    ap.add_argument("--src-layer-a", type=int, default=0)
    ap.add_argument("--src-layer-b", type=int, default=7)
    ap.add_argument("--hidden-dim", type=int, default=16)
    ap.add_argument("--n-shortcuts", type=int, default=3)
    ap.add_argument("--noise-std", type=float, default=0.6,
                    help="段階1のプローブで選定済みの、denseが解けてlocal-chainが崩れる強度")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--control1-threshold", type=float, default=0.1,
                    help="denseがこの正解率を下回れば、課題設定自体が解けないとみなす")
    ap.add_argument("--out", type=Path,
                    default=Path("results/small_world_layers/small_world_vs_controls.json"))
    args = ap.parse_args()

    spec = ChainSpec(n_layers=args.n_layers, src_layer_a=args.src_layer_a,
                     src_layer_b=args.src_layer_b)
    t0 = time.time()

    probe = probe_noise_levels(spec, args.hidden_dim, [0.0, 0.15, 0.3, 0.5, 0.8],
                               steps=400, batch_size=64, lr=0.05, n_eval=1000, seeds=3)

    all_accs: dict[str, list[float]] = {c: [] for c in CONDITIONS}
    for condition in CONDITIONS:
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            g = torch.Generator().manual_seed(seed + 90_000)
            adj = build_adjacency(condition, spec, args.n_shortcuts, g)
            model = LayerChainNet(spec, args.hidden_dim, adj, args.noise_std)
            train(model, spec, args.steps, args.batch_size, args.lr, g)
            acc = accuracy(model, spec, args.n_eval, torch.Generator().manual_seed(seed + 1))
            all_accs[condition].append(acc)

    summary: dict[str, dict] = {}
    for condition in CONDITIONS:
        accs = torch.tensor(all_accs[condition])
        summary[condition] = {
            "mean": float(accs.mean()),
            "std": float(accs.std(unbiased=False)),
            "per_seed": accs.tolist(),
        }

    # 統制1: dense（参考上限）がチャンスを明確に上回ること（課題設定自体が解けることの確認）。
    dense_mean = summary["dense"]["mean"]
    control1_ok = dense_mean > (0.5 + args.control1_threshold)
    if not control1_ok:
        print(f"警告: 統制1（課題の可解性）が崩れています。denseの正解率: {dense_mean:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "ステップ11 皮質バックボーン（層間結合のスモールワールド化）: "
                "local-chain（統制1）対 small-world（本設計）対 random-wire（統制3）対 "
                "dense（参考上限）の、残差ストリームなし・情報劣化ありの逐次層連鎖での"
                "XORパリティ課題比較（12.6.30節）",
        "条件": vars(args) | {"out": str(args.out)},
        "noise_probe": probe,
        "control1_ok": control1_ok,
        "summary": summary,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n書き出し: {args.out}")
    print("-- ノイズ強度プローブ（dense条件） --")
    for noise_std, r in probe.items():
        print(f"  noise_std={noise_std}: mean_acc={r['mean']:.4f}")
    print("-- 本実験 --")
    for condition in CONDITIONS:
        s = summary[condition]
        print(f"{condition:>12}  acc={s['mean']:.3f}±{s['std']:.3f}  per_seed={[round(v, 3) for v in s['per_seed']]}")


if __name__ == "__main__":
    main()
