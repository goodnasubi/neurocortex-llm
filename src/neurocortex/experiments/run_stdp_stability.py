"""ステップ26（12.6.60節の設計）: STDPパターン分離層の学習安定性検証。

`hippocampus.py`の`fit_stdp`・`PatternSeparator`はコード変更せず（`fit_stdp`には
epoch末コールバックのみ追加）、`run_hippocampus_stdp.py`の`encode_spikes`・
`build_backbone`をそのまま流用する。`a_plus`・`a_minus`（比0.8固定）を既定値の
0.1倍・1倍・3倍・10倍に振り、`tau=0.9`固定・epochs=20・シード5通りで、各epoch末の
`sep.weight`のフロベニウスノルム・NaN/Inf有無・縮退の兆候（重みベクトル間の
ペアワイズコサイン類似度の平均・最大値、一意な重みベクトル数）を記録する。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_stdp_stability
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..hippocampus import HippocampalMemory, fit_stdp, max_pairwise_cosine
from ..tasks import FactSpec, make_fact_batch
from .run_hippocampus import build_backbone
from .run_hippocampus_stdp import encode_spikes

# a_plus・a_minus の倍率グリッド（既定 a_plus=0.01, a_minus=0.008, 比0.8固定）
MULTIPLIERS = [0.1, 1.0, 3.0, 10.0]


def degeneracy_stats(weight: torch.Tensor) -> dict:
    """縮退（少数ユニットへの重みベクトルの集中）の兆候を測る。

    - `mean_pairwise_cos` / `max_pairwise_cos`: 正規化した重みベクトル間の
      ペアワイズコサイン類似度（対角除く）の平均・最大。1に近いほど多くの
      ユニットが同じ方向に集まっている（縮退）。
    - `n_unique_rows`: 小数丸め後に一意な重みベクトルの本数（M本中）。
      これが大きく減っていれば「勝者総取り」的な縮退が起きている。
    """
    w = weight.detach()
    m = w.shape[0]
    wn = w / w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sims = wn @ wn.T
    mean_cos = float(sims[~torch.eye(m, dtype=torch.bool)].mean())
    max_cos = max_pairwise_cosine(w)  # steps 26-28: 共通ヘルパー（hippocampus.py）に委譲
    rounded = torch.round(wn * 1e4) / 1e4
    n_unique = int(len({tuple(row.tolist()) for row in rounded}))
    return {
        "mean_pairwise_cos": mean_cos,
        "max_pairwise_cos": max_cos,
        "n_unique_rows": n_unique,
        "n_units": m,
    }


def run_one(train_s: torch.Tensor, d_model: int, n_units: int, k: int,
            backbone_seed: int, stdp_seed: int, a_plus: float, a_minus: float,
            tau: float, epochs: int) -> dict:
    """1（ハイパラ点・シード）ぶんの安定性計測を行う。"""
    mem = HippocampalMemory(d_model, value_dim=d_model, n_units=n_units, k=k,
                             mode="sdr", seed=backbone_seed)
    g = torch.Generator().manual_seed(stdp_seed + 7_000)

    epoch_records: list[dict] = []

    def callback(ep: int, sep) -> None:
        w = sep.weight.detach()
        finite = torch.isfinite(w).all()
        norm = float(w.norm()) if finite else float("nan")
        rec = {
            "epoch": ep,
            "frobenius_norm": norm,
            "has_nan": bool(torch.isnan(w).any()),
            "has_inf": bool(torch.isinf(w).any()),
        }
        if finite:
            rec |= degeneracy_stats(w)
        epoch_records.append(rec)

    fit_stdp(mem.separator, train_s, epochs=epochs, a_plus=a_plus, a_minus=a_minus,
              tau=tau, generator=g, callback=callback)

    diverged = any(r["has_nan"] or r["has_inf"] for r in epoch_records)
    return {"epoch_records": epoch_records, "diverged": diverged}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-units", type=int, default=2048)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--a-plus", type=float, default=0.01)
    ap.add_argument("--a-minus", type=float, default=0.008)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--n-prefix", type=int, default=8)
    ap.add_argument("--n-suffix", type=int, default=4096)
    ap.add_argument("--n-object", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--train-steps", type=int, default=400)
    ap.add_argument("--multipliers", type=float, nargs="+", default=MULTIPLIERS)
    ap.add_argument("--out", type=Path,
                     default=Path("results/stdp_stability/stability.json"))
    args = ap.parse_args()

    spec = FactSpec(n_prefix=args.n_prefix, n_suffix=args.n_suffix, n_object=args.n_object)
    t0 = time.time()
    rows: list[dict] = []

    for seed in range(args.seeds):
        model = build_backbone(spec, args.d_model, 3, 4, seed, args.train_steps, 3e-3, 64)
        g = torch.Generator().manual_seed(seed + 1_000)
        tr_prompts, _ = make_fact_batch(spec, args.n_train, g)
        train_s = encode_spikes(model, tr_prompts)

        for mult in args.multipliers:
            a_plus = args.a_plus * mult
            a_minus = args.a_minus * mult
            t1 = time.time()
            result = run_one(train_s, args.d_model, args.n_units, args.k,
                              backbone_seed=seed, stdp_seed=seed, a_plus=a_plus,
                              a_minus=a_minus, tau=args.tau, epochs=args.epochs)
            row = {
                "seed": seed, "multiplier": mult, "a_plus": a_plus, "a_minus": a_minus,
                "tau": args.tau, "diverged": result["diverged"],
                "epoch_records": result["epoch_records"],
                "sec": round(time.time() - t1, 1),
            }
            rows.append(row)
            last = result["epoch_records"][-1]
            print(f"seed={seed} mult={mult:>5g} diverged={result['diverged']} "
                  f"final_norm={last['frobenius_norm']:.4f} "
                  f"n_unique={last.get('n_unique_rows', '-')}/{args.n_units} "
                  f"({row['sec']}s)", flush=True)

    # --- 集計 ---------------------------------------------------------------
    def norm_change_rate(records: list[dict], lo: int, hi: int) -> float:
        """[lo, hi) epoch区間の |Δnorm| の平均変化率。"""
        deltas = []
        for i in range(max(lo, 1), min(hi, len(records))):
            prev, cur = records[i - 1]["frobenius_norm"], records[i]["frobenius_norm"]
            if prev == prev and cur == cur:  # NaN除外
                deltas.append(abs(cur - prev))
        return float(torch.tensor(deltas).mean()) if deltas else float("nan")

    summary = {}
    for mult in args.multipliers:
        sel = [r for r in rows if r["multiplier"] == mult]
        n_diverged = sum(r["diverged"] for r in sel)
        summary[str(mult)] = {
            "n_seeds": len(sel),
            "n_diverged": n_diverged,
            "diverged_fraction": n_diverged / len(sel),
            "early_rate_mean": float(torch.tensor(
                [norm_change_rate(r["epoch_records"], 1, 6) for r in sel
                 if not r["diverged"]]).mean()) if any(not r["diverged"] for r in sel) else float("nan"),
            "late_rate_mean": float(torch.tensor(
                [norm_change_rate(r["epoch_records"], 16, 21) for r in sel
                 if not r["diverged"]]).mean()) if any(not r["diverged"] for r in sel) else float("nan"),
            "final_n_unique_mean": float(torch.tensor(
                [r["epoch_records"][-1].get("n_unique_rows", args.n_units) for r in sel
                 if not r["diverged"]]).float().mean()) if any(not r["diverged"] for r in sel) else float("nan"),
            "final_max_pairwise_cos_mean": float(torch.tensor(
                [r["epoch_records"][-1].get("max_pairwise_cos", float("nan")) for r in sel
                 if not r["diverged"]]).mean()) if any(not r["diverged"] for r in sel) else float("nan"),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "STDPパターン分離層の学習安定性検証（12.6.60節の設計、ステップ26）",
        "条件": vars(args) | {"out": str(args.out)},
        "summary": summary,
        "rows": rows,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n書き出し: {args.out}")
    for mult in args.multipliers:
        s = summary[str(mult)]
        print(f"mult={mult:>5g}  diverged={s['n_diverged']}/{s['n_seeds']}  "
              f"early_rate={s['early_rate_mean']:.5f}  late_rate={s['late_rate_mean']:.5f}  "
              f"n_unique(final)={s['final_n_unique_mean']:.1f}/{args.n_units}  "
              f"max_cos(final)={s['final_max_pairwise_cos_mean']:.4f}")


if __name__ == "__main__":
    main()
