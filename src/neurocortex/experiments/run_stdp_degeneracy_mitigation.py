"""ステップ28（12.6.64節の設計）: STDP縮退への対策の実装・効果測定。

ステップ27（`run_stdp_degeneracy_impact.py`）で実害が確定した高学習率域
（3倍・10倍点）の縮退に対し、2種類の独立した対策を検証する。

  (a) k-WTA競合強度調整: `PatternSeparator.k`を既定40から80・160へ拡大
  (b) 正則化項の追加: `fit_stdp`の新規引数`weight_decay`を0（既定=ステップ27相当）
      から1e-4・1e-3へ拡大（`k=40`固定）

対照群はステップ27の`results/stdp_stability/degeneracy_impact.json`
（k=40・weight_decay=0.0相当）をそのまま参照し、再実行しない。
（a)(b)いずれも既定点（k=40, weight_decay=0.0）は上記対照群と重複するため
本スクリプトでは再実行しない。

`hippocampus.py`のSTDPアルゴリズム自体・`run_stdp_degeneracy_impact.py`は
変更しない（`fit_stdp`への`weight_decay`引数追加のみ）。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_stdp_degeneracy_mitigation
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..hippocampus import HippocampalMemory, fit_stdp
from ..tasks import FactSpec, make_fact_batch
from .run_hippocampus import build_backbone, encode_keys
from .run_hippocampus_stdp import encode_spikes, evaluate
from .run_stdp_stability import degeneracy_stats

MULTIPLIERS = [0.1, 1.0, 3.0, 10.0]
BASELINE_JSON = Path("results/stdp_stability/degeneracy_impact.json")


def run_point(spec: FactSpec, args, multiplier: float, seed: int, k: int,
              weight_decay: float) -> dict:
    """1（設定・倍率・シード）ぶんの学習→評価を行う（run_stdp_degeneracy_impact.pyのrun_impact_pointと同型）。"""
    model = build_backbone(spec, args.d_model, 3, 4, seed, args.train_steps, 3e-3, 64)
    g = torch.Generator().manual_seed(seed + 1_000)
    tr_prompts, _ = make_fact_batch(spec, args.n_train, g)
    train_x = encode_keys(model, tr_prompts)
    train_s = encode_spikes(model, tr_prompts)

    mem = HippocampalMemory(args.d_model, value_dim=args.d_model, n_units=args.n_units,
                             k=k, mode="sdr", beta=args.beta, seed=seed)
    stdp_g = torch.Generator().manual_seed(seed + 7_000)
    fit_stdp(mem.separator, train_s, epochs=args.epochs_b,
              a_plus=args.a_plus * multiplier, a_minus=args.a_minus * multiplier,
              tau=args.tau, generator=stdp_g, weight_decay=weight_decay)
    dstats = degeneracy_stats(mem.separator.weight)

    prompts, objects = make_fact_batch(spec, args.n_facts, g)
    hw = encode_keys(model, prompts)
    sw = encode_spikes(model, prompts)[:, -1]
    scale = args.cue_noise / args.d_model ** 0.5
    hq = hw + scale * hw.norm(dim=-1, keepdim=True) * torch.randn(hw.shape, generator=g)
    sq = sw + scale * sw.norm(dim=-1, keepdim=True).clamp_min(1e-6) * torch.randn(
        sw.shape, generator=g)

    r = evaluate(model, spec, "spike-stdp", mem, hw, hq, sw, sq, objects)
    r = dict(r, accuracy=r["n_correct"] / r["n_queries"], multiplier=multiplier, seed=seed,
              k=k, weight_decay=weight_decay, **dstats)
    return r


def measure_raw_update_norm(spec: FactSpec, args, seed: int, k: int,
                             weight_decay: float) -> dict:
    """既定点（1倍）での生の更新量ノルム推移を測る（主張(c)の収束傾向判定用）。"""
    model = build_backbone(spec, args.d_model, 3, 4, seed, args.train_steps, 3e-3, 64)
    g = torch.Generator().manual_seed(seed + 1_000)
    tr_prompts, _ = make_fact_batch(spec, args.n_train, g)
    train_s = encode_spikes(model, tr_prompts)

    mem = HippocampalMemory(args.d_model, value_dim=args.d_model, n_units=args.n_units,
                             k=k, mode="sdr", seed=seed)
    stdp_g = torch.Generator().manual_seed(seed + 7_000)
    records: list[dict] = []

    def raw_cb(ep: int, raw_norm: float, records=records) -> None:
        records.append({"epoch": ep, "raw_update_norm": raw_norm})

    fit_stdp(mem.separator, train_s, epochs=args.epochs,
              a_plus=args.a_plus, a_minus=args.a_minus,
              tau=args.tau, generator=stdp_g, raw_update_callback=raw_cb,
              weight_decay=weight_decay)
    early = [r["raw_update_norm"] for r in records[:5]]
    late = [r["raw_update_norm"] for r in records[-5:]]
    return {"records": records, "early_mean": sum(early) / len(early),
            "late_mean": sum(late) / len(late)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3,
                     help="シード数（設計文書の基準は5だが、実行時間短縮のため既定3。"
                          "理由はdocs/brain-structure-research.mdの実行時間注記を参照")
    ap.add_argument("--n-units", type=int, default=2048)
    ap.add_argument("--k-values", type=int, nargs="+", default=[80, 160],
                     help="対策(a)のk（既定40はベースラインと重複するため含めない）")
    ap.add_argument("--weight-decay-values", type=float, nargs="+", default=[1e-4, 1e-3],
                     help="対策(b)のweight_decay（既定0.0はベースラインと重複するため含めない）")
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--epochs-b", type=int, default=20)
    ap.add_argument("--a-plus", type=float, default=0.01)
    ap.add_argument("--a-minus", type=float, default=0.008)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--n-prefix", type=int, default=8)
    ap.add_argument("--n-suffix", type=int, default=4096)
    ap.add_argument("--n-object", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--train-steps", type=int, default=400)
    ap.add_argument("--n-facts", type=int, default=2000)
    ap.add_argument("--cue-noise", type=float, default=0.02)
    ap.add_argument("--beta", type=float, default=1e4)
    ap.add_argument("--multipliers", type=float, nargs="+", default=MULTIPLIERS)
    ap.add_argument("--baseline-json", type=Path, default=BASELINE_JSON)
    ap.add_argument("--out", type=Path,
                     default=Path("results/stdp_stability/degeneracy_mitigation.json"))
    args = ap.parse_args()

    spec = FactSpec(n_prefix=args.n_prefix, n_suffix=args.n_suffix, n_object=args.n_object)
    t0 = time.time()

    baseline = json.loads(args.baseline_json.read_text(encoding="utf-8")) \
        if args.baseline_json.exists() else None

    # --- 対策(a)(b)の効果測定 --------------------------------------------
    conditions = [("k", k, 0.0) for k in args.k_values] + \
                 [("weight_decay", 40, wd) for wd in args.weight_decay_values]

    rows: list[dict] = []
    for kind, k, wd in conditions:
        for seed in range(args.seeds):
            for mult in args.multipliers:
                t1 = time.time()
                r = run_point(spec, args, mult, seed, k, wd)
                r["sec"] = round(time.time() - t1, 1)
                r["kind"] = kind
                rows.append(r)
                print(f"[{kind}] k={k} wd={wd:g} seed={seed} mult={mult:>5g} "
                      f"acc={r['accuracy']:.4f} max_cos={r.get('max_pairwise_cos', float('nan')):.5f} "
                      f"({r['sec']}s)", flush=True)

    def summarize(kind: str, k: int, wd: float) -> dict:
        out = {}
        for mult in args.multipliers:
            sel = [r for r in rows if r["kind"] == kind and r["k"] == k
                   and r["weight_decay"] == wd and r["multiplier"] == mult]
            if not sel:
                continue
            out[str(mult)] = {
                "n_seeds": len(sel),
                "accuracy_mean": float(torch.tensor([r["accuracy"] for r in sel]).mean()),
                "max_pairwise_cos_mean": float(torch.tensor(
                    [r["max_pairwise_cos"] for r in sel]).mean()),
                "n_unique_rows_mean": float(torch.tensor(
                    [r["n_unique_rows"] for r in sel], dtype=torch.float).mean()),
                "margin_mean": float(torch.tensor([r["margin"] for r in sel]).mean()),
                "winner_retention_mean": float(
                    torch.tensor([r["winner_retention"] for r in sel]).mean()),
            }
        return out

    summaries = {}
    for kind, k, wd in conditions:
        key = f"k={k}" if kind == "k" else f"weight_decay={wd:g}"
        summaries[key] = summarize(kind, k, wd)

    # --- 主張(c)用: 既定点（1倍）の生の更新量ノルム比 -----------------------
    raw_norm_summaries = {}
    for kind, k, wd in conditions:
        key = f"k={k}" if kind == "k" else f"weight_decay={wd:g}"
        recs = [measure_raw_update_norm(spec, args, seed, k, wd) for seed in range(args.seeds)]
        early = float(torch.tensor([r["early_mean"] for r in recs]).mean())
        late = float(torch.tensor([r["late_mean"] for r in recs]).mean())
        raw_norm_summaries[key] = {
            "per_seed": recs, "early_mean": early, "late_mean": late,
            "ratio": early / max(late, 1e-8),
        }
        print(f"[raw_norm] {key} early={early:.4f} late={late:.4f} "
              f"ratio={early / max(late, 1e-8):.2f}x", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "STDP縮退への対策の実装・効果測定（12.6.64節の設計、ステップ28）",
        "条件": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "baseline_ref": str(args.baseline_json),
        "baseline_summary_b": baseline["目的b"]["summary"] if baseline else None,
        "summaries": summaries,
        "raw_update_norm_at_default": raw_norm_summaries,
        "rows": rows,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n書き出し: {args.out}")
    for key, s in summaries.items():
        for mult, v in s.items():
            print(f"{key} mult={mult} acc={v['accuracy_mean']:.4f} "
                  f"max_cos={v['max_pairwise_cos_mean']:.5f}")


if __name__ == "__main__":
    main()
