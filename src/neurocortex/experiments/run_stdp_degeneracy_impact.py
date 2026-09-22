"""ステップ27（12.6.62節の設計）: STDP安定性の未確認事項の解消。

2つの独立した問いに答える:

(a) 既定ハイパーパラメータ（1倍点）で、正規化の影響を受けない代替指標
    （epochごとの一意な重みベクトル数`n_unique_rows`、正規化前の生の重み更新量
    ノルム）で収束傾向を再判定する。`n_unique_rows`はステップ26の
    `results/stdp_stability/stability.json`のepoch_recordsを再集計し、
    生の更新量ノルムは`fit_stdp`の新規引数`raw_update_callback`付きで
    既定点・5シードを再実行して測る。
(b) ステップ26で観察された高学習率域（3倍・10倍）の縮退が、実際に想起精度・
    分離診断（`margin`・`winner_retention`）を悪化させるかを、
    `run_hippocampus_stdp.py`の評価パイプラインに接続して検証する。

`hippocampus.py`のSTDPアルゴリズム自体・`run_stdp_stability.py`・
`run_hippocampus_stdp.py`は変更しない（`fit_stdp`への`raw_update_callback`
追加のみ）。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_stdp_degeneracy_impact
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..hippocampus import HippocampalMemory, fit_stdp, separation_diagnostics
from ..tasks import FactSpec, make_fact_batch
from .run_hippocampus import build_backbone, encode_keys
from .run_hippocampus_stdp import encode_spikes, evaluate

STABILITY_JSON = Path("results/stdp_stability/stability.json")
MULTIPLIERS_B = [0.1, 1.0, 3.0, 10.0]


# --- 目的(a): 代替指標での収束判定 ------------------------------------------

def recompute_n_unique_from_stability(path: Path, multiplier: float = 1.0) -> dict:
    """ステップ26の`stability.json`から`n_unique_rows`のepoch1→epoch20推移を再集計する。"""
    if not path.exists():
        return {"available": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in data["rows"] if r["multiplier"] == multiplier and not r["diverged"]]
    per_seed_first = []
    per_seed_last = []
    for r in rows:
        recs = r["epoch_records"]
        per_seed_first.append(recs[0]["n_unique_rows"])
        per_seed_last.append(recs[-1]["n_unique_rows"])
    return {
        "available": True,
        "n_seeds": len(rows),
        "epoch1_n_unique_mean": float(torch.tensor(per_seed_first, dtype=torch.float).mean()),
        "epoch20_n_unique_mean": float(torch.tensor(per_seed_last, dtype=torch.float).mean()),
        "epoch1_n_unique_per_seed": per_seed_first,
        "epoch20_n_unique_per_seed": per_seed_last,
    }


def measure_raw_update_norms(spec: FactSpec, args, multiplier: float = 1.0) -> dict:
    """既定点（1倍）・複数シードで、epochごとの生の更新量ノルムを測る。"""
    per_seed_records: list[list[dict]] = []
    for seed in range(args.seeds_a):
        model = build_backbone(spec, args.d_model, 3, 4, seed, args.train_steps, 3e-3, 64)
        g = torch.Generator().manual_seed(seed + 1_000)
        tr_prompts, _ = make_fact_batch(spec, args.n_train, g)
        train_s = encode_spikes(model, tr_prompts)

        mem = HippocampalMemory(args.d_model, value_dim=args.d_model, n_units=args.n_units,
                                 k=args.k, mode="sdr", seed=seed)
        stdp_g = torch.Generator().manual_seed(seed + 7_000)
        records: list[dict] = []

        def raw_cb(ep: int, raw_norm: float, records=records) -> None:
            records.append({"epoch": ep, "raw_update_norm": raw_norm})

        fit_stdp(mem.separator, train_s, epochs=args.epochs,
                  a_plus=args.a_plus * multiplier, a_minus=args.a_minus * multiplier,
                  tau=args.tau, generator=stdp_g, raw_update_callback=raw_cb)
        per_seed_records.append(records)
        print(f"[a] seed={seed} raw_update_norm epoch1={records[0]['raw_update_norm']:.4f} "
              f"epoch{args.epochs}={records[-1]['raw_update_norm']:.4f}", flush=True)

    early_means = []
    late_means = []
    for records in per_seed_records:
        early = [r["raw_update_norm"] for r in records[:5]]
        late = [r["raw_update_norm"] for r in records[-5:]]
        early_means.append(sum(early) / len(early))
        late_means.append(sum(late) / len(late))
    return {
        "n_seeds": args.seeds_a,
        "per_seed_records": per_seed_records,
        "early_mean_of_seed_means": float(torch.tensor(early_means).mean()),
        "late_mean_of_seed_means": float(torch.tensor(late_means).mean()),
    }


# --- 目的(b): 縮退の実害検証 -------------------------------------------------

def run_impact_point(spec: FactSpec, args, multiplier: float, seed: int) -> dict:
    """1（倍率・シード）ぶんの学習→評価を行う。"""
    model = build_backbone(spec, args.d_model, 3, 4, seed, args.train_steps, 3e-3, 64)
    g = torch.Generator().manual_seed(seed + 1_000)
    tr_prompts, _ = make_fact_batch(spec, args.n_train, g)
    train_x = encode_keys(model, tr_prompts)
    train_s = encode_spikes(model, tr_prompts)

    mem = HippocampalMemory(args.d_model, value_dim=args.d_model, n_units=args.n_units,
                             k=args.k, mode="sdr", beta=args.beta, seed=seed)
    stdp_g = torch.Generator().manual_seed(seed + 7_000)
    fit_stdp(mem.separator, train_s, epochs=args.epochs_b,
              a_plus=args.a_plus * multiplier, a_minus=args.a_minus * multiplier,
              tau=args.tau, generator=stdp_g)

    prompts, objects = make_fact_batch(spec, args.n_facts, g)
    hw = encode_keys(model, prompts)
    sw = encode_spikes(model, prompts)[:, -1]
    scale = args.cue_noise / args.d_model ** 0.5
    hq = hw + scale * hw.norm(dim=-1, keepdim=True) * torch.randn(hw.shape, generator=g)
    sq = sw + scale * sw.norm(dim=-1, keepdim=True).clamp_min(1e-6) * torch.randn(
        sw.shape, generator=g)

    r = evaluate(model, spec, "spike-stdp", mem, hw, hq, sw, sq, objects)
    r = dict(r, accuracy=r["n_correct"] / r["n_queries"], multiplier=multiplier, seed=seed)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-a", type=int, default=5, help="目的(a)のシード数")
    ap.add_argument("--seeds-b", type=int, default=5, help="目的(b)のシード数")
    ap.add_argument("--n-units", type=int, default=2048)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=20, help="目的(a)のepoch数（ステップ26と同一）")
    ap.add_argument("--epochs-b", type=int, default=20, help="目的(b)のepoch数（設計文書の固定値）")
    ap.add_argument("--a-plus", type=float, default=0.01)
    ap.add_argument("--a-minus", type=float, default=0.008)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--n-prefix", type=int, default=8)
    ap.add_argument("--n-suffix", type=int, default=4096)
    ap.add_argument("--n-object", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--train-steps", type=int, default=400)
    ap.add_argument("--n-facts", type=int, default=2000,
                     help="想起精度評価の書き込み件数（n_unitsの100%前後、ステップ21準拠）")
    ap.add_argument("--cue-noise", type=float, default=0.02)
    ap.add_argument("--beta", type=float, default=1e4)
    ap.add_argument("--multipliers-b", type=float, nargs="+", default=MULTIPLIERS_B)
    ap.add_argument("--stability-json", type=Path, default=STABILITY_JSON)
    ap.add_argument("--out", type=Path,
                     default=Path("results/stdp_stability/degeneracy_impact.json"))
    args = ap.parse_args()

    spec = FactSpec(n_prefix=args.n_prefix, n_suffix=args.n_suffix, n_object=args.n_object)
    t0 = time.time()

    # --- 目的(a) -------------------------------------------------------
    print("=== 目的(a): 代替指標での収束判定 ===", flush=True)
    n_unique_summary = recompute_n_unique_from_stability(args.stability_json, multiplier=1.0)
    raw_norm_summary = measure_raw_update_norms(spec, args, multiplier=1.0)

    # --- 目的(b) -------------------------------------------------------
    print("\n=== 目的(b): 縮退の実害検証 ===", flush=True)
    rows_b: list[dict] = []
    for seed in range(args.seeds_b):
        for mult in args.multipliers_b:
            t1 = time.time()
            r = run_impact_point(spec, args, mult, seed)
            r["sec"] = round(time.time() - t1, 1)
            rows_b.append(r)
            print(f"[b] seed={seed} mult={mult:>5g} acc={r['accuracy']:.4f} "
                  f"margin={r.get('margin', float('nan')):.5f} "
                  f"winner_retention={r.get('winner_retention', float('nan')):.3f} "
                  f"({r['sec']}s)", flush=True)

    summary_b = {}
    for mult in args.multipliers_b:
        sel = [r for r in rows_b if r["multiplier"] == mult]
        summary_b[str(mult)] = {
            "n_seeds": len(sel),
            "accuracy_mean": float(torch.tensor([r["accuracy"] for r in sel]).mean()),
            "margin_mean": float(torch.tensor([r["margin"] for r in sel]).mean()),
            "winner_retention_mean": float(
                torch.tensor([r["winner_retention"] for r in sel]).mean()),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "STDP安定性の未確認事項の解消: 代替指標での収束判定と縮退の実害検証"
                "（12.6.62節の設計、ステップ27）",
        "条件": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "目的a": {
            "n_unique_rows_from_stability_json": n_unique_summary,
            "raw_update_norm": raw_norm_summary,
        },
        "目的b": {"summary": summary_b, "rows": rows_b},
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n書き出し: {args.out}")
    if n_unique_summary.get("available"):
        print(f"[a] n_unique_rows epoch1={n_unique_summary['epoch1_n_unique_mean']:.1f} "
              f"-> epoch20={n_unique_summary['epoch20_n_unique_mean']:.1f}")
    print(f"[a] raw_update_norm early(1-5)={raw_norm_summary['early_mean_of_seed_means']:.4f} "
          f"late(16-20)={raw_norm_summary['late_mean_of_seed_means']:.4f} "
          f"ratio={raw_norm_summary['early_mean_of_seed_means'] / max(raw_norm_summary['late_mean_of_seed_means'], 1e-8):.2f}x")
    for mult in args.multipliers_b:
        s = summary_b[str(mult)]
        print(f"[b] mult={mult:>5g} acc={s['accuracy_mean']:.4f} "
              f"margin={s['margin_mean']:.5f} winner_retention={s['winner_retention_mean']:.3f}")


if __name__ == "__main__":
    main()
