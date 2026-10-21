#!/usr/bin/env python3
"""ステップ44.8: 縮小規模の vanilla 対照（フェーズ V）と A/B/C の最終 PPL を比較する。

判定は既存（step44_reduced_scale_analysis.py）と同じ: 平均差 > 2×SE で差あり。paired と unpaired の両方。
比較は V→A, V→B, V→C（差 = フェーズ - V。正なら PPL 増加＝モジュール付きが悪い）。
使い方: python step44_vanilla_analysis.py [縮小規模 A/B/C のディレクトリ] [V のディレクトリ] [出力先]
"""
import json
import sys
from pathlib import Path

import numpy as np

SEEDS = (0, 1, 2)
METRICS = {"val": lambda r: r["val_ppls"][-1], "test": lambda r: r["test_ppl"]}


def verdict(diff, se):
    return "悪化" if diff > 2 * se else "改善" if diff < -2 * se else "差なし"


def analyze(abc_dir, v_dir, seeds=SEEDS):
    runs = {p: [json.loads((Path(d) / f"phase{p}_seed{s}.json").read_text()) for s in seeds]
            for p, d in [("V", v_dir), ("A", abc_dir), ("B", abc_dir), ("C", abc_dir)]}
    out = {"phases": {}, "comparisons": {}}
    for p, rs in runs.items():
        out["phases"][p] = {k: {"per_seed": [f(r) for r in rs], "mean": float(np.mean([f(r) for r in rs])),
                                "sd": float(np.std([f(r) for r in rs], ddof=1))} for k, f in METRICS.items()}
    for b in "ABC":
        for k, f in METRICS.items():
            xv = np.array([f(r) for r in runs["V"]]); xb = np.array([f(r) for r in runs[b]])
            d = xb - xv
            se_p = d.std(ddof=1) / np.sqrt(len(d))
            se_u = np.sqrt(xv.var(ddof=1) / len(xv) + xb.var(ddof=1) / len(xb))
            out["comparisons"][f"V->{b}_{k}"] = {
                "mean_diff": float(d.mean()), "rel_pct": float(d.mean() / xv.mean() * 100), "per_seed": d.tolist(),
                "se_paired": float(se_p), "verdict_paired": verdict(d.mean(), se_p),
                "se_unpaired": float(se_u), "verdict_unpaired": verdict(d.mean(), se_u),
            }
    return out


def main():
    abc = sys.argv[1] if len(sys.argv) > 1 else "results/step44_reduced_scale"
    v = sys.argv[2] if len(sys.argv) > 2 else "results/step44_reduced_scale_vanilla"
    dest = Path(sys.argv[3] if len(sys.argv) > 3 else v)
    out = analyze(abc, v)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for p, m in out["phases"].items():
        print(f"{p}: val {m['val']['mean']:.3f} (sd {m['val']['sd']:.3f}) test {m['test']['mean']:.3f}")
    for key, c in out["comparisons"].items():
        print(f"{key:10s} {c['mean_diff']:+.3f} ({c['rel_pct']:+.2f}%) paired SE {c['se_paired']:.3f} -> {c['verdict_paired']} | "
              f"unpaired SE {c['se_unpaired']:.3f} -> {c['verdict_unpaired']}")


if __name__ == "__main__":
    main()
