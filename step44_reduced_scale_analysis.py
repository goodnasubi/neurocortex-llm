#!/usr/bin/env python3
"""ステップ44.4 縮小版の判定: フェーズ間の最終 PPL の平均差を、差の標準誤差の2倍と比べる。

同じシードでは初期値とデータ順が共通なので、シードごとの対応あり（paired）の SE を主とし、
対応なし（unpaired）の SE も併記する。
"""
import json
import sys
from pathlib import Path

import numpy as np

R = Path(sys.argv[1] if len(sys.argv) > 1 else "results/step44_reduced_scale")


def verdict(diff, se):
    return "悪化" if diff > 2 * se else "改善" if diff < -2 * se else "差なし"


def main():
    runs = {p: [json.loads((R / f"phase{p}_seed{s}.json").read_text()) for s in range(3)] for p in "ABC"}
    metrics = {"val": lambda r: r["val_ppls"][-1], "test": lambda r: r["test_ppl"]}
    out = {"phases": {}, "comparisons": {}}
    for p, rs in runs.items():
        out["phases"][p] = {k: {"per_seed": [f(r) for r in rs], "mean": float(np.mean([f(r) for r in rs])),
                                "sd": float(np.std([f(r) for r in rs], ddof=1))} for k, f in metrics.items()}
    for a, b in [("A", "B"), ("B", "C"), ("A", "C")]:
        for k, f in metrics.items():
            xa = np.array([f(r) for r in runs[a]]); xb = np.array([f(r) for r in runs[b]])
            d = xb - xa
            se_p = d.std(ddof=1) / np.sqrt(len(d))
            se_u = np.sqrt(xa.var(ddof=1) / len(xa) + xb.var(ddof=1) / len(xb))
            out["comparisons"][f"{a}->{b}_{k}"] = {
                "mean_diff": float(d.mean()), "rel_pct": float(d.mean() / xa.mean() * 100), "per_seed": d.tolist(),
                "se_paired": float(se_p), "verdict_paired": verdict(d.mean(), se_p),
                "se_unpaired": float(se_u), "verdict_unpaired": verdict(d.mean(), se_u),
            }
    (R / "analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for key, c in out["comparisons"].items():
        print(f"{key:10s} {c['mean_diff']:+.3f} ({c['rel_pct']:+.2f}%) paired SE {c['se_paired']:.3f} -> {c['verdict_paired']} | "
              f"unpaired SE {c['se_unpaired']:.3f} -> {c['verdict_unpaired']}")


if __name__ == "__main__":
    main()
