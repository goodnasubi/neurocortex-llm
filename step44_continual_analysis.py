#!/usr/bin/env python3
"""ステップ44.9: 継続学習の忘却指標の V→A/B/C 比較（paired・unpaired）と、事前登録した仮説の判定。

使い方: python step44_continual_analysis.py <結果ディレクトリ> <seed...> [--out analysis.json]
差 = フェーズ - V。2×SE を超えたら差あり（44.8 と同じ規則）。向きは指標ごとに「小さいほどよい」か「大きいほどよい」。
"""
import json
import sys
from pathlib import Path

import numpy as np

LOWER_BETTER = ("F_ce", "dCE", "ACC_ce", "LA_ce", "F_acc")
HIGHER_BETTER = ("ACC_acc", "BWT_acc", "LA_acc")
METRICS = LOWER_BETTER + HIGHER_BETTER
PRIMARY = "F_ce"


def verdict(diff, se, lower_better=True):
    """差あり（2×SE 超）のとき、改善か悪化か。"""
    if abs(diff) <= 2 * se:
        return "差なし"
    return "改善" if (diff < 0) == lower_better else "悪化"


def compare(xv, xb, lower_better):
    xv, xb = np.asarray(xv, float), np.asarray(xb, float)
    d = xb - xv
    se_p = d.std(ddof=1) / np.sqrt(len(d))
    se_u = np.sqrt(xv.var(ddof=1) / len(xv) + xb.var(ddof=1) / len(xb))
    return {"mean_diff": float(d.mean()), "per_seed": d.tolist(), "se_paired": float(se_p), "se_unpaired": float(se_u),
            "verdict_paired": verdict(d.mean(), se_p, lower_better),
            "verdict_unpaired": verdict(d.mean(), se_u, lower_better)}


def h1(fc, la):
    """事前登録 H1: 忘却 F_ce が V より paired・unpaired の両方で 2×SE 超に小さく、かつ LA_ce が両方で悪化でない。"""
    both = lambda c, v: c["verdict_paired"] == v and c["verdict_unpaired"] == v
    any_ = lambda c, v: v in (c["verdict_paired"], c["verdict_unpaired"])
    if both(fc, "改善"):
        return "支持" if not any_(la, "悪化") else "忘却は減ったが可塑性が低下（学習不足による見かけの軽減の疑い。不支持）"
    if both(fc, "悪化"):
        return "逆（モジュール付きの方が忘れる）"
    if any_(fc, "改善") or any_(fc, "悪化"):
        return "片方の検定のみ差あり（傾向。不支持）"
    return "差なし（不支持）"


def analyze(d, seeds):
    d = Path(d)
    runs = {p: [json.loads((d / f"phase{p}_seed{s}.json").read_text()) for s in seeds] for p in "VABC"}
    # 同一 seed の並びは全フェーズで同じであること（paired の前提）
    for i, s in enumerate(seeds):
        assert len({tuple(runs[p][i]["order"]) for p in "VABC"}) == 1, f"seed {s}: タスク順が一致しない"
    out = {"seeds": list(seeds), "phases": {}, "comparisons": {}, "h1": {}}
    for p, rs in runs.items():
        out["phases"][p] = {m: {"per_seed": [r["metrics"][m] for r in rs],
                                "mean": float(np.mean([r["metrics"][m] for r in rs])),
                                "sd": float(np.std([r["metrics"][m] for r in rs], ddof=1))} for m in METRICS}
        out["phases"][p]["n_params"] = runs[p][0]["n_params"]
        out["phases"][p]["train_seconds_mean"] = float(np.mean([r["train_seconds"] for r in rs]))
    for b in "ABC":
        for m in METRICS:
            out["comparisons"][f"V->{b}_{m}"] = compare([r["metrics"][m] for r in runs["V"]],
                                                        [r["metrics"][m] for r in runs[b]], m in LOWER_BETTER)
        out["h1"][b] = h1(out["comparisons"][f"V->{b}_{PRIMARY}"], out["comparisons"][f"V->{b}_LA_ce"])
    return out


def to_markdown(out):
    lines = ["| 指標 | " + " | ".join(f"{p} 平均(SD)" for p in "VABC") + " |", "|---|---|---|---|---|"]
    for m in METRICS:
        lines.append(f"| {m} | " + " | ".join(f"{out['phases'][p][m]['mean']:.4f} ({out['phases'][p][m]['sd']:.4f})"
                                              for p in "VABC") + " |")
    lines += ["", "| 比較（差 = フェーズ − V） | 平均差 | paired SE | paired | unpaired SE | unpaired |", "|---|---|---|---|---|---|"]
    for k, c in out["comparisons"].items():
        lines.append(f"| {k} | {c['mean_diff']:+.4f} | {c['se_paired']:.4f} | {c['verdict_paired']} | "
                     f"{c['se_unpaired']:.4f} | {c['verdict_unpaired']} |")
    lines += ["", "H1（" + PRIMARY + " が V より小さく、かつ LA_ce が悪化しない）: " + "; ".join(f"{b}: {v}" for b, v in out["h1"].items())]
    return "\n".join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    out_path = None
    if "--out" in argv:
        i = argv.index("--out")
        out_path = argv[i + 1]
        del argv[i:i + 2]
    out = analyze(argv[0], [int(s) for s in argv[1:]])
    md = to_markdown(out)
    print(md)
    if out_path:
        Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        Path(out_path).with_suffix(".md").write_text(md + "\n")


if __name__ == "__main__":
    main()
