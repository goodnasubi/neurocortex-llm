#!/usr/bin/env python3
"""ステップ44.10: 条件 R（A+replay）と V・A（44.9 の結果を再利用）の比較と、事前登録した H1 の判定。

使い方: python step44_replay_analysis.py <R の結果ディレクトリ> <V・A の結果ディレクトリ> <seed...> [--out analysis.json]
比較は V->R（H1）、A->R（リプレイだけの効果、副次）、V->A（44.9 の再掲）。差 = 後者 − 前者、2×SE 超で差あり。
再利用の前提（設定・環境・タスク順の一致、R の最初の行が A と一致）を検査して結果に記録する。
"""
import json
import sys
from pathlib import Path

import numpy as np

import step44_continual_analysis as ca

PAIRS = (("V", "R"), ("A", "R"), ("V", "A"))
SAME_CONFIG_KEYS = ("langs", "per_lang_vocab", "n_train", "n_eval", "epochs", "hidden_size", "num_layers", "nhead",
                    "max_seq_length", "batch_size", "grad_accum", "lr", "threads")
SAME_ENV_KEYS = ("python", "torch", "numpy")


def load(r_dir, base_dir, seeds):
    runs = {p: [json.loads((Path(base_dir) / f"phase{p}_seed{s}.json").read_text()) for s in seeds] for p in "VA"}
    runs["R"] = [json.loads((Path(r_dir) / f"phaseR_seed{s}.json").read_text()) for s in seeds]
    return runs


def check_reuse(runs, seeds):
    """V・A を再利用する前提の検査。食い違いの一覧（空なら一致）と、R と A の最初の行の CE の最大差を返す。"""
    issues = []
    for i, s in enumerate(seeds):
        if len({tuple(runs[p][i]["order"]) for p in runs}) != 1:
            issues.append(f"seed {s}: タスク順が一致しない")
        for p in "VA":
            for k in SAME_CONFIG_KEYS:
                if runs[p][i]["config"].get(k) != runs["R"][i]["config"].get(k):
                    issues.append(f"seed {s} {p}: config.{k} が R と異なる")
            for k in SAME_ENV_KEYS:
                if runs[p][i]["env"].get(k) != runs["R"][i]["env"].get(k):
                    issues.append(f"seed {s} {p}: env.{k} が R と異なる")
    row0 = max(abs(a - b) for ra, rr in zip(runs["A"], runs["R"]) for a, b in zip(ra["ce"][0], rr["ce"][0]))
    return issues, float(row0)


def analyze(r_dir, base_dir, seeds):
    runs = load(r_dir, base_dir, seeds)
    issues, row0 = check_reuse(runs, seeds)
    out = {"seeds": list(seeds), "reuse_issues": issues, "R_vs_A_first_row_ce_maxdiff": row0,
           "phases": {}, "comparisons": {}, "h1": {}}
    for p, rs in runs.items():
        out["phases"][p] = {m: {"per_seed": [r["metrics"][m] for r in rs],
                                "mean": float(np.mean([r["metrics"][m] for r in rs])),
                                "sd": float(np.std([r["metrics"][m] for r in rs], ddof=1))} for m in ca.METRICS}
        out["phases"][p]["n_params"] = rs[0]["n_params"]
        out["phases"][p]["train_seconds_mean"] = float(np.mean([r["train_seconds"] for r in rs]))
    for a, b in PAIRS:
        for m in ca.METRICS:
            out["comparisons"][f"{a}->{b}_{m}"] = ca.compare([r["metrics"][m] for r in runs[a]],
                                                             [r["metrics"][m] for r in runs[b]], m in ca.LOWER_BETTER)
        out["h1"][f"{a}->{b}"] = ca.h1(out["comparisons"][f"{a}->{b}_{ca.PRIMARY}"],
                                       out["comparisons"][f"{a}->{b}_LA_ce"])
    out["H1_verdict"] = out["h1"]["V->R"]
    return out


def to_markdown(out):
    ph = list(out["phases"])
    lines = ["| 指標 | " + " | ".join(f"{p} 平均(SD)" for p in ph) + " |", "|---" * (len(ph) + 1) + "|"]
    for m in ca.METRICS:
        lines.append(f"| {m} | " + " | ".join(f"{out['phases'][p][m]['mean']:.4f} ({out['phases'][p][m]['sd']:.4f})"
                                              for p in ph) + " |")
    lines += ["", "| 比較（差 = 後 − 前） | 平均差 | paired SE | paired | unpaired SE | unpaired |", "|---|---|---|---|---|---|"]
    for k, c in out["comparisons"].items():
        lines.append(f"| {k} | {c['mean_diff']:+.4f} | {c['se_paired']:.4f} | {c['verdict_paired']} | "
                     f"{c['se_unpaired']:.4f} | {c['verdict_unpaired']} |")
    lines += ["", f"H1（V->R の {ca.PRIMARY} が小さく、かつ LA_ce が悪化しない）: {out['H1_verdict']}",
              "参考（同じ規則）: " + "; ".join(f"{k}: {v}" for k, v in out["h1"].items()),
              f"再利用の検査: 食い違い {len(out['reuse_issues'])} 件 {out['reuse_issues']}、"
              f"R と A の最初の行の CE の最大差 {out['R_vs_A_first_row_ce_maxdiff']:.2e}"]
    return "\n".join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    out_path = None
    if "--out" in argv:
        i = argv.index("--out")
        out_path = argv[i + 1]
        del argv[i:i + 2]
    out = analyze(argv[0], argv[1], [int(s) for s in argv[2:]])
    md = to_markdown(out)
    print(md)
    if out_path:
        Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        Path(out_path).with_suffix(".md").write_text(md + "\n")


if __name__ == "__main__":
    main()
