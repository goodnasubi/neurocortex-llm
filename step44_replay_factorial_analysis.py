#!/usr/bin/env python3
"""ステップ44.12: 保存数 M と再生数 r の 2×2 要因計画（Rsmall=M8r2, M8r16, M64r2, R=M64r16）の解析と H1〜H3 の判定。

使い方:
  python step44_replay_factorial_analysis.py <新条件のディレクトリ> <Rsmall のディレクトリ（44.11）> <R のディレクトリ（44.10）>
         <V・A のディレクトリ（44.9）> <seed...> [--out analysis.json] [--europarl-dir data/europarl_raw]
- H1（多様性の主効果）: Rsmall->M64r2 と M8r16->R の F_ce がそれぞれ paired・unpaired の両方で「改善」。両方で支持、一方で部分支持。
- H2（再生量の主効果）: Rsmall->M8r16 と M64r2->R に同じ規則。
- H3（文脈情報は多様性で決まる）: excess_1gram（過去の言語）が 0 より 2×SE 超で負（保持）なのが M64r2・R の両方で、
  Rsmall・M8r16 のどちらでもない場合に支持。
再利用の前提（設定・環境・タスク順の一致、新条件の最初の行が A とビット一致）を検査して記録する。
"""
import json
import sys
from pathlib import Path

import numpy as np

import step44_continual_analysis as ca
import step44_replay_ablation_analysis as ab

NEW = ("M64r2", "M8r16")
CELLS = {"Rsmall": (8, 2), "M8r16": (8, 16), "M64r2": (64, 2), "R": (64, 16)}
H1_PAIRS = (("Rsmall", "M64r2"), ("M8r16", "R"))  # r 固定で M 8->64
H2_PAIRS = (("Rsmall", "M8r16"), ("M64r2", "R"))  # M 固定で r 2->16
PAIRS = H1_PAIRS + H2_PAIRS + (("V", "M64r2"), ("V", "M8r16"), ("A", "M64r2"), ("A", "M8r16"))
VARYING_KEYS = ab.VARYING_KEYS
SAME_ENV_KEYS = ab.SAME_ENV_KEYS


def load(new_dir, small_dir, r_dir, base_dir, seeds):
    rd = lambda d, p, s: json.loads((Path(d) / f"phase{p}_seed{s}.json").read_text())
    runs = {p: [rd(base_dir, p, s) for s in seeds] for p in "VA"}
    runs["R"] = [rd(r_dir, "R", s) for s in seeds]
    runs["Rsmall"] = [rd(small_dir, "Rsmall", s) for s in seeds]
    for p in NEW:
        runs[p] = [rd(new_dir, p, s) for s in seeds]
    return runs


def check_reuse(runs, seeds, ref="M64r2"):
    """全条件で設定（M・r・source・label 以外）・環境・タスク順が一致するか。M・r が条件の定義どおりか。
    新条件と再利用条件の最初の行の CE と A の最大差も返す。"""
    issues = []
    for i, s in enumerate(seeds):
        if len({tuple(runs[p][i]["order"]) for p in runs}) != 1:
            issues.append(f"seed {s}: タスク順が一致しない")
        rc = runs[ref][i]["config"]
        for p in runs:
            if p in CELLS:
                c = runs[p][i]["config"]
                if (c.get("memory_per_task"), c.get("replay_size")) != CELLS[p]:
                    issues.append(f"seed {s} {p}: (M, r) が {CELLS[p]} でない")
            if p == ref:
                continue
            c = runs[p][i]["config"]
            for k in sorted((set(c) | set(rc)) - set(VARYING_KEYS)):
                if c.get(k) != rc.get(k):
                    issues.append(f"seed {s} {p}: config.{k} が {ref} と異なる")
            for k in SAME_ENV_KEYS:
                if runs[p][i]["env"].get(k) != runs[ref][i]["env"].get(k):
                    issues.append(f"seed {s} {p}: env.{k} が {ref} と異なる")
    row0 = {p: float(max(abs(a - b) for ra, rp in zip(runs["A"], runs[p]) for a, b in zip(ra["ce"][0], rp["ce"][0])))
            for p in CELLS}
    return issues, row0


def main_effect(cmps):
    """2 つの比較それぞれが paired・unpaired の両方で「改善」か。"""
    ok = [c["verdict_paired"] == c["verdict_unpaired"] == "改善" for c in cmps]
    return "支持" if all(ok) else "部分支持" if any(ok) else "不支持"


def one_sample(xs):
    xs = np.asarray(xs, float)
    m, se = float(xs.mean()), float(xs.std(ddof=1) / np.sqrt(len(xs)))
    return {"mean": m, "se": se, "retains_context": bool(m < 0 and abs(m) > 2 * se)}


def h3_verdict(ctx):
    big = all(ctx[p]["retains_context"] for p in ("M64r2", "R"))
    small = any(ctx[p]["retains_context"] for p in ("Rsmall", "M8r16"))
    return "支持" if big and not small else "不支持"


def interaction(runs, metric="F_ce"):
    """seed ごとの (R − M8r16) − (M64r2 − Rsmall)。M の効果が r によって違うか（記述的）。"""
    f = lambda p: np.array([r["metrics"][metric] for r in runs[p]], float)
    d = (f("R") - f("M8r16")) - (f("M64r2") - f("Rsmall"))
    return {"mean": float(d.mean()), "se_paired": float(d.std(ddof=1) / np.sqrt(len(d))), "per_seed": d.tolist()}


def analyze(new_dir, small_dir, r_dir, base_dir, seeds, uni=None):
    runs = load(new_dir, small_dir, r_dir, base_dir, seeds)
    issues, row0 = check_reuse(runs, seeds)
    out = {"seeds": list(seeds), "cells": CELLS, "reuse_issues": issues, "first_row_ce_maxdiff_vs_A": row0,
           "phases": {}, "comparisons": {}, "rule_h1_44_9": {}}
    for p in ("V", "A") + tuple(CELLS):
        rs = runs[p]
        out["phases"][p] = {m: {"per_seed": [r["metrics"][m] for r in rs],
                                "mean": float(np.mean([r["metrics"][m] for r in rs])),
                                "sd": float(np.std([r["metrics"][m] for r in rs], ddof=1))} for m in ca.METRICS}
        out["phases"][p]["train_seconds_mean"] = float(np.mean([r["train_seconds"] for r in rs]))
        out["phases"][p]["n_replayed_per_task"] = rs[0].get("n_replayed_per_task")
    for a, b in PAIRS:
        for m in ca.METRICS:
            out["comparisons"][f"{a}->{b}_{m}"] = ca.compare([r["metrics"][m] for r in runs[a]],
                                                             [r["metrics"][m] for r in runs[b]], m in ca.LOWER_BETTER)
        out["rule_h1_44_9"][f"{a}->{b}"] = ca.h1(out["comparisons"][f"{a}->{b}_F_ce"],
                                                 out["comparisons"][f"{a}->{b}_LA_ce"])
    out["H1_verdict"] = main_effect([out["comparisons"][f"{a}->{b}_F_ce"] for a, b in H1_PAIRS])
    out["H2_verdict"] = main_effect([out["comparisons"][f"{a}->{b}_F_ce"] for a, b in H2_PAIRS])
    out["interaction_F_ce"] = interaction(runs)
    if uni is not None:
        out["unigram_ce"] = uni
        ex = {p: [ab.excess_1gram(r, uni)["past"] for r in runs[p]] for p in ("V", "A") + tuple(CELLS)}
        out["excess_1gram"] = {p: {"past_per_seed": v, "past_mean": float(np.mean(v)),
                                   "past_sd": float(np.std(v, ddof=1))} for p, v in ex.items()}
        out["context_retention"] = {p: one_sample(ex[p]) for p in CELLS}
        out["H3_verdict"] = h3_verdict(out["context_retention"])
        out["excess_1gram_comparisons"] = {f"{a}->{b}": ca.compare(ex[a], ex[b], True) for a, b in H1_PAIRS + H2_PAIRS}
    return out


def to_markdown(out):
    ph = list(out["phases"])
    lines = ["| 指標 | " + " | ".join(f"{p} 平均(SD)" for p in ph) + " |", "|---" * (len(ph) + 1) + "|"]
    for m in ca.METRICS:
        lines.append(f"| {m} | " + " | ".join(f"{out['phases'][p][m]['mean']:.4f} ({out['phases'][p][m]['sd']:.4f})"
                                              for p in ph) + " |")
    if "excess_1gram" in out:
        lines.append("| excess_1gram（過去の言語） | " + " | ".join(
            f"{out['excess_1gram'][p]['past_mean']:+.4f} ({out['excess_1gram'][p]['past_sd']:.4f})" for p in ph) + " |")
    lines += ["", "| 比較（差 = 後 − 前） | 平均差 | paired SE | paired | unpaired SE | unpaired |", "|---|---|---|---|---|---|"]
    for k, c in out["comparisons"].items():
        lines.append(f"| {k} | {c['mean_diff']:+.4f} | {c['se_paired']:.4f} | {c['verdict_paired']} | "
                     f"{c['se_unpaired']:.4f} | {c['verdict_unpaired']} |")
    for k, c in out.get("excess_1gram_comparisons", {}).items():
        lines.append(f"| {k}_excess_1gram | {c['mean_diff']:+.4f} | {c['se_paired']:.4f} | {c['verdict_paired']} | "
                     f"{c['se_unpaired']:.4f} | {c['verdict_unpaired']} |")
    it = out["interaction_F_ce"]
    lines += ["", f"H1（多様性 M の主効果）: {out['H1_verdict']}", f"H2（再生量 r の主効果）: {out['H2_verdict']}"]
    if "H3_verdict" in out:
        lines.append(f"H3（文脈情報は M=64 のみ保持）: {out['H3_verdict']}（" + "; ".join(
            f"{p}: {c['mean']:+.4f}±{c['se']:.4f} {'保持' if c['retains_context'] else '非保持'}"
            for p, c in out["context_retention"].items()) + "）")
    lines += [f"交互作用 F_ce (R−M8r16)−(M64r2−Rsmall): {it['mean']:+.4f}（paired SE {it['se_paired']:.4f}）",
              "参考（44.9 の規則）: " + "; ".join(f"{k}: {v}" for k, v in out["rule_h1_44_9"].items()),
              f"再利用の検査: 食い違い {len(out['reuse_issues'])} 件 {out['reuse_issues']}、"
              f"最初の行の CE の A との最大差 {out['first_row_ce_maxdiff_vs_A']}"]
    if "unigram_ce" in out:
        lines.append("1-gram CE: " + ", ".join(f"{k} {v:.3f}" for k, v in out["unigram_ce"].items()))
    return "\n".join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    opts = {}
    for key in ("--out", "--europarl-dir"):
        if key in argv:
            i = argv.index(key)
            opts[key] = argv[i + 1]
            del argv[i:i + 2]
    seeds = [int(s) for s in argv[4:]]
    uni = None
    if "--europarl-dir" in opts:
        cfg = json.loads((Path(argv[0]) / f"phase{NEW[0]}_seed{seeds[0]}.json").read_text())["config"]
        uni = ab.compute_unigram(opts["--europarl-dir"], cfg)
    out = analyze(argv[0], argv[1], argv[2], argv[3], seeds, uni)
    md = to_markdown(out)
    print(md)
    if "--out" in opts:
        Path(opts["--out"]).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        Path(opts["--out"]).with_suffix(".md").write_text(md + "\n")


if __name__ == "__main__":
    main()
