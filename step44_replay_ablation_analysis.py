#!/usr/bin/env python3
"""ステップ44.11: リプレイの計算量対照（C1）と保存数の下限（Rmid・Rsmall）の解析と、事前登録した H1・H2 の判定。

使い方:
  python step44_replay_ablation_analysis.py <新条件のディレクトリ> <R のディレクトリ> <V・A のディレクトリ> <seed...>
         [--out analysis.json] [--europarl-dir data/europarl_raw]
- H1: R->C1 の F_ce が paired・unpaired の両方で「悪化」（C1 の方が忘れる）なら支持。
- H2: V->Rmid（H2a）・V->Rsmall（H2b）に 44.9 と同じ規則（step44_continual_analysis.h1）。
- 用量反応: 保存数 M = 0（A）, 8, 16, 64 の F_ce（記述的）。
- 出力分布の寄与: 各言語の 1-gram（加算 1 平滑化）の保持評価 CE との差 excess_1gram（記述的。--europarl-dir 指定時）。
V・A・R の再利用の前提（設定・環境・タスク順の一致、新条件の最初の行が A とビット一致）を検査して記録する。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

import step44_continual_analysis as ca

NEW = ("C1", "Rmid", "Rsmall")
DOSE = (("A", 0), ("Rsmall", 8), ("Rmid", 16), ("R", 64))
PAIRS = (("R", "C1"), ("V", "C1"), ("A", "C1"), ("V", "Rmid"), ("V", "Rsmall"), ("V", "R"),
         ("A", "Rsmall"), ("Rsmall", "Rmid"), ("Rmid", "R"))
VARYING_KEYS = ("seeds", "phases", "out_dir", "memory_per_task", "replay_size", "replay_source", "label", "europarl_dir")
SAME_ENV_KEYS = ("python", "torch", "numpy")


def load(new_dir, r_dir, base_dir, seeds):
    runs = {p: [json.loads((Path(base_dir) / f"phase{p}_seed{s}.json").read_text()) for s in seeds] for p in "VA"}
    runs["R"] = [json.loads((Path(r_dir) / f"phaseR_seed{s}.json").read_text()) for s in seeds]
    for p in NEW:
        runs[p] = [json.loads((Path(new_dir) / f"phase{p}_seed{s}.json").read_text()) for s in seeds]
    return runs


def check_reuse(runs, seeds):
    """全条件で設定（M・r・source 以外）・環境・タスク順が一致するか。新条件の最初の行と A の CE の最大差も返す。"""
    issues = []
    ref = "C1"
    for i, s in enumerate(seeds):
        if len({tuple(runs[p][i]["order"]) for p in runs}) != 1:
            issues.append(f"seed {s}: タスク順が一致しない")
        rc = runs[ref][i]["config"]
        for p in runs:
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
            for p in ("R",) + NEW}
    return issues, row0


def unigram_ce(train_ids: torch.Tensor, eval_chunks: torch.Tensor, vocab_size: int) -> float:
    """学習語の 1-gram（加算 1 平滑化）の、保持評価断片の次トークン位置（各断片の 2 語目以降）での CE（nats）。"""
    counts = torch.bincount(train_ids.reshape(-1), minlength=vocab_size).double() + 1.0
    logp = torch.log(counts / counts.sum())
    return float(-logp[eval_chunks[:, 1:].reshape(-1)].mean())


def excess_1gram(run: dict, uni: dict) -> dict:
    """最終 CE − 1-gram CE を言語ごとに。past は最後のタスク以外（過去の言語）の平均。"""
    order, last = run["order"], run["ce"][-1]
    per = {lang: last[j] - uni[lang] for j, lang in enumerate(order)}
    return {"per_lang": per, "past": float(np.mean([per[l] for l in order[:-1]])), "all": float(np.mean(list(per.values())))}


def dose_response(runs, seeds):
    means = {p: float(np.mean([r["metrics"]["F_ce"] for r in runs[p]])) for p, _ in DOSE if p in runs}
    mono = sum(all(runs[DOSE[k][0]][i]["metrics"]["F_ce"] > runs[DOSE[k + 1][0]][i]["metrics"]["F_ce"]
                   for k in range(len(DOSE) - 1)) for i in range(len(seeds)))
    return {"M": [m for _, m in DOSE], "phases": [p for p, _ in DOSE], "F_ce_mean": [means[p] for p, _ in DOSE],
            "n_seeds_monotone_decreasing": int(mono), "n_seeds": len(seeds)}


def h1_compute(c):
    """H1: R->C1 の F_ce が両方の検定で「悪化」（C1 の方が忘れる）なら支持。"""
    v = (c["verdict_paired"], c["verdict_unpaired"])
    if v == ("悪化", "悪化"):
        return "支持"
    if "悪化" in v:
        return "片方の検定のみ差あり（傾向。不支持）"
    if v == ("改善", "改善"):
        return "逆（C1 の方が忘れない）"
    return "差なし（不支持）"


def analyze(new_dir, r_dir, base_dir, seeds, uni=None):
    runs = load(new_dir, r_dir, base_dir, seeds)
    issues, row0 = check_reuse(runs, seeds)
    out = {"seeds": list(seeds), "reuse_issues": issues, "first_row_ce_maxdiff_vs_A": row0,
           "phases": {}, "comparisons": {}, "rule_h1_44_9": {}}
    for p in ("V", "A", "R") + NEW:
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
    out["H1_verdict"] = h1_compute(out["comparisons"]["R->C1_F_ce"])
    h2a, h2b = out["rule_h1_44_9"]["V->Rmid"], out["rule_h1_44_9"]["V->Rsmall"]
    out["H2a_verdict"], out["H2b_verdict"] = h2a, h2b
    out["H2_verdict"] = ("支持" if h2a == h2b == "支持" else
                         "部分支持" if "支持" in (h2a, h2b) else "不支持")
    out["dose_response"] = dose_response(runs, seeds)
    if uni is not None:
        out["unigram_ce"] = uni
        out["excess_1gram"] = {}
        for p in ("V", "A", "R") + NEW:
            ex = [excess_1gram(r, uni) for r in runs[p]]
            out["excess_1gram"][p] = {"past_per_seed": [e["past"] for e in ex],
                                      "past_mean": float(np.mean([e["past"] for e in ex])),
                                      "past_sd": float(np.std([e["past"] for e in ex], ddof=1)),
                                      "all_mean": float(np.mean([e["all"] for e in ex]))}
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
    d = out["dose_response"]
    lines += ["", f"H1（R->C1 の F_ce が両方で悪化）: {out['H1_verdict']}",
              f"H2: {out['H2_verdict']}（H2a Rmid: {out['H2a_verdict']}; H2b Rsmall: {out['H2b_verdict']}）",
              "参考（44.9 の規則）: " + "; ".join(f"{k}: {v}" for k, v in out["rule_h1_44_9"].items()),
              "用量反応 F_ce: " + ", ".join(f"M={m}({p}) {f:.4f}" for m, p, f in zip(d["M"], d["phases"], d["F_ce_mean"]))
              + f"; 単調減少の seed {d['n_seeds_monotone_decreasing']}/{d['n_seeds']}",
              f"再利用の検査: 食い違い {len(out['reuse_issues'])} 件 {out['reuse_issues']}、"
              f"最初の行の CE の A との最大差 {out['first_row_ce_maxdiff_vs_A']}"]
    if "unigram_ce" in out:
        lines.append("1-gram CE: " + ", ".join(f"{k} {v:.3f}" for k, v in out["unigram_ce"].items()))
    return "\n".join(lines)


def compute_unigram(europarl_dir, cfg):
    import step44_continual_forgetting as cf
    data = cf.build_tasks(Path(europarl_dir), tuple(cfg["langs"]), cfg["per_lang_vocab"], cfg["n_train"],
                          cfg["n_eval"], cfg["max_seq_length"])
    return {l: unigram_ce(data["tasks"][l]["train"].chunks, data["tasks"][l]["eval"].chunks, data["vocab_size"])
            for l in data["langs"]}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    opts = {}
    for key in ("--out", "--europarl-dir"):
        if key in argv:
            i = argv.index(key)
            opts[key] = argv[i + 1]
            del argv[i:i + 2]
    seeds = [int(s) for s in argv[3:]]
    uni = None
    if "--europarl-dir" in opts:
        cfg = json.loads((Path(argv[0]) / f"phaseC1_seed{seeds[0]}.json").read_text())["config"]
        uni = compute_unigram(opts["--europarl-dir"], cfg)
    out = analyze(argv[0], argv[1], argv[2], seeds, uni)
    md = to_markdown(out)
    print(md)
    if "--out" in opts:
        Path(opts["--out"]).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        Path(opts["--out"]).with_suffix(".md").write_text(md + "\n")


if __name__ == "__main__":
    main()
