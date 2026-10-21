#!/usr/bin/env python3
"""ステップ44.4 フル規模の最終判定: フェーズ間の最終エポック val PPL の平均差を、差の標準誤差の2倍と比べる。

判定基準は縮小規模（step44_reduced_scale_analysis.py）と同じ。同じシードでは初期値とデータ順が共通なので、
両フェーズがそろったシードだけで対応あり（paired）の SE を出し、対応なし（unpaired）の SE も併記する。
欠けている (phase, seed) があっても、そろっている分だけで解析する。n<2 の比較は判定不能とする。

使い方: python step44_full_scale_analysis.py [結果ディレクトリ] [縮小規模の analysis.json]
  → <結果ディレクトリ>/analysis.json を書き、docs に貼れる Markdown を標準出力に出す。
"""
import json
import sys
from pathlib import Path

import numpy as np

PHASES = "ABC"
SEEDS = range(3)
PAIRS = [("A", "B"), ("B", "C"), ("A", "C")]
UNDECIDABLE = "判定不能"


def verdict(diff, se):
    # PPL は小さいほど良いので、増加が「悪化」
    return "悪化" if diff > 2 * se else "改善" if diff < -2 * se else "差なし"


def load_runs(root):
    """{phase: {seed: run}}。ファイルがない (phase, seed) は含めない。"""
    runs = {p: {} for p in PHASES}
    for p in PHASES:
        for s in SEEDS:
            f = root / f"phase{p}_seed{s}.json"
            if f.exists():
                runs[p][s] = json.loads(f.read_text())
    return runs


def compare(xa, xb, per_seed_diff):
    """xa, xb: 各フェーズの値（対応なし用）。per_seed_diff: 共通シードの b-a（対応あり用）。"""
    out = {"n_a": len(xa), "n_b": len(xb), "n_paired": len(per_seed_diff), "per_seed": per_seed_diff}
    if len(xa) >= 2 and len(xb) >= 2:
        d = float(np.mean(xb) - np.mean(xa))
        se = float(np.sqrt(np.var(xa, ddof=1) / len(xa) + np.var(xb, ddof=1) / len(xb)))
        out.update(mean_diff_unpaired=d, rel_pct_unpaired=d / float(np.mean(xa)) * 100,
                   se_unpaired=se, verdict_unpaired=verdict(d, se))
    else:
        out.update(mean_diff_unpaired=None, rel_pct_unpaired=None, se_unpaired=None, verdict_unpaired=UNDECIDABLE)
    # 平均差は共通シードが1つでも参考として出すが、判定は n>=2 のみ
    d = np.array(per_seed_diff)
    out.update(mean_diff=float(d.mean()) if len(d) else None,
               rel_pct=float(d.mean() / np.mean(xa) * 100) if len(d) else None, se_paired=None, verdict_paired=UNDECIDABLE)
    if len(d) >= 2:
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        out.update(se_paired=se, verdict_paired=verdict(d.mean(), se))
    return out


def analyze(root, reduced_path=None):
    runs = load_runs(root)
    final = {p: {s: r["val_ppls"][-1] for s, r in rs.items()} for p, rs in runs.items()}
    out = {"metric": "最終エポックの val_ppls[-1]（backbone lm_head の非重み付き PPL）",
           "missing": [f"{p}{s}" for p in PHASES for s in SEEDS if s not in runs[p]],
           "phases": {}, "epochs": {}, "comparisons": {}, "reduced_scale_agreement": {}}
    for p in PHASES:
        v = list(final[p].values())
        out["phases"][p] = {"seeds": sorted(final[p]), "per_seed": {str(s): x for s, x in sorted(final[p].items())},
                            "n": len(v), "mean": float(np.mean(v)) if v else None,
                            "sd": float(np.std(v, ddof=1)) if len(v) >= 2 else None}
        # エポックごとの平均（そのエポックまで進んだシードだけ）
        n_ep = max((len(r["val_ppls"]) for r in runs[p].values()), default=0)
        out["epochs"][p] = [{"epoch": e + 1, "n": len(xs), "mean": float(np.mean(xs))}
                            for e in range(n_ep) for xs in [[r["val_ppls"][e] for r in runs[p].values() if len(r["val_ppls"]) > e]]]
    for a, b in PAIRS:
        common = sorted(set(final[a]) & set(final[b]))
        c = compare(list(final[a].values()), list(final[b].values()), [final[b][s] - final[a][s] for s in common])
        c["paired_seeds"] = common
        out["comparisons"][f"{a}->{b}_val"] = c
    if reduced_path is not None and Path(reduced_path).exists():
        red = json.loads(Path(reduced_path).read_text())["comparisons"]
        for key, c in out["comparisons"].items():
            if key in red:
                full_v, red_v = c["verdict_paired"], red[key]["verdict_paired"]
                out["reduced_scale_agreement"][key] = {
                    "reduced_verdict_paired": red_v, "reduced_rel_pct": red[key]["rel_pct"], "full_verdict_paired": full_v,
                    "agree": None if full_v == UNDECIDABLE else full_v == red_v}
    return out


def fmt(x, f="{:.3f}"):
    return "—" if x is None else f.format(x)


def to_markdown(out):
    L = [f"評価指標: {out['metric']}。判定: 平均差が 2×SE を超えたら差あり（PPL 増加＝悪化）。",
         f"欠損 (phase, seed): {', '.join(out['missing']) if out['missing'] else 'なし'}", "",
         "| フェーズ | n | seed0 | seed1 | seed2 | 平均 | SD |", "|---|---|---|---|---|---|---|"]
    for p, ph in out["phases"].items():
        cells = [fmt(ph["per_seed"].get(str(s)), "{:.2f}") for s in SEEDS]
        L.append(f"| {p} | {ph['n']} | {' | '.join(cells)} | {fmt(ph['mean'], '{:.2f}')} | {fmt(ph['sd'])} |")
    n_ep = max((len(e) for e in out["epochs"].values()), default=0)
    L += ["", "エポックごとの平均 val PPL（括弧内は n）:", "",
          "| フェーズ | " + " | ".join(f"E{e + 1}" for e in range(n_ep)) + " |", "|---|" + "---|" * n_ep]
    for p, eps in out["epochs"].items():
        cells = [f"{e['mean']:.2f} ({e['n']})" for e in eps] + ["—"] * (n_ep - len(eps))
        L.append(f"| {p} | " + " | ".join(cells) + " |")
    L += ["", "| 比較 | paired n（seed） | 平均差 | 相対 | paired SE | 判定 | unpaired n | unpaired SE | 判定 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for key, c in out["comparisons"].items():
        seeds = ",".join(map(str, c["paired_seeds"])) or "—"
        L.append(f"| {key.replace('_val', '')} | {c['n_paired']} ({seeds}) | {fmt(c['mean_diff'], '{:+.3f}')} | "
                 f"{fmt(c['rel_pct'], '{:+.2f}%')} | {fmt(c['se_paired'])} | {c['verdict_paired']} | "
                 f"{c['n_a']}/{c['n_b']} | {fmt(c['se_unpaired'])} | {c['verdict_unpaired']} |")
    if out["reduced_scale_agreement"]:
        L += ["", "縮小規模との方向の一致（paired 判定）:", "", "| 比較 | 縮小規模 | フル規模 | 一致 |", "|---|---|---|---|"]
        for key, g in out["reduced_scale_agreement"].items():
            agree = "—" if g["agree"] is None else "一致" if g["agree"] else "不一致"
            L.append(f"| {key.replace('_val', '')} | {g['reduced_verdict_paired']} ({g['reduced_rel_pct']:+.2f}%) | "
                     f"{g['full_verdict_paired']} | {agree} |")
    return "\n".join(L)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[0] if argv else "results/step44_full_scale_rerun")
    reduced = Path(argv[1] if len(argv) > 1 else "results/step44_reduced_scale/analysis.json")
    out = analyze(root, reduced)
    (root / "analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(to_markdown(out))
    return out


if __name__ == "__main__":
    main()
