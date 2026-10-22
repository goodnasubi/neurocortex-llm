#!/usr/bin/env python3
"""ステップ44.11: 圧縮表現の再生（H2/H4）・生の断片（R）・頻度だけ（S）・計算量の対照（C）・下限（R_M4/R_r2）の比較と、
事前登録した仮説 H1〜H5 の判定。

使い方: python step44_compressed_analysis.py <結果ディレクトリ> <seed...> [--out analysis.json] [--ref44 <44.9 と 44.10 の eval>]
差 = 後 − 前、2×SE 超で差あり（44.9/44.10 と同じ step44_continual_analysis.compare）。
"""
import json
import sys
from pathlib import Path

import numpy as np

import step44_continual_analysis as ca

COND = {"V": "V", "R": "R_M64_r16", "S": "S_M64_r16", "C": "C_M64_r16", "H2": "H2_M64_r16_pq16x8",
        "H4": "H4_M64_r16_pq16x8", "R_M4": "R_M4_r16", "R_r2": "R_M64_r2"}
DECOMP = ("dCE_marg", "dG", "G_last_past", "G_diag", "CE_marg_last_past")
LOWER = ca.LOWER_BETTER + ("dCE_marg", "CE_marg_last_past")  # dG・G は大きいほどよい
METRICS = ca.METRICS + DECOMP
PAIRS = (("V", "H2"), ("R", "H2"), ("R", "H4"), ("V", "H4"), ("H2", "H4"), ("V", "R"), ("V", "S"), ("S", "R"),
         ("V", "C"), ("C", "R"), ("V", "R_M4"), ("V", "R_r2"), ("R_M4", "R"), ("R_r2", "R"))


def load(d, seeds):
    runs = {}
    for k, tag in COND.items():
        paths = [Path(d) / f"{tag}_seed{s}.json" for s in seeds]
        if all(p.exists() for p in paths):
            runs[k] = [json.loads(p.read_text()) for p in paths]
    return runs


def share(fv, fx, fr):
    """保護の割合 (F_V − F_X)/(F_V − F_R) の seed ごとの値。"""
    return [(v - x) / (v - r) for v, x, r in zip(fv, fx, fr)]


def analyze(d, seeds):
    runs = load(d, seeds)
    for i, s in enumerate(seeds):
        assert len({tuple(runs[k][i]["order"]) for k in runs}) == 1, f"seed {s}: タスク順が一致しない"
    out = {"seeds": list(seeds), "conditions": {k: COND[k] for k in runs}, "phases": {}, "comparisons": {},
           "hypotheses": {}}
    for k, rs in runs.items():
        out["phases"][k] = {m: {"per_seed": [r["metrics"][m] for r in rs],
                                "mean": float(np.mean([r["metrics"][m] for r in rs])),
                                "sd": float(np.std([r["metrics"][m] for r in rs], ddof=1))} for m in METRICS}
        out["phases"][k]["memory_bytes_total"] = rs[0]["memory_bytes_total"]
        out["phases"][k]["train_seconds_mean"] = float(np.mean([r["train_seconds"] for r in rs]))
        if k.startswith("H"):
            out["phases"][k]["pq_recon_rel_err_mean"] = float(np.mean([np.mean(r["pq_recon_rel_err"]) for r in rs]))
            # 最終時点の過去タスク（最後に保存したものを除く）の隠れ状態の古さ
            out["phases"][k]["drift_final_past_mean"] = float(np.mean([np.mean(r["drift_final"][:-1]) for r in rs]))
    for a, b in PAIRS:
        if a in runs and b in runs:
            for m in METRICS:
                out["comparisons"][f"{a}->{b}_{m}"] = ca.compare([r["metrics"][m] for r in runs[a]],
                                                                 [r["metrics"][m] for r in runs[b]], m in LOWER)
    c, h = out["comparisons"], out["hypotheses"]
    F = lambda k: [r["metrics"]["F_ce"] for r in runs[k]]
    both = lambda x, v: x["verdict_paired"] == v and x["verdict_unpaired"] == v
    if "H2" in runs:
        h["H1_V->H2"] = ca.h1(c["V->H2_F_ce"], c["V->H2_LA_ce"])
    for hk in ("H2", "H4"):
        if hk in runs and "R" in runs:
            h[f"H2_R->{hk}"] = ("H は R より多く守る" if both(c[f"R->{hk}_F_ce"], "改善")
                                else "R が H を支配する（小さい保存で同等以上に守る）")
            h[f"H2_bytes_ratio_{hk}/R"] = out["phases"][hk]["memory_bytes_total"] / out["phases"]["R"]["memory_bytes_total"]
    if {"V", "S", "R"} <= set(runs):
        sf = share(F("V"), F("S"), F("R"))
        h["H3a_share_freq_per_seed"] = sf
        h["H3a_share_freq_mean"] = float(np.mean(sf))
        h["H3a"] = "大部分は出力分布（頻度）の保護" if np.mean(sf) >= 0.9 else "頻度だけでは R の軽減の 9 割に届かない"
        h["H3b"] = ("R は頻度を超えて文脈の利得も守る" if both(c["S->R_G_last_past"], "改善")
                    else "文脈の利得の保護は S と差が示せない")
    if {"C", "R"} <= set(runs):
        h["H4_C->R"] = ("軽減は再生の内容によるもので計算量によらない" if both(c["C->R_F_ce"], "改善")
                        else "計算量の対照と差が示せない")
    for k in ("R_M4", "R_r2"):
        if k in runs and "R" in runs:
            h[f"H5_V->{k}"] = ca.h1(c[f"V->{k}_F_ce"], c[f"V->{k}_LA_ce"])
            h[f"H5_share_{k}"] = float(np.mean(share(F("V"), F(k), F("R"))))
    for k in ("H2", "H4", "C"):
        if k in runs and {"V", "R"} <= set(runs):
            h[f"share_{k}"] = float(np.mean(share(F("V"), F(k), F("R"))))
    return out


def compare_ref(runs_dir, ref49, ref410, seeds):
    """V・R の再計算と 44.9/44.10 の結果の CE 行列の最大差（同一性の確認）。"""
    res = {}
    for k, ref, name in (("V", ref49, "phaseV"), ("R", ref410, "phaseR")):
        diffs = []
        for s in seeds:
            new = json.loads((Path(runs_dir) / f"{COND[k]}_seed{s}.json").read_text())["ce"]
            old = json.loads((Path(ref) / f"{name}_seed{s}.json").read_text())["ce"]
            diffs.append(max(abs(a - b) for ra, rb in zip(new, old) for a, b in zip(ra, rb)))
        res[k] = {"max_abs_diff_ce": float(max(diffs)), "per_seed": diffs}
    return res


def to_markdown(out):
    ph = list(out["phases"])
    lines = ["| 指標 | " + " | ".join(ph) + " |", "|---" * (len(ph) + 1) + "|"]
    for m in METRICS:
        lines.append(f"| {m} | " + " | ".join(f"{out['phases'][p][m]['mean']:.3f} ({out['phases'][p][m]['sd']:.3f})"
                                              for p in ph) + " |")
    lines.append("| 保存 B（3 タスク計） | " + " | ".join(str(out["phases"][p]["memory_bytes_total"]) for p in ph) + " |")
    lines += ["", "| 比較（差 = 後 − 前） | 平均差 | paired SE | paired | unpaired SE | unpaired |", "|---|---|---|---|---|---|"]
    for k, c in out["comparisons"].items():
        if any(k.endswith("_" + m) for m in ("F_ce", "LA_ce", "ACC_ce", "dCE_marg", "dG", "G_last_past")):
            lines.append(f"| {k} | {c['mean_diff']:+.4f} | {c['se_paired']:.4f} | {c['verdict_paired']} | "
                         f"{c['se_unpaired']:.4f} | {c['verdict_unpaired']} |")
    lines += ["", "仮説: " + "; ".join(f"{k}: {v}" for k, v in out["hypotheses"].items())]
    if "ref44" in out:
        lines.append(f"44.9/44.10 との CE の最大差: {out['ref44']}")
    return "\n".join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    opts = {}
    for key in ("--out", "--ref44"):
        if key in argv:
            i = argv.index(key)
            opts[key] = argv[i + 1]
            del argv[i:i + 2]
    seeds = [int(s) for s in argv[1:]]
    out = analyze(argv[0], seeds)
    if "--ref44" in opts:
        base = Path(opts["--ref44"])
        out["ref44"] = compare_ref(argv[0], base / "step44_continual_forgetting" / "eval",
                                   base / "step44_continual_replay" / "eval", seeds)
    md = to_markdown(out)
    print(md)
    if "--out" in opts:
        Path(opts["--out"]).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        Path(opts["--out"]).with_suffix(".md").write_text(md + "\n")


if __name__ == "__main__":
    main()
