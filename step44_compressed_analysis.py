#!/usr/bin/env python3
"""ステップ44.12: 圧縮表現の再生（H2/H4）・生の断片（R）・頻度だけ（S）・計算量の対照（C）・下限（R_M4/R_r2）の比較と、
事前登録した仮説 H1〜H5 の判定。ステップ44.13 の条件（HR: 再固定化、G: 生成的リプレイ）と仮説 s13_H1〜s13_H5 の判定、
ステップ44.14 の条件（X4: 露出制限、D4: 多様性最大化の保存選択、C4: 切り出し再生）と仮説 s14_H0〜s14_H4 の判定も扱う。
44.14 の主判定は F_ce ではなく文脈の利得（G_last_past）の回復である（44.13 の H4 の留保による）。

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
COND13 = {"HR": "HR2_M64_r16_pq16x8", "G": "G_M64_r16"}  # 44.13 の条件（結果ファイルがある場合だけ読み込む）
COND14 = {"X4": "X4_M4_rdiv4", "D4": "D4_M4_r16", "C4": "C4_M4_r16"}  # 44.14 の条件（同上）
DECOMP = ("dCE_marg", "dG", "G_last_past", "G_diag", "CE_marg_last_past")
LOWER = ca.LOWER_BETTER + ("dCE_marg", "CE_marg_last_past")  # dG・G は大きいほどよい
METRICS = ca.METRICS + DECOMP
PAIRS = (("V", "H2"), ("R", "H2"), ("R", "H4"), ("V", "H4"), ("H2", "H4"), ("V", "R"), ("V", "S"), ("S", "R"),
         ("V", "C"), ("C", "R"), ("V", "R_M4"), ("V", "R_r2"), ("R_M4", "R"), ("R_r2", "R"),
         ("H2", "HR"), ("V", "HR"), ("HR", "R"), ("V", "G"), ("S", "G"), ("G", "R"), ("H2", "G"),
         ("R_M4", "X4"), ("R_M4", "D4"), ("R_M4", "C4"), ("X4", "R"), ("D4", "R"), ("C4", "R"),
         ("V", "X4"), ("V", "D4"), ("V", "C4"), ("D4", "C4"))


def load(d, seeds):
    runs = {}
    for k, tag in {**COND, **COND13, **COND14}.items():
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
    out = {"seeds": list(seeds), "conditions": {k: {**COND, **COND13, **COND14}[k] for k in runs}, "phases": {},
           "comparisons": {}, "hypotheses": {}, "drift_dose_response": []}
    for k, rs in runs.items():
        out["phases"][k] = {m: {"per_seed": [r["metrics"][m] for r in rs],
                                "mean": float(np.mean([r["metrics"][m] for r in rs])),
                                "sd": float(np.std([r["metrics"][m] for r in rs], ddof=1))} for m in METRICS}
        out["phases"][k]["memory_bytes_total"] = rs[0]["memory_bytes_total"]
        out["phases"][k]["train_seconds_mean"] = float(np.mean([r["train_seconds"] for r in rs]))
        if rs[0].get("stored_gap"):  # 44.14: 保存断片への過適合の直接指標（記述的）
            out["phases"][k]["stored_gap_ce_mean"] = float(np.mean([r["stored_gap"]["gap_ce_past"] for r in rs]))
            out["phases"][k]["stored_gap_gain_mean"] = float(np.mean([r["stored_gap"]["gap_gain_past"] for r in rs]))
        if rs[0].get("stored_pairwise_jaccard"):
            out["phases"][k]["stored_pairwise_jaccard_mean"] = float(
                np.mean([np.mean(r["stored_pairwise_jaccard"]) for r in rs]))
        if rs[0].get("replay_counts_per_task"):  # 44.14: 1 断片あたりの実再生回数（全フェーズ・全断片の平均）
            cnts = [v for r in rs for ph in r["replay_counts_per_task"] for t in ph for v in t]
            if cnts:
                out["phases"][k]["replay_per_fragment_mean"] = float(np.mean(cnts))
        if k.startswith("H"):
            out["phases"][k]["pq_recon_rel_err_mean"] = float(np.mean([np.mean(r["pq_recon_rel_err"]) for r in rs]))
            # 最終時点の過去タスク（最後に保存したものを除く）の隠れ状態の古さ
            dfp = [float(np.mean(r["drift_final"][:-1])) for r in rs]
            out["phases"][k]["drift_final_past_per_seed"] = dfp
            out["phases"][k]["drift_final_past_mean"] = float(np.mean(dfp))
            for i, s in enumerate(seeds):  # drift の用量反応（記述的）: 古さと F_ce を seed ごとに並べる
                out["drift_dose_response"].append({"condition": k, "seed": s, "drift_final_past": dfp[i],
                                                   "F_ce": rs[i]["metrics"]["F_ce"]})
        if k == "HR":
            after = [float(np.mean([v for e in r.get("refresh_drift", []) for v in e["after"]])) for r in rs]
            before = [float(np.mean([v for e in r.get("refresh_drift", []) for v in e["before"]])) for r in rs]
            if all(np.isfinite(after)):
                out["phases"][k]["drift_after_refresh_mean"] = float(np.mean(after))
                out["phases"][k]["drift_before_refresh_mean"] = float(np.mean(before))
    for a, b in PAIRS:
        if a in runs and b in runs:
            for m in METRICS:
                out["comparisons"][f"{a}->{b}_{m}"] = ca.compare([r["metrics"][m] for r in runs[a]],
                                                                 [r["metrics"][m] for r in runs[b]], m in LOWER)
    if {"H2", "HR"} <= set(runs):  # 古さそのものの比較（drift は小さいほどよい）
        out["comparisons"]["H2->HR_drift_final_past"] = ca.compare(
            out["phases"]["H2"]["drift_final_past_per_seed"], out["phases"]["HR"]["drift_final_past_per_seed"], True)
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
    # ---- ステップ44.13 の事前登録（s13_H1〜s13_H5）。判定基準は docs の 44.13 設計の通りで変えない。
    if {"H2", "HR"} <= set(runs):
        d_after = out["phases"]["HR"].get("drift_after_refresh_mean")
        pq_level = d_after is not None and d_after <= 0.19  # 44.12 の H2 の PQ 相対復元誤差の範囲（0.02〜0.19）
        drift_down = both(c["H2->HR_drift_final_past"], "改善")
        h["s13_H1_mechanism"] = ("再固定化は古さを消す機構として働いた" if pq_level and drift_down else
                                 "再固定化が古さを消す機構として働いていない"
                                 f"（作り直し直後の古さ {d_after} が PQ 水準 0.19 以下: {pq_level}、"
                                 f"最終の古さが H2 より両検定で小さい: {drift_down}）")
        h["s13_H2_effect"] = ca.h1(c["H2->HR_F_ce"], c["H2->HR_LA_ce"])
        if {"V", "R"} <= set(runs):
            share_hr = float(np.mean(share(F("V"), F("HR"), F("R"))))
            h["s13_H2_share_HR"] = share_hr
            improved = both(c["H2->HR_F_ce"], "改善")
            if not (pq_level and drift_down):
                h["s13_H2_interpretation"] = "H1（機構確認）が満たされないため、効果の有無ではなく機構の失敗として扱う"
            elif improved and share_hr >= 0.8:
                h["s13_H2_interpretation"] = "44.12 で H が R に劣った主因は表現の古さだった、を支持する"
            elif improved:
                h["s13_H2_interpretation"] = "再固定化は忘却を減らすが、保護の割合は 0.8 に届かない（部分的な回復）"
            else:
                h["s13_H2_interpretation"] = ("古さは主因ではなく、再生で下位層を学ばないこと等の古さ以外の要因が主因、"
                                              "と解釈を移す")
    if "G" in runs and "V" in runs:
        h["s13_H3_V->G"] = ca.h1(c["V->G_F_ce"], c["V->G_LA_ce"])
    if {"S", "G"} <= set(runs):
        h["s13_H4"] = ("生成された文脈は、頻度を超えて文脈の利得も守る" if both(c["S->G_G_last_past"], "改善")
                       else "生成された文脈の利得の保護は S と差が示せない")
        h["s13_H4_share_freq_G"] = float(np.mean(share(F("V"), F("S"), F("G"))))  # (F_V − F_S)/(F_V − F_G)
    if {"V", "G", "R"} <= set(runs):
        share_g = float(np.mean(share(F("V"), F("G"), F("R"))))
        h["s13_H5_share_G"] = share_g
        h["s13_H5"] = ("索引＋生成は、索引＋実文保存に迫る効率を持つ（保存は R の 1/8 で保護の割合 0.8 以上）"
                       if share_g >= 0.8 else "索引＋生成の保護の割合は 0.8 に届かない")
    # ---- ステップ44.14 の事前登録（s14_H0〜s14_H4）。判定は文脈の利得 G_last_past の回復で行い、基準は docs の 44.14 設計の通りで変えない。
    G14 = lambda k: out["phases"][k]["G_last_past"]["mean"]
    recov = lambda k: (f"R_M4->{k}_G_last_past" in c and both(c[f"R_M4->{k}_G_last_past"], "改善"))
    if {"R", "R_M4"} <= set(runs):
        neg = G14("R_M4") < 0
        below_r = both(c["R_M4->R_G_last_past"], "改善")  # R_M4 が R より両検定で小さい
        h["s14_H0_reproduced"] = bool(neg and below_r)
        h["s14_H0"] = ("文脈への過適合を再現した（R_M4 の G_last_past が負で、R より両検定で小さい）"
                       if neg and below_r else
                       "再現失敗（R_M4 の G_last_past が負: " + str(neg) + "、R より両検定で小さい: " + str(below_r) +
                       "）。対策の判定より先に、環境差で現象が消えた可能性を優先して解釈する")
    for k, hk, name in (("D4", "s14_H1", "保存の多様性最大化は、保存数が少なくても文脈への過適合を防ぐ"),
                        ("C4", "s14_H2", "再生のたびに提示文脈を変えることは、文脈への過適合を防ぐ")):
        if k in runs and "R_M4" in runs:
            ok = recov(k) and G14(k) >= 0
            h[hk] = (name if ok else
                     f"{name}、とは言えない（R_M4 より両検定で大きい: {recov(k)}、{k} の G_last_past が 0 以上: {G14(k) >= 0}）")
            if {"V", "R"} <= set(runs):
                h[f"{hk}_share_{k}"] = float(np.mean(share(F("V"), F(k), F("R"))))  # 記述的（判定には使わない）
    if "X4" in runs and "R_M4" in runs:
        h["s14_H3"] = ("病理の主因は 1 断片あたりの逐語再生の回数である、を支持する"
                       if recov("X4") else "X4 では回復しない（反復回数仮説は直接には支持されない）")
    if "R_M4" in runs:
        flags = {k: recov(k) for k in ("X4", "D4", "C4") if k in runs}
        if flags:
            if any(flags.values()):
                interp = ("いずれかの対策で回復した: 「逐語反復の回避」が共通因子である読みを結論の中心に置く"
                          if flags.get("X4") else
                          "X4 で回復しないのに D4 か C4 で回復した: 因子は回数ではなく提示文脈のカバー範囲"
                          "（D4 のみ回復）または同一性（C4 のみ回復）であると解釈を移す")
            else:
                interp = ("いずれでも回復しない: 因子は反復の側にはなく、保存が 4 本であること自体（代表できる文脈の"
                          "絶対量の不足）に解釈を移し、残課題 (2) は「少数保存では文脈は守れない」という否定的な形で締める")
            h["s14_H3_recovery"] = flags
            h["s14_H3_interpretation"] = interp
        # H4（記述的）: 対策条件の F_ce の改善を dCE_marg（頻度）と dG（文脈）に分解し、文脈の回復が頻度の犠牲の上にないかを見る
        for k in ("X4", "D4", "C4"):
            if k in runs:
                h[f"s14_H4_{k}_dCE_marg"] = (f"paired {c[f'R_M4->{k}_dCE_marg']['verdict_paired']} / "
                                             f"unpaired {c[f'R_M4->{k}_dCE_marg']['verdict_unpaired']}")
                h[f"s14_H4_{k}_dG"] = (f"paired {c[f'R_M4->{k}_dG']['verdict_paired']} / "
                                       f"unpaired {c[f'R_M4->{k}_dG']['verdict_unpaired']}")
                h[f"s14_H4_{k}_marg_worsened"] = bool(both(c[f"R_M4->{k}_dCE_marg"], "悪化"))
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
    if any("stored_gap_ce_mean" in out["phases"][p] for p in ph):
        f2 = lambda p, key: (f"{out['phases'][p][key]:.3f}" if key in out["phases"][p] else "-")
        lines.append("| 般化ギャップ CE（保持−保存、44.14） | " + " | ".join(f2(p, "stored_gap_ce_mean") for p in ph) + " |")
        lines.append("| 般化ギャップ 利得（保存−保持、44.14） | " + " | ".join(f2(p, "stored_gap_gain_mean") for p in ph) + " |")
        lines.append("| 保存断片の平均ペアワイズ Jaccard | " + " | ".join(f2(p, "stored_pairwise_jaccard_mean") for p in ph) + " |")
        lines.append("| 1 断片あたり実再生回数（平均） | " + " | ".join(f2(p, "replay_per_fragment_mean") for p in ph) + " |")
    if any("drift_final_past_mean" in out["phases"][p] for p in ph):
        f = lambda p, key: (f"{out['phases'][p][key]:.3f}" if key in out["phases"][p] else "-")
        lines.append("| 古さ drift_final_past | " + " | ".join(f(p, "drift_final_past_mean") for p in ph) + " |")
        lines.append("| 作り直し直後の古さ（HR） | " + " | ".join(f(p, "drift_after_refresh_mean") for p in ph) + " |")
    if out.get("drift_dose_response"):
        lines += ["", "| 古さの用量反応（記述的） | seed | drift_final_past | F_ce |", "|---|---|---|---|"]
        for e in out["drift_dose_response"]:
            lines.append(f"| {e['condition']} | {e['seed']} | {e['drift_final_past']:.3f} | {e['F_ce']:.3f} |")
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
