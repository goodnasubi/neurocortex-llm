#!/usr/bin/env python3
"""ステップ44.16: フル規模での再検証の解析と、事前登録した仮説 H0-F〜H3-F の判定。

設計の固定版は docs/brain-structure-research.md のステップ44.16 設計の節。
主判定は F_ce ではなく文脈の利得（G_last_past）で行う（44.14/44.15 と同じ）。
絶対値は小規模と並べて比較せず、符号・条件間差・保護の割合だけを見る（概念的再検証）。

解析の前に、各結果 JSON の真正性チェック（git_commit が null／dirty でないこと、
config_sha256 が設定から再生成できること、全指標が有限であること、時刻の前後関係）を
自動で行い、違反があれば判定を出さずに報告する（44.4 撤回の再発防止、設計 §4.3）。

使い方: python step44_fullscale_analysis.py <結果ディレクトリ> <seed...> [--out analysis.json]
差 = 後 − 前、2×SE 超で差あり（step44_continual_analysis.compare）。
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import step44_continual_analysis as ca
import step44_fullscale_replay as fs

COND16 = {"V": "V", "R": "R_M64_E7.75", "R_M4": "R_M4_E124", "X4": "X4_M4_E15.5",
          "C4": "C4_M4_E124_w128", "CX4": "CX4_M4_E15.5_w128"}
DECOMP = ("dCE_marg", "dG", "G_last_past", "G_diag", "CE_marg_last_past")
LOWER = ca.LOWER_BETTER + ("dCE_marg", "CE_marg_last_past")
METRICS = ca.METRICS + DECOMP
PAIRS = (("V", "R"), ("V", "R_M4"), ("R_M4", "R"), ("R_M4", "X4"), ("R_M4", "C4"), ("R_M4", "CX4"),
         ("X4", "CX4"), ("C4", "CX4"), ("CX4", "R"), ("V", "CX4"))


def load(d, seeds):
    runs = {}
    for k, tag in COND16.items():
        paths = [Path(d) / f"{tag}_seed{s}.json" for s in seeds]
        if all(p.exists() for p in paths):
            runs[k] = [json.loads(p.read_text()) for p in paths]
    return runs


def validate_run(r: dict) -> list[str]:
    """結果 JSON 1件の真正性チェック。違反の文字列の列を返す（空なら合格）。"""
    problems = []
    env = r.get("env") or {}
    commit = env.get("git_commit")
    if commit is None:
        problems.append("env.git_commit が null")
    elif str(commit).endswith("+dirty"):
        problems.append(f"env.git_commit が dirty（{commit}）")
    cfg = r.get("config")
    if cfg is None or r.get("config_sha256") != fs.config_sha256(cfg):
        problems.append("config_sha256 が設定から再生成できない")
    for k, v in (r.get("metrics") or {}).items():
        if not np.isfinite(v):
            problems.append(f"metrics.{k} が有限でない")
    if r.get("started_utc") and r.get("finished_utc") and r["finished_utc"] < r["started_utc"]:
        problems.append("finished_utc が started_utc より前")
    audit = (r.get("exposure_audit") or {}).get("per_stage") or []
    recorded = r.get("n_presented_per_task") or []
    for st in audit:
        if recorded and st["stage"] < len(recorded) and st["total_presentations"] != recorded[st["stage"]]:
            problems.append(f"露出監査の総提示回数が記録と一致しない（stage {st['stage']}）")
    return problems


def exposure_ok(r: dict, tol_frac: float = 0.10) -> bool:
    """実提示回数（1断片平均）が目標 E の近傍に収まっているか（操作の検証）。"""
    a = r.get("exposure_audit") or {}
    E = a.get("target_E")
    if E is None:
        return True  # V は再生なし
    for st in a.get("per_stage", []):
        if st["pool"] and abs(st["mean_per_fragment"] - E) > max(2.0, tol_frac * E):
            return False
    return True


def analyze(d, seeds):
    runs = load(d, seeds)
    problems = {k: [p for r in rs for p in validate_run(r)] for k, rs in runs.items()}
    out = {"seeds": list(seeds), "conditions": {k: COND16[k] for k in runs},
           "validation": problems, "phases": {}, "comparisons": {}, "hypotheses": {}}
    for k, rs in runs.items():
        out["phases"][k] = {m: {"per_seed": [r["metrics"][m] for r in rs],
                                "mean": float(np.mean([r["metrics"][m] for r in rs])),
                                "sd": float(np.std([r["metrics"][m] for r in rs], ddof=1))} for m in METRICS}
        out["phases"][k]["train_seconds_mean"] = float(np.mean([r["train_seconds"] for r in rs]))
        out["phases"][k]["exposure_ok"] = all(exposure_ok(r) for r in rs)
        means = [(r.get("exposure_audit") or {}).get("per_stage") or [] for r in rs]
        flat = [st["mean_per_fragment"] for ms in means for st in ms if st["pool"]]
        if flat:
            out["phases"][k]["presentations_per_fragment_mean"] = float(np.mean(flat))
    for a, b in PAIRS:
        if a in runs and b in runs:
            for m in METRICS:
                out["comparisons"][f"{a}->{b}_{m}"] = ca.compare([r["metrics"][m] for r in runs[a]],
                                                                 [r["metrics"][m] for r in runs[b]], m in LOWER)
    c, h = out["comparisons"], out["hypotheses"]
    both = lambda x, v: x["verdict_paired"] == v and x["verdict_unpaired"] == v
    G = lambda k: [r["metrics"]["G_last_past"] for r in runs[k]]
    F = lambda k: [r["metrics"]["F_ce"] for r in runs[k]]
    if any(problems[k] for k in problems):
        h["s16_validation"] = "真正性チェックに違反があるため判定しない: " + "; ".join(
            f"{k}: {p}" for k in problems for p in problems[k])
        return out
    if {"V", "R_M4", "X4", "CX4"} <= set(runs):
        # H0-F（関門）: R_M4 の G_last_past が負で過半の seed でも負、G_diag が正、露出監査が目標どおり
        g_rm4 = G("R_M4")
        h["s16_H0F_majority_negative"] = bool(sum(1 for v in g_rm4 if v < 0) * 2 > len(g_rm4))
        h["s16_H0F_G_diag_positive"] = bool(all(r["metrics"]["G_diag"] > 0 for k in runs for r in runs[k]))
        h["s16_H0F_exposure_ok"] = bool(out["phases"]["R_M4"]["exposure_ok"])
        h["s16_H0F_reproduced"] = bool(np.mean(g_rm4) < 0 and h["s16_H0F_majority_negative"]
                                       and h["s16_H0F_G_diag_positive"] and h["s16_H0F_exposure_ok"])
        h["s16_H0F"] = ("フル規模でも文脈への過適合が再現した（R_M4 の G_last_past が負）"
                        if h["s16_H0F_reproduced"] else
                        "前提が成立しなかったため H1-F／H2-F は判定不能（規模・学習条件が変わると病理が出ない、という境界条件の知見として扱う）")
        if h["s16_H0F_reproduced"]:
            # H1-F: CX4 の G_last_past > 0 かつ対 R_M4 が両検定で改善
            h["s16_H1F_gain_positive"] = bool(np.mean(G("CX4")) > 0)
            h["s16_H1F_better_than_R_M4"] = bool(both(c["R_M4->CX4_G_last_past"], "改善"))
            h["s16_H1F_prevented"] = bool(h["s16_H1F_gain_positive"] and h["s16_H1F_better_than_R_M4"])
            h["s16_H1F"] = ("CX4 はフル規模でも文脈への過適合を防ぐ" if h["s16_H1F_prevented"]
                            else "CX4 がフル規模で文脈への過適合を防ぐとは言えない")
            share = [(x - m) / (v - m) for x, m, v in zip(G("CX4"), g_rm4, G("V"))]
            h["s16_H1F_share_CX4_per_seed"] = share
            h["s16_H1F_share_CX4"] = float(np.mean(share))
            # H2-F: H1-F 成立かつ CX4 の F_ce 平均が X4 以下なら実用候補は CX4 のまま
            h["s16_H2F_F_not_worse_than_X4"] = bool(np.mean(F("CX4")) <= np.mean(F("X4")))
            if h["s16_H1F_prevented"] and h["s16_H2F_F_not_worse_than_X4"]:
                h["s16_H2F_candidate"] = "CX4"
                h["s16_H2F"] = "実用候補は CX4 のまま維持される（H1-F 成立、かつ CX4 の F_ce 平均が X4 以下）"
            elif h["s16_H1F_prevented"]:
                h["s16_H2F_candidate"] = "X4"
                h["s16_H2F"] = "CX4 は文脈は守るが忘却の保護で X4 に劣り、実用候補は X4 へ移す"
            else:
                h["s16_H2F_candidate"] = None
                h["s16_H2F"] = "H1-F が不成立のため実用候補の判定は行わない（併用は棄却し、解釈は規模・学習条件への依存へ移す）"
    if "C4" in runs and {"R_M4", "X4", "CX4"} <= set(runs):  # H3-F（拡張時のみ、記述的）
        for m, key in (("G_last_past", "G"), ("F_ce", "F")):
            vals = lambda k: np.array([r["metrics"][m] for r in runs[k]])
            inter = (vals("CX4") - vals("C4")) - (vals("X4") - vals("R_M4"))
            h[f"s16_H3F_interaction_{key}_per_seed"] = inter.tolist()
            h[f"s16_H3F_interaction_{key}_mean"] = float(np.mean(inter))
            h[f"s16_H3F_interaction_{key}_se"] = float(inter.std(ddof=1) / np.sqrt(len(inter)))
    return out


def to_markdown(out) -> str:
    lines = ["| 条件 | G_last_past | F_ce | dCE_marg | dG |", "|---|---|---|---|---|"]
    for k, ph in out["phases"].items():
        lines.append(f"| {k} | {ph['G_last_past']['mean']:+.3f}±{ph['G_last_past']['sd']:.3f} "
                     f"| {ph['F_ce']['mean']:.3f}±{ph['F_ce']['sd']:.3f} "
                     f"| {ph['dCE_marg']['mean']:+.3f} | {ph['dG']['mean']:+.3f} |")
    for key in sorted(out["hypotheses"]):
        if isinstance(out["hypotheses"][key], str):
            lines.append(f"\n- {key}: {out['hypotheses'][key]}")
    return "\n".join(lines)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    out_path = None
    if "--out" in args:
        i = args.index("--out")
        out_path = args[i + 1]
        args = args[:i] + args[i + 2:]
    d, seeds = args[0], [int(s) for s in args[1:]]
    out = analyze(d, seeds)
    print(to_markdown(out))
    if out_path:
        Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
