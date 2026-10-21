#!/usr/bin/env python3
"""ステップ44.5 結果統合分析: 縮小規模（256d/4層/WikiText-2）とフル規模（768d/12層/WikiText-103）のステップ44 結果を1つの表にまとめる。

新規の学習は行わず、既存の phase{A,B,C}_seed{0-2}.json だけを読む。判定基準は step44_full_scale_analysis.py と同じ
（最終エポック val PPL の平均差 > 2×SE で差あり。対応あり／なしの両方）。結果 JSON に無い量（計算時間・パラメータ数など）は
「データなし」と明記し、推定や補完はしない。

使い方: python step44_integrated_analysis.py [縮小規模ディレクトリ] [フル規模ディレクトリ] [出力ディレクトリ]
  → <出力ディレクトリ>/analysis.json を書き、docs に貼れる Markdown を標準出力に出す。
"""
import json
import sys
from pathlib import Path

import numpy as np

from step44_full_scale_analysis import PAIRS, PHASES, SEEDS, UNDECIDABLE, analyze, fmt, load_runs

NO_DATA = "データなし"
SCALES = {"reduced": "縮小規模（256d/4層/WikiText-2 ローカルデータ・語彙上限10000）", "full": "フル規模（768d/12層/WikiText-103）"}
NOT_RECORDED = ["学習時間（秒）", "ステップ当たり時間", "パラメータ数", "FLOPs", "ピークメモリ"]


def sign(x):
    return None if x is None else (x > 0) - (x < 0)


def aux_costs(runs):
    """補助損失の最終エポック値（ヘッド別 CE の平均・SD）と、設定上の損失重み。ヘッドを持たないフェーズは含めない。"""
    out = {}
    for p in PHASES:
        rs = list(runs[p].values())
        heads = {}
        for r in rs:
            for k, v in (r["val_head_ce"][-1] if r.get("val_head_ce") else {}).items():
                heads.setdefault(k, []).append(v)
        cfg = rs[0]["config"] if rs and "config" in rs[0] else {}
        out[p] = {"head_ce_final": {k: {"n": len(v), "mean": float(np.mean(v)),
                                         "sd": float(np.std(v, ddof=1)) if len(v) >= 2 else None}
                                    for k, v in heads.items()},
                  "loss_weights": {k: cfg.get(k) for k in ("hippocampus_loss_weight", "basal_ganglia_loss_weight",
                                                           "cerebellum_loss_weight") if k in cfg} or NO_DATA}
    return out


def test_ppl_summary(runs):
    out = {}
    for p in PHASES:
        v = [r["test_ppl"] for r in runs[p].values() if "test_ppl" in r]
        out[p] = {"n": len(v), "mean": float(np.mean(v)) if v else None,
                  "sd": float(np.std(v, ddof=1)) if len(v) >= 2 else None} if v else NO_DATA
    return out


def analyze_scale(root):
    root = Path(root)
    a = analyze(root)
    a.pop("reduced_scale_agreement", None)
    runs = load_runs(root)
    cfg = next((r["config"] for rs in runs.values() for r in rs.values() if "config" in r), {})
    a["config"] = {k: cfg.get(k) for k in ("dataset_config", "hidden_size", "num_layers", "num_epochs", "max_seq_length",
                                           "batch_size", "learning_rate") if k in cfg}
    a["aux"] = aux_costs(runs)
    a["test_ppl"] = test_ppl_summary(runs)
    a["cost"] = {k: NO_DATA for k in NOT_RECORDED}
    return a


def direction_agreement(red, full):
    out = {}
    for key in red["comparisons"]:
        r, f = red["comparisons"][key], full["comparisons"][key]
        sr, sf = sign(r["mean_diff"]), sign(f["mean_diff"])
        out[key] = {"reduced": {"mean_diff": r["mean_diff"], "rel_pct": r["rel_pct"], "verdict_paired": r["verdict_paired"],
                                "verdict_unpaired": r["verdict_unpaired"]},
                    "full": {"mean_diff": f["mean_diff"], "rel_pct": f["rel_pct"], "verdict_paired": f["verdict_paired"],
                             "verdict_unpaired": f["verdict_unpaired"]},
                    "sign_agree": None if sr is None or sf is None else sr == sf,
                    "verdict_paired_agree": None if UNDECIDABLE in (r["verdict_paired"], f["verdict_paired"])
                    else r["verdict_paired"] == f["verdict_paired"],
                    "verdict_unpaired_agree": None if UNDECIDABLE in (r["verdict_unpaired"], f["verdict_unpaired"])
                    else r["verdict_unpaired"] == f["verdict_unpaired"],
                    "improved_anywhere": any(c["verdict_paired"] == "改善" or c["verdict_unpaired"] == "改善" for c in (r, f))}
    return out


def integrate(reduced_dir, full_dir):
    sc = {"reduced": analyze_scale(reduced_dir), "full": analyze_scale(full_dir)}
    return {"description": "ステップ44.5 スコープ1〜2（既存データのみ）。vanilla Transformer との比較・勾配寄与度は結果 JSON に無く未実施。",
            "scales": sc, "scale_labels": SCALES,
            "direction_agreement": direction_agreement(sc["reduced"], sc["full"]),
            "not_done": ["vanilla Transformer との PPL 比較（Phase A は海馬ヘッド付きのため vanilla ではない）",
                         "モジュール別の勾配寄与度分析", "計算コスト（時間・パラメータ数・FLOPs）の計測"]}


def to_markdown(out):
    L = []
    for sk, label in SCALES.items():
        s = out["scales"][sk]
        L += [f"#### {label}（設定: {json.dumps(s['config'], ensure_ascii=False)}）", "",
              "| フェーズ | n | 平均 val PPL | SD | test PPL 平均 |", "|---|---|---|---|---|"]
        for p in PHASES:
            ph, t = s["phases"][p], s["test_ppl"][p]
            L.append(f"| {p} | {ph['n']} | {fmt(ph['mean'], '{:.2f}')} | {fmt(ph['sd'])} | "
                     f"{NO_DATA if isinstance(t, str) else fmt(t['mean'], '{:.2f}')} |")
        n_ep = max(len(e) for e in s["epochs"].values())
        L += ["", "エポック別 平均 val PPL:", "", "| フェーズ | " + " | ".join(f"E{e + 1}" for e in range(n_ep)) + " |",
              "|---|" + "---|" * n_ep]
        for p in PHASES:
            L.append(f"| {p} | " + " | ".join(f"{e['mean']:.2f}" for e in s["epochs"][p]) + " |")
        L += ["", "| 比較 | 平均差 | 相対 | paired SE | paired 判定 | unpaired SE | unpaired 判定 |", "|---|---|---|---|---|---|---|"]
        for key, c in s["comparisons"].items():
            L.append(f"| {key.replace('_val', '')} | {fmt(c['mean_diff'], '{:+.3f}')} | {fmt(c['rel_pct'], '{:+.2f}%')} | "
                     f"{fmt(c['se_paired'])} | {c['verdict_paired']} | {fmt(c['se_unpaired'])} | {c['verdict_unpaired']} |")
        L += ["", "補助損失（最終エポックの各ヘッド CE 平均、損失重み）:", "", "| フェーズ | ヘッド別 CE | 損失重み |", "|---|---|---|"]
        for p in PHASES:
            a = s["aux"][p]
            heads = ", ".join(f"{k.replace('_loss', '')} {v['mean']:.3f}" for k, v in a["head_ce_final"].items()) or NO_DATA
            w = a["loss_weights"] if isinstance(a["loss_weights"], str) else ", ".join(f"{k.replace('_loss_weight', '')} {v}" for k, v in a["loss_weights"].items())
            L.append(f"| {p} | {heads} | {w} |")
        L += ["", "計算コスト: " + "、".join(f"{k}={v}" for k, v in s["cost"].items()), ""]
    L += ["#### 規模間の方向一致", "",
          "| 比較 | 縮小 平均差（相対） | フル 平均差（相対） | 符号 | paired 判定（縮小/フル） | unpaired 判定（縮小/フル） |",
          "|---|---|---|---|---|---|"]
    for key, g in out["direction_agreement"].items():
        r, f = g["reduced"], g["full"]
        ag = "—" if g["sign_agree"] is None else "一致" if g["sign_agree"] else "不一致"
        L.append(f"| {key.replace('_val', '')} | {r['mean_diff']:+.3f} ({r['rel_pct']:+.2f}%) | {f['mean_diff']:+.3f} ({f['rel_pct']:+.2f}%) | "
                 f"{ag} | {r['verdict_paired']}/{f['verdict_paired']} | {r['verdict_unpaired']}/{f['verdict_unpaired']} |")
    return "\n".join(L)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    red = Path(argv[0] if argv else "results/step44_reduced_scale")
    full = Path(argv[1] if len(argv) > 1 else "results/step44_full_scale_rerun")
    dest = Path(argv[2] if len(argv) > 2 else "results/step44_integrated")
    out = integrate(red, full)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(to_markdown(out))
    return out


if __name__ == "__main__":
    main()
