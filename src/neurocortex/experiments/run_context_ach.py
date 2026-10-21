"""CA3 のパターン補完の強さを、想起の不確かさでゲートする（アセチルコリンによる符号化と想起の切り替え。12.6.187〜188節）。

12.6.183〜184節の補完（反復 3 回、発火の割合 2.5%）は、上書きのある組で少ない手がかりの利得を取り戻したが、上書きの
ない組で想起を典型的な痕跡に集めて害を生んだ（H3b 不合格）。ここでは、補完なしで想起した痕跡の保存時の文脈の分布の
正規化エントロピー u（記憶が示す文脈の事後分布の不確かさ。context_lm.recall_uncertainty）で補完の強さ λ を決め、
嗅内皮質からの入力を保ったまま再帰結合の寄与を λ 倍にする（active_replay_ach、λ = u）。対照は、閾値で切り替える
active_replay_ach_hard（λ = 1[u > 0.5]）、λ をエピソードの間で並べ替えた active_replay_ach_shuf、逆向きの
active_replay_ach_inv（λ = 1 − u）、補完なしの
active_replay_tcm、常に補完する active_replay_ca3。統制は保留サンプルで同じくゲートする active_holdout_ach。

    --phase dev（開発用シード 100〜109）/ eval（評価用シード 0〜9 で一度だけ）
    結果は results/context_lm/context_ach_{phase}.json、集計と判定は --analyze <結果ファイル>

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_ach --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_context_ach --analyze results/context_lm/context_ach_dev.json
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import torch

from ..context_lm import ACH_RULES, ACH_THRESHOLD, TrainConfig, evaluate, make_languages, train_models
from ..run_env import env_info
from .run_context_ca3 import CA3_ALL_RULES, CA3_ITERS, CA3_SPARSITY, analyze_ca3
from .run_context_lm import BUFFER, CUE_NS, DEV_SEEDS, EPISODES, EVAL_SEEDS, FAMILIES, POOL, REGIMES, _mean
from .run_context_tcm import BAD_FRAC, BAD_RUN, CAPTURE, FIX1_BETA, FIX1_ITEM, NO_HARM, NONE_ABS, OWN_MIN

ACH_ALL_RULES = CA3_ALL_RULES + ACH_RULES
NS = ("1", "2", "4", "8", "16")


def h4_ach(runs: list[dict], g: dict) -> dict:
    """12.6.187節の H4-T1・T2・H4b・H4-G（ゲートの寄与）・H4-C（補完の寄与が残る）と、操作の確認・参考を計算する。"""
    names = list(g)
    over = [nm for nm in names if g[nm]["分類"] == "上書きあり"]
    none = [nm for nm in names if g[nm]["分類"] == "上書きなし"]
    over_runs = [r for r in runs if f"{r['family']}-{r['regime']}" in over]
    rules = {"ach": "active_replay_ach", "ca3": "active_replay_ca3", "tcm": "active_replay_tcm",
             "own": "active_replay_own", "holdout_ach": "active_holdout_ach", "hard": "active_replay_ach_hard",
             "shuf": "active_replay_ach_shuf", "inv": "active_replay_ach_inv"}
    pur_keys = {"ach": "replay_ach", "ca3": "replay_ca3", "tcm": "replay_tcm", "hard": "replay_ach_hard",
                "shuf": "replay_ach_shuf", "inv": "replay_ach_inv"}
    out: dict = {"上書きあり": over, "上書きなし": none, "手がかりごと": {}}
    for n in NS:
        row = {}
        for nm in names:
            rs = [r for r in runs if f"{r['family']}-{r['regime']}" == nm]
            row[nm] = {f"gain_{a}": g[nm]["acc"][n][rule] - g[nm]["acc"][n]["active"] for a, rule in rules.items()}
            row[nm].update({f"purity_{a}": _mean([r["cues"][n]["tcm_purity"][key] for r in rs])
                            for a, key in pur_keys.items()})
            row[nm]["lambda_ach"] = _mean([r["cues"][n]["ach_lambda"]["replay_ach"] for r in rs])
            row[nm]["lambda_hard"] = _mean([r["cues"][n]["ach_lambda"]["replay_ach_hard"] for r in rs])
        out["手がかりごと"][n] = row

    def avg_over(nms: list[str], n: str, key: str) -> float:
        return _mean([out["手がかりごと"][n][nm][key] for nm in nms])

    if not over:
        for key in ("H4-T1", "H4-T2", "H4-C"):
            out[key] = "検証できず（上書きありの組がない）"
    else:
        bad = {n: sum(r["cues"][n]["acc"]["active_replay_ach"] - r["cues"][n]["acc"]["active"] < BAD_RUN
                      for r in over_runs) for n in ("8", "16")}
        m = {n: avg_over(over, n, "gain_ach") for n in ("2", "4", "8", "16")}
        t1 = all(x >= NO_HARM for x in m.values()) and all(b <= BAD_FRAC * len(over_runs) for b in bad.values())
        out["H4-T1"] = {"gain_ach の平均": m, "−0.02 を下回る回": {n: f"{b}/{len(over_runs)}" for n, b in bad.items()},
                        "合格": t1}
        avg = {a: _mean([avg_over(over, n, f"gain_{a}") for n in ("2", "4", "8")]) for a in rules}
        if avg["own"] < OWN_MIN:
            t2: object = "検証できず（診断の利得が 0.01 未満）"
        else:
            t2 = avg["ach"] >= CAPTURE * avg["own"] and avg["holdout_ach"] <= avg["ach"] / 2
        ratio = {a: avg[a] / avg["own"] if avg["own"] else float("nan") for a in ("tcm", "ca3", "ach", "hard", "shuf", "inv")}
        out["H4-T2"] = {"gain_ach（2・4・8 の平均）": avg["ach"], "gain_own": avg["own"], "取り戻した割合": ratio["ach"],
                        "参考_割合": ratio, "gain_holdout_ach": avg["holdout_ach"], "合格": t2}
        d_c = _mean([avg_over(over, n, "gain_ach") - avg_over(over, n, "gain_tcm") for n in ("2", "4")])
        out["H4-C"] = {"gain_ach − gain_tcm（上書きあり、2・4 の平均）": d_c, "合格": d_c > 0}
    if none:
        m_none = {n: avg_over(none, n, "gain_ach") for n in ("8", "16")}
        out["H4b"] = {"上書きなしの gain_ach": m_none, "参考_ca3": {n: avg_over(none, n, "gain_ca3") for n in ("8", "16")},
                      "合格": all(abs(x) <= NONE_ABS for x in m_none.values())}
        d_g = {n: avg_over(none, n, "gain_ach") - avg_over(none, n, "gain_shuf") for n in ("8", "16")}
        out["H4-G"] = {"gain_ach − gain_shuf（上書きなし）": d_g, "合格": all(x > 0 for x in d_g.values())}
    else:
        out["H4b"] = out["H4-G"] = "検証できず"
    ok = [out[key] if isinstance(out[key], str) else out[key]["合格"] for key in ("H4-T1", "H4-T2", "H4b")]
    out["失敗が解けた"] = all(x is True for x in ok)
    extra = [out[key] if isinstance(out[key], str) else out[key]["合格"] for key in ("H4-C", "H4-G")]
    out["ゲートによる解決"] = out["失敗が解けた"] and all(x is True for x in extra)
    groups = {"上書きあり": over, "上書きなし": none}
    out["操作の確認_λ"] = {grp: {n: avg_over(nms, n, "lambda_ach") for n in NS} for grp, nms in groups.items() if nms}
    out["参考_規則ごと"] = {grp: {n: {a: {"gain": avg_over(nms, n, f"gain_{a}"),
                                          **({"純度": avg_over(nms, n, f"purity_{a}")} if a in pur_keys else {})}
                                      for a in ("tcm", "ca3", "ach", "hard", "shuf", "inv", "own")} for n in NS}
                         for grp, nms in groups.items() if nms}
    out["参考_組ごとの gain（手がかり8個）"] = {nm: {a: out["手がかりごと"]["8"][nm][f"gain_{a}"]
                                                 for a in ("tcm", "ca3", "ach", "hard", "shuf", "inv")} for nm in names}
    return out


def analyze_ach(runs: list[dict]) -> dict:
    out = analyze_ca3(runs)
    out["判定"]["H4_ACh"] = h4_ach(runs, out["組"])
    return out


def run_one(args: tuple[str, str, int], steps: int = 1500, episodes: int = EPISODES, cue_ns=CUE_NS) -> dict:
    fam_name, regime, seed = args
    torch.set_num_threads(1)
    fam, k = FAMILIES[fam_name]
    g = torch.Generator().manual_seed(seed)
    trans, _ = make_languages(fam, g)
    cfg = TrainConfig(regime=regime, steps_per_lang=steps, buffer=BUFFER, k=k, arch="gain", tcm_beta=FIX1_BETA,
                      tcm_item=FIX1_ITEM, ca3_iters=CA3_ITERS, ca3_sparsity=CA3_SPARSITY)
    trained = train_models(fam, trans, cfg, g)
    ev = evaluate(fam, trans, trained, cfg, tuple(cue_ns), g, episodes=episodes, pool=POOL, rules=ACH_ALL_RULES)
    ev.update({"family": fam_name, "regime": regime, "seed": seed, "final_loss": trained["final_loss"]})
    ev.pop("native_per_lang", None)  # 結果ファイルを小さくする（集計には使わない）
    return ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", type=Path, nargs="+", help="結果ファイルを集計して判定する")
    ap.add_argument("--phase", choices=("dev", "eval"), default="dev")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.analyze:
        runs = [r for p in args.analyze for r in json.loads(p.read_text(encoding="utf-8"))["runs"]]
        print(json.dumps(analyze_ach(runs)["判定"]["H4_ACh"], ensure_ascii=False, indent=1))
        return
    seeds = args.seeds or (DEV_SEEDS if args.phase == "dev" else EVAL_SEEDS)
    jobs = [(f, reg, s) for f in args.families for reg in REGIMES for s in seeds]
    with Pool(args.workers) as pool:
        runs = pool.map(run_one, jobs)
    out = {"環境": env_info(), "設定": {"arch": "gain", "tcm_beta": FIX1_BETA, "item": f"{FIX1_ITEM}文字組の one-hot",
                                     "ca3_iters": CA3_ITERS, "ca3_sparsity": CA3_SPARSITY,
                                     "ach": "λ = u（想起の正規化エントロピー）", "ach_threshold": ACH_THRESHOLD,
                                     "families": {k: [v[0].__dict__, v[1]] for k, v in FAMILIES.items()},
                                     "buffer": BUFFER, "episodes": EPISODES, "pool": POOL, "cue_ns": CUE_NS,
                                     "rules": ACH_ALL_RULES, "seeds": list(seeds)},
           "runs": runs}
    out["集計"] = analyze_ach(runs)
    path = args.out or Path("results/context_lm") / f"context_ach_{args.phase}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["集計"]["判定"]["H4_ACh"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
