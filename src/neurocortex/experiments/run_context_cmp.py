"""CA3 のパターン補完の強さを、CA1 の比較器の食い違いで決める（12.6.191〜192節）。

12.6.187〜188節の不確かさのゲート（λ = u）は上書きのない組の害を除いたが、λ をエピソードの間で並べ替えても同じくらい
よく（H4-G 不合格）、エピソードごとのゲートの寄与は示せなかった。ここでは、CA3 が想起した文脈と嗅内皮質から直接入る
入力（今の文脈）の食い違い δ（context_lm.ca1_mismatch）が大きいほどアセチルコリンを上げて補完を弱める
（active_replay_cmp、λ = 1 − δ。Hasselmo, Wyble & Wallenstein 1996）。対照は λ をエピソードの間で並べ替えた
active_replay_cmp_shuf、補完なしの active_replay_tcm、常に補完する active_replay_ca3、不確かさのゲート
active_replay_ach（λ = u）。統制は保留サンプルで同じくゲートする active_holdout_cmp。参考は λ = u·(1 − δ) の
active_replay_cmp_u。

    --phase dev（開発用シード 100〜109）/ eval（評価用シード 0〜9 で一度だけ）
    結果は results/context_lm/context_cmp_{phase}.json、集計と判定は --analyze <結果ファイル>

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_cmp --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_context_cmp --analyze results/context_lm/context_cmp_dev.json
"""

from __future__ import annotations

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import torch

from ..context_lm import CMP_RULES, TrainConfig, evaluate, make_languages, train_models
from ..run_env import env_info
from .run_context_ach import ACH_ALL_RULES, NS, analyze_ach
from .run_context_ca3 import CA3_ITERS, CA3_SPARSITY
from .run_context_lm import BUFFER, CUE_NS, DEV_SEEDS, EPISODES, EVAL_SEEDS, FAMILIES, POOL, REGIMES, _mean
from .run_context_tcm import BAD_FRAC, BAD_RUN, CAPTURE, FIX1_BETA, FIX1_ITEM, NO_HARM, NONE_ABS, OWN_MIN

CMP_ALL_RULES = ACH_ALL_RULES + CMP_RULES
T_MIN = 2.0  # H5-G: 1回ごとの差の平均の t 値（平均 / 標準誤差）がこれより大きい


def _t(xs: list[float]) -> float:
    """平均の t 値（平均 / 標準誤差）。ばらつきが 0 なら平均の符号で ±∞。"""
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m / (sd / math.sqrt(n)) if sd > 0 else math.copysign(math.inf, m) if m else 0.0


def _paired(runs: list[dict], a: str, b: str, ns: tuple[str, ...]) -> dict:
    """1回（組 × シード）ごとに、手がかり ns の平均の acc(a) − acc(b) を取り、平均と t 値を返す。"""
    d = [_mean([r["cues"][n]["acc"][a] - r["cues"][n]["acc"][b] for n in ns]) for r in runs]
    return {"平均": _mean(d), "t": _t(d), "回数": len(d)}


def h5_cmp(runs: list[dict], g: dict) -> dict:
    """12.6.191節の H5-T1・T2・H5b・H5-C・H5-G・H5-Gover と、操作の確認・参考を計算する。"""
    names = list(g)
    over = [nm for nm in names if g[nm]["分類"] == "上書きあり"]
    none = [nm for nm in names if g[nm]["分類"] == "上書きなし"]
    over_runs = [r for r in runs if f"{r['family']}-{r['regime']}" in over]
    none_runs = [r for r in runs if f"{r['family']}-{r['regime']}" in none]
    rules = {"cmp": "active_replay_cmp", "cmp_shuf": "active_replay_cmp_shuf", "cmp_u": "active_replay_cmp_u",
             "holdout_cmp": "active_holdout_cmp", "ach": "active_replay_ach", "ach_shuf": "active_replay_ach_shuf",
             "ca3": "active_replay_ca3", "tcm": "active_replay_tcm", "own": "active_replay_own"}
    pur_keys = {"cmp": "replay_cmp", "cmp_shuf": "replay_cmp_shuf", "cmp_u": "replay_cmp_u", "ach": "replay_ach",
                "ca3": "replay_ca3", "tcm": "replay_tcm"}
    out: dict = {"上書きあり": over, "上書きなし": none, "手がかりごと": {}}
    for n in NS:
        row = {}
        for nm in names:
            rs = [r for r in runs if f"{r['family']}-{r['regime']}" == nm]
            row[nm] = {f"gain_{a}": g[nm]["acc"][n][rule] - g[nm]["acc"][n]["active"] for a, rule in rules.items()}
            row[nm].update({f"purity_{a}": _mean([r["cues"][n]["tcm_purity"][key] for r in rs])
                            for a, key in pur_keys.items()})
            row[nm]["lambda_cmp"] = _mean([r["cues"][n]["ach_lambda"]["replay_cmp"] for r in rs])
            row[nm]["lambda_ach"] = _mean([r["cues"][n]["ach_lambda"]["replay_ach"] for r in rs])
            row[nm]["lambda_cmp_u"] = _mean([r["cues"][n]["ach_lambda"]["replay_cmp_u"] for r in rs])
        out["手がかりごと"][n] = row

    def avg_over(nms: list[str], n: str, key: str) -> float:
        return _mean([out["手がかりごと"][n][nm][key] for nm in nms])

    if not over:
        for key in ("H5-T1", "H5-T2", "H5-C", "H5-Gover"):
            out[key] = "検証できず（上書きありの組がない）"
    else:
        bad = {n: sum(r["cues"][n]["acc"]["active_replay_cmp"] - r["cues"][n]["acc"]["active"] < BAD_RUN
                      for r in over_runs) for n in ("8", "16")}
        m = {n: avg_over(over, n, "gain_cmp") for n in ("2", "4", "8", "16")}
        t1 = all(x >= NO_HARM for x in m.values()) and all(b <= BAD_FRAC * len(over_runs) for b in bad.values())
        out["H5-T1"] = {"gain_cmp の平均": m, "−0.02 を下回る回": {n: f"{b}/{len(over_runs)}" for n, b in bad.items()},
                        "合格": t1}
        avg = {a: _mean([avg_over(over, n, f"gain_{a}") for n in ("2", "4", "8")]) for a in rules}
        if avg["own"] < OWN_MIN:
            t2: object = "検証できず（診断の利得が 0.01 未満）"
        else:
            t2 = avg["cmp"] >= CAPTURE * avg["own"] and avg["holdout_cmp"] <= avg["cmp"] / 2
        ratio = {a: avg[a] / avg["own"] if avg["own"] else float("nan")
                 for a in ("tcm", "ca3", "ach", "ach_shuf", "cmp", "cmp_shuf", "cmp_u")}
        out["H5-T2"] = {"gain_cmp（2・4・8 の平均）": avg["cmp"], "gain_own": avg["own"], "取り戻した割合": ratio["cmp"],
                        "参考_割合": ratio, "gain_holdout_cmp": avg["holdout_cmp"], "合格": t2}
        d_c = _mean([avg_over(over, n, "gain_cmp") - avg_over(over, n, "gain_tcm") for n in ("2", "4")])
        out["H5-C"] = {"gain_cmp − gain_tcm（上書きあり、2・4 の平均）": d_c, "合格": d_c > 0}
        p_over = _paired(over_runs, "active_replay_cmp", "active_replay_cmp_shuf", ("2", "4", "8"))
        out["H5-Gover"] = {"cmp − cmp_shuf（上書きあり、2・4・8、1回ごと）": p_over, "合格": p_over["平均"] > 0,
                           "参考_ach − ach_shuf": _paired(over_runs, "active_replay_ach", "active_replay_ach_shuf",
                                                          ("2", "4", "8"))}
    if none:
        m_none = {n: avg_over(none, n, "gain_cmp") for n in ("8", "16")}
        out["H5b"] = {"上書きなしの gain_cmp": m_none,
                      "参考_ca3": {n: avg_over(none, n, "gain_ca3") for n in ("8", "16")},
                      "参考_ach": {n: avg_over(none, n, "gain_ach") for n in ("8", "16")},
                      "合格": all(abs(x) <= NONE_ABS for x in m_none.values())}
        p_none = _paired(none_runs, "active_replay_cmp", "active_replay_cmp_shuf", ("8", "16"))
        d_g = {n: avg_over(none, n, "gain_cmp") - avg_over(none, n, "gain_cmp_shuf") for n in ("8", "16")}
        out["H5-G"] = {"cmp − cmp_shuf（上書きなし、8・16、1回ごと）": p_none, "手がかりごとの差": d_g,
                       "合格": p_none["平均"] > 0 and p_none["t"] > T_MIN,
                       "参考_ach − ach_shuf": _paired(none_runs, "active_replay_ach", "active_replay_ach_shuf",
                                                      ("8", "16"))}
    else:
        out["H5b"] = out["H5-G"] = "検証できず"
    ok = [out[key] if isinstance(out[key], str) else out[key]["合格"] for key in ("H5-T1", "H5-T2", "H5b")]
    out["失敗が解けた"] = all(x is True for x in ok)
    extra = [out[key] if isinstance(out[key], str) else out[key]["合格"] for key in ("H5-C", "H5-G", "H5-Gover")]
    out["比較器のゲートによる解決"] = out["失敗が解けた"] and all(x is True for x in extra)
    groups = {"上書きあり": over, "上書きなし": none}
    out["操作の確認_λ"] = {grp: {n: {a: avg_over(nms, n, f"lambda_{a}") for a in ("cmp", "ach", "cmp_u")} for n in NS}
                         for grp, nms in groups.items() if nms}
    out["参考_規則ごと"] = {grp: {n: {a: {"gain": avg_over(nms, n, f"gain_{a}"),
                                          **({"純度": avg_over(nms, n, f"purity_{a}")} if a in pur_keys else {})}
                                      for a in ("tcm", "ca3", "ach", "ach_shuf", "cmp", "cmp_shuf", "cmp_u", "own")}
                                  for n in NS}
                         for grp, nms in groups.items() if nms}
    out["参考_cmp_u − ach"] = {"上書きあり（2・4・8）": _paired(over_runs, "active_replay_cmp_u", "active_replay_ach",
                                                             ("2", "4", "8")) if over_runs else None,
                              "上書きなし（8・16）": _paired(none_runs, "active_replay_cmp_u", "active_replay_ach",
                                                          ("8", "16")) if none_runs else None}
    out["参考_組ごとの gain（手がかり8個）"] = {nm: {a: out["手がかりごと"]["8"][nm][f"gain_{a}"]
                                                 for a in ("tcm", "ca3", "ach", "cmp", "cmp_shuf", "cmp_u")}
                                             for nm in names}
    fam_b = [nm for nm in names if nm.startswith("B-")]
    out["族Bの害"] = {nm: {n: {a: out["手がかりごと"][n][nm][f"gain_{a}"] for a in ("ca3", "ach", "cmp", "cmp_shuf")}
                           for n in ("8", "16")} for nm in fam_b}
    return out


def analyze_cmp(runs: list[dict]) -> dict:
    out = analyze_ach(runs)
    out["判定"]["H5_CMP"] = h5_cmp(runs, out["組"])
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
    ev = evaluate(fam, trans, trained, cfg, tuple(cue_ns), g, episodes=episodes, pool=POOL, rules=CMP_ALL_RULES)
    ev.update({"family": fam_name, "regime": regime, "seed": seed, "final_loss": trained["final_loss"]})
    ev.pop("native_per_lang", None)  # 結果ファイルを小さくする（集計には使わない）
    for res in ev["cues"].values():
        res.pop("acc_per_lang", None)  # 同上
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
        print(json.dumps(analyze_cmp(runs)["判定"]["H5_CMP"], ensure_ascii=False, indent=1))
        return
    seeds = args.seeds or (DEV_SEEDS if args.phase == "dev" else EVAL_SEEDS)
    jobs = [(f, reg, s) for f in args.families for reg in REGIMES for s in seeds]
    with Pool(args.workers) as pool:
        runs = pool.map(run_one, jobs)
    out = {"環境": env_info(), "設定": {"arch": "gain", "tcm_beta": FIX1_BETA, "item": f"{FIX1_ITEM}文字組の one-hot",
                                     "ca3_iters": CA3_ITERS, "ca3_sparsity": CA3_SPARSITY,
                                     "cmp": "λ = 1 − δ、δ = 1 − cos(s_S, t̂)/cos(s_1, t̂)",
                                     "families": {k: [v[0].__dict__, v[1]] for k, v in FAMILIES.items()},
                                     "buffer": BUFFER, "episodes": EPISODES, "pool": POOL, "cue_ns": CUE_NS,
                                     "rules": CMP_ALL_RULES, "seeds": list(seeds)},
           "runs": runs}
    out["集計"] = analyze_cmp(runs)
    path = args.out or Path("results/context_lm") / f"context_cmp_{args.phase}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["集計"]["判定"]["H5_CMP"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
