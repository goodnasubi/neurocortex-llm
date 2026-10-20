"""歯状回の成体神経新生による、時間的に近い痕跡の符号の共有（12.6.203〜204節）。

12.6.198節では、固定の歯状回の分離（疎な拡張と k-WTA）で族 B の害は減ったが、疎性を揃えた乱数射影でも同じくらい減り
（H6-R 不合格）、分離に特有の効果とは言えなかった。ここでは、時間とともに入れ替わる若い顆粒細胞の集団に、その時刻に
保存する痕跡を優先して割り当て、若い細胞の入力の結合を保存のときに Hebb 則で学ぶ（context_lm.ng_codes。Aimone, Wiles &
Gage 2006, 2009）。時間的に近い痕跡は同じ若い細胞を共有し、遠い痕跡は別の集団になる。CA3 の補完をその符号の空間で行い
（context_lm.ca3_recall_codes）、上書きのある組で古い痕跡と新しい痕跡を分けられるかを調べる。比べるのは、比較器の
ゲート（active_replay_cmp）、固定の歯状回（active_replay_dgcmp）、疎性を揃えた乱数射影（active_replay_rpcmp）、
神経新生（active_replay_ngcmp、主）、若さの割り当てを時刻と無関係に並べ替えた対照（active_replay_ngshufcmp）。
統制は保留サンプルで同じく想起する active_holdout_ngcmp。参考は比較器なし（ng・ngshuf）と Hebb 則なし（ngnohebbcmp）。

    --phase dev（開発用シード 100〜109）/ eval（評価用シード 0〜9 で一度だけ）
    結果は results/context_lm/context_ng_{phase}.json、集計と判定は --analyze <結果ファイル>

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_ng --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_context_ng --analyze results/context_lm/context_ng_dev.json
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import torch

from ..context_lm import NG_RULES, TrainConfig, evaluate, make_languages, train_models
from ..run_env import env_info
from .run_context_ach import NS
from .run_context_ca3 import CA3_ITERS, CA3_SPARSITY
from .run_context_cmp import T_MIN, _paired
from .run_context_dg import B_HARM, DG_ALL_RULES, DG_CONN, DG_EXPAND, DG_SPARSITY, FAM_B, analyze_dg
from .run_context_lm import BUFFER, CUE_NS, DEV_SEEDS, EPISODES, EVAL_SEEDS, FAMILIES, POOL, REGIMES, _mean
from .run_context_tcm import BAD_FRAC, BAD_RUN, CAPTURE, FIX1_BETA, FIX1_ITEM, NO_HARM, NONE_ABS, OWN_MIN

NG_ALL_RULES = DG_ALL_RULES + NG_RULES
NG_YOUNG, NG_ALLOC, NG_TAU, NG_ETA = 0.05, 0.5, 1.0, 0.01  # 12.6.203節で計測前に固定
JOINT_SAME = 0.05  # H7-P: joint の組で、符号の比 between/within の ng と ngshuf の差の絶対値がこれ以下
MAIN_NS = ("2", "4", "8")


def h7_ng(runs: list[dict], g: dict) -> dict:
    """12.6.203節の H7-R・H7-S（主）、H7-T1・T2・H7b・H7-D・H7-C・H7-B と、操作の確認（H7-P）・参考を計算する。"""
    names = list(g)
    over = [nm for nm in names if g[nm]["分類"] == "上書きあり"]
    none = [nm for nm in names if g[nm]["分類"] == "上書きなし"]
    over_runs = [r for r in runs if f"{r['family']}-{r['regime']}" in over]
    b_runs = [r for r in runs if f"{r['family']}-{r['regime']}" == FAM_B]
    rules = {"ngcmp": "active_replay_ngcmp", "ng": "active_replay_ng", "holdout_ngcmp": "active_holdout_ngcmp",
             "ngshufcmp": "active_replay_ngshufcmp", "ngshuf": "active_replay_ngshuf",
             "ngnohebbcmp": "active_replay_ngnohebbcmp", "dgcmp": "active_replay_dgcmp", "dg": "active_replay_dg",
             "rpcmp": "active_replay_rpcmp", "rp": "active_replay_rp", "cmp": "active_replay_cmp",
             "ca3": "active_replay_ca3", "tcm": "active_replay_tcm", "own": "active_replay_own"}
    pur = [a for a in rules if a not in ("own", "holdout_ngcmp")]
    lam = ("ngcmp", "ngshufcmp", "ngnohebbcmp", "dgcmp", "rpcmp", "cmp")
    out: dict = {"上書きあり": over, "上書きなし": none, "手がかりごと": {}}
    for n in NS:
        row = {}
        for nm in names:
            rs = [r for r in runs if f"{r['family']}-{r['regime']}" == nm]
            row[nm] = {f"gain_{a}": g[nm]["acc"][n][rule] - g[nm]["acc"][n]["active"] for a, rule in rules.items()}
            row[nm].update({f"purity_{a}": _mean([r["cues"][n]["tcm_purity"][f"replay_{a}"] for r in rs])
                            for a in pur})
            row[nm].update({f"lambda_{a}": _mean([r["cues"][n]["ach_lambda"][f"replay_{a}"] for r in rs])
                            for a in lam})
        out["手がかりごと"][n] = row

    def avg_over(nms: list[str], n: str, key: str) -> float:
        return _mean([out["手がかりごと"][n][nm][key] for nm in nms])

    if not over:
        for key in ("H7-R", "H7-S", "H7-T1", "H7-T2", "H7-D", "H7-C"):
            out[key] = "検証できず（上書きありの組がない）"
    else:
        for key, other in (("H7-R", "rpcmp"), ("H7-S", "ngshufcmp"), ("H7-D", "dgcmp"), ("H7-C", "cmp")):
            p = _paired(over_runs, rules["ngcmp"], rules[other], MAIN_NS)
            out[key] = {f"ngcmp − {other}（上書きあり、2・4・8、1回ごと）": p, "合格": p["平均"] > 0 and p["t"] > T_MIN}
        bad = {n: sum(r["cues"][n]["acc"][rules["ngcmp"]] - r["cues"][n]["acc"]["active"] < BAD_RUN
                      for r in over_runs) for n in ("8", "16")}
        m = {n: avg_over(over, n, "gain_ngcmp") for n in ("2", "4", "8", "16")}
        t1 = all(x >= NO_HARM for x in m.values()) and all(b <= BAD_FRAC * len(over_runs) for b in bad.values())
        out["H7-T1"] = {"gain_ngcmp の平均": m, "−0.02 を下回る回": {n: f"{b}/{len(over_runs)}" for n, b in bad.items()},
                        "合格": t1}
        avg = {a: _mean([avg_over(over, n, f"gain_{a}") for n in MAIN_NS]) for a in rules}
        if avg["own"] < OWN_MIN:
            t2: object = "検証できず（診断の利得が 0.01 未満）"
        else:
            t2 = avg["ngcmp"] >= CAPTURE * avg["own"] and avg["holdout_ngcmp"] <= avg["ngcmp"] / 2
        out["H7-T2"] = {"gain_ngcmp（2・4・8 の平均）": avg["ngcmp"], "gain_own": avg["own"],
                        "取り戻した割合": avg["ngcmp"] / avg["own"] if avg["own"] else float("nan"),
                        "参考_割合": {a: avg[a] / avg["own"] if avg["own"] else float("nan")
                                  for a in rules if a not in ("own", "holdout_ngcmp")},
                        "gain_holdout_ngcmp": avg["holdout_ngcmp"], "合格": t2}
        out["参考_比較器なし（上書きあり、2・4・8）"] = {
            "ng − rp": _paired(over_runs, rules["ng"], rules["rp"], MAIN_NS),
            "ng − ngshuf": _paired(over_runs, rules["ng"], rules["ngshuf"], MAIN_NS),
            "ngcmp − ngnohebbcmp": _paired(over_runs, rules["ngcmp"], rules["ngnohebbcmp"], MAIN_NS)}
    if none:
        m_none = {n: avg_over(none, n, "gain_ngcmp") for n in ("8", "16")}
        out["H7b"] = {"上書きなしの gain_ngcmp": m_none,
                      "参考": {a: {n: avg_over(none, n, f"gain_{a}") for n in ("8", "16")}
                             for a in ("tcm", "ca3", "cmp", "dgcmp", "rpcmp", "ng", "ngshufcmp", "ngnohebbcmp")},
                      "合格": all(abs(x) <= NONE_ABS for x in m_none.values())}
    else:
        out["H7b"] = "検証できず"
    if FAM_B in names:
        mb = {n: out["手がかりごと"][n][FAM_B]["gain_ngcmp"] for n in ("8", "16")}
        out["H7-B"] = {"B-joint の gain_ngcmp": mb, "合格": all(x >= B_HARM for x in mb.values())}
    else:
        out["H7-B"] = "検証できず（B-joint がない）"
    # 操作の確認: seq では時刻が言語とそろうので、神経新生の符号は言語どうしを分ける（比が並べ替え・固定の歯状回より
    # 小さい）。joint では時刻が言語と無関係なので、並べ替えと同じ
    ov = {nm: {c: _mean([r["ng_overlap"][c]["ratio"] for r in runs if f"{r['family']}-{r['regime']}" == nm])
               for c in ("ng", "ngshuf", "ngnohebb")} for nm in names}
    for nm in names:
        ov[nm]["dg"] = _mean([r["dg_overlap"]["dg"]["ratio"] for r in runs if f"{r['family']}-{r['regime']}" == nm])
        ov[nm]["raw"] = _mean([r["dg_overlap"]["raw"]["ratio"] for r in runs
                               if f"{r['family']}-{r['regime']}" == nm])
    seq = [nm for nm in names if nm.endswith("-seq")]
    joint = [nm for nm in names if nm.endswith("-joint")]
    p_ok = (all(ov[nm]["ng"] < ov[nm]["ngshuf"] and ov[nm]["ng"] < ov[nm]["dg"] for nm in seq)
            and all(abs(ov[nm]["ng"] - ov[nm]["ngshuf"]) <= JOINT_SAME for nm in joint))
    out["H7-P"] = {"符号の比 between/within（組ごと）": ov, "合格": p_ok}
    keys = ("H7-R", "H7-S", "H7-T1", "H7b")
    ok = [out[key] if isinstance(out[key], str) else out[key]["合格"] for key in keys]
    out["神経新生の時間的な割り当てで古い痕跡と新しい痕跡を分けた"] = all(x is True for x in ok)
    groups = {"上書きあり": over, "上書きなし": none}
    out["操作の確認_λ"] = {grp: {n: {a: avg_over(nms, n, f"lambda_{a}") for a in lam} for n in NS}
                         for grp, nms in groups.items() if nms}
    out["参考_規則ごと"] = {grp: {n: {a: {"gain": avg_over(nms, n, f"gain_{a}"),
                                          **({"純度": avg_over(nms, n, f"purity_{a}")} if a in pur else {})}
                                      for a in rules if a != "holdout_ngcmp"}
                                  for n in NS}
                         for grp, nms in groups.items() if nms}
    show = ("tcm", "ca3", "cmp", "dgcmp", "rpcmp", "ng", "ngcmp", "ngshufcmp", "ngnohebbcmp")
    out["参考_組ごとの gain（手がかり 2・8 個）"] = {
        nm: {n: {a: out["手がかりごと"][n][nm][f"gain_{a}"] for a in show} for n in ("2", "8")} for nm in names}
    out["参考_ngcmp − ngshufcmp（組ごと、2・4・8、1回ごと）"] = {
        nm: _paired([r for r in runs if f"{r['family']}-{r['regime']}" == nm], rules["ngcmp"], rules["ngshufcmp"],
                    MAIN_NS) for nm in names}
    return out


def analyze_ng(runs: list[dict]) -> dict:
    out = analyze_dg(runs)
    out["判定"]["H7_NG"] = h7_ng(runs, out["組"])
    return out


def run_one(args: tuple[str, str, int], steps: int = 1500, episodes: int = EPISODES, cue_ns=CUE_NS) -> dict:
    fam_name, regime, seed = args
    torch.set_num_threads(1)
    fam, k = FAMILIES[fam_name]
    g = torch.Generator().manual_seed(seed)
    trans, _ = make_languages(fam, g)
    cfg = TrainConfig(regime=regime, steps_per_lang=steps, buffer=BUFFER, k=k, arch="gain", tcm_beta=FIX1_BETA,
                      tcm_item=FIX1_ITEM, ca3_iters=CA3_ITERS, ca3_sparsity=CA3_SPARSITY, dg_expand=DG_EXPAND,
                      dg_conn=DG_CONN, dg_sparsity=DG_SPARSITY, ng_young=NG_YOUNG, ng_alloc=NG_ALLOC, ng_tau=NG_TAU,
                      ng_eta=NG_ETA)
    trained = train_models(fam, trans, cfg, g)
    ev = evaluate(fam, trans, trained, cfg, tuple(cue_ns), g, episodes=episodes, pool=POOL, rules=NG_ALL_RULES)
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
        print(json.dumps(analyze_ng(runs)["判定"]["H7_NG"], ensure_ascii=False, indent=1))
        return
    seeds = args.seeds or (DEV_SEEDS if args.phase == "dev" else EVAL_SEEDS)
    jobs = [(f, reg, s) for f in args.families for reg in REGIMES for s in seeds]
    with Pool(args.workers) as pool:
        runs = pool.map(run_one, jobs, chunksize=1)
    out = {"環境": env_info(), "設定": {"arch": "gain", "tcm_beta": FIX1_BETA, "item": f"{FIX1_ITEM}文字組の one-hot",
                                     "ca3_iters": CA3_ITERS, "ca3_sparsity": CA3_SPARSITY,
                                     "cmp": "λ = 1 − δ、δ = 1 − cos(r_S, t̂)/cos(r_1, t̂)",
                                     "dg": {"expand": DG_EXPAND, "conn": DG_CONN, "sparsity": DG_SPARSITY},
                                     "ng": {"young": NG_YOUNG, "alloc": NG_ALLOC, "tau": NG_TAU, "eta": NG_ETA},
                                     "families": {k: [v[0].__dict__, v[1]] for k, v in FAMILIES.items()},
                                     "buffer": BUFFER, "episodes": EPISODES, "pool": POOL, "cue_ns": CUE_NS,
                                     "rules": NG_ALL_RULES, "seeds": list(seeds)},
           "runs": runs}
    out["集計"] = analyze_ng(runs)
    path = args.out or Path("results/context_lm") / f"context_ng_{args.phase}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["集計"]["判定"]["H7_NG"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
