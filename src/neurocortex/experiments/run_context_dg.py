"""歯状回のパターン分離で CA3 の補完の害（族 B）を除けるか（12.6.197〜198節）。

12.6.192節の比較器のゲート（λ = 1 − δ）は上書きのある組で診断の利得の 93% を取り戻したが、7言語が重なる族 B の joint
では、どのゲートでも補完の害（手がかり8個で約 −0.014）が残った。ここでは、痕跡を保存する前に歯状回で疎な高次元の展開と
競合（k-WTA）を通して符号化し（context_lm.dg_codes）、CA3 の補完をその符号の空間で行う（context_lm.ca3_recall_codes。
手がかりは貫通路から生のタグで入る）。比べるのは、比較器のゲート（active_replay_cmp）、歯状回の分離のみ
（active_replay_dg、λ = 1）、歯状回の分離＋比較器のゲート（active_replay_dgcmp、主）、疎性を合わせた乱数射影
（active_replay_rp・active_replay_rpcmp。競合なしで幾何を保つ対照）、補完なし（active_replay_tcm）。統制は保留サンプルで
同じく想起する active_holdout_dgcmp。

    --phase dev（開発用シード 100〜109）/ eval（評価用シード 0〜9 で一度だけ）
    結果は results/context_lm/context_dg_{phase}.json、集計と判定は --analyze <結果ファイル>

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_dg --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_context_dg --analyze results/context_lm/context_dg_dev.json
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import torch

from ..context_lm import DG_RULES, TrainConfig, evaluate, make_languages, train_models
from ..run_env import env_info
from .run_context_ach import NS
from .run_context_ca3 import CA3_ITERS, CA3_SPARSITY
from .run_context_cmp import CMP_ALL_RULES, T_MIN, _paired, analyze_cmp
from .run_context_lm import BUFFER, CUE_NS, DEV_SEEDS, EPISODES, EVAL_SEEDS, FAMILIES, POOL, REGIMES, _mean
from .run_context_tcm import BAD_FRAC, BAD_RUN, CAPTURE, FIX1_BETA, FIX1_ITEM, NO_HARM, NONE_ABS, OWN_MIN

DG_ALL_RULES = CMP_ALL_RULES + DG_RULES
DG_EXPAND, DG_CONN, DG_SPARSITY = 4, 0.05, 0.05  # 12.6.197節で計測前に固定
KEEP = 0.8  # H6-K: gain_dgcmp が gain_cmp（12.6.191節の規則、同じシード）の 80% 以上
B_HARM = -NONE_ABS  # H6-B: 族 B の joint で gain_dgcmp がこれ以上（手がかり 8・16 個のそれぞれ）
FAM_B = "B-joint"


def h6_dg(runs: list[dict], g: dict) -> dict:
    """12.6.197節の H6-T1・T2・H6-K・H6b・H6-B・H6-R と、操作の確認（H6-P）・参考を計算する。"""
    names = list(g)
    over = [nm for nm in names if g[nm]["分類"] == "上書きあり"]
    none = [nm for nm in names if g[nm]["分類"] == "上書きなし"]
    over_runs = [r for r in runs if f"{r['family']}-{r['regime']}" in over]
    b_runs = [r for r in runs if f"{r['family']}-{r['regime']}" == FAM_B]
    rules = {"dgcmp": "active_replay_dgcmp", "dg": "active_replay_dg", "holdout_dgcmp": "active_holdout_dgcmp",
             "rp": "active_replay_rp", "rpcmp": "active_replay_rpcmp", "cmp": "active_replay_cmp",
             "ca3": "active_replay_ca3", "tcm": "active_replay_tcm", "own": "active_replay_own"}
    pur_keys = {"dgcmp": "replay_dgcmp", "dg": "replay_dg", "rp": "replay_rp", "rpcmp": "replay_rpcmp",
                "cmp": "replay_cmp", "ca3": "replay_ca3", "tcm": "replay_tcm"}
    out: dict = {"上書きあり": over, "上書きなし": none, "手がかりごと": {}}
    for n in NS:
        row = {}
        for nm in names:
            rs = [r for r in runs if f"{r['family']}-{r['regime']}" == nm]
            row[nm] = {f"gain_{a}": g[nm]["acc"][n][rule] - g[nm]["acc"][n]["active"] for a, rule in rules.items()}
            row[nm].update({f"purity_{a}": _mean([r["cues"][n]["tcm_purity"][key] for r in rs])
                            for a, key in pur_keys.items()})
            for a in ("dgcmp", "rpcmp", "cmp"):
                row[nm][f"lambda_{a}"] = _mean([r["cues"][n]["ach_lambda"][f"replay_{a}"] for r in rs])
        out["手がかりごと"][n] = row

    def avg_over(nms: list[str], n: str, key: str) -> float:
        return _mean([out["手がかりごと"][n][nm][key] for nm in nms])

    if not over:
        for key in ("H6-T1", "H6-T2", "H6-K"):
            out[key] = "検証できず（上書きありの組がない）"
    else:
        bad = {n: sum(r["cues"][n]["acc"]["active_replay_dgcmp"] - r["cues"][n]["acc"]["active"] < BAD_RUN
                      for r in over_runs) for n in ("8", "16")}
        m = {n: avg_over(over, n, "gain_dgcmp") for n in ("2", "4", "8", "16")}
        t1 = all(x >= NO_HARM for x in m.values()) and all(b <= BAD_FRAC * len(over_runs) for b in bad.values())
        out["H6-T1"] = {"gain_dgcmp の平均": m, "−0.02 を下回る回": {n: f"{b}/{len(over_runs)}" for n, b in bad.items()},
                        "合格": t1}
        avg = {a: _mean([avg_over(over, n, f"gain_{a}") for n in ("2", "4", "8")]) for a in rules}
        if avg["own"] < OWN_MIN:
            t2: object = "検証できず（診断の利得が 0.01 未満）"
        else:
            t2 = avg["dgcmp"] >= CAPTURE * avg["own"] and avg["holdout_dgcmp"] <= avg["dgcmp"] / 2
        ratio = {a: avg[a] / avg["own"] if avg["own"] else float("nan")
                 for a in ("tcm", "ca3", "cmp", "dg", "dgcmp", "rp", "rpcmp")}
        out["H6-T2"] = {"gain_dgcmp（2・4・8 の平均）": avg["dgcmp"], "gain_own": avg["own"],
                        "取り戻した割合": ratio["dgcmp"], "参考_割合": ratio,
                        "gain_holdout_dgcmp": avg["holdout_dgcmp"], "合格": t2}
        if avg["cmp"] <= 0:
            k: object = "検証できず（比較器の利得が正でない）"
        else:
            k = avg["dgcmp"] >= KEEP * avg["cmp"]
        out["H6-K"] = {"gain_dgcmp": avg["dgcmp"], "gain_cmp": avg["cmp"],
                       "比": avg["dgcmp"] / avg["cmp"] if avg["cmp"] else float("nan"), "合格": k,
                       "参考_dgcmp − cmp（上書きあり、2・4・8、1回ごと）":
                           _paired(over_runs, "active_replay_dgcmp", "active_replay_cmp", ("2", "4", "8"))}
    if none:
        m_none = {n: avg_over(none, n, "gain_dgcmp") for n in ("8", "16")}
        out["H6b"] = {"上書きなしの gain_dgcmp": m_none,
                      "参考": {a: {n: avg_over(none, n, f"gain_{a}") for n in ("8", "16")}
                             for a in ("tcm", "ca3", "cmp", "dg", "rp", "rpcmp")},
                      "合格": all(abs(x) <= NONE_ABS for x in m_none.values())}
    else:
        out["H6b"] = "検証できず"
    if FAM_B in names:
        mb = {n: out["手がかりごと"][n][FAM_B]["gain_dgcmp"] for n in ("8", "16")}
        out["H6-B"] = {"B-joint の gain_dgcmp": mb, "合格": all(x >= B_HARM for x in mb.values()),
                       "参考_dgcmp − cmp（B-joint、8・16、1回ごと）":
                           _paired(b_runs, "active_replay_dgcmp", "active_replay_cmp", ("8", "16"))}
        p_r = _paired(b_runs, "active_replay_dgcmp", "active_replay_rpcmp", ("8", "16"))
        out["H6-R"] = {"dgcmp − rpcmp（B-joint、8・16、1回ごと）": p_r, "合格": p_r["平均"] > 0 and p_r["t"] > T_MIN,
                       "参考_dg − rp（B-joint、8・16）": _paired(b_runs, "active_replay_dg", "active_replay_rp",
                                                               ("8", "16")),
                       "参考_dgcmp − rpcmp（上書きあり、2・4・8）":
                           _paired(over_runs, "active_replay_dgcmp", "active_replay_rpcmp", ("2", "4", "8"))
                           if over_runs else None}
        ov = {c: {q: _mean([r["dg_overlap"][c][q] for r in b_runs]) for q in ("within", "between", "ratio")}
              for c in ("raw", "dg", "rp")}
        out["H6-P"] = {"B-joint の符号の重なり": ov,
                       "合格": ov["dg"]["ratio"] < ov["raw"]["ratio"] and ov["dg"]["ratio"] < ov["rp"]["ratio"]}
    else:
        out["H6-B"] = out["H6-R"] = out["H6-P"] = "検証できず（B-joint がない）"
    keys = ("H6-T1", "H6-T2", "H6-K", "H6b", "H6-B")
    ok = [out[key] if isinstance(out[key], str) else out[key]["合格"] for key in keys]
    out["族Bの害を除き利得を保った"] = all(x is True for x in ok)
    r_ok = out["H6-R"] if isinstance(out["H6-R"], str) else out["H6-R"]["合格"]
    out["分離による解決"] = out["族Bの害を除き利得を保った"] and r_ok is True
    groups = {"上書きあり": over, "上書きなし": none}
    out["操作の確認_λ"] = {grp: {n: {a: avg_over(nms, n, f"lambda_{a}") for a in ("dgcmp", "rpcmp", "cmp")}
                                for n in NS} for grp, nms in groups.items() if nms}
    out["参考_規則ごと"] = {grp: {n: {a: {"gain": avg_over(nms, n, f"gain_{a}"),
                                          **({"純度": avg_over(nms, n, f"purity_{a}")} if a in pur_keys else {})}
                                      for a in rules if a != "holdout_dgcmp"}
                                  for n in NS}
                         for grp, nms in groups.items() if nms}
    out["参考_組ごとの gain（手がかり8個）"] = {nm: {a: out["手がかりごと"]["8"][nm][f"gain_{a}"]
                                                 for a in ("tcm", "ca3", "cmp", "dg", "dgcmp", "rp", "rpcmp")}
                                             for nm in names}
    out["参考_族B"] = {nm: {n: {a: out["手がかりごと"][n][nm][f"gain_{a}"]
                               for a in ("tcm", "ca3", "cmp", "dg", "dgcmp", "rp", "rpcmp")}
                           for n in ("8", "16")} for nm in names if nm.startswith("B-")}
    out["参考_符号の重なり（組ごと）"] = {nm: {c: _mean([r["dg_overlap"][c]["ratio"] for r in runs
                                                 if f"{r['family']}-{r['regime']}" == nm])
                                         for c in ("raw", "dg", "rp")} for nm in names}
    return out


def analyze_dg(runs: list[dict]) -> dict:
    out = analyze_cmp(runs)
    out["判定"]["H6_DG"] = h6_dg(runs, out["組"])
    return out


def run_one(args: tuple[str, str, int], steps: int = 1500, episodes: int = EPISODES, cue_ns=CUE_NS) -> dict:
    fam_name, regime, seed = args
    torch.set_num_threads(1)
    fam, k = FAMILIES[fam_name]
    g = torch.Generator().manual_seed(seed)
    trans, _ = make_languages(fam, g)
    cfg = TrainConfig(regime=regime, steps_per_lang=steps, buffer=BUFFER, k=k, arch="gain", tcm_beta=FIX1_BETA,
                      tcm_item=FIX1_ITEM, ca3_iters=CA3_ITERS, ca3_sparsity=CA3_SPARSITY, dg_expand=DG_EXPAND,
                      dg_conn=DG_CONN, dg_sparsity=DG_SPARSITY)
    trained = train_models(fam, trans, cfg, g)
    ev = evaluate(fam, trans, trained, cfg, tuple(cue_ns), g, episodes=episodes, pool=POOL, rules=DG_ALL_RULES)
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
        print(json.dumps(analyze_dg(runs)["判定"]["H6_DG"], ensure_ascii=False, indent=1))
        return
    seeds = args.seeds or (DEV_SEEDS if args.phase == "dev" else EVAL_SEEDS)
    jobs = [(f, reg, s) for f in args.families for reg in REGIMES for s in seeds]
    with Pool(args.workers) as pool:
        runs = pool.map(run_one, jobs)
    out = {"環境": env_info(), "設定": {"arch": "gain", "tcm_beta": FIX1_BETA, "item": f"{FIX1_ITEM}文字組の one-hot",
                                     "ca3_iters": CA3_ITERS, "ca3_sparsity": CA3_SPARSITY,
                                     "cmp": "λ = 1 − δ、δ = 1 − cos(r_S, t̂)/cos(r_1, t̂)",
                                     "dg": {"expand": DG_EXPAND, "conn": DG_CONN, "sparsity": DG_SPARSITY},
                                     "families": {k: [v[0].__dict__, v[1]] for k, v in FAMILIES.items()},
                                     "buffer": BUFFER, "episodes": EPISODES, "pool": POOL, "cue_ns": CUE_NS,
                                     "rules": DG_ALL_RULES, "seeds": list(seeds)},
           "runs": runs}
    out["集計"] = analyze_dg(runs)
    path = args.out or Path("results/context_lm") / f"context_dg_{args.phase}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["集計"]["判定"]["H6_DG"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
