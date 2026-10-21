"""海馬 CA3 の再帰結合によるパターン補完で、少ない手がかりからの文脈依存の想起を純化する（12.6.183〜184節）。

12.6.179〜180節の時間文脈のタグ（修正1: β = 0.005・3文字組の項目）をそのまま使い、情報源が示したラベルなしの窓から
作った部分的な今の文脈を、保存したタグを記憶したアトラクタの網の反復（k-WTA の抑制つき。context_lm.ca3_complete）
で補完してから、近いタグの痕跡を想起して候補にし、期待情報量で選ぶ（active_replay_ca3、反復 3 回）。
対照は、補完なしの active_replay_tcm（12.6.180節）、本来の文脈のバッファだけから取る診断 active_replay_own、反復の
回数を変えたもの（active_replay_ca3_s1・_s10）。統制は、リプレイしていない保留サンプルで同じく補完・想起する
active_holdout_ca3。

    --phase dev（開発用シード 100〜109）/ eval（評価用シード 0〜9 で一度だけ）
    結果は results/context_lm/context_ca3_{phase}.json、集計と判定は --analyze <結果ファイル>

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_ca3 --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_context_ca3 --analyze results/context_lm/context_ca3_dev.json
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import torch

from ..context_lm import CA3_RULES, TrainConfig, evaluate, make_languages, train_models
from ..run_env import env_info
from .run_context_lm import BUFFER, CUE_NS, DEV_SEEDS, EPISODES, EVAL_SEEDS, FAMILIES, POOL, REGIMES, _mean
from .run_context_tcm import (BAD_FRAC, BAD_RUN, CAPTURE, FIX1_BETA, FIX1_ITEM, NO_HARM, NONE_ABS, OWN_MIN,
                              TCM_ALL_RULES, analyze_tcm)

CA3_ITERS = 3  # 反復の回数（12.6.183節で計測前に固定）
CA3_SPARSITY = 0.025  # 発火する記憶の単位の割合（痕跡の総数の 2.5%。A・C で 200 個、B で 350 個）
CA3_ALL_RULES = TCM_ALL_RULES + CA3_RULES


def h3_ca3(runs: list[dict], g: dict) -> dict:
    """12.6.183節の H3-T1・T2・H3b・H3-P（補完の寄与）・H3-CE と、反復の回数の比較（参考）を計算する。"""
    names = list(g)
    over = [nm for nm in names if g[nm]["分類"] == "上書きあり"]
    none = [nm for nm in names if g[nm]["分類"] == "上書きなし"]
    over_runs = [r for r in runs if f"{r['family']}-{r['regime']}" in over]
    rules = {"ca3": "active_replay_ca3", "tcm": "active_replay_tcm", "own": "active_replay_own",
             "holdout_ca3": "active_holdout_ca3", "ca3_s1": "active_replay_ca3_s1",
             "ca3_s10": "active_replay_ca3_s10", "replay": "active_replay"}
    pur_keys = {"ca3": "replay_ca3", "tcm": "replay_tcm", "holdout_ca3": "holdout_ca3", "ca3_s1": "replay_ca3_s1",
                "ca3_s10": "replay_ca3_s10"}
    out: dict = {"上書きあり": over, "上書きなし": none, "手がかりごと": {}}
    for n in ("1", "2", "4", "8", "16"):
        row = {}
        for nm in names:
            rs = [r for r in runs if f"{r['family']}-{r['regime']}" == nm]
            row[nm] = {f"gain_{a}": g[nm]["acc"][n][rule] - g[nm]["acc"][n]["active"] for a, rule in rules.items()}
            row[nm].update({f"purity_{a}": _mean([r["cues"][n]["tcm_purity"][key] for r in rs])
                            for a, key in pur_keys.items()})
            row[nm].update({"ce_active": g[nm]["ce"][n]["active"], "ce_ca3": g[nm]["ce"][n]["active_replay_ca3"]})
        out["手がかりごと"][n] = row
    if not over:
        for key in ("H3-T1", "H3-T2", "H3-P", "H3-CE"):
            out[key] = "検証できず（上書きありの組がない）"
    else:
        keys = list(out["手がかりごと"]["2"][over[0]])
        m = {n: {key: _mean([out["手がかりごと"][n][nm][key] for nm in over]) for key in keys}
             for n in ("1", "2", "4", "8", "16")}
        bad = {n: sum(r["cues"][n]["acc"]["active_replay_ca3"] - r["cues"][n]["acc"]["active"] < BAD_RUN
                      for r in over_runs) for n in ("8", "16")}
        t1 = all(m[n]["gain_ca3"] >= NO_HARM for n in ("2", "4", "8", "16")) \
            and all(b <= BAD_FRAC * len(over_runs) for b in bad.values())
        out["H3-T1"] = {"gain_ca3 の平均": {n: m[n]["gain_ca3"] for n in ("2", "4", "8", "16")},
                        "−0.02 を下回る回": {n: f"{b}/{len(over_runs)}" for n, b in bad.items()}, "合格": t1}
        ns = ("2", "4", "8")
        avg = {a: _mean([m[n][f"gain_{a}"] for n in ns]) for a in rules}
        if avg["own"] < OWN_MIN:
            t2: object = "検証できず（診断の利得が 0.01 未満）"
        else:
            t2 = avg["ca3"] >= CAPTURE * avg["own"] and avg["holdout_ca3"] <= avg["ca3"] / 2
        out["H3-T2"] = {"gain_ca3（2・4・8 の平均）": avg["ca3"], "gain_own": avg["own"],
                        "取り戻した割合": avg["ca3"] / avg["own"] if avg["own"] else float("nan"),
                        "参考_補完なし（tcm）の割合": avg["tcm"] / avg["own"] if avg["own"] else float("nan"),
                        "gain_holdout_ca3": avg["holdout_ca3"], "合格": t2}
        few = ("2", "4")
        d_gain = _mean([m[n]["gain_ca3"] - m[n]["gain_tcm"] for n in few])
        d_pur = {n: m[n]["purity_ca3"] - m[n]["purity_tcm"] for n in few}
        out["H3-P"] = {"gain_ca3 − gain_tcm（2・4 の平均）": d_gain, "純度の差（ca3 − tcm）": d_pur,
                       "合格": d_gain > 0 and all(x > 0 for x in d_pur.values())}
        out["H3-CE"] = {"CE（active − ca3）": {n: m[n]["ce_active"] - m[n]["ce_ca3"] for n in ("8", "16")},
                        "合格": all(m[n]["ce_active"] - m[n]["ce_ca3"] > 0 for n in ("8", "16"))}
        out["参考_反復の回数"] = {
            n: {a: {"gain": m[n][f"gain_{a}"], "純度": m[n][f"purity_{a}"]} for a in ("tcm", "ca3_s1", "ca3", "ca3_s10")}
            for n in ("1", "2", "4", "8", "16")}
        out["参考_反復の回数_2・4・8 の平均の割合"] = {
            a: avg[a] / avg["own"] if avg["own"] else float("nan") for a in ("tcm", "ca3_s1", "ca3", "ca3_s10")}
    if none:
        m_none = {n: _mean([out["手がかりごと"][n][nm]["gain_ca3"] for nm in none]) for n in ("8", "16")}
        out["H3b"] = {"上書きなしの gain_ca3": m_none, "合格": all(abs(x) <= NONE_ABS for x in m_none.values())}
    else:
        out["H3b"] = "検証できず"
    t1_ok, t2_ok, b_ok = (out[key] if isinstance(out[key], str) else out[key]["合格"] for key in ("H3-T1", "H3-T2", "H3b"))
    out["失敗が解けた"] = t1_ok is True and t2_ok is True and b_ok is True
    p_ok = out["H3-P"] if isinstance(out["H3-P"], str) else out["H3-P"]["合格"]
    out["補完による解決"] = out["失敗が解けた"] and p_ok is True
    return out


def analyze_ca3(runs: list[dict]) -> dict:
    out = analyze_tcm(runs)
    out["判定"]["H3_CA3"] = h3_ca3(runs, out["組"])
    return out


def run_one(args: tuple[str, str, int], steps: int = 1500, episodes: int = EPISODES, cue_ns=CUE_NS,
            iters: int = CA3_ITERS, sparsity: float = CA3_SPARSITY) -> dict:
    fam_name, regime, seed = args
    torch.set_num_threads(1)
    fam, k = FAMILIES[fam_name]
    g = torch.Generator().manual_seed(seed)
    trans, _ = make_languages(fam, g)
    cfg = TrainConfig(regime=regime, steps_per_lang=steps, buffer=BUFFER, k=k, arch="gain", tcm_beta=FIX1_BETA,
                      tcm_item=FIX1_ITEM, ca3_iters=iters, ca3_sparsity=sparsity)
    trained = train_models(fam, trans, cfg, g)
    ev = evaluate(fam, trans, trained, cfg, tuple(cue_ns), g, episodes=episodes, pool=POOL, rules=CA3_ALL_RULES)
    ev.update({"family": fam_name, "regime": regime, "seed": seed, "final_loss": trained["final_loss"]})
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
        print(json.dumps(analyze_ca3(runs)["判定"]["H3_CA3"], ensure_ascii=False, indent=1))
        return
    seeds = args.seeds or (DEV_SEEDS if args.phase == "dev" else EVAL_SEEDS)
    jobs = [(f, reg, s) for f in args.families for reg in REGIMES for s in seeds]
    with Pool(args.workers) as pool:
        runs = pool.map(run_one, jobs)
    out = {"環境": env_info(), "設定": {"arch": "gain", "tcm_beta": FIX1_BETA, "item": f"{FIX1_ITEM}文字組の one-hot",
                                     "ca3_iters": CA3_ITERS, "ca3_sparsity": CA3_SPARSITY,
                                     "families": {k: [v[0].__dict__, v[1]] for k, v in FAMILIES.items()},
                                     "buffer": BUFFER, "episodes": EPISODES, "pool": POOL, "cue_ns": CUE_NS,
                                     "rules": CA3_ALL_RULES, "seeds": list(seeds)},
           "runs": runs}
    out["集計"] = analyze_ca3(runs)
    path = args.out or Path("results/context_lm") / f"context_ca3_{args.phase}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["集計"]["判定"]["H3_CA3"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
