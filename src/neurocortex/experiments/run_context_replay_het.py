"""ヘテロシナプスの競合による共有の可塑性の配分で、加えた痕跡の固定化と新しい写像の共有の重みの上の競合を変え、
取り戻した一般化と新しいタスクの学習を両立できるか（12.6.237〜238節）。

12.6.163〜170・225〜236節では、リプレイの配分を変えても、カスケード＋捕獲（効果は新しさに依らなかった）でも、
keep と ach_c を結ぶ一般化と保護の交換の曲線を確かには越えなかった。ここではリプレイの配分を変えず、共有の重みの
シナプスごとに、加えた痕跡のリプレイの勾配と新しい入力の勾配が逆向きの所でだけ痕跡側の変化を弱める:
    keep_het       … keep（加えた痕跡を一様にリプレイ）＋競合（γ = 0）
    ach_c_het      … ach_c（連続的な新しさの信号で固定化を配分）＋競合（γ = 0。主）
    ach_c_het_perm … ach_c＋弱める位置を無作為にした対照（数はステップごとに同じ）
base・keep・ach_c は 12.6.167〜168節の結果ファイル（同じシード・同じ実装で再現する）と対にして使う。

    --phase dev（開発用シード 100〜104）/ eval（評価用シード 0〜4 で一度だけ）
    結果は results/context_inference/replay_het_{phase}.json

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_replay_het --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_context_replay_het --analyze <結果ファイル> <12.6.167〜168節の結果ファイル>
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import torch

from ..run_env import env_info
from . import run_context_replay_cascade as cas
from .run_context_episodic_readout import FAMILIES, SELECTS
from .run_context_replay_novelty import SUMMARY_KEYS
from .run_context_replay_split import CONFIGS as SPLIT_CONFIGS
from .run_context_replay_split import _criteria
from .run_context_writeback import NEW_TASK, SELECT

DEV_SEEDS = list(range(100, 105))
EVAL_SEEDS = list(range(0, 5))
# 競合の利得（計測前に決めた。γ = 0: 逆向きのシナプスでは加えた痕跡による変化を止める）
HET = {"het_gain": 0.0}
CONFIGS: dict[str, dict] = {
    "base": SPLIT_CONFIGS["base"],
    "keep": SPLIT_CONFIGS["keep"],
    "ach_c": SPLIT_CONFIGS["ach_c"],
    "keep_het": {**SPLIT_CONFIGS["keep"], **HET},
    "ach_c_het": {**SPLIT_CONFIGS["ach_c"], **HET},
    "ach_c_het_perm": {**SPLIT_CONFIGS["ach_c"], **HET, "het_mode": "perm"},
}
REUSED = ("base", "keep", "ach_c")  # 12.6.167〜168節の結果ファイルから読む
NEW = ("keep_het", "ach_c_het", "ach_c_het_perm")
# 修正（12.6.237節。上限3回）。修正1: 逆向きのシナプスで常に痕跡側を止めると一般化が無作為の位置より大きく下がった
# （開発用 ach_c_het の上書きありの end +0.060、perm +0.084）。逆向きの所では大きさの大きい方の信号が勝ち、小さい方を
# 止める対称な競合（強い誘導が弱い誘導を抑える）にする
CONFIGS.update({f"{c}_wta": {**CONFIGS[c], "het_mode": "wta"} for c in ("keep_het", "ach_c_het")})
FIXES: tuple[str, ...] = ("keep_het_wta", "ach_c_het_wta")
# 確定の規則（修正1の計測の前に決めた）: 開発用シードで D1〜D3 を満たせば H_fix の候補にする。判定の基準は変えない
FIX_CANDIDATES: tuple[str, ...] = ()
FRONTIER = ("keep_het", "ach_c_het", "ach_c_het_perm")  # 交換の曲線からの残差を測る条件
RULES = ("keep", "ach_c", *NEW, *FIXES)


def run_one(job: tuple, table: dict) -> dict:
    return cas.run_job(job, table[job[3]])


def analyze(paths: list[Path], rules=RULES) -> dict:
    """12.6.233節の集計（新しい条件を測ったシードに限る）に、12.6.237節の判定を付ける。"""
    old = (cas.NEW, cas.FIXES)
    cas.NEW, cas.FIXES = NEW, FIXES
    try:
        res = cas.analyze(paths, rules)
    finally:
        cas.NEW, cas.FIXES = old
    res["verdict"] = verdict(res)
    return res


def verdict(res: dict) -> dict:
    """12.6.237節で計測前に固定した判定条件。"""
    o, n, a = res["over_14bit"], res["not_over"], res["all"]
    out: dict = {}
    for w in ("keep", "ach_c", *NEW, *FIXES):
        if o.get(f"end_{w}") is None:
            continue
        v = _criteria(o, n, a, w)
        if w not in ("keep", "ach_c") and o.get("end_ach_c") is not None:
            v["D4"] = bool(o[f"end_{w}"] - o["end_ach_c"] >= 0.02 and a[f"bad_{w}"] <= a["bad_ach_c"])
        r = cas.frontier_residual(o, a, w) if w in FRONTIER or w in FIXES else None
        if r is not None:
            v["residual"] = r
            v["F"] = bool(r >= 0.02)
        out[w] = v
    out["H"] = out.get("ach_c_het", {}).get("success")
    fr = [w for w in ("keep_het", "ach_c_het") if w in out]
    out["F"] = any(out[w].get("F") for w in fr) if fr else None
    # P（競合の位置が効く）: ach_c_het の残差が、弱める数を揃えた無作為の位置の対照より +0.01 以上大きい
    rh, rp = out.get("ach_c_het", {}).get("residual"), out.get("ach_c_het_perm", {}).get("residual")
    out["P"] = bool(rh - rp >= 0.01) if rh is not None and rp is not None else None
    out["H_fix"] = any(out.get(w, {}).get("success") for w in FIX_CANDIDATES) if FIX_CANDIDATES else None
    # F_fix（修正1の計測の前に加えた）: keep_het_wta・ach_c_het_wta のいずれかが交換の曲線より +0.02 以上外側
    fx = [w for w in FIXES if w in out]
    out["F_fix"] = any(out[w].get("F") for w in fx) if fx else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", type=Path, nargs="+", help="結果ファイル（12.6.167〜168節のものも）を集計して判定する")
    ap.add_argument("--configs", nargs="+", default=[*NEW, *FIXES])
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--phase", choices=("dev", "eval"), default="dev")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.analyze:
        res = analyze(args.analyze)
        print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False, indent=1))
        return
    seeds = args.seeds or (EVAL_SEEDS if args.phase == "eval" else DEV_SEEDS)
    out_path = args.out or Path(f"results/context_inference/replay_het_{args.phase}.json")
    all_jobs = [(*cfg, s) for cfg in cas.jobs(tuple(args.configs), tuple(args.families)) for s in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(partial(run_one, table=CONFIGS), all_jobs, chunksize=1)
    out: dict = {"環境": env_info(),
                 "説明": f"ヘテロシナプスの競合による共有の可塑性の配分（12.6.237〜238節）。phase={args.phase}",
                 "seeds": seeds, "select": {SELECT: SELECTS[SELECT]}, "new_task": {k: repr(v) for k, v in NEW_TASK.items()},
                 "configs": {k: CONFIGS[k] for k in args.configs}, "runs": {}}
    for key in dict.fromkeys(x["key"] for x in results):
        rows = [x["row"] for x in results if x["key"] == key]
        summ = {}
        for k in (*SUMMARY_KEYS, "het_frac", "het_conflict"):
            vals = [float(x[k]) for x in rows if x.get(k) is not None]
            if vals:
                t = torch.tensor(vals)
                summ[k] = {"mean": float(t.mean()), "std": float(t.std(unbiased=False)), "n": len(vals)}
        out["runs"][key] = {"summary": summ, "per_seed": rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out_path, len(results), "runs", flush=True)


if __name__ == "__main__":
    main()
