"""一般化と保護の交換の曲線は共有の層の容量で決まるか — 共有の隠れ層の幅を 32 → 64 → 128 にして、keep と ach_c の
点（交換の曲線）そのものが外へ動くかを測る（12.6.241〜242節）。

12.6.163〜238節では、リプレイの配分・可塑性の時間構造・シナプスごとの競合のどれも、隠れ層 32 の共有の重みの上では
keep と ach_c を結ぶ交換の曲線を確かには越えなかった。ここでは規則を変えず、共有の層の幅だけを変える:
    base_h{W}・keep_h{W}・ach_c_h{W} … 12.6.167〜168節の base・keep・ach_c と同じ設定で、隠れ層の幅を W にする
幅 32 の base・keep・ach_c は 12.6.167〜168節の結果ファイル（同じシード・同じ実装で再現する）から読む。

集計は二つの枠で行う:
    in  … 同じ幅の base との差（組の分類も同じ幅の base の gap）。規則の効果の交換の曲線
    abs … 幅 32 の base との差（組の分類も幅 32 の base）。幅を広げたこと自体の効果を含む

    --phase dev（開発用シード 100〜104）/ eval（評価用シード 0〜4 で一度だけ）
    結果は results/context_inference/capacity_curve_{phase}.json

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_capacity_curve --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_context_capacity_curve --analyze <結果ファイル> <12.6.167〜168節の結果ファイル>
"""

from __future__ import annotations

import argparse
import json
import tempfile
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
from .run_context_replay_split import analyze as _split_analyze
from .run_context_writeback import NEW_TASK, SELECT

DEV_SEEDS = list(range(100, 105))
EVAL_SEEDS = list(range(0, 5))
BASE_W = 32  # 12.6.167〜170節の共有の隠れ層の幅
WIDTHS = (64, 128)  # 計測前に決めた（2倍・4倍）
RULES0 = ("base", "keep", "ach_c")
CONFIGS: dict[str, dict] = {r: SPLIT_CONFIGS[r] for r in RULES0}
CONFIGS.update({f"{r}_h{w}": {**SPLIT_CONFIGS[r], "hidden": w} for w in WIDTHS for r in RULES0})
NEW = tuple(f"{r}_h{w}" for w in WIDTHS for r in RULES0)
# 修正（上限3回）。規則の調整はしない計測なので、修正は実装の誤りに限る（計測前に決めた）
FIXES: tuple[str, ...] = ()


def run_one(job: tuple, table: dict) -> dict:
    return cas.run_job(job, table[job[3]])


def _frame(runs: dict, seeds: set[int], width: int, absolute: bool) -> dict:
    """幅 width の base・keep・ach_c を、規則の名前 keep_h{W}・ach_c_h{W} で、対にする base（in は同じ幅、abs は幅 32）
    と一緒に 12.6.169節の集計にかける。"""
    sfx = "" if width == BASE_W else f"_h{width}"
    base_cfg = "base" if absolute else f"base{sfx}"
    sel: dict = {}
    for k, v in runs.items():
        cb, cfg = k.rsplit("/", 1)
        if cfg == base_cfg:
            name = "base"
        elif cfg in (f"keep{sfx}", f"ach_c{sfx}"):
            name = cfg
        else:
            continue
        sel[f"{cb}/{name}"] = {"per_seed": [x for x in v["per_seed"] if x["seed"] in seeds]}
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "runs.json"
        f.write_text(json.dumps({"runs": sel}), encoding="utf-8")
        res = _split_analyze([f], (f"keep{sfx}", f"ach_c{sfx}"))
    res.pop("verdict", None)
    return res


def _point(res: dict, w: str) -> dict:
    o, n, a = res["over_14bit"], res["not_over"], res["all"]
    return {"end": o.get(f"end_{w}"), "intact": n.get(f"end_{w}"), "new": a.get(f"new_{w}"),
            "min_new": a.get(f"min_new_{w}"), "argmin_new": a.get(f"argmin_new_{w}"), "bad": a.get(f"bad_{w}"),
            "bad_frac": a.get(f"bad_frac_{w}"), "q05": a.get(f"q05_new_{w}"), "n_over": o.get("n"),
            "n_intact": n.get("n"), "crit": _criteria(o, n, a, w)}


def residual(line: tuple[dict, dict], p: dict) -> float | None:
    """幅 32 の keep と ach_c を結ぶ直線（横軸 = 30組の new の平均、縦軸 = 上書きありの組の end）からの点 p の end の残差
    （12.6.233節の frontier_residual と同じ。正なら曲線の外側）。"""
    k, c = line
    if None in (k["end"], c["end"], k["new"], c["new"], p["end"], p["new"]) or abs(k["new"] - c["new"]) < 1e-9:
        return None
    return p["end"] - (c["end"] + (k["end"] - c["end"]) * (p["new"] - c["new"]) / (k["new"] - c["new"]))


def analyze(paths: list[Path]) -> dict:
    """幅ごと・枠ごとの keep と ach_c の点と、幅 32 の曲線からの残差に、12.6.241節の判定を付ける。
    幅を広げた条件を測ったシードに限る（10シードの結果ファイルと対にしても、ここで測ったシードだけを使う）。"""
    runs: dict = {}
    seeds: set[int] = set()
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        if any(k.endswith(NEW) for k in d["runs"]):
            seeds.update(d["seeds"])
        runs.update(d["runs"])
    out: dict = {"seeds": sorted(seeds), "points": {}}
    r32 = _frame(runs, seeds, BASE_W, False)
    out["pre_identical"] = {"32": r32["pre_identical"]}
    out["points"]["32"] = {r: _point(r32, r) for r in ("keep", "ach_c")}
    line = (out["points"]["32"]["keep"], out["points"]["32"]["ach_c"])
    for w in WIDTHS:
        for fr in ("in", "abs"):
            res = _frame(runs, seeds, w, fr == "abs")
            if not res["rows"]:
                continue
            if fr == "in":
                out["pre_identical"][str(w)] = res["pre_identical"]
            pts = {r: _point(res, f"{r}_h{w}") for r in ("keep", "ach_c")}
            for p in pts.values():
                p["residual"] = residual(line, p)
            out["points"][f"{w}_{fr}"] = pts
    out["verdict"] = verdict(out["points"])
    return out


def _mean_res(pts: dict | None) -> float | None:
    if not pts or any(p.get("residual") is None for p in pts.values()):
        return None
    return sum(p["residual"] for p in pts.values()) / len(pts)


def verdict(points: dict) -> dict:
    """12.6.241節で計測前に固定した判定条件。"""
    v: dict = {f"res_{k}": _mean_res(points.get(k)) for k in (f"{w}_{fr}" for w in WIDTHS for fr in ("in", "abs"))}
    top = WIDTHS[-1]
    r_in, r_abs = v[f"res_{top}_in"], v[f"res_{top}_abs"]
    # H（容量で曲線が外へ動く）: 幅 128 の keep と ach_c の残差の平均（in の枠）が +0.02 以上
    v["H"] = None if r_in is None else bool(r_in >= 0.02)
    # A（幅を広げたこと自体を含めて外へ動く）: 同じ量を abs の枠で
    v["A"] = None if r_abs is None else bool(r_abs >= 0.02)
    # M（単調）: in の枠で、幅 64 の残差の平均が 0 以上、幅 128 が幅 64 以上
    r64 = v[f"res_{WIDTHS[0]}_in"]
    v["M"] = None if None in (r64, r_in) else bool(r64 >= 0.0 and r_in >= r64)
    # K（加えた痕跡の保護の代価が縮む）: in の枠で、ach_c と keep の new の差が幅 32 の半分以下
    p32, pt = points.get("32"), points.get(f"{top}_in")
    if p32 and pt and None not in (p32["ach_c"]["new"], p32["keep"]["new"], pt["ach_c"]["new"], pt["keep"]["new"]):
        c32, ct = p32["ach_c"]["new"] - p32["keep"]["new"], pt["ach_c"]["new"] - pt["keep"]["new"]
        v["cost_32"], v[f"cost_{top}"] = c32, ct
        v["K"] = bool(ct <= c32 / 2)
    else:
        v["K"] = None
    # 参考: D1〜D3 を満たす点（両立）
    v["joint"] = sorted(f"{k}/{r}" for k, pts in points.items() for r, p in pts.items() if p["crit"]["success"])
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", type=Path, nargs="+", help="結果ファイル（12.6.167〜168節のものも）を集計して判定する")
    ap.add_argument("--configs", nargs="+", default=list(NEW))
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--phase", choices=("dev", "eval"), default="dev")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.analyze:
        print(json.dumps(analyze(args.analyze), ensure_ascii=False, indent=1))
        return
    seeds = args.seeds or (EVAL_SEEDS if args.phase == "eval" else DEV_SEEDS)
    out_path = args.out or Path(f"results/context_inference/capacity_curve_{args.phase}.json")
    all_jobs = [(*cfg, s) for cfg in cas.jobs(tuple(args.configs), tuple(args.families)) for s in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(partial(run_one, table=CONFIGS), all_jobs, chunksize=1)
    out: dict = {"環境": env_info(),
                 "説明": f"共有の隠れ層の幅と交換の曲線（12.6.241〜242節）。phase={args.phase}",
                 "seeds": seeds, "select": {SELECT: SELECTS[SELECT]}, "new_task": {k: repr(v) for k, v in NEW_TASK.items()},
                 "configs": {k: CONFIGS[k] for k in args.configs}, "runs": {}}
    for key in dict.fromkeys(x["key"] for x in results):
        rows = [x["row"] for x in results if x["key"] == key]
        summ = {}
        for k in SUMMARY_KEYS:
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
