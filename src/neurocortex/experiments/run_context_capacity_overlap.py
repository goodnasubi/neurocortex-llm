"""共有の層の容量の効果は表現の重なりの減少から来るか — 幅ごとの隠れ層の表現と勾配の重なりの測定と、k-WTA による
疎な符号化の因果的な対照（12.6.243〜244節）。

12.6.241〜242節では、規則を変えずに共有の隠れ層を 32 → 64 → 128 に広げると、keep と ach_c の点が幅 32 の交換の
曲線より外へ出た（加えた痕跡のリプレイが新しいタスクに払う代価が消えた）。ここでは、その効果が「表現の重なりの
減少」（相補的学習系の説、McClelland et al. 1995; French 1999）から来るかを測る:
    {r}_h{W}  … 12.6.241節と同じ密な隠れ層（W = 32・64・128）に、表現と勾配の診断を付けて測り直す（値は同じに再現する）
    {r}_k{W}  … 同じ幅で、隠れ層を k-WTA（前活性の大きい k = W/4 個だけが活動する疎な符号化）にする（W = 32・128）
r は base・keep・ach_c（12.6.167〜168節と同じ設定）。

診断（新しいタスクの学習の間・最後に測る。学習の乱数と .grad には触れない）:
    gcos_tr  … 共有の重みの上の、新しい入力の勾配と加えた痕跡の勾配の cos（10 ステップごとの平均）
    gcos_old … 同じく、古い文脈の覚醒中の痕跡の勾配との cos
    rep_overlap … 最後の、古いタスクの本来の文脈と新しいタスクの文脈の隠れ層の活動の cos（同じ探査入力）
    rep_drift   … 新しいタスクの学習の間の、古いタスクの本来の文脈の隠れ層の活動の動き（1 − cos）
    rep_sparse・rep_active … Treves–Rolls の母集団の疎さ・|h| > 0.5 のユニットの割合

    --phase dev（開発用シード 100〜104）/ eval（評価用シード 0〜4 で一度だけ）
    結果は results/context_inference/capacity_overlap_{phase}.json

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_capacity_overlap --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_context_capacity_overlap --analyze <結果ファイル> \
        [--ref <12.6.167〜168節と 12.6.241〜242節の結果ファイル（再現の確認）>]
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
from .run_context_capacity_curve import residual
from .run_context_episodic_readout import FAMILIES, SELECTS
from .run_context_replay_novelty import SUMMARY_KEYS
from .run_context_replay_split import CONFIGS as SPLIT_CONFIGS
from .run_context_replay_split import _criteria
from .run_context_replay_split import analyze as _split_analyze
from .run_context_writeback import NEW_TASK, SELECT

DEV_SEEDS = list(range(100, 105))
EVAL_SEEDS = list(range(0, 5))
RULES0 = ("base", "keep", "ach_c")
DENSE = (32, 64, 128)  # 12.6.241節の幅
SPARSE = {32: 8, 128: 32}  # k-WTA の幅と k（k = W/4。幅 128 の k = 32 は幅 32 の密な層と同じ数の活動するユニット）
CONFIGS: dict[str, dict] = {}
CONFIGS.update({f"{r}_h{w}": {**SPLIT_CONFIGS[r], "hidden": w, "rep_diag": True} for w in DENSE for r in RULES0})
CONFIGS.update({f"{r}_k{w}": {**SPLIT_CONFIGS[r], "hidden": w, "kwta": k, "rep_diag": True}
                for w, k in SPARSE.items() for r in RULES0})
NEW = tuple(CONFIGS)
REP_KEYS = ("rep_drift", "rep_overlap", "rep_sparse", "rep_active", "gcos_tr", "gcos_old")
# 修正（上限3回）。k と幅は計測前に決めた。修正は実装の誤りに限る
FIXES: tuple[str, ...] = ()


def run_one(job: tuple, table: dict) -> dict:
    return cas.run_job(job, table[job[3]])


def _frame(runs: dict, seeds: set[int], base_cfg: str, sfx: str) -> dict:
    """規則 keep{sfx}・ach_c{sfx} を base_cfg と対にして 12.6.169節の集計にかける。"""
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
            "q05": a.get(f"q05_new_{w}"), "n_over": o.get("n"), "crit": _criteria(o, n, a, w)}


def rep_summary(runs: dict, seeds: set[int]) -> dict:
    """設定ごとの診断の平均（全組・全シード）。abs_gcos_tr は回ごとの |gcos_tr| の平均。生の new・古いタスクの end も。"""
    out: dict = {}
    for cfg in NEW:
        rows = [x for k, v in runs.items() if k.rsplit("/", 1)[1] == cfg for x in v["per_seed"] if x["seed"] in seeds]
        if not rows:
            continue
        s: dict = {"n": len(rows)}
        for key in (*REP_KEYS, "new_nat"):
            vals = [x[key] for x in rows if x.get(key) is not None]
            s[key] = sum(vals) / len(vals) if vals else None
        vals = [abs(x["gcos_tr"]) for x in rows if x.get("gcos_tr") is not None]
        s["abs_gcos_tr"] = sum(vals) / len(vals) if vals else None
        s["end_raw"] = sum(sum(x["end_nat"]) / len(x["end_nat"]) for x in rows) / len(rows)
        out[cfg] = s
    return out


def reproduce(runs: dict, refs: dict) -> dict:
    """密な層の測り直しが 12.6.167〜168節（幅 32）・12.6.241〜242節（幅 64・128）の結果と一致するか（new と end の値）。"""
    n = same = 0
    for k, v in runs.items():
        cb, cfg = k.rsplit("/", 1)
        r, _, w = cfg.rpartition("_h")
        if not r or cfg not in NEW:
            continue
        ref = refs.get(f"{cb}/{r}" if w == "32" else f"{cb}/{cfg}")
        if not ref:
            continue
        rs = {x["seed"]: x for x in ref["per_seed"]}
        for x in v["per_seed"]:
            if x["seed"] in rs:
                n += 1
                same += x["new_nat"] == rs[x["seed"]]["new_nat"] and x["end_nat"] == rs[x["seed"]]["end_nat"]
    return {"n": n, "identical": same}


def analyze(paths: list[Path], refs: list[Path] | None = None) -> dict:
    runs: dict = {}
    seeds: set[int] = set()
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        seeds.update(d["seeds"])
        runs.update(d["runs"])
    out: dict = {"seeds": sorted(seeds), "points": {}, "pre_identical": {}}
    frames = {f"h{w}": (f"base_h{w}", f"_h{w}") for w in DENSE}
    for w in SPARSE:
        frames[f"k{w}_in"] = (f"base_k{w}", f"_k{w}")
        frames[f"k{w}_abs"] = ("base_h32", f"_k{w}")
    for name, (b, sfx) in frames.items():
        res = _frame(runs, seeds, b, sfx)
        if not res["rows"]:
            continue
        out["pre_identical"][name] = res["pre_identical"]
        out["points"][name] = {r: _point(res, f"{r}{sfx}") for r in ("keep", "ach_c")}
    if "h32" in out["points"]:
        line = (out["points"]["h32"]["keep"], out["points"]["h32"]["ach_c"])
        for name, pts in out["points"].items():
            if name != "h32":
                for p in pts.values():
                    p["residual"] = residual(line, p)
    out["rep"] = rep_summary(runs, seeds)
    if refs:
        ref_runs: dict = {}
        for p in refs:
            ref_runs.update(json.loads(p.read_text(encoding="utf-8"))["runs"])
        out["reproduce"] = reproduce(runs, ref_runs)
    out["verdict"] = verdict(out["points"], out["rep"])
    return out


def _mean_res(pts: dict | None) -> float | None:
    if not pts or any(p.get("residual") is None for p in pts.values()):
        return None
    return sum(p["residual"] for p in pts.values()) / len(pts)


def verdict(points: dict, rep: dict) -> dict:
    """12.6.243節で計測前に固定した判定条件。"""
    v: dict = {f"res_{k}": _mean_res(points.get(k)) for k in ("h64", "h128", "k32_in", "k32_abs", "k128_in", "k128_abs")}
    # S（疎な符号化が曲線を外へ動かす）: 幅 32 の k-WTA の keep・ach_c の残差の平均（in の枠）が +0.02 以上
    v["S"] = None if v["res_k32_in"] is None else bool(v["res_k32_in"] >= 0.02)
    # Q（同じ幅で疎さが上積みする）: 幅 128 の k-WTA の残差の平均（in）が幅 128 の密な層より +0.01 以上大きい
    v["Q"] = None if None in (v["res_k128_in"], v["res_h128"]) else bool(v["res_k128_in"] >= v["res_h128"] + 0.01)
    g = {c: rep.get(f"keep_{c}", {}).get("abs_gcos_tr") for c in ("h32", "h64", "h128", "k32")}
    v["abs_gcos_tr_keep"] = g
    # O（幅で勾配の重なりが減る）: keep の |gcos_tr| が幅 32 > 64 > 128 で単調に減り、幅 128 で幅 32 の 0.8 倍以下
    if None in (g["h32"], g["h64"], g["h128"]):
        v["O"] = None
    else:
        v["O"] = bool(g["h32"] > g["h64"] > g["h128"] and g["h128"] <= 0.8 * g["h32"])
    # P（k-WTA が重なりを減らす）: 幅 32 の k-WTA の keep の |gcos_tr| が密な層の 0.8 倍以下、かつ rep_overlap が小さい
    ov = {c: rep.get(f"keep_{c}", {}).get("rep_overlap") for c in ("h32", "k32")}
    v["rep_overlap_keep"] = ov
    if None in (g["h32"], g["k32"], ov["h32"], ov["k32"]):
        v["P"] = None
    else:
        v["P"] = bool(g["k32"] <= 0.8 * g["h32"] and ov["k32"] < ov["h32"])
    v["joint"] = sorted(f"{k}/{r}" for k, pts in points.items() for r, p in pts.items() if p["crit"]["success"])
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", type=Path, nargs="+", help="結果ファイルを集計して判定する")
    ap.add_argument("--ref", type=Path, nargs="+", help="再現の確認に使う 12.6.167〜168節・12.6.241〜242節の結果ファイル")
    ap.add_argument("--configs", nargs="+", default=list(NEW))
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--phase", choices=("dev", "eval"), default="dev")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.analyze:
        print(json.dumps(analyze(args.analyze, args.ref), ensure_ascii=False, indent=1))
        return
    seeds = args.seeds or (EVAL_SEEDS if args.phase == "eval" else DEV_SEEDS)
    out_path = args.out or Path(f"results/context_inference/capacity_overlap_{args.phase}.json")
    all_jobs = [(*cfg, s) for cfg in cas.jobs(tuple(args.configs), tuple(args.families)) for s in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(partial(run_one, table=CONFIGS), all_jobs, chunksize=1)
    out: dict = {"環境": env_info(),
                 "説明": f"容量の効果と表現の重なり・k-WTA（12.6.243〜244節）。phase={args.phase}",
                 "seeds": seeds, "select": {SELECT: SELECTS[SELECT]}, "new_task": {k: repr(v) for k, v in NEW_TASK.items()},
                 "configs": {k: CONFIGS[k] for k in args.configs}, "runs": {}}
    for key in dict.fromkeys(x["key"] for x in results):
        rows = [x["row"] for x in results if x["key"] == key]
        summ = {}
        for k in (*SUMMARY_KEYS, *REP_KEYS):
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
