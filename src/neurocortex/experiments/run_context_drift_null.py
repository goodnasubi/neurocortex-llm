"""表現の動き（drift）は読み出しの零空間に入るか — 幅ごとの、古いタスクの表現の動きの読み出しの出力に効く方向と
零空間への分解と、読み出しの可塑性による補償（12.6.247〜248節）。

12.6.244節では、共有の隠れ層を 32 → 128 に広げると古いタスクの表現の動き（1 − cos）が増えたのに、古いタスクの
読み出し（end）は損なわれなかった。神経科学では、表現が日々動いても行動が安定なのは、動きが読み出しの零空間
（出力に効かない方向）に入るから（Rule et al. 2019; Kaufman et al. 2014 の output-null 空間）とも、読み出しが動きに
追従して補償するから（Rule & O'Leary 2022 の自己修復する符号）とも説明される。ここでは 12.6.243節の密な層の条件
（{r}_h{W}、r = base・keep・ach_c、W = 32・64・128）を同じ乱数で測り直し、表現の動き Δh を、新しいタスクの学習の
直前の読み出しのロジットの差の方向 d̂0 とその零空間に分ける（null_stats。学習の乱数と .grad には触れない）。

    --phase dev（開発用シード 100〜104）/ eval（評価用シード 0〜4 で一度だけ）
    結果は results/context_inference/drift_null_{phase}.json

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_drift_null --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_context_drift_null --analyze <結果ファイル> \
        [--ref <12.6.243〜244節の結果ファイル（再現の確認）>]
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
from .run_context_capacity_overlap import CONFIGS as OVERLAP_CONFIGS
from .run_context_capacity_overlap import DENSE, REP_KEYS, RULES0
from .run_context_episodic_readout import FAMILIES, SELECTS
from .run_context_replay_novelty import SUMMARY_KEYS
from .run_context_writeback import NEW_TASK, SELECT

DEV_SEEDS = list(range(100, 105))
EVAL_SEEDS = list(range(0, 5))
CONFIGS: dict[str, dict] = {f"{r}_h{w}": {**OVERLAP_CONFIGS[f"{r}_h{w}"], "null_diag": True} for w in DENSE for r in RULES0}
NEW = tuple(CONFIGS)
NULL_KEYS = ("nl_d_tot", "nl_f_pot", "nl_e_iso", "nl_e_rep", "nl_f_pot1", "nl_f_h0", "nl_pot_rel", "nl_harm_rep",
             "nl_harm_ro", "nl_comp", "nl_acc0", "nl_acc1")
# 修正（上限3回）。判定の条件は計測前に決めた。修正は実装の誤りに限る
FIXES: tuple[str, ...] = ()


def run_one(job: tuple, table: dict) -> dict:
    return cas.run_job(job, table[job[3]])


def _rows(runs: dict, cfg: str, seeds: set[int]) -> list[dict]:
    return [x for k, v in runs.items() if k.rsplit("/", 1)[1] == cfg for x in v["per_seed"] if x["seed"] in seeds]


def _old_change(x: dict) -> float:
    """古いタスクの読み出しの変化（新しいタスクの学習の間の、本来の文脈での end − pre の平均）。"""
    return sum(e - p for e, p in zip(x["end_nat"], x["pre_nat"])) / len(x["end_nat"])


def summary(runs: dict, seeds: set[int]) -> dict:
    """設定ごとの平均（全組・全シード）。"""
    out: dict = {}
    for cfg in NEW:
        rows = _rows(runs, cfg, seeds)
        if not rows:
            continue
        s: dict = {"n": len(rows)}
        for key in (*NULL_KEYS, "rep_drift", "gcos_tr", "new_nat"):
            vals = [x[key] for x in rows if x.get(key) is not None]
            s[key] = sum(vals) / len(vals) if vals else None
        s["old_change"] = sum(_old_change(x) for x in rows) / len(rows)
        out[cfg] = s
    return out


def _rank(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    ra, rb = torch.tensor(_rank(a)), torch.tensor(_rank(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = float(ra.norm() * rb.norm())
    return float((ra * rb).sum()) / den if den > 0 else None


def per_seed(runs: dict, seeds: set[int]) -> dict:
    """シードごとの幅 128 − 幅 32 の差（全組の平均、規則ごと）と、正の数。"""
    out: dict = {}
    for r in RULES0:
        for key in ("nl_d_tot", "nl_pot_rel", "nl_e_iso", "nl_comp", "nl_harm_rep", "nl_harm_ro"):
            diffs = []
            for s in sorted(seeds):
                a = [x[key] for x in _rows(runs, f"{r}_h32", {s})]
                b = [x[key] for x in _rows(runs, f"{r}_h128", {s})]
                if a and b:
                    diffs.append(sum(b) / len(b) - sum(a) / len(a))
            if diffs:
                t = torch.tensor(diffs)
                sd = float(t.std()) if len(diffs) > 1 else 0.0
                out[f"{r}/{key}"] = {"mean": float(t.mean()), "n_pos": int((t > 0).sum()), "n": len(diffs),
                                     "t": float(t.mean()) / (sd / len(diffs) ** 0.5) if sd > 0 else None}
    return out


def reproduce(runs: dict, refs: dict) -> dict:
    """12.6.243〜244節の同じ設定の結果と、new・end・表現の診断が一致するか。"""
    n = same = 0
    for k, v in runs.items():
        ref = refs.get(k)
        if not ref:
            continue
        rs = {x["seed"]: x for x in ref["per_seed"]}
        for x in v["per_seed"]:
            if x["seed"] in rs:
                y = rs[x["seed"]]
                n += 1
                same += all(x.get(q) == y.get(q) for q in ("new_nat", "end_nat", *REP_KEYS))
    return {"n": n, "identical": same}


def _avg(summ: dict, w: int, key: str) -> float | None:
    vals = [summ.get(f"{r}_h{w}", {}).get(key) for r in RULES0]
    return None if any(v is None for v in vals) else sum(vals) / len(vals)


def verdict(summ: dict, runs: dict, seeds: set[int]) -> dict:
    """12.6.247節で計測前に固定した判定条件（値は base・keep・ach_c の平均）。"""
    v: dict = {f"{k}_h{w}": _avg(summ, w, f"nl_{k}") for w in DENSE
               for k in ("d_tot", "pot_rel", "e_iso", "comp", "harm_rep", "harm_ro")}
    if None in (v["d_tot_h32"], v["d_tot_h128"], v["pot_rel_h32"], v["pot_rel_h128"]):
        v["Z"] = None
    else:
        # Z（主仮説。動きの増えた分は零空間に入る）: 幅 128 で表現の動きの大きさは増え、出力に効く動きは増えない
        v["Z"] = bool(v["d_tot_h128"] > v["d_tot_h32"] and v["pot_rel_h128"] <= v["pot_rel_h32"])
    # E（零空間に偏る）: 幅 128 の濃縮 e_iso < 1、かつ幅 32 より小さい
    v["E"] = None if None in (v["e_iso_h32"], v["e_iso_h128"]) else bool(v["e_iso_h128"] < 1 and v["e_iso_h128"] < v["e_iso_h32"])
    # K（読み出しの補償）: 幅 128 の comp ≥ 0.5
    v["K"] = None if v["comp_h128"] is None else bool(v["comp_h128"] >= 0.5)
    # F（測度の妥当性）: 全回で、マージンの変化の向き（harm_rep + harm_ro）と古いタスクの end − pre の順位相関 ≥ 0.3
    xs, ys = [], []
    for cfg in NEW:
        for x in _rows(runs, cfg, seeds):
            if x.get("nl_harm_rep") is not None:
                xs.append(x["nl_harm_rep"] + x["nl_harm_ro"])
                ys.append(_old_change(x))
    v["F_rho"] = spearman(xs, ys)
    v["F"] = None if v["F_rho"] is None else bool(v["F_rho"] >= 0.3)
    return v


def analyze(paths: list[Path], refs: list[Path] | None = None) -> dict:
    runs: dict = {}
    seeds: set[int] = set()
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        seeds.update(d["seeds"])
        for k, v in d["runs"].items():  # 開発用と評価用のファイルは同じキーを持つので、回を連結する
            runs.setdefault(k, {"per_seed": []})["per_seed"] += v["per_seed"]
    summ = summary(runs, seeds)
    out: dict = {"seeds": sorted(seeds), "summary": summ, "per_seed": per_seed(runs, seeds)}
    if refs:
        ref_runs: dict = {}
        for p in refs:
            ref_runs.update(json.loads(p.read_text(encoding="utf-8"))["runs"])
        out["reproduce"] = reproduce(runs, ref_runs)
    out["verdict"] = verdict(summ, runs, seeds)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", type=Path, nargs="+", help="結果ファイルを集計して判定する")
    ap.add_argument("--ref", type=Path, nargs="+", help="再現の確認に使う 12.6.243〜244節の結果ファイル")
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
    out_path = args.out or Path(f"results/context_inference/drift_null_{args.phase}.json")
    all_jobs = [(*cfg, s) for cfg in cas.jobs(tuple(args.configs), tuple(args.families)) for s in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(partial(run_one, table=CONFIGS), all_jobs, chunksize=1)
    out: dict = {"環境": env_info(),
                 "説明": f"表現の動きの読み出しの零空間への分解（12.6.247〜248節）。phase={args.phase}",
                 "seeds": seeds, "select": {SELECT: SELECTS[SELECT]}, "new_task": {k: repr(v) for k, v in NEW_TASK.items()},
                 "configs": {k: CONFIGS[k] for k in args.configs}, "runs": {}}
    for key in dict.fromkeys(x["key"] for x in results):
        rows = [x["row"] for x in results if x["key"] == key]
        summ = {}
        for k in (*SUMMARY_KEYS, *REP_KEYS, *NULL_KEYS):
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
