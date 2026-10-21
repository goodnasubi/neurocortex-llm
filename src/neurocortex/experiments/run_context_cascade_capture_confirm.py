"""カスケード＋新しさによる捕獲（ach_c_cas_cap、12.6.233〜234節）の事前登録した確証実験と、捕獲の駆動の対照
（12.6.235〜236節）。

12.6.234節で ach_c_cas_cap は評価用シードで D1〜D3 を満たし、keep と ach_c を結ぶ交換の曲線の外に出たが、開発用シード
では (c) が1回足りず、事前の規則で候補にしていなかった。ここでは
    1. 新しいシード 200〜209（計測の前に決めた。これまでの結果ファイルで使っていない）で base・keep・ach_c・
       ach_c_cas_cap を測り、D1〜D4 と曲線からの残差を確かめる（確証）。
    2. 捕獲の駆動 1 − f を、同じ組・同じシードの ach_c_cas_cap の実行で記録した列の
         ach_c_cas_cap_perm … 並べ替え（量の分布を保ち、新しさとの時間的な対応だけを壊す）
         ach_c_cas_cap_mean … 平均の定数（量だけを保ち、時間の構造をなくす）
       に置き換えた対照で、効果が新しさに結びついた捕獲によるかを切り分ける。

    --phase dev（開発用シード 100〜104。対照の確定）/ eval（評価用シード 0〜4 の対照を一度だけ）/
    confirm（確証用シード 200〜209 で全条件を一度だけ）
    結果は results/context_inference/cascade_capture_{phase}.json

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_cascade_capture_confirm --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_context_cascade_capture_confirm --analyze <結果ファイル>...
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import torch

from ..run_env import env_info
from .run_context_episodic_readout import FAMILIES, LS, MECHS_USED, SELECTS
from .run_context_replay_cascade import CONFIGS as CAS_CONFIGS
from .run_context_replay_cascade import frontier_residual, run_job
from .run_context_replay_novelty import SUMMARY_KEYS
from .run_context_replay_split import _criteria
from .run_context_replay_split import analyze as _split_analyze
from .run_context_writeback import SELECT

DEV_SEEDS = list(range(100, 105))
EVAL_SEEDS = list(range(0, 5))
CONFIRM_SEEDS = list(range(200, 210))  # 12.6.235節で計測の前に決めた確証用のシード
PERM_SEED = 70_000  # 並べ替えの乱数（シードに足す）
CAP = "ach_c_cas_cap"
CONTROLS = (f"{CAP}_perm", f"{CAP}_mean")
REFS = ("base", "keep", "ach_c")
CONFIGS: dict[str, dict] = {k: CAS_CONFIGS[k] for k in (*REFS, CAP)}
RULES = ("keep", "ach_c", CAP, *CONTROLS)
PHASE_CONFIGS = {"dev": ("capctl",), "eval": ("capctl",), "confirm": (*REFS, "capctl")}
SEEDS = {"dev": DEV_SEEDS, "eval": EVAL_SEEDS, "confirm": CONFIRM_SEEDS}


def control_drives(drive: list[float], seed: int) -> dict[str, list[float]]:
    """記録した 1 − f の列から、対照の駆動を作る（perm: 並べ替え、mean: 平均の定数）。"""
    g = torch.Generator().manual_seed(seed + PERM_SEED)
    perm = torch.randperm(len(drive), generator=g).tolist()
    m = sum(drive) / len(drive)
    return {f"{CAP}_perm": [drive[i] for i in perm], f"{CAP}_mean": [m] * len(drive)}


def run_one(job: tuple) -> list[dict]:
    """job = (課題, 機構, L, 設定, シード)。設定 capctl では ach_c_cas_cap を測り、記録した駆動で対照の2つを続けて測る。"""
    fam, mech, L, cfg, seed = job
    if cfg != "capctl":
        return [run_job(job, CONFIGS[cfg])]
    cap = run_job((fam, mech, L, CAP, seed), CONFIGS[CAP])
    drive = cap.pop("drive")
    out = [cap]
    for name, drv in control_drives(drive, seed).items():
        r = run_job((fam, mech, L, name, seed), {**CONFIGS[CAP], "cascade_capture_drive": drv})
        r["row"]["drive_len_match"] = len(r.pop("drive")) == len(drv)
        out.append(r)
    return out


def jobs(phase: str, seeds: list[int], families=FAMILIES) -> list[tuple]:
    return [(f, m, L, c, s) for c in PHASE_CONFIGS[phase] for m in MECHS_USED for f in families for L in LS
            for s in seeds]


def per_seed_end(runs: dict, combos: list[str], w: str) -> dict[int, float]:
    """固定した組の集合（例: 全シードで決めた 14 ビットの上書きありの組）での、シードごとの end の差（base との差）。"""
    acc: dict[int, list[float]] = {}
    for cb in combos:
        base = {x["seed"]: x for x in runs.get(f"{cb}/base", {}).get("per_seed", [])}
        for x in runs.get(f"{cb}/{w}", {}).get("per_seed", []):
            b = base.get(x["seed"])
            if b is not None:
                de = [a - c for a, c in zip(x["end_nat"], b["end_nat"])]
                acc.setdefault(x["seed"], []).append(sum(de) / len(de))
    return {s: sum(v) / len(v) for s, v in acc.items()}


def per_seed_new(runs: dict, w: str) -> dict[int, float]:
    """30組の new の差（base との差）のシードごとの平均。"""
    acc: dict[int, list[float]] = {}
    for k, v in runs.items():
        cb, r = k.rsplit("/", 1)
        if r != w:
            continue
        base = {x["seed"]: x for x in runs.get(f"{cb}/base", {}).get("per_seed", [])}
        for x in v["per_seed"]:
            if x["seed"] in base:
                acc.setdefault(x["seed"], []).append(x["new_nat"] - base[x["seed"]]["new_nat"])
    return {s: sum(v) / len(v) for s, v in acc.items()}


def paired(d: dict[int, float]) -> dict:
    """シードごとの対の差の平均・正の数・t 値。"""
    v = list(d.values())
    n = len(v)
    if n == 0:
        return {"n": 0}
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    return {"n": n, "mean": m, "pos": sum(x > 0 for x in v), "t": m / (sd / math.sqrt(n)) if sd > 0 else None}


def analyze(paths: list[Path], seeds: list[int] | None = None) -> dict:
    """同じシードの base との差で集計し（12.6.169節と同じ）、12.6.235節で計測前に固定した判定を付ける。
    seeds を与えればそのシードだけを使う。"""
    runs: dict = {}
    for p in paths:
        for k, v in json.loads(p.read_text(encoding="utf-8"))["runs"].items():
            per = [x for x in v["per_seed"] if seeds is None or x["seed"] in seeds]
            if per:
                old = runs.get(k, {"per_seed": []})["per_seed"]
                got = {x["seed"] for x in per}
                runs[k] = {"per_seed": [x for x in old if x["seed"] not in got] + per}
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "runs.json"
        f.write_text(json.dumps({"runs": runs}), encoding="utf-8")
        res = _split_analyze([f], RULES)
    res["seeds"] = sorted({x["seed"] for v in runs.values() for x in v["per_seed"]})
    o, a = res["over_14bit"], res["all"]
    slope = None
    if None not in (o.get("end_keep"), o.get("end_ach_c"), a.get("new_keep"), a.get("new_ach_c")):
        slope = (o["end_keep"] - o["end_ach_c"]) / (a["new_keep"] - a["new_ach_c"])
    res["paired"] = {}
    for w in (CAP, *CONTROLS):
        e_w, n_w = per_seed_end(runs, o["combos"], w), per_seed_new(runs, w)
        e_a = per_seed_end(runs, o["combos"], "ach_c")
        res["paired"][f"end_{w}-ach_c"] = paired({s: e_w[s] - e_a[s] for s in e_w if s in e_a})
        if w != CAP and slope is not None:
            # 残差の差（cap − 対照）= end の差 − 傾き × new の差。シードごとに
            e_c, n_c = per_seed_end(runs, o["combos"], CAP), per_seed_new(runs, CAP)
            res["paired"][f"resid_{CAP}-{w}"] = paired({s: (e_c[s] - e_w[s]) - slope * (n_c[s] - n_w[s])
                                                       for s in e_w if s in e_c and s in n_w and s in n_c})
    # 参考: 移行の長さ L ごとの上書きありの end と 30組のうちその L の組の new の平均
    res["by_L"] = {}
    for L in LS:
        rows = {k: v for k, v in res["rows"].items() if k.split("/")[1].endswith(f"_L{L}")}
        res["by_L"][L] = {f"{q}_{w}": _mean([v.get(f"{q}_{w}") for k, v in rows.items()
                                             if q == "new" or k in o["combos"]])
                          for w in RULES for q in ("end", "new")}
    res["drive"] = {w: _mean([x.get("cap_drive") for k, v in runs.items() if k.endswith("/" + w)
                              for x in v["per_seed"]]) for w in (CAP, *CONTROLS)}
    res["drive_len_match"] = all(x.get("drive_len_match", True) for v in runs.values() for x in v["per_seed"])
    res["verdict"] = verdict(res)
    return res


def _mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def verdict(res: dict) -> dict:
    """12.6.235節で計測前に固定した判定条件。"""
    o, n, a = res["over_14bit"], res["not_over"], res["all"]
    out: dict = {}
    for w in RULES:
        if o.get(f"end_{w}") is None:
            continue
        v = _criteria(o, n, a, w)
        if w not in ("keep", "ach_c") and o.get("end_ach_c") is not None:
            v["D4"] = bool(o[f"end_{w}"] - o["end_ach_c"] >= 0.02 and a[f"bad_{w}"] <= a["bad_ach_c"])
            r = frontier_residual(o, a, w)
            if r is not None:
                v["residual"] = r
                v["F"] = bool(r >= 0.02)
        out[w] = v
    c = out.get(CAP, {})
    # C（確証）: ach_c_cas_cap が D1〜D3 を満たし、交換の曲線から +0.02 以上外側
    out["C"] = bool(c.get("success") and c.get("F")) if c else None
    # N（新しさとの対応が要る）: 2つの対照のどちらに対しても、残差が ach_c_cas_cap より 0.02 以上小さい
    rs = [out.get(w, {}).get("residual") for w in CONTROLS]
    if c.get("residual") is not None and None not in rs:
        out["N"] = all(c["residual"] - r >= 0.02 for r in rs)
        out["N_each"] = {w: bool(c["residual"] - r >= 0.02) for w, r in zip(CONTROLS, rs)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", type=Path, nargs="+", help="結果ファイルを集計して判定する")
    ap.add_argument("--seeds", type=int, nargs="+", help="集計・計測に使うシード")
    ap.add_argument("--phase", choices=tuple(PHASE_CONFIGS), default="dev")
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.analyze:
        res = analyze(args.analyze, args.seeds)
        print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False, indent=1))
        return
    seeds = args.seeds or SEEDS[args.phase]
    out_path = args.out or Path(f"results/context_inference/cascade_capture_{args.phase}.json")
    with Pool(args.workers) as pool:
        results = [r for rs in pool.imap(run_one, jobs(args.phase, seeds, tuple(args.families)), chunksize=1)
                   for r in rs]
    out: dict = {"環境": env_info(),
                 "説明": f"カスケード＋新しさによる捕獲の確証と駆動の対照（12.6.235〜236節）。phase={args.phase}",
                 "seeds": seeds, "select": {SELECT: SELECTS[SELECT]},
                 "configs": {**CONFIGS, **{w: {**CONFIGS[CAP], "cascade_capture_drive": "記録した列から"}
                                           for w in CONTROLS}},
                 "perm_seed": PERM_SEED, "runs": {}}
    for key in dict.fromkeys(x["key"] for x in results):
        rows = [x["row"] for x in results if x["key"] == key]
        summ = {}
        for k in (*SUMMARY_KEYS, "cas_gap", "cas_move", "cap_drive"):
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
