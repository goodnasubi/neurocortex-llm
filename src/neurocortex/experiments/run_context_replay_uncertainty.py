"""不確かさによる新しさの信号で、取り戻した一般化と新しいタスクの学習を両立できるか（12.6.225〜226節）。

12.6.168節の ach_c（文脈ごとの損失のゆっくりした移動平均で固定化の配分を連続的に増やす）は、新しいタスクの保護を最も
改善したが、一般化とのトレードオフの曲線の上を動くだけだった。12.6.168節の残る課題 (2) に対し、なじみの度合いを
不確かさの他の推定で求める:
    ach_u   … モデル自身の予測のエントロピーの移動平均（ラベルを使わない、予期された不確かさ。Yu & Dayan 2005 の
              アセチルコリン）
    ach_v   … 損失の移動平均に、その揺らぎ（移動平均からのずれの標準偏差）を足す（予期しない不確かさ・変動性）
    ach_uv  … 予測のエントロピーと損失の揺らぎの和（アセチルコリンとノルアドレナリンの両方の不確かさ）
を、base・keep・ach_need2・ach_c（12.6.163・167節）と比べる。組・新しいタスク・集計は 12.6.167節と同じ。

    --phase dev（開発用シード 100〜104）/ eval（評価用シード 0〜4 で一度だけ）
    結果は results/context_inference/replay_uncertainty_{phase}.json

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_replay_uncertainty --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_context_replay_uncertainty --analyze <結果ファイル>
"""

from __future__ import annotations

import argparse
import json
import math
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import torch

from ..run_env import env_info
from .run_context_episodic_readout import FAMILIES, SELECTS
from .run_context_replay_novelty import CONFIGS as NOV_CONFIGS
from .run_context_replay_novelty import SUMMARY_KEYS, _criteria, analyze, jobs, run_one
from .run_context_evidence_info import SELECTION_TRIALS
from .run_context_writeback import NEW_TASK, SELECT

DEV_SEEDS = list(range(100, 105))
EVAL_SEEDS = list(range(0, 5))
CONFIGS: dict[str, dict] = {k: NOV_CONFIGS[k] for k in ("base", "keep", "ach_need2", "ach_c")}
CONFIGS.update({w: {**NOV_CONFIGS["ach_c"], "replay_rule": w} for w in ("ach_u", "ach_v", "ach_uv")})
NEW_RULES = ("ach_u", "ach_v", "ach_uv")
RULES = ("keep", "ach_need2", "ach_c", *NEW_RULES)
# 修正は開発用シードの結果を見て追加する（上限3回）


def _pareto(o: dict, a: dict, w: str, ref: str = "ach_c") -> bool | None:
    """B6（12.6.225節）: ref の点から、保護か一般化の一方を悪くせずに他方を基準以上よくする。"""
    if o.get(f"end_{w}") is None or o.get(f"end_{ref}") is None:
        return None
    d_end = o[f"end_{w}"] - o[f"end_{ref}"]
    bad, bad_ref = a[f"bad_{w}"], a[f"bad_{ref}"]
    more_gen = d_end >= 0.02 and bad <= bad_ref
    more_protect = bad_ref >= 2 and bad <= bad_ref / 2 and d_end >= -0.005
    return bool(more_gen or more_protect)


def verdict(res: dict) -> dict:
    """12.6.225節で計測前に固定した判定条件。"""
    o, n, a = res["over_14bit"], res["not_over"], res["all"]
    out: dict = {}
    for w in ("ach_c", *NEW_RULES):
        if o.get(f"end_{w}") is None:
            continue
        v = _criteria(o, n, a, w)
        v["B6"] = None if w == "ach_c" else _pareto(o, a, w)
        out[w] = v
    out["H_u"] = out.get("ach_u", {}).get("success")
    out["H"] = any(out.get(w, {}).get("success") for w in NEW_RULES)
    out["H_P"] = any(out.get(w, {}).get("B6") for w in NEW_RULES)
    return out


def paired_over_end(paths: list[Path], w: str, ref: str = "ach_c") -> dict:
    """参考: 14 ビットの上書きありの組の end のシードごとの差（w − ref）の平均と t 値。組の分類は analyze と同じ。"""
    runs: dict = {}
    for p in paths:
        runs.update(json.loads(p.read_text(encoding="utf-8"))["runs"])
    res = analyze(paths, RULES, verdict)
    combos = res["over_14bit"]["combos"]
    per: dict[int, list[float]] = {}
    for cb in combos:
        base = {x["seed"]: x for x in runs[f"{cb}/base"]["per_seed"]}
        xs = {x["seed"]: x for x in runs.get(f"{cb}/{w}", {}).get("per_seed", [])}
        rs = {x["seed"]: x for x in runs.get(f"{cb}/{ref}", {}).get("per_seed", [])}
        for s, b in base.items():
            if s in xs and s in rs:
                over = [i for i, g in enumerate(b["gap"]) if g >= 0.10]
                d = sum(xs[s]["end_nat"][i] - rs[s]["end_nat"][i] for i in over) / max(len(over), 1)
                per.setdefault(s, []).append(d)
    d = [sum(v) / len(v) for v in per.values()]
    if len(d) < 2:
        return {"n": len(d)}
    m = sum(d) / len(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
    return {"n": len(d), "mean": m, "t": m / (sd / math.sqrt(len(d))) if sd > 0 else math.inf}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", type=Path, nargs="+", help="結果ファイルを集計して判定に使う量を表示する")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--phase", choices=("dev", "eval"), default="dev")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.analyze:
        res = analyze(args.analyze, RULES, verdict)
        res["paired_over_end_vs_ach_c"] = {w: paired_over_end(args.analyze, w) for w in NEW_RULES}
        print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False, indent=1))
        return
    seeds = args.seeds or (EVAL_SEEDS if args.phase == "eval" else DEV_SEEDS)
    out_path = args.out or Path(f"results/context_inference/replay_uncertainty_{args.phase}.json")
    all_jobs = [(*cfg, s) for cfg in jobs(tuple(args.configs), tuple(args.families)) for s in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(partial(run_one, table=CONFIGS), all_jobs, chunksize=1)
    out: dict = {"環境": env_info(),
                 "説明": f"不確かさによる新しさの信号（12.6.225〜226節）。phase={args.phase}",
                 "seeds": seeds, "select": {SELECT: SELECTS[SELECT]}, "new_task": {k: repr(v) for k, v in NEW_TASK.items()},
                 "configs": {k: CONFIGS[k] for k in args.configs}, "selection_trials": SELECTION_TRIALS, "runs": {}}
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
