"""海馬の痕跡のリプレイの配分で、取り戻した一般化と新しいタスクの学習を両立できるか（12.6.163〜164節）。

12.6.162節では、保留サンプルをリプレイに加えると（wb_none_keep）新しいタスクの後にも一般化が +0.145 残ったが、新しい
タスクの学習が最悪の組で −0.29 損なわれた。ここでは同じ痕跡（バッファ + 保留サンプルの先頭50個）をリプレイし続けたまま、
リプレイの配分の規則を変える: 予測誤差による優先度（gain）、新旧の比率（need）、覚醒と睡眠に似た段階の交互（sleep）。
睡眠の段階の追加の学習そのものの効果と分けるため、加えた痕跡なしで同じ睡眠の段階を置く対照（sleep_buf）も測る。

    --phase dev（開発用シード）/ eval（評価用シードで一度だけ）
    結果は results/context_inference/replay_alloc_{phase}.json

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_replay_alloc --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_context_replay_alloc --analyze <結果ファイル>
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import torch

from ..context_inference import run_context_sequence
from ..run_env import env_info
from .run_context_episodic_readout import FAMILIES, LS, MECHS_USED, SELECTS, specs_for
from .run_context_evidence_info import MECHS, SELECTION_TRIALS
from .run_context_gated_long import DEV_SEEDS, EVAL_SEEDS, native_contexts
from .run_context_writeback import CUE_N, NEW_TASK, SELECT, WB_KW, _avg, _exp

KEEP = {"writeback": "none", "writeback_keep": True, **WB_KW}
CONFIGS: dict[str, dict] = {
    "base": {"writeback": "none", **WB_KW},  # 保留サンプルをリプレイしない（12.6.162節の wb_none）
    "keep": KEEP,  # 一様なリプレイ（12.6.162節の wb_none_keep）
    "gain": {**KEEP, "replay_rule": "gain", "replay_beta": 0.5},
    "need": {**KEEP, "replay_rule": "need", "replay_need": 0.5},
    "sleep": {**KEEP, "replay_rule": "sleep", "sleep_every": 50, "sleep_steps": 25},
    "sleep_buf": {"writeback": "none", **WB_KW, "replay_rule": "sleep", "sleep_every": 50, "sleep_steps": 25},
}
RULES = ("keep", "gain", "need", "sleep", "sleep_buf")
# 修正は開発用シードの結果を見て追加する（上限3回）


def jobs(configs=tuple(CONFIGS), families=FAMILIES) -> list[tuple]:
    return [(f, m, L, cfg) for m in MECHS_USED for f in families for L in LS for cfg in configs]


def run_job(job: tuple, kw: dict) -> dict:
    """job = (課題, 機構, L, 設定の名前, シード)。12.6.161節の run_job の書き戻しの部分と同じ量に、リプレイの内訳を加える。"""
    fam, mech, L, cfg, seed = job
    torch.set_num_threads(1)
    specs = specs_for(fam)
    T0 = len(specs)
    specs = specs + [NEW_TASK[fam]]
    T = len(specs)
    torch.manual_seed(seed)
    r = run_context_sequence("inferred_gated", specs, 32, torch.Generator().manual_seed(seed + 50_000),
                             torch.Generator().manual_seed(seed + 90_000), transition_steps=L, steps=500,
                             diagnostics=True, cue_n=CUE_N, selection_trials=SELECTION_TRIALS, **MECHS[mech],
                             **SELECTS[SELECT], writeback_at=T0, **kw)
    d = r["diag"]
    nat = native_contexts(r["assign"], T)
    ch = r["selection_choices"]
    acc = d["ctx_acc"][-1]
    orig = range(T0)
    w = d["writeback"]
    ci = [w["used"].index(nat[j]) for j in orig]
    mix = d.get("replay_mix", {}).get(T0, {})
    row = {"seed": seed, "n_contexts": r["n_contexts"], "native": nat,
           "selection_acc": sum(c == nat[j] for j in orig for c in ch[j]) / (T0 * SELECTION_TRIALS),
           "exp_end": _exp(acc, ch, orig), "end_nat": [2 * acc[j][nat[j]] - 1 for j in orig],
           "pre_nat": [2 * w["pre"]["fresh"][j][ci[j]] - 1 for j in orig],
           "gap": [w["pre"]["own_buf"][j][ci[j]] - w["pre"]["fresh"][j][ci[j]] for j in orig],
           "new_nat": 2 * acc[T0][nat[T0]] - 1, "new_exp": _exp(acc, ch, [T0]), "new_ctx": nat[T0] not in w["used"],
           "mix_own": mix.get("own"), "mix_kept": mix.get("kept"), "mix_size": mix.get("size"), "sleep_steps": mix.get("sleep")}
    return {"key": f"{fam}/{mech}_L{L}/{cfg}", "row": row}


def run_one(job: tuple, table: dict) -> dict:
    return run_job(job, table[job[3]])


def analyze(paths: list[Path]) -> dict:
    """各規則は同じシードの base と対にして差をとる。組の分類は 12.6.161節と同じ（base の gap、正確度の単位）。"""
    runs: dict = {}
    for p in paths:
        runs.update(json.loads(p.read_text(encoding="utf-8"))["runs"])
    combos = sorted({k.rsplit("/", 1)[0] for k in runs})
    rows: dict = {}
    for cb in combos:
        base = {x["seed"]: x for x in runs.get(f"{cb}/base", {}).get("per_seed", [])}
        if not base:
            continue
        r: dict = {"gap": _avg([_avg(x["gap"]) for x in base.values()]),
                   "end_base": _avg([_avg(x["end_nat"]) for x in base.values()]),
                   "new_base": _avg([x["new_nat"] for x in base.values()]), "pre_identical": True}
        for w in RULES:
            per = {x["seed"]: x for x in runs.get(f"{cb}/{w}", {}).get("per_seed", [])}
            if not per:
                continue
            end, over_end, intact_end, exp, new_d, newexp_d, sel_d = ([] for _ in range(7))
            for s, x in per.items():
                b = base[s]
                r["pre_identical"] &= x["pre_nat"] == b["pre_nat"]
                de = [a - c for a, c in zip(x["end_nat"], b["end_nat"])]
                end.append(_avg(de))
                over_end.append(_avg([de[i] for i, g in enumerate(b["gap"]) if g >= 0.10]))
                intact_end.append(_avg([de[i] for i, g in enumerate(b["gap"]) if g <= 0.03]))
                exp.append(x["exp_end"] - b["exp_end"])
                new_d.append(x["new_nat"] - b["new_nat"])
                newexp_d.append(x["new_exp"] - b["new_exp"])
                sel_d.append(x["selection_acc"] - b["selection_acc"])
            r.update({f"end_{w}": _avg(end), f"over_end_{w}": _avg(over_end), f"intact_end_{w}": _avg(intact_end),
                      f"exp_{w}": _avg(exp), f"new_{w}": _avg(new_d), f"new_exp_{w}": _avg(newexp_d),
                      f"sel_{w}": _avg(sel_d), f"new_worst_seed_{w}": min(new_d),
                      f"mix_own_{w}": _avg([x["mix_own"] for x in per.values()]),
                      f"mix_kept_{w}": _avg([x["mix_kept"] for x in per.values()])})
        rows[cb] = r

    keys = sorted({k for r in rows.values() for k in r if k != "pre_identical"})

    def group(pred) -> dict:
        g = {k: v for k, v in rows.items() if pred(k, v)}
        out: dict = {"n": len(g), "combos": sorted(g)}
        for k in keys:
            vals = [v[k] for v in g.values() if v.get(k) is not None]
            out[k] = _avg(vals)
            out[f"min_{k}"] = min(vals) if vals else None
            if k.startswith("new_") and vals:
                out[f"argmin_{k}"] = min((v[k], c) for c, v in g.items() if v.get(k) is not None)[1]
        return out

    res = {"rows": rows, "pre_identical": all(r["pre_identical"] for r in rows.values()),
           "over_14bit": group(lambda k, v: v["gap"] >= 0.10 and k.startswith(("flip_m1", "flip7_m1"))),
           "not_over": group(lambda k, v: v["gap"] <= 0.03), "all": group(lambda k, v: True)}
    res["verdict"] = verdict(res)
    return res


def verdict(res: dict) -> dict:
    """12.6.163節で計測前に固定した判定条件。"""
    o, n, a = res["over_14bit"], res["not_over"], res["all"]
    def ge(x, t):  # 値がなければ判定しない（None）
        return None if x is None else bool(x >= t)

    def diff(x, y):
        return None if x is None or y is None else x - y

    out: dict = {}
    for w in ("gain", "need", "sleep"):
        if o.get(f"end_{w}") is None:
            continue
        a1 = ge(o[f"end_{w}"], 0.10)
        a2 = ge(a[f"new_{w}"], -0.02) and ge(a[f"min_new_{w}"], -0.05)
        a3 = ge(n.get(f"end_{w}"), -0.005)
        a4 = ge(diff(a[f"min_new_{w}"], a.get("min_new_keep")), 0.10)
        out[w] = {"A1": a1, "A2": a2, "A3": a3, "A4": a4, "success": bool(a1 and a2 and a3)}
    if "sleep" in out:
        out["sleep"]["A5"] = ge(diff(o["end_sleep"], o.get("end_sleep_buf")), 0.05)
    return out


SUMMARY_KEYS = ("selection_acc", "exp_end", "new_nat", "new_exp", "mix_own", "mix_kept", "mix_size")


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
        res = analyze(args.analyze)
        print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False, indent=1))
        return
    seeds = args.seeds or (EVAL_SEEDS if args.phase == "eval" else DEV_SEEDS)
    out_path = args.out or Path(f"results/context_inference/replay_alloc_{args.phase}.json")
    all_jobs = [(*cfg, s) for cfg in jobs(tuple(args.configs), tuple(args.families)) for s in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(partial(run_one, table=CONFIGS), all_jobs, chunksize=1)
    out: dict = {"環境": env_info(),
                 "説明": f"海馬の痕跡のリプレイの配分（12.6.163〜164節）。phase={args.phase}",
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
        print(key, " ".join(f"{k}={v['mean']:.3f}" for k, v in summ.items()), flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
