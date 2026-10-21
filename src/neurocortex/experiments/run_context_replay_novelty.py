"""連続的な新しさの信号と、新しい記憶を優先する睡眠の段階で、取り戻した一般化と新しいタスクの学習を両立できるか（12.6.167〜168節）。

12.6.164節では、リプレイの配分を変えても新しいタスクの最悪の組を守れなかった。最善の ach_need2 も、新しい文脈が確立した
直後の脆い時期を二値の驚きでは区別できず、評価用シードの2シードで新しい写像が削られた。ここでは:
    ach_c       … 連続的な新しさの信号（文脈ごとの損失のゆっくりした移動平均。アセチルコリンの連続的な調節）で、
                  痕跡の固定化の配分を少しずつ増やす
    nrem        … 新しい記憶（新しい文脈の痕跡）の NREM に似たリプレイと、古い記憶の REM に似たリプレイを交互に置く
                  睡眠の段階（Singh, Norman & Schapiro 2022）
    ach_c_nrem  … 両者の組み合わせ
    nrem_buf    … nrem の対照（保留サンプルを加えない）
を、base（保留サンプルをリプレイしない）・keep（一様）・ach_need2（12.6.163節の修正2）と比べる。
最悪の組の判定が1〜2シードに依存しないよう、シードを 10 個に増やし、1回ごとの差の分布でも判定する（12.6.167節）。

    --phase dev（開発用シード 100〜109）/ eval（評価用シード 0〜9 で一度だけ）
    結果は results/context_inference/replay_novelty_{phase}.json

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_replay_novelty --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_context_replay_novelty --analyze <結果ファイル>
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
from .run_context_gated_long import native_contexts
from .run_context_replay_alloc import CONFIGS as ALLOC_CONFIGS
from .run_context_replay_alloc import KEEP
from .run_context_writeback import CUE_N, NEW_TASK, SELECT, WB_KW, _avg, _exp

DEV_SEEDS = list(range(100, 110))  # 12.6.164節の5シードから増やす（計測前に固定）
EVAL_SEEDS = list(range(0, 10))
SLEEP = {"sleep_every": 50, "sleep_steps": 25}  # 12.6.163節の sleep と同じ量
CONFIGS: dict[str, dict] = {
    "base": ALLOC_CONFIGS["base"],
    "keep": ALLOC_CONFIGS["keep"],
    "ach_need2": ALLOC_CONFIGS["ach_need2"],
    "ach_c": {**KEEP, "replay_rule": "ach_c", "replay_need": 0.5, "ach_ema": 0.98},
    "nrem": {**KEEP, "replay_rule": "nrem", **SLEEP},
    "nrem_buf": {"writeback": "none", **WB_KW, "replay_rule": "nrem", **SLEEP},
    "ach_c_nrem": {**KEEP, "replay_rule": "ach_c_nrem", "replay_need": 0.5, "ach_ema": 0.98, **SLEEP},
}
# 修正1（12.6.167節）: nrem は加えた痕跡（まだ学習していない保留サンプル）を REM のステップでまとめて学び、新しいタスクを
# 大きく損なった（nrem_buf は中立）。睡眠の段階の痕跡の出どころを Singh et al. の海馬と新皮質の分業に合わせる
CONFIGS.update({"nrem2": {**CONFIGS["nrem"], "replay_rule": "nrem2"},
                "ach_c_nrem2": {**CONFIGS["ach_c_nrem"], "replay_rule": "ach_c_nrem2"}})
FIX1 = ("nrem2", "ach_c_nrem2")
NEW_RULES = ("ach_c", "nrem", "ach_c_nrem", *FIX1)
RULES = ("keep", "ach_need2", *NEW_RULES, "nrem_buf")
BAD = -0.10  # 1回ごとの新しいタスクの差がこれを下回ると「大きく損なった」回とする
# 修正は開発用シードの結果を見て追加する（上限3回）


def jobs(configs=tuple(c for c in CONFIGS if c not in FIX1), families=FAMILIES) -> list[tuple]:
    return [(f, m, L, cfg) for m in MECHS_USED for f in families for L in LS for cfg in configs]


def run_job(job: tuple, kw: dict) -> dict:
    """job = (課題, 機構, L, 設定の名前, シード)。12.6.163節の run_job と同じ量に、なじみの度合いを加える。"""
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
           "mix_own": mix.get("own"), "mix_kept": mix.get("kept"), "mix_size": mix.get("size"),
           "mix_fam": mix.get("fam"), "sleep_steps": mix.get("sleep")}
    return {"key": f"{fam}/{mech}_L{L}/{cfg}", "row": row}


def run_one(job: tuple, table: dict) -> dict:
    return run_job(job, table[job[3]])


def _group(rows: dict, runs_d: dict, pred, rules: tuple = RULES) -> dict:
    """組の集合ごとの平均・最悪の組・1回ごとの差の分布。"""
    g = {k: v for k, v in rows.items() if pred(k, v)}
    out: dict = {"n": len(g), "combos": sorted(g)}
    keys = sorted({k for v in g.values() for k in v if k != "pre_identical"})
    for k in keys:
        vals = [v[k] for v in g.values() if v.get(k) is not None]
        out[k] = _avg(vals)
        if k.startswith("new_") and vals:
            out[f"min_{k}"] = min(vals)
            out[f"argmin_{k}"] = min((v[k], c) for c, v in g.items() if v.get(k) is not None)[1]
    for w in rules:
        d = sorted(x for c in g for x in runs_d.get(c, {}).get(w, []))
        if d:
            n = len(d)
            out[f"runs_{w}"] = n
            out[f"bad_{w}"] = sum(x < BAD for x in d)
            out[f"bad_frac_{w}"] = sum(x < BAD for x in d) / n
            out[f"q05_new_{w}"] = d[int(0.05 * n)]
            out[f"q10_new_{w}"] = d[int(0.10 * n)]
            out[f"min_run_new_{w}"] = d[0]
    return out


def analyze(paths: list[Path], rules: tuple = RULES, verdict_fn=None) -> dict:
    """各規則は同じシードの base と対にして差をとる。組の分類は 12.6.161節と同じ（base の gap、正確度の単位）。
    rules・verdict_fn は 12.6.225節（run_context_replay_uncertainty）が別の規則の組で使う。"""
    runs: dict = {}
    for p in paths:
        runs.update(json.loads(p.read_text(encoding="utf-8"))["runs"])
    combos = sorted({k.rsplit("/", 1)[0] for k in runs})
    rows: dict = {}
    runs_d: dict = {}  # 組 → 規則 → 1回ごとの新しいタスクの差
    for cb in combos:
        base = {x["seed"]: x for x in runs.get(f"{cb}/base", {}).get("per_seed", [])}
        if not base:
            continue
        r: dict = {"gap": _avg([_avg(x["gap"]) for x in base.values()]),
                   "end_base": _avg([_avg(x["end_nat"]) for x in base.values()]),
                   "new_base": _avg([x["new_nat"] for x in base.values()]), "pre_identical": True}
        runs_d[cb] = {}
        for w in rules:
            per = {x["seed"]: x for x in runs.get(f"{cb}/{w}", {}).get("per_seed", [])}
            if not per:
                continue
            end, over_end, intact_end, exp, new_d, sel_d = ([] for _ in range(6))
            for s, x in per.items():
                b = base[s]
                r["pre_identical"] &= x["pre_nat"] == b["pre_nat"]
                de = [a - c for a, c in zip(x["end_nat"], b["end_nat"])]
                end.append(_avg(de))
                over_end.append(_avg([de[i] for i, g in enumerate(b["gap"]) if g >= 0.10]))
                intact_end.append(_avg([de[i] for i, g in enumerate(b["gap"]) if g <= 0.03]))
                exp.append(x["exp_end"] - b["exp_end"])
                new_d.append(x["new_nat"] - b["new_nat"])
                sel_d.append(x["selection_acc"] - b["selection_acc"])
            runs_d[cb][w] = new_d
            r.update({f"end_{w}": _avg(end), f"over_end_{w}": _avg(over_end), f"intact_end_{w}": _avg(intact_end),
                      f"exp_{w}": _avg(exp), f"new_{w}": _avg(new_d), f"sel_{w}": _avg(sel_d),
                      f"mix_own_{w}": _avg([x["mix_own"] for x in per.values()]),
                      f"mix_kept_{w}": _avg([x["mix_kept"] for x in per.values()]),
                      f"mix_fam_{w}": _avg([x.get("mix_fam") for x in per.values()]),
                      f"sleep_{w}": _avg([x.get("sleep_steps") for x in per.values()])})
        rows[cb] = r

    def over(k, v):
        return v["gap"] >= 0.10 and k.startswith(("flip_m1", "flip7_m1"))

    def intact(k, v):
        return v["gap"] <= 0.03

    def mech(m):
        return lambda k, v: k.split("/")[1].startswith(f"{m}_")

    res = {"rows": rows, "pre_identical": all(r["pre_identical"] for r in rows.values()),
           "over_14bit": _group(rows, runs_d, over, rules), "not_over": _group(rows, runs_d, intact, rules),
           "all": _group(rows, runs_d, lambda k, v: True, rules)}
    for m in MECHS_USED:
        res[f"{m}_over_14bit"] = _group(rows, runs_d, lambda k, v, f=mech(m): f(k, v) and over(k, v), rules)
        res[f"{m}_not_over"] = _group(rows, runs_d, lambda k, v, f=mech(m): f(k, v) and intact(k, v), rules)
        res[f"{m}_all"] = _group(rows, runs_d, mech(m), rules)
    res["verdict"] = (verdict_fn or verdict)(res)
    return res


def _ge(x, t):  # 値がなければ判定しない（None）
    return None if x is None else bool(x >= t)


def _criteria(o: dict, n: dict, a: dict, w: str) -> dict:
    """B1〜B3（12.6.167節）。o・n・a は上書きあり・上書きなし・全体の組の集計。"""
    b1 = _ge(o.get(f"end_{w}"), 0.10)
    b2 = None
    if a.get(f"new_{w}") is not None:
        b2 = bool(a[f"new_{w}"] >= -0.02 and a[f"min_new_{w}"] >= -0.05 and a[f"bad_frac_{w}"] <= 0.01)
    b3 = _ge(n.get(f"end_{w}"), -0.005)
    return {"B1": b1, "B2": b2, "B3": b3, "success": bool(b1 and b2 and b3)}


def verdict(res: dict) -> dict:
    """12.6.167節で計測前に固定した判定条件。"""
    o, n, a = res["over_14bit"], res["not_over"], res["all"]
    out: dict = {}
    for w in ("ach_need2", *NEW_RULES):
        if o.get(f"end_{w}") is None:
            continue
        v = _criteria(o, n, a, w)
        ref = a.get("bad_ach_need2")
        v["B4"] = None if ref is None or w == "ach_need2" else bool(a[f"bad_{w}"] <= ref / 2)
        sp = _criteria(res["split_over_14bit"], res["split_not_over"], res["split_all"], w)
        v["H3_split"] = sp
        out[w] = v
    if o.get("end_nrem") is not None and o.get("end_nrem_buf") is not None:
        for w in ("nrem", "nrem2"):  # nrem2 の痕跡なしの版は nrem_buf と同じ学習になる
            if w in out:
                out[w]["B5"] = bool(o[f"end_{w}"] - o["end_nrem_buf"] >= 0.05)
    out["H1"] = out.get("ach_c", {}).get("success")
    out["H2"] = out.get("nrem", {}).get("success")
    out["H"] = any(out.get(w, {}).get("success") for w in NEW_RULES)
    out["H3"] = any(out.get(w, {}).get("H3_split", {}).get("success") for w in NEW_RULES)
    return out


SUMMARY_KEYS = ("selection_acc", "exp_end", "new_nat", "new_exp", "mix_own", "mix_kept", "mix_size", "mix_fam")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", type=Path, nargs="+", help="結果ファイルを集計して判定に使う量を表示する")
    ap.add_argument("--configs", nargs="+", default=[c for c in CONFIGS if c not in FIX1])
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
    out_path = args.out or Path(f"results/context_inference/replay_novelty_{args.phase}.json")
    all_jobs = [(*cfg, s) for cfg in jobs(tuple(args.configs), tuple(args.families)) for s in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(partial(run_one, table=CONFIGS), all_jobs, chunksize=1)
    out: dict = {"環境": env_info(),
                 "説明": f"連続的な新しさの信号と新しい記憶を優先する睡眠の段階（12.6.167〜168節）。phase={args.phase}",
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
