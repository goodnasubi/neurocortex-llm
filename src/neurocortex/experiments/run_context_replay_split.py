"""連続的な新しさの信号（ach_c）と、文脈固有と共有の可塑性の分離の組み合わせ（12.6.169〜170節）。

12.6.168節の ach_c は新しいタスクの保護を大きく改善したが、取り戻した一般化は +0.081 に下がった（基準 +0.10）。
12.6.164節は、新しい文脈の側の可塑性を上げる機構との組み合わせが必要だと結論した。ここでは、12.6.129・132節の
可塑性の分離（新しい文脈の列の可塑性を上げ、共有の重みの可塑性を下げる）を、固定の期間ではなく ach_c と同じ
なじみの度合い f で続ける（nov_split。新しい記憶の割り当てと興奮性、メタ可塑性の単純化）。
    ach_c_nov      … ach_c + nov_split（主仮説）
    base_nov       … 保留サンプルを加えずに nov_split だけ（対照）
    ach_c_nov_ctx  … 新しい文脈の列の可塑性を上げるだけ（切り分け）
    ach_c_nov_sh   … 共有の重みの可塑性を下げるだけ（切り分け）
    ach_c_nov_half … 修正2（共有の利得 0.5）
base・keep・ach_c は 12.6.167〜168節の結果ファイル（同じシード・同じ実装で再現する）と対にして使う。

    --phase dev（開発用シード 100〜109）/ eval（評価用シード 0〜9 で一度だけ）
    結果は results/context_inference/replay_split_{phase}.json

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_replay_split --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_context_replay_split --analyze <結果ファイル> <12.6.167〜168節の結果ファイル>
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
from .run_context_replay_novelty import CONFIGS as NOVELTY_CONFIGS
from .run_context_replay_novelty import DEV_SEEDS, EVAL_SEEDS, SUMMARY_KEYS
from .run_context_writeback import CUE_N, NEW_TASK, SELECT, _avg, _exp

# 12.6.129節の利得（g_ctx = 2.0、g_shared = 0.2）と 12.6.132節の下限 0.01（lr = 0.05 では効かない）。計測前に固定
NOV = {"nov_split": True, "nov_ctx_gain": 2.0, "nov_shared_gain": 0.2, "split_shared_lr_floor": 0.01}
CONFIGS: dict[str, dict] = {
    "base": NOVELTY_CONFIGS["base"],
    "keep": NOVELTY_CONFIGS["keep"],
    "ach_c": NOVELTY_CONFIGS["ach_c"],
    "ach_c_nov": {**NOVELTY_CONFIGS["ach_c"], **NOV},
    "base_nov": {**NOVELTY_CONFIGS["base"], **NOV},
    "ach_c_nov_ctx": {**NOVELTY_CONFIGS["ach_c"], **NOV, "nov_shared_gain": 1.0},
    "ach_c_nov_sh": {**NOVELTY_CONFIGS["ach_c"], **NOV, "nov_ctx_gain": 1.0},
}
REUSED = ("base", "keep", "ach_c")  # 12.6.167〜168節の結果ファイルから読む
NEW = ("ach_c_nov", "base_nov", "ach_c_nov_ctx", "ach_c_nov_sh")
# 修正（12.6.169節。上限3回）。修正1: 切り分けとして測った ach_c_nov_ctx（新しい文脈の列の可塑性を上げるだけ）を
# 候補にする（追加の計測なし）。修正2: 共有の利得を 0.2 と 1 の中間の 0.5 にする（新しいタスクの学習の遅れを減らし、
# 共有を抑えることによる一般化の上積みを一部残す）
CONFIGS["ach_c_nov_half"] = {**CONFIGS["ach_c_nov"], "nov_shared_gain": 0.5}
FIXES: tuple[str, ...] = ("ach_c_nov_ctx", "ach_c_nov_half")
# 確定の規則（修正2の計測の前に決めた）: 開発用シードで D1〜D3 を満たした修正だけを H_fix の候補にする。
# 修正2は −0.10 未満が 7 回で D2 不合格だったので、評価用シードでは参考として測るだけ
FIX_CANDIDATES: tuple[str, ...] = ("ach_c_nov_ctx",)
RULES = ("keep", "ach_c", *NEW, "ach_c_nov_half")
BAD = -0.10  # 1回ごとの新しいタスクの差がこれを下回ると「大きく損なった」回とする（12.6.167節と同じ）


def jobs(configs=NEW, families=FAMILIES) -> list[tuple]:
    return [(f, m, L, cfg) for m in MECHS_USED for f in families for L in LS for cfg in configs]


def run_job(job: tuple, kw: dict) -> dict:
    """job = (課題, 機構, L, 設定の名前, シード)。12.6.167節の run_job と同じ量に、分離の利得の診断を加える。"""
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
    nov = d.get("nov_split", {}).get(T0, {})
    row = {"seed": seed, "n_contexts": r["n_contexts"], "native": nat,
           "selection_acc": sum(c == nat[j] for j in orig for c in ch[j]) / (T0 * SELECTION_TRIALS),
           "exp_end": _exp(acc, ch, orig), "end_nat": [2 * acc[j][nat[j]] - 1 for j in orig],
           "pre_nat": [2 * w["pre"]["fresh"][j][ci[j]] - 1 for j in orig],
           "gap": [w["pre"]["own_buf"][j][ci[j]] - w["pre"]["fresh"][j][ci[j]] for j in orig],
           "new_nat": 2 * acc[T0][nat[T0]] - 1, "new_exp": _exp(acc, ch, [T0]), "new_ctx": nat[T0] not in w["used"],
           "mix_own": mix.get("own"), "mix_kept": mix.get("kept"), "mix_size": mix.get("size"),
           "mix_fam": mix.get("fam"), "nov_steps": nov.get("steps"), "nov_g_shared": nov.get("g_shared")}
    return {"key": f"{fam}/{mech}_L{L}/{cfg}", "row": row}


def run_one(job: tuple, table: dict) -> dict:
    return run_job(job, table[job[3]])


def _group(rows: dict, runs_d: dict, pred, rules=RULES) -> dict:
    """組の集合ごとの平均・最悪の組・1回ごとの差の分布（12.6.167節と同じ）。"""
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
            out[f"min_run_new_{w}"] = d[0]
    return out


def analyze(paths: list[Path], rules=RULES) -> dict:
    """各規則は同じシードの base と対にして差をとる。組の分類は 12.6.161節と同じ（base の gap、正確度の単位）。
    同じキーが複数のファイルにあれば、後のファイルを優先する。"""
    runs: dict = {}
    for p in paths:
        runs.update(json.loads(p.read_text(encoding="utf-8"))["runs"])
    combos = sorted({k.rsplit("/", 1)[0] for k in runs})
    rows: dict = {}
    runs_d: dict = {}
    for cb in combos:
        base = {x["seed"]: x for x in runs.get(f"{cb}/base", {}).get("per_seed", [])}
        if not base:
            continue
        r: dict = {"gap": _avg([_avg(x["gap"]) for x in base.values()]),
                   "end_base": _avg([_avg(x["end_nat"]) for x in base.values()]),
                   "new_base": _avg([x["new_nat"] for x in base.values()]), "pre_identical": True}
        runs_d[cb] = {}
        for w in rules:
            per = {x["seed"]: x for x in runs.get(f"{cb}/{w}", {}).get("per_seed", []) if x["seed"] in base}
            if not per:
                continue
            end, over_end, intact_end, new_d = [], [], [], []
            for s, x in per.items():
                b = base[s]
                r["pre_identical"] &= x["pre_nat"] == b["pre_nat"]
                de = [a - c for a, c in zip(x["end_nat"], b["end_nat"])]
                end.append(_avg(de))
                over_end.append(_avg([de[i] for i, g in enumerate(b["gap"]) if g is not None and g >= 0.10]))
                intact_end.append(_avg([de[i] for i, g in enumerate(b["gap"]) if g is not None and g <= 0.03]))
                new_d.append(x["new_nat"] - b["new_nat"])
            runs_d[cb][w] = new_d
            r.update({f"end_{w}": _avg(end), f"over_end_{w}": _avg(over_end), f"intact_end_{w}": _avg(intact_end),
                      f"new_{w}": _avg(new_d), f"seeds_{w}": len(per),
                      f"mix_kept_{w}": _avg([x["mix_kept"] for x in per.values()]),
                      f"mix_fam_{w}": _avg([x.get("mix_fam") for x in per.values()]),
                      f"nov_steps_{w}": _avg([x.get("nov_steps") for x in per.values()]),
                      f"nov_g_{w}": _avg([x.get("nov_g_shared") for x in per.values()])})
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
    res["verdict"] = verdict(res)
    return res


def _ge(x, t):  # 値がなければ判定しない（None）
    return None if x is None else bool(x >= t)


def _criteria(o: dict, n: dict, a: dict, w: str) -> dict:
    """D1〜D3（12.6.169節。12.6.167節の B1〜B3 と同じ基準）。"""
    d1 = _ge(o.get(f"end_{w}"), 0.10)
    d2 = None
    if a.get(f"new_{w}") is not None:
        d2 = bool(a[f"new_{w}"] >= -0.02 and a[f"min_new_{w}"] >= -0.05 and a[f"bad_frac_{w}"] <= 0.01)
    d3 = _ge(n.get(f"end_{w}"), -0.005)
    return {"D1": d1, "D2": d2, "D3": d3, "success": bool(d1 and d2 and d3)}


def verdict(res: dict) -> dict:
    """12.6.169節で計測前に固定した判定条件。"""
    o, n, a = res["over_14bit"], res["not_over"], res["all"]
    out: dict = {}
    for w in dict.fromkeys(("ach_c", "ach_c_nov", "ach_c_nov_ctx", "ach_c_nov_sh", *FIXES)):
        if o.get(f"end_{w}") is None:
            continue
        v = _criteria(o, n, a, w)
        if w != "ach_c" and o.get("end_ach_c") is not None:
            # D4: ach_c からの上積み（一般化 +0.02 以上、かつ大きく損なった回が増えない）
            v["D4"] = bool(o[f"end_{w}"] - o["end_ach_c"] >= 0.02 and a[f"bad_{w}"] <= a["bad_ach_c"])
        out[w] = v
    if o.get("end_ach_c_nov") is not None and o.get("end_base_nov") is not None:
        out["ach_c_nov"]["D5"] = bool(o["end_ach_c_nov"] - o["end_base_nov"] >= 0.05)
    out["H"] = out.get("ach_c_nov", {}).get("success")
    out["H_fix"] = any(out.get(w, {}).get("success") for w in FIX_CANDIDATES) if FIX_CANDIDATES else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", type=Path, nargs="+", help="結果ファイル（12.6.167〜168節のものも）を集計して判定する")
    ap.add_argument("--configs", nargs="+", default=[*NEW, "ach_c_nov_half"])
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
    out_path = args.out or Path(f"results/context_inference/replay_split_{args.phase}.json")
    all_jobs = [(*cfg, s) for cfg in jobs(tuple(args.configs), tuple(args.families)) for s in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(partial(run_one, table=CONFIGS), all_jobs, chunksize=1)
    out: dict = {"環境": env_info(),
                 "説明": f"ach_c と文脈固有・共有の可塑性の分離の組み合わせ（12.6.169〜170節）。phase={args.phase}",
                 "seeds": seeds, "select": {SELECT: SELECTS[SELECT]}, "new_task": {k: repr(v) for k, v in NEW_TASK.items()},
                 "configs": {k: CONFIGS[k] for k in args.configs}, "selection_trials": SELECTION_TRIALS, "runs": {}}
    for key in dict.fromkeys(x["key"] for x in results):
        rows = [x["row"] for x in results if x["key"] == key]
        summ = {}
        for k in (*SUMMARY_KEYS, "nov_steps", "nov_g_shared"):
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
