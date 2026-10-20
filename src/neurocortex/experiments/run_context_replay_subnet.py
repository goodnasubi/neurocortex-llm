"""文脈でゲートした文脈固有の部分ネットワーク（相補的学習系の容量）と、ach_c・可塑性の分離の組み合わせ（12.6.219〜220節）。

12.6.170節は、共有の重みの可塑性を下げると一般化は保たれるが新しいタスクの学習が遅れ、新しい文脈の列（one-hot の
入力重み）の可塑性を上げてもほとんど変わらないことを示し、原因を「文脈固有の表現の容量が小さすぎて、新しい写像の
大部分を共有の重みでしか表せない」と結論した。ここでは、文脈の枠ごとに K 個の隠れユニットを持ち、文脈の one-hot が
指す枠だけを働かせる部分ネットワーク（context_subnet.GatedModel）を加える。
    ach_c_sub_nov  … ach_c + 部分ネットワーク + nov_split（主仮説。新しい文脈の部分ネットワークを速く、共有を遅く）
    ach_c_sub      … ach_c + 部分ネットワーク（可塑性の分離なし。容量だけの効果）
    base_sub_nov   … 保留サンプルを加えずに部分ネットワーク + nov_split（対照。D5）
    ach_c_wide_nov … ach_c + nov_split で、共有の隠れ層を同じパラメータ数まで広げる（ゲートの対照。G）
base・keep・ach_c は 12.6.167〜168節、ach_c_nov は 12.6.169〜170節の結果ファイル（同じシード・同じ実装で再現する）と
対にして使う。

    --phase dev（開発用シード 100〜104）/ eval（評価用シード 0〜4 で一度だけ）
    結果は results/context_inference/replay_subnet_{phase}.json

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_replay_subnet --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_context_replay_subnet --analyze <結果ファイル> <12.6.167〜170節の結果ファイル>
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
from .run_context_replay_novelty import SUMMARY_KEYS
from .run_context_replay_split import CONFIGS as SPLIT_CONFIGS
from .run_context_replay_split import NOV, _criteria
from .run_context_replay_split import analyze as _split_analyze
from .run_context_writeback import CUE_N, NEW_TASK, SELECT, _exp

DEV_SEEDS = list(range(100, 105))
EVAL_SEEDS = list(range(0, 5))
HIDDEN = 32  # 共有の隠れ層（12.6.167〜170節と同じ）
K = 16  # 文脈の枠ごとの部分ネットワークのユニット数（計測前に固定。共有の隠れ層の半分）
SUB = {"ctx_units": K}
WIDE = {"wide": True}  # 共有の隠れ層を HIDDEN + 枠の数 × K にする（パラメータ数が部分ネットワークと一致）
CONFIGS: dict[str, dict] = {
    "base": SPLIT_CONFIGS["base"],
    "keep": SPLIT_CONFIGS["keep"],
    "ach_c": SPLIT_CONFIGS["ach_c"],
    "ach_c_nov": SPLIT_CONFIGS["ach_c_nov"],
    "ach_c_sub_nov": {**SPLIT_CONFIGS["ach_c"], **NOV, **SUB},
    "ach_c_sub": {**SPLIT_CONFIGS["ach_c"], **SUB},
    "base_sub_nov": {**SPLIT_CONFIGS["base"], **NOV, **SUB},
    "ach_c_wide_nov": {**SPLIT_CONFIGS["ach_c"], **NOV, **WIDE},
}
REUSED = ("base", "keep", "ach_c", "ach_c_nov")  # 12.6.167〜170節の結果ファイルから読む
NEW = ("ach_c_sub_nov", "ach_c_sub", "base_sub_nov", "ach_c_wide_nov")
FIXES: tuple[str, ...] = ()  # 修正（上限3回）は開発用シードの結果を見て追加する
FIX_CANDIDATES: tuple[str, ...] = ()
RULES = ("keep", "ach_c", "ach_c_nov", *NEW)


def jobs(configs=NEW, families=FAMILIES) -> list[tuple]:
    return [(f, m, L, cfg) for m in MECHS_USED for f in families for L in LS for cfg in configs]


def n_slots(fam: str) -> int:
    """run_context_sequence の文脈の枠の数（n_slots 未指定なら タスク数 + 2）。"""
    return len(specs_for(fam)) + 1 + 2


def run_job(job: tuple, kw: dict) -> dict:
    """job = (課題, 機構, L, 設定の名前, シード)。12.6.169節の run_job と同じ量。"""
    fam, mech, L, cfg, seed = job
    torch.set_num_threads(1)
    kw = dict(kw)
    hidden = HIDDEN + n_slots(fam) * K if kw.pop("wide", False) else HIDDEN
    specs = specs_for(fam)
    T0 = len(specs)
    specs = specs + [NEW_TASK[fam]]
    T = len(specs)
    torch.manual_seed(seed)
    r = run_context_sequence("inferred_gated", specs, hidden, torch.Generator().manual_seed(seed + 50_000),
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


def analyze(paths: list[Path], rules=RULES) -> dict:
    """12.6.169節の集計（同じシードの base との差、組の分類も同じ）に、12.6.219節の判定を付ける。
    base のシードに限るので、10シードの結果ファイルと対にしても、ここで測ったシードだけを使う。"""
    seeds: set[int] = set()
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        if any(k.endswith(NEW) for k in d["runs"]):
            seeds.update(d["seeds"])
    tmp = []
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        if seeds:
            for v in d["runs"].values():
                v["per_seed"] = [x for x in v["per_seed"] if x["seed"] in seeds]
        tmp.append(d)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        fs = []
        for i, d in enumerate(tmp):
            f = Path(td) / f"{i}.json"
            f.write_text(json.dumps({"runs": d["runs"]}), encoding="utf-8")
            fs.append(f)
        res = _split_analyze(fs, rules)
    res["seeds"] = sorted(seeds)
    res["verdict"] = verdict(res)
    return res


def verdict(res: dict) -> dict:
    """12.6.219節で計測前に固定した判定条件。"""
    o, n, a = res["over_14bit"], res["not_over"], res["all"]
    out: dict = {}
    for w in dict.fromkeys(("ach_c", "ach_c_nov", *NEW, *FIXES)):
        if o.get(f"end_{w}") is None:
            continue
        v = _criteria(o, n, a, w)
        if w != "ach_c" and o.get("end_ach_c") is not None:
            # D4: ach_c からの上積み（一般化 +0.02 以上、かつ大きく損なった回が増えない）
            v["D4"] = bool(o[f"end_{w}"] - o["end_ach_c"] >= 0.02 and a[f"bad_{w}"] <= a["bad_ach_c"])
        out[w] = v
    m = "ach_c_sub_nov"
    if m in out:
        if o.get("end_base_sub_nov") is not None:
            # D5: 一般化の利得が加えた痕跡から来る
            out[m]["D5"] = bool(o[f"end_{m}"] - o["end_base_sub_nov"] >= 0.05)
        if a.get("bad_ach_c_wide_nov") is not None:
            # G: ゲートが要る（同じパラメータ数の広い共有層より、大きく損なった回が半分以下で、一般化を 0.02 より失わない）
            out[m]["G"] = bool(a[f"bad_{m}"] <= a["bad_ach_c_wide_nov"] / 2
                               and o[f"end_{m}"] >= o["end_ach_c_wide_nov"] - 0.02)
        if a.get("bad_ach_c_nov") is not None:
            # C: 容量が効く（12.6.169節の ach_c_nov より大きく損なった回が半分以下で、一般化を 0.02 より失わない）
            out[m]["C"] = bool(a[f"bad_{m}"] <= a["bad_ach_c_nov"] / 2 and o[f"end_{m}"] >= o["end_ach_c_nov"] - 0.02)
    out["H"] = out.get(m, {}).get("success")
    out["H_sub"] = out.get("ach_c_sub", {}).get("success")
    out["H_fix"] = any(out.get(w, {}).get("success") for w in FIX_CANDIDATES) if FIX_CANDIDATES else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", type=Path, nargs="+", help="結果ファイル（12.6.167〜170節のものも）を集計して判定する")
    ap.add_argument("--configs", nargs="+", default=list(NEW))
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
    out_path = args.out or Path(f"results/context_inference/replay_subnet_{args.phase}.json")
    all_jobs = [(*cfg, s) for cfg in jobs(tuple(args.configs), tuple(args.families)) for s in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(partial(run_one, table=CONFIGS), all_jobs, chunksize=1)
    out: dict = {"環境": env_info(),
                 "説明": f"文脈でゲートした部分ネットワークと ach_c・可塑性の分離（12.6.219〜220節）。phase={args.phase}",
                 "seeds": seeds, "select": {SELECT: SELECTS[SELECT]}, "new_task": {k: repr(v) for k, v in NEW_TASK.items()},
                 "hidden": HIDDEN, "ctx_units": K,
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
