"""海馬のバッファから候補を取る効果の検証と、対立仮説が複数残る場面での検証つきの打ち切り（12.6.151〜152節）。

(1) 12.6.148節の探索的な結果（gen_replay で上書きされた文脈の過信が減る）を計測前の仮説として確かめる。候補の
出どころを、与えられた手がかり（active）・海馬のバッファ（gen_replay）・リプレイに使わない保留サンプル
（gen_holdout。同じ時点・同じ整合フィルタで保存し、リプレイだけしない）で比べ、最終モデルで各文脈の正確度を入力の
出どころ（自分のバッファ・他の文脈のバッファ・保留サンプル・新しい入力）ごとに切り分ける。課題は 5 族 × 作り直し
（reset）・分離（split）× L = 0/250/500。

(2) 対立仮説が複数残る場面（似たタスクを 7 個に増やした flip7_m1）と従来の flip_m1 で、検証の作り方（top2・all・
each）と、閾値を上げるだけの統制（α = 1e-5・1e-7・1e-9・1e-12。速さと正確さの曲線）を比べる。

    --part replay / verify、--phase dev（開発用シード）/ eval（評価用シードで一度だけ）
    結果は results/context_inference/replay_verify_{part}_{phase}.json

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_replay_verify --part replay --phase dev
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import torch

from ..context_inference import FlipSpec, run_context_sequence
from ..continual_metrics import forgetting_metrics
from ..run_env import env_info
from .run_context_evidence_info import FIX2, MECHS, SELECTION_TRIALS
from .run_context_gated_long import DEV_SEEDS, EVAL_SEEDS, native_contexts
from .run_context_split_generalize import family_specs

HOLDOUT = 50  # 保留サンプル（文脈ごと。バッファと同じ数）
ACTIVE = {**FIX2, "active_pool": 4, "reliability_holdout": HOLDOUT}
LS = (0, 250, 500)

# (1) 候補の出どころ
REPLAY_FAMILIES = ("parity", "flip_m1", "flip_m2", "cue_q0.9", "cue_q0.75")
REPLAY_MECHS = ("reset", "split")
REPLAY_CUE_NS = (4, 8, 16)
REPLAY_SELECTS: dict[str, dict] = {
    "batch": {"cue_select": "batch", "reliability_holdout": HOLDOUT},
    "active": ACTIVE,
    "gen_replay": {**ACTIVE, "active_source": "replay"},
    "gen_holdout": {**ACTIVE, "active_source": "holdout"},
}

# (2) 検証つきの打ち切り
VERIFY_SCENARIOS = (("flip_m1", "reset"), ("flip7_m1", "reset"))
VERIFY_CUE_NS = (4, 8, 16)
CAUTIOUS_ALPHAS = (1e-5, 1e-7, 1e-9, 1e-12)
VERIFY_SELECTS: dict[str, dict] = {
    "batch": {"cue_select": "batch", "reliability_holdout": HOLDOUT},
    "active": ACTIVE,
    "verify_top2": {**ACTIVE, "active_verify": 2},
    "verify_all": {**ACTIVE, "active_verify": 2, "verify_mode": "all"},
    "verify_each": {**ACTIVE, "active_verify": 1, "verify_mode": "each"},
    **{f"cautious_{a:.0e}": {**ACTIVE, "evidence_alpha": a} for a in CAUTIOUS_ALPHAS},
}
# 修正は開発用シードの結果を見て追加する（上限3回）
PARTS = {"replay": REPLAY_SELECTS, "verify": VERIFY_SELECTS}


def specs_for(family: str) -> list:
    """flip7_m1 … 似たタスク m = 1 を 7 個に増やした課題（基本タスクと、ビット 2〜7 のそれぞれで反転する 6 個）。"""
    if family == "flip7_m1":
        return [FlipSpec(14)] + [FlipSpec(14, flip=(2 + i,)) for i in range(6)]
    return family_specs(family)[0]


def jobs(part: str, selects) -> list[tuple]:
    if part == "replay":
        combos = [(f, m) for m in REPLAY_MECHS for f in REPLAY_FAMILIES]
        cue_ns = REPLAY_CUE_NS
    else:
        combos, cue_ns = list(VERIFY_SCENARIOS), VERIFY_CUE_NS
    return [(fam, mech, L, n, sel) for fam, mech in combos for L in LS for n in cue_ns for sel in selects]


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def run_job(job: tuple, select_kw: dict) -> dict:
    """job = (課題, 機構, L, 手がかりの数, 規則の名前, シード)。"""
    fam, mech, L, cue_n, sel, seed = job
    torch.set_num_threads(1)
    specs = specs_for(fam)
    torch.manual_seed(seed)
    r = run_context_sequence("inferred_gated", specs, 32, torch.Generator().manual_seed(seed + 50_000),
                             torch.Generator().manual_seed(seed + 90_000), transition_steps=L, steps=500,
                             diagnostics=True, cue_n=cue_n, selection_trials=SELECTION_TRIALS, **MECHS[mech],
                             **select_kw)
    d, T = r["diag"], len(specs)
    nat = native_contexts(r["assign"], T)
    acc = d["ctx_acc"][-1]
    ch, cu = r["selection_choices"], r["selection_cues"]
    n_tr = T * SELECTION_TRIALS
    m = forgetting_metrics(r["R"], chance=0.5)
    oa = d["origin_acc"]
    row = {"seed": seed, "n_contexts": r["n_contexts"], "ACC": m["ACC"], "native": nat,
           "selection_acc": sum(c == nat[j] for j in range(T) for c in ch[j]) / n_tr,
           "cues_used": sum(sum(x) for x in cu) / n_tr,
           "expected_acc": sum(2 * acc[j][c] - 1 for j in range(T) for c in ch[j]) / n_tr,
           "native_acc": sum(2 * acc[j][nat[j]] - 1 for j in range(T)) / T,
           # 本来の文脈の、入力の出どころごとの正確度（タスクの平均。fresh は新しい入力）
           "nat_fresh": _mean([acc[j][nat[j]] for j in range(T)]),
           **{f"nat_{k}": _mean([oa[j][nat[j]][k] for j in range(T)]) for k in ("own_buf", "other_buf", "holdout")}}
    if "selection_rival_share" in r:
        sh = [x for j in range(T) for x in r["selection_rival_share"][j] if x is not None]
        row["rival_share"] = _mean(sh)
        wrong = [(j, o) for j in range(T) for c, o in zip(ch[j], r["selection_order"][j])
                 if c != nat[j] and o is not None]
        row["n_wrong_hit"] = len(wrong)
        row["native_second_frac"] = (sum(o[1] == nat[j] for j, o in wrong if len(o) > 1) / len(wrong)
                                     if wrong else None)
        row["wrong_native_rank"] = _mean([o.index(nat[j]) + 1 for j, o in wrong if nat[j] in o])
    return {"key": f"{fam}/{mech}_L{L}/cue{cue_n}/{sel}", "row": row}


def run_one(job: tuple, table: dict) -> dict:
    return run_job(job, table[job[4]])


SUMMARY_KEYS = ("ACC", "selection_acc", "cues_used", "expected_acc", "native_acc", "nat_fresh", "nat_own_buf",
                "nat_other_buf", "nat_holdout", "rival_share", "native_second_frac", "wrong_native_rank")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=tuple(PARTS), default="replay")
    ap.add_argument("--phase", choices=("dev", "eval"), default="dev")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--selects", nargs="+", default=None)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    table = PARTS[args.part]
    selects = args.selects or list(table)
    seeds = args.seeds or (EVAL_SEEDS if args.phase == "eval" else DEV_SEEDS)
    out_path = args.out or Path(f"results/context_inference/replay_verify_{args.part}_{args.phase}.json")
    all_jobs = [(*cfg, s) for cfg in jobs(args.part, selects) for s in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(partial(run_one, table=table), all_jobs, chunksize=1)
    out: dict = {"環境": env_info(),
                 "説明": f"バッファの候補の検証と複数の対立仮説での検証（12.6.151〜152節）。part={args.part} phase={args.phase}",
                 "seeds": seeds, "selects": {k: table[k] for k in selects},
                 "selection_trials": SELECTION_TRIALS, "runs": {}}
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
