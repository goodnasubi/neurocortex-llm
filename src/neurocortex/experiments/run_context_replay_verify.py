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


def _m(runs: dict, key: str, k: str) -> float | None:
    v = runs.get(key, {}).get("summary", {}).get(k)
    return v["mean"] if v else None


def _spearman(x: list[float], y: list[float]) -> float:
    rank = lambda v: torch.tensor(v).argsort().argsort().float()  # noqa: E731  （同順位はほぼない）
    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm()))


def analyze_replay(path: Path) -> dict:
    """12.6.151節の H1〜H3 の判定に使う量。"""
    runs = json.loads(path.read_text(encoding="utf-8"))["runs"]
    combos = []
    for mech in REPLAY_MECHS:
        for fam in REPLAY_FAMILIES:
            for L in LS:
                c = f"{fam}/{mech}_L{L}"
                k16 = f"{c}/cue16/active"
                row = {"combo": c, **{k: _m(runs, k16, k) for k in ("nat_own_buf", "nat_fresh", "nat_other_buf",
                                                                      "nat_holdout", "native_acc")}}
                row["gap"] = row["nat_own_buf"] - row["nat_fresh"]
                for n in REPLAY_CUE_NS:
                    for sel in REPLAY_SELECTS:
                        row[f"{sel}_{n}"] = _m(runs, f"{c}/cue{n}/{sel}", "selection_acc")
                        row[f"{sel}_{n}_exp"] = _m(runs, f"{c}/cue{n}/{sel}", "expected_acc")
                    row[f"gain_replay_{n}"] = row[f"gen_replay_{n}"] - row[f"active_{n}"]
                    row[f"gain_holdout_{n}"] = row[f"gen_holdout_{n}"] - row[f"active_{n}"]
                combos.append(row)
    over = [r for r in combos if r["gap"] >= 0.10]
    intact = [r for r in combos if r["gap"] <= 0.03]
    rep = [r for r in over if r["combo"].split("_L")[0] != "flip_m1/reset"]
    avg = lambda rs, k: sum(r[k] for r in rs) / len(rs) if rs else None  # noqa: E731
    out = {"combos": combos, "n_over": len(over), "n_intact": len(intact), "n_replication": len(rep),
           "over": [r["combo"] for r in over], "replication": [r["combo"] for r in rep]}
    for n in (8, 16):
        out[f"over_gain_replay_{n}"] = avg(over, f"gain_replay_{n}")
        out[f"rep_gain_replay_{n}"] = avg(rep, f"gain_replay_{n}")
        out[f"intact_gain_replay_{n}"] = avg(intact, f"gain_replay_{n}")
        out[f"over_gain_holdout_{n}"] = avg(over, f"gain_holdout_{n}")
    out["spearman_gap_gain"] = _spearman([r["gap"] for r in combos],
                                         [(r["gain_replay_8"] + r["gain_replay_16"]) / 2 for r in combos])
    for k in ("nat_own_buf", "nat_fresh", "nat_other_buf", "nat_holdout"):
        out[f"over_{k}"] = avg(over, k)
    return out


def frontier(points: list[tuple[float, float]], x: float) -> float:
    """(手がかりの数, 選択精度) の点を手がかりの数の順に結んだ折れ線の x での値（範囲の外は端の値）。"""
    pts = sorted(points)
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else max(y0, y1)
    return pts[-1][1]


def analyze_verify(path: Path) -> dict:
    """12.6.151節の場面の分類と H4・H5 の判定に使う量。"""
    runs = json.loads(path.read_text(encoding="utf-8"))["runs"]
    out: dict = {}
    for fam, mech in VERIFY_SCENARIOS:
        rows = []
        for L in LS:
            for n in VERIFY_CUE_NS:
                c = f"{fam}/{mech}_L{L}/cue{n}"
                row = {"L": L, "cue": n, **{sel: _m(runs, f"{c}/{sel}", "selection_acc") for sel in VERIFY_SELECTS},
                       **{f"{sel}_cues": _m(runs, f"{c}/{sel}", "cues_used") for sel in VERIFY_SELECTS},
                       "rival_share": _m(runs, f"{c}/active", "rival_share"),
                       "native_second_frac": _m(runs, f"{c}/active", "native_second_frac"),
                       "wrong_native_rank": _m(runs, f"{c}/active", "wrong_native_rank")}
                pts = [(row[f"{s}_cues"], row[s]) for s in VERIFY_SELECTS if s == "active" or s.startswith("cautious")]
                for v in ("verify_top2", "verify_all", "verify_each"):
                    row[f"{v}_over_frontier"] = row[v] - frontier(pts, row[f"{v}_cues"])
                rows.append(row)
        big = [r for r in rows if r["cue"] in (8, 16)]
        avg = lambda k: (sum(r[k] for r in big if r[k] is not None)  # noqa: E731
                         / max(1, sum(r[k] is not None for r in big)))
        s = {k: avg(k) for k in ("rival_share", "native_second_frac", "wrong_native_rank", "verify_top2_over_frontier",
                                 "verify_all_over_frontier", "verify_each_over_frontier")}
        s["each_minus_top2"] = avg("verify_each") - avg("verify_top2")
        s["dispersed"] = s["rival_share"] <= 0.7 and s["native_second_frac"] <= 0.7
        out[f"{fam}/{mech}"] = {"summary": s, "rows": rows}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", type=Path, help="結果ファイルを集計して判定に使う量を表示する")
    ap.add_argument("--part", choices=tuple(PARTS), default="replay")
    ap.add_argument("--phase", choices=("dev", "eval"), default="dev")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--selects", nargs="+", default=None)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.analyze:
        res = (analyze_replay if args.part == "replay" else analyze_verify)(args.analyze)
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return
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
