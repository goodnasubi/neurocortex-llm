"""再固定化の学び直しを共有の重みに書き戻す（系統的な再固定化）と、バッファでの学び直しが効く理由の切り分け
（12.6.161〜162節）。

12.6.160節では、選んだ文脈の痕跡で最終モデルの複製を学び直すと、上書きされた一般化がかなり戻った（テスト時の一時的な
複製）。ここでは (1) 各文脈を一度ずつ再活性化し、その痕跡での学び直しを共有の重みに恒久的に書き戻した後、新しいタスクを
学習し（リプレイも続ける）、取り戻した一般化が保たれるか、他の文脈が壊れないかを、書き戻さない場合・一つの文脈の痕跡
だけで書き戻す場合と比べる。(2) 固定化済みのバッファでの学び直しが効く理由を、一つの文脈への集中を除いた変種
（recon_buf_inter）と、確信度のマージンを広げる学習を除いた誤り駆動の変種（recon_buf_err）で切り分ける。

    --phase dev（開発用シード）/ eval（評価用シードで一度だけ）
    結果は results/context_inference/writeback_{phase}.json

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_writeback --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_context_writeback --analyze <結果ファイル>
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import torch

from ..basal_ganglia_core import ParitySpec
from ..context_inference import RECON_EXTRA_READOUTS, RECON_READOUTS, WRITEBACKS, CueSpec, FlipSpec, run_context_sequence
from ..run_env import env_info
from .run_context_episodic_readout import EPI_KW, FAMILIES, LS, MECHS_USED, SELECTS, specs_for
from .run_context_evidence_info import MECHS, SELECTION_TRIALS
from .run_context_gated_long import DEV_SEEDS, EVAL_SEEDS, native_contexts
from .run_context_replay_verify import _mean

CUE_N = 16
SELECT = "gen_replay"  # 12.6.151節の設定（P = 4、バッファから候補を取る）
# 書き戻しの後に学習する新しいタスク（各課題と同じ構造で、まだ使っていない反転ビット・ビットの組）
NEW_TASK = {"flip_m1": FlipSpec(14, flip=(5,)), "flip7_m1": FlipSpec(14, flip=(8,)), "flip8_m1": FlipSpec(8, flip=(5,)),
            "parity": ParitySpec(n_inputs=8, relevant=(1, 2)), "cue_q0.9": CueSpec(16, (1, 2), (9, 10), 0.9)}
WB_KW = {"writeback_steps": 100, "recon_lr": 0.01, "recon_batch": 32, "writeback_eval": 2000,
         "episodic_hold": EPI_KW["episodic_hold"]}
SEP_KW = {**EPI_KW, "recon_steps": 100, "recon_lr": 0.01, "recon_batch": 32, "recon_extra": True}
SEP_READOUTS = ("param", *RECON_READOUTS, *RECON_EXTRA_READOUTS)
CONFIGS: dict[str, dict] = {**{f"wb_{w}": {"writeback": w, **WB_KW} for w in WRITEBACKS}, "sep": SEP_KW}
# 修正は開発用シードの結果を見て追加する（上限3回）


def jobs(configs=tuple(CONFIGS), families=FAMILIES) -> list[tuple]:
    return [(f, m, L, cfg) for m in MECHS_USED for f in families for L in LS for cfg in configs]


def _exp(acc_row: list, ch: list, tasks) -> float:
    """選んだ文脈での期待 ACC（2·正確度 − 1 の選択の試行の平均）。acc_row[j][c]。"""
    vals = [2 * acc_row[j][c] - 1 for j in tasks for c in ch[j]]
    return sum(vals) / len(vals)


def run_job(job: tuple, kw: dict) -> dict:
    """job = (課題, 機構, L, 設定の名前, シード)。"""
    fam, mech, L, cfg, seed = job
    torch.set_num_threads(1)
    specs = specs_for(fam)
    T0 = len(specs)
    wb = cfg.startswith("wb_")
    if wb:
        specs = specs + [NEW_TASK[fam]]
        kw = {**kw, "writeback_at": T0}
    T = len(specs)
    torch.manual_seed(seed)
    r = run_context_sequence("inferred_gated", specs, 32, torch.Generator().manual_seed(seed + 50_000),
                             torch.Generator().manual_seed(seed + 90_000), transition_steps=L, steps=500,
                             diagnostics=True, cue_n=CUE_N, selection_trials=SELECTION_TRIALS, **MECHS[mech],
                             **SELECTS[SELECT], **kw)
    d = r["diag"]
    nat = native_contexts(r["assign"], T)
    ch = r["selection_choices"]
    acc = d["ctx_acc"][-1]
    orig = range(T0)
    row = {"seed": seed, "n_contexts": r["n_contexts"], "native": nat,
           "selection_acc": sum(c == nat[j] for j in orig for c in ch[j]) / (T0 * SELECTION_TRIALS),
           "exp_end": _exp(acc, ch, orig), "end_nat": [2 * acc[j][nat[j]] - 1 for j in orig]}
    if wb:
        w = d["writeback"]
        ci = [w["used"].index(nat[j]) for j in orig]
        # 書き戻しの前後の本来の文脈での期待 ACC（2·正確度 − 1）と、タスクごとの上書きの大きさ（正確度の単位）
        row.update(pre_nat=[2 * w["pre"]["fresh"][j][ci[j]] - 1 for j in orig],
                   post_nat=[2 * w["post"]["fresh"][j][ci[j]] - 1 for j in orig],
                   gap=[w["pre"]["own_buf"][j][ci[j]] - w["pre"]["fresh"][j][ci[j]] for j in orig],
                   post_own_buf=[w["post"]["own_buf"][j][ci[j]] for j in orig],
                   new_nat=2 * acc[T0][nat[T0]] - 1, new_exp=_exp(acc, ch, [T0]),
                   new_ctx=nat[T0] not in w["used"])
    else:
        ea = d["episodic_acc"]
        n_tr = T0 * SELECTION_TRIALS
        for k in SEP_READOUTS:
            row[f"exp_{k}"] = sum(2 * ea[j][c][k] - 1 for j in orig for c in ch[j]) / n_tr
            row[f"nat_{k}"] = sum(2 * ea[j][nat[j]][k] - 1 for j in orig) / T0
        mg = d["recon_margin"]
        for k in ("loss", "acc", "p_correct"):
            row[f"buf_{k}"] = _mean([mg[nat[j]][k] for j in orig])
        oa = d["origin_acc"]
        row["gap_combo"] = _mean([oa[j][nat[j]]["own_buf"] for j in orig]) - _mean([acc[j][nat[j]] for j in orig])
    return {"key": f"{fam}/{mech}_L{L}/{cfg}", "row": row}


def run_one(job: tuple, table: dict) -> dict:
    return run_job(job, table[job[3]])


def _avg(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def analyze(paths: list[Path]) -> dict:
    """書き戻し（wb_*）は同じシードの wb_none と対にして差をとる。組の分類は書き戻しの前の gap（正確度の単位、
    本来の文脈のバッファでの正確度 − 新しい入力での正確度、タスクとシードの平均）で決める。"""
    runs: dict = {}
    for p in paths:
        runs.update(json.loads(p.read_text(encoding="utf-8"))["runs"])
    combos = sorted({k.rsplit("/", 1)[0] for k in runs})
    rows: dict = {}
    for cb in combos:
        base = {x["seed"]: x for x in runs.get(f"{cb}/wb_none", {}).get("per_seed", [])}
        r: dict = {}
        if base:
            r["gap"] = _avg([_avg(x["gap"]) for x in base.values()])
            r["pre"] = _avg([_avg(x["pre_nat"]) for x in base.values()])
            r["end_none"] = _avg([_avg(x["end_nat"]) for x in base.values()])
            r["exp_end_none"] = _avg([x["exp_end"] for x in base.values()])
            r["pre_identical"] = True
            for w in ("all", "inter"):
                per = {x["seed"]: x for x in runs.get(f"{cb}/wb_{w}", {}).get("per_seed", [])}
                post, end, exp, intact_post, intact_end, over_post, over_end, new_d, sel_d, newexp_d = ([] for _ in range(10))
                for s, x in per.items():
                    b = base[s]
                    r["pre_identical"] &= x["pre_nat"] == b["pre_nat"]
                    dp = [a - c for a, c in zip(x["post_nat"], x["pre_nat"])]
                    de = [a - c for a, c in zip(x["end_nat"], b["end_nat"])]
                    post.append(_avg(dp))
                    end.append(_avg(de))
                    exp.append(x["exp_end"] - b["exp_end"])
                    intact = [i for i, g in enumerate(b["gap"]) if g <= 0.03]
                    over = [i for i, g in enumerate(b["gap"]) if g >= 0.10]
                    intact_post.append(_avg([dp[i] for i in intact]))
                    intact_end.append(_avg([de[i] for i in intact]))
                    over_post.append(_avg([dp[i] for i in over]))
                    over_end.append(_avg([de[i] for i in over]))
                    new_d.append(x["new_nat"] - b["new_nat"])
                    newexp_d.append(x["new_exp"] - b["new_exp"])
                    sel_d.append(x["selection_acc"] - b["selection_acc"])
                r.update({f"post_{w}": _avg(post), f"end_{w}": _avg(end), f"exp_{w}": _avg(exp),
                          f"intact_post_{w}": _avg(intact_post), f"intact_end_{w}": _avg(intact_end),
                          f"over_post_{w}": _avg(over_post), f"over_end_{w}": _avg(over_end),
                          f"new_{w}": _avg(new_d), f"new_exp_{w}": _avg(newexp_d), f"sel_{w}": _avg(sel_d)})
        sep = runs.get(f"{cb}/sep", {}).get("per_seed", [])
        if sep:
            r["sep_gap"] = _avg([x["gap_combo"] for x in sep])
            for k in SEP_READOUTS[1:]:
                r[f"d_{k}"] = _avg([x[f"exp_{k}"] - x["exp_param"] for x in sep])
                r[f"dnat_{k}"] = _avg([x[f"nat_{k}"] - x["nat_param"] for x in sep])
            for k in ("buf_loss", "buf_acc", "buf_p_correct"):
                r[k] = _avg([x[k] for x in sep])
        rows[cb] = r

    keys = sorted({k for r in rows.values() for k in r if k not in ("pre_identical",)})

    def group(pred) -> dict:
        g = {k: v for k, v in rows.items() if pred(k, v)}
        out: dict = {"n": len(g), "combos": sorted(g)}
        for k in keys:
            vals = [v[k] for v in g.values() if v.get(k) is not None]
            out[k] = _avg(vals)
            out[f"min_{k}"] = min(vals) if vals else None
        return out

    def gap_of(v: dict) -> float | None:
        return v.get("gap", v.get("sep_gap"))

    over = lambda k, v: gap_of(v) is not None and gap_of(v) >= 0.10  # noqa: E731
    return {"rows": rows, "pre_identical": all(r.get("pre_identical", True) for r in rows.values()),
            "over_14bit": group(lambda k, v: over(k, v) and k.startswith(("flip_m1", "flip7_m1"))),
            "not_over": group(lambda k, v: gap_of(v) is not None and gap_of(v) <= 0.03),
            "all": group(lambda k, v: True)}


SUMMARY_KEYS = ("selection_acc", "exp_end", "new_nat", "new_exp", *(f"exp_{k}" for k in SEP_READOUTS),
                *(f"nat_{k}" for k in SEP_READOUTS), "buf_loss", "buf_acc", "buf_p_correct", "gap_combo")


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
    out_path = args.out or Path(f"results/context_inference/writeback_{args.phase}.json")
    all_jobs = [(*cfg, s) for cfg in jobs(tuple(args.configs), tuple(args.families)) for s in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(partial(run_one, table=CONFIGS), all_jobs, chunksize=1)
    out: dict = {"環境": env_info(),
                 "説明": f"学び直しの共有の重みへの書き戻しと、バッファでの学び直しの切り分け（12.6.161〜162節）。phase={args.phase}",
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
