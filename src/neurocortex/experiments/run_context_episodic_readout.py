"""選んだ文脈の予測と、再生された痕跡のラベルを使う記憶に基づく予測の組み合わせ（12.6.155〜156節）。

12.6.152節では、海馬のバッファから候補を取ると本来の文脈を正しく選べるが、その文脈が新しい入力で誤るため期待
ACC はほとんど上がらなかった。ここでは選択の後の予測を、パラメトリックな予測（選んだ文脈のネットワーク）と、選んだ
文脈の痕跡（バッファ＝リプレイした痕跡、保留サンプル＝保存だけでリプレイしなかった痕跡）からの読み出しの組み合わせ
（痕跡との近さで重み付けした混合・記憶の信頼度による切り替え）にして、期待 ACC を比べる。

    --phase dev（開発用シード）/ eval（評価用シードで一度だけ）
    結果は results/context_inference/episodic_readout_{phase}.json

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_episodic_readout --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_context_episodic_readout --analyze <結果ファイル>
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import torch

from ..context_inference import EPISODIC_READOUTS, FlipSpec, run_context_sequence
from ..continual_metrics import forgetting_metrics
from ..run_env import env_info
from .run_context_evidence_info import MECHS, SELECTION_TRIALS
from .run_context_gated_long import DEV_SEEDS, EVAL_SEEDS, native_contexts
from .run_context_replay_verify import ACTIVE, _mean
from .run_context_replay_verify import specs_for as _specs_for

HOLD_SMALL = 50  # 保留サンプルの先頭（12.6.151〜152節の保留サンプルと同じ数）
HOLD_LARGE = 200  # 痕跡の量を増やした場合（mix_all200）
EPI_N = 2000  # 読み出しの評価に使う新しい入力の数（タスクごと）
FAMILIES = ("flip_m1", "flip7_m1", "parity", "cue_q0.9", "flip8_m1")
# 修正1（12.6.155節）: 設計は似たタスク m = 1 を 8 ビットと誤認していた（実際は 14 ビットで、痕跡は入力の 0.6% しか覆わない）。
# H1〜H4(a) が想定した入力空間の小さい似たタスクとして、同じ構造の 8 ビット版 flip8_m1 を加える
MECHS_USED = ("reset", "split")
LS = (0, 250, 500)
CUE_N = 16
SELECTS: dict[str, dict] = {
    "gen_replay": {**ACTIVE, "active_source": "replay", "reliability_holdout": HOLD_LARGE},
    "batch": {"cue_select": "batch", "reliability_holdout": HOLD_LARGE},
}
EPI_KW = {"episodic_eval": EPI_N, "episodic_hold": HOLD_SMALL, "episodic_lambda": 0.5, "episodic_kappa": 1.0}
# 修正は開発用シードの結果を見て追加する（上限3回）


def specs_for(family: str) -> list:
    """flip8_m1 … 似たタスク m = 1 の 8 ビット版（基本タスクと、ビット 2・3・4 のそれぞれで反転する 3 個）。"""
    if family == "flip8_m1":
        return [FlipSpec(8)] + [FlipSpec(8, flip=(2 + i,)) for i in range(3)]
    return _specs_for(family)


def jobs(selects, families=FAMILIES) -> list[tuple]:
    return [(f, m, L, CUE_N, sel) for m in MECHS_USED for f in families for L in LS for sel in selects]


def run_job(job: tuple, select_kw: dict, epi_kw: dict = EPI_KW) -> dict:
    """job = (課題, 機構, L, 手がかりの数, 規則の名前, シード)。"""
    fam, mech, L, cue_n, sel, seed = job
    torch.set_num_threads(1)
    specs = specs_for(fam)
    torch.manual_seed(seed)
    r = run_context_sequence("inferred_gated", specs, 32, torch.Generator().manual_seed(seed + 50_000),
                             torch.Generator().manual_seed(seed + 90_000), transition_steps=L, steps=500,
                             diagnostics=True, cue_n=cue_n, selection_trials=SELECTION_TRIALS, **MECHS[mech],
                             **select_kw, **epi_kw)
    d, T = r["diag"], len(specs)
    nat = native_contexts(r["assign"], T)
    acc = d["ctx_acc"][-1]
    ea = d["episodic_acc"]
    ch = r["selection_choices"]
    n_tr = T * SELECTION_TRIALS
    oa = d["origin_acc"]
    row = {"seed": seed, "n_contexts": r["n_contexts"], "ACC": forgetting_metrics(r["R"], chance=0.5)["ACC"],
           "native": nat, "selection_acc": sum(c == nat[j] for j in range(T) for c in ch[j]) / n_tr,
           "expected_acc": sum(2 * acc[j][c] - 1 for j in range(T) for c in ch[j]) / n_tr,
           "nat_fresh": _mean([acc[j][nat[j]] for j in range(T)]),
           "nat_own_buf": _mean([oa[j][nat[j]]["own_buf"] for j in range(T)]),
           "nat_holdout": _mean([oa[j][nat[j]]["holdout"] for j in range(T)]),
           "cover_sel": sum(d["episodic_cover"][j][c] for j in range(T) for c in ch[j]) / n_tr,
           "cover_nat": _mean([d["episodic_cover"][j][nat[j]] for j in range(T)]),
           "purity_nat": _mean([d["store_purity"][j][nat[j]] for j in range(T)])}
    for k in EPISODIC_READOUTS:
        # 選んだ文脈での期待 ACC（2·正確度 − 1 の選択の試行の平均）と、本来の文脈での値（選択が完全な場合）
        row[f"exp_{k}"] = sum(2 * ea[j][c][k] - 1 for j in range(T) for c in ch[j]) / n_tr
        row[f"nat_{k}"] = sum(2 * ea[j][nat[j]][k] - 1 for j in range(T)) / T
    return {"key": f"{fam}/{mech}_L{L}/cue{cue_n}/{sel}", "row": row}


def run_one(job: tuple, table: dict) -> dict:
    return run_job(job, table[job[4]])


SUMMARY_KEYS = ("ACC", "selection_acc", "expected_acc", "nat_fresh", "nat_own_buf", "nat_holdout", "cover_sel",
                "cover_nat", "purity_nat", *(f"exp_{k}" for k in EPISODIC_READOUTS),
                *(f"nat_{k}" for k in EPISODIC_READOUTS))


def _m(runs: dict, key: str, k: str) -> float | None:
    s = runs.get(key, {}).get("summary", {}).get(k)
    return s["mean"] if s else None


def analyze(paths: list[Path], sel: str = "gen_replay") -> dict:
    """組ごとに gap（12.6.151節の上書きの大きさ）と、各読み出しの期待 ACC の param との差を求め、判定に使う平均を返す。"""
    runs: dict = {}
    for p in paths:
        runs.update(json.loads(p.read_text(encoding="utf-8"))["runs"])
    rows = {}
    for key in runs:
        fam, ml, cue, s = key.split("/")
        if s != sel:
            continue
        own, fresh = _m(runs, key, "nat_own_buf"), _m(runs, key, "nat_fresh")
        r = {"gap": own - fresh if own is not None and fresh is not None else None,
             "sel": _m(runs, key, "selection_acc"), "cover_nat": _m(runs, key, "cover_nat"),
             "purity_nat": _m(runs, key, "purity_nat"), "exp_param": _m(runs, key, "exp_param"),
             "nat_param": _m(runs, key, "nat_param")}
        for k in EPISODIC_READOUTS[1:]:
            r[f"d_{k}"] = _m(runs, key, f"exp_{k}") - r["exp_param"]
            r[f"dnat_{k}"] = _m(runs, key, f"nat_{k}") - r["nat_param"]
        rows[f"{fam}/{ml}"] = r

    def group(pred) -> dict:
        g = {k: v for k, v in rows.items() if pred(k, v)}
        out = {"n": len(g), "combos": sorted(g)}
        for k in EPISODIC_READOUTS[1:]:
            vals = [v[f"d_{k}"] for v in g.values()]
            out[f"d_{k}"] = _mean(vals)
            out[f"min_d_{k}"] = min(vals) if vals else None
        return out

    over = lambda k, v: v["gap"] is not None and v["gap"] >= 0.10  # noqa: E731
    return {"select": sel, "rows": rows,
            "flip_m1_reset": group(lambda k, v: k.startswith("flip_m1/reset")),
            "flip8_m1_reset": group(lambda k, v: k.startswith("flip8_m1/reset")),
            "over_14bit": group(lambda k, v: over(k, v) and k.startswith(("flip_m1", "flip7_m1"))),
            "over_flip7": group(lambda k, v: over(k, v) and k.startswith("flip7_m1")),
            "over": group(over),
            "not_over": group(lambda k, v: v["gap"] is not None and v["gap"] <= 0.03)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", type=Path, nargs="+", help="結果ファイルを集計して判定に使う量を表示する")
    ap.add_argument("--select", default="gen_replay")
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--phase", choices=("dev", "eval"), default="dev")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--selects", nargs="+", default=None)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.analyze:
        print(json.dumps(analyze(args.analyze, args.select), ensure_ascii=False, indent=1))
        return
    selects = args.selects or list(SELECTS)
    seeds = args.seeds or (EVAL_SEEDS if args.phase == "eval" else DEV_SEEDS)
    out_path = args.out or Path(f"results/context_inference/episodic_readout_{args.phase}.json")
    all_jobs = [(*cfg, s) for cfg in jobs(selects, tuple(args.families)) for s in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(partial(run_one, table=SELECTS), all_jobs, chunksize=1)
    out: dict = {"環境": env_info(),
                 "説明": f"選んだ文脈の予測と記憶に基づく予測の組み合わせ（12.6.155〜156節）。phase={args.phase}",
                 "seeds": seeds, "selects": {k: SELECTS[k] for k in selects}, "episodic": EPI_KW,
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
        print(key, " ".join(f"{k}={v['mean']:.3f}" for k, v in summ.items() if k.startswith(("sel", "exp_"))),
              flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
