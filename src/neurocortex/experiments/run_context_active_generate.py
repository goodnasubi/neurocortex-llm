"""能動的な手がかりの選択の候補の数、自分で候補を作る形、検証つきの打ち切り（12.6.147〜148節）。

12.6.143節の能動的な選択（active、P = 4）について、(1) 候補の数 P = 1/2/4/8/16、(2) 候補を与えずに自分で作る形
（入力空間から一様に: gen_uniform、海馬のバッファから: gen_replay、期待情報量の山登りで組み立てる: gen_search）、
(3) 打ち切りの規則（先頭の仮説を反証しうる手がかりで検証する: active_verify、閾値を上げるだけの統制: active_cautious）を、
batch・evidence_fix2 と手がかり 1/2/4/8/16 個で比べる。機構・課題・補足は 12.6.143節と同じ。

    --phase dev（開発用シード, active_generate_dev.json）/ eval（評価用シードで一度だけ, active_generate_eval.json）

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_active_generate --phase dev
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import torch

from ..run_env import env_info
from .run_context_evidence_info import FIX2, FAMILIES, SUPPLEMENT, CUE_NS, LS, SELECTION_TRIALS, run_job
from .run_context_gated_long import DEV_SEEDS, EVAL_SEEDS

ACTIVE = {**FIX2, "active_pool": 4}
# 選択の規則。修正は開発用シードの結果を見て追加する（上限3回）
SELECTS: dict[str, dict] = {
    "batch": {"cue_select": "batch"},
    "evidence_fix2": FIX2,
    "active": ACTIVE,
    **{f"active_P{p}": {**FIX2, "active_pool": p} for p in (1, 2, 8, 16)},
    "gen_uniform": {**ACTIVE, "active_source": "uniform"},
    "gen_replay": {**ACTIVE, "active_source": "replay"},
    "gen_search": {**ACTIVE, "active_source": "search"},
    "active_verify": {**ACTIVE, "active_verify": 2},
    "active_cautious": {**ACTIVE, "evidence_alpha": 1e-5},
}
BASE_SELECTS = tuple(SELECTS)
EVAL_SELECTS = BASE_SELECTS


def jobs(selects=BASE_SELECTS) -> list[tuple]:
    combos = [(fam, "split") for fam in FAMILIES] + list(SUPPLEMENT)
    return [(fam, mech, L, n, sel) for fam, mech in combos for L in LS for n in CUE_NS for sel in selects]


def run_one(job: tuple, table: dict) -> dict:
    return run_job(job, table[job[4]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("dev", "eval"), default="dev")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--selects", nargs="+", choices=list(SELECTS), default=None)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.selects is None:
        args.selects = list(EVAL_SELECTS if args.phase == "eval" else BASE_SELECTS)
    seeds = args.seeds or (EVAL_SEEDS if args.phase == "eval" else DEV_SEEDS)
    out_path = args.out or Path(f"results/context_inference/active_generate_{args.phase}.json")
    all_jobs = [(*cfg, s) for cfg in jobs(args.selects) for s in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(partial(run_one, table=SELECTS), all_jobs, chunksize=1)
    out: dict = {"環境": env_info(),
                 "説明": f"能動的な選択の候補の数・自己生成の候補・検証つきの打ち切り（12.6.147〜148節）。phase={args.phase}",
                 "seeds": seeds, "selects": {k: SELECTS[k] for k in args.selects},
                 "selection_trials": SELECTION_TRIALS, "runs": {}}
    keys = ("ACC", "final_selection_correct", "selection_acc", "cues_used", "expected_acc", "native_acc", "cue_info")
    for key in dict.fromkeys(x["key"] for x in results):
        rows = [x["row"] for x in results if x["key"] == key]
        summ = {k: {"mean": float(torch.tensor([float(x[k]) for x in rows]).mean()),
                    "std": float(torch.tensor([float(x[k]) for x in rows]).std(unbiased=False))}
                for k in keys if all(k in x for x in rows)}
        out["runs"][key] = {"summary": summ, "per_seed": rows}
        print(key, " ".join(f"{k}={v['mean']:.3f}" for k, v in summ.items()), flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
