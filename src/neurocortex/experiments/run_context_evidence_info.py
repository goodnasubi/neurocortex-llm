"""手がかりの情報量による証拠の重み付けと、能動的な手がかりの選択（active sensing）（12.6.143〜144節）。

従来の選択（batch）、12.6.135節の証拠の蓄積（evidence_fix2）、(a) 手がかりの情報量（文脈の間の予測の不一致）で
証拠に重みを付ける形（info）、(b) ラベルなしの候補から期待情報量が最大の手がかりを選んで正解を観測する形（active）、
(a)+(b)（active_info）を、手がかり 1/2/4/8/16 個で比べる。機構・課題・補足は 12.6.139節と同じ。

    --phase dev（開発用シード, evidence_info_dev.json）/ eval（評価用シードで一度だけ, evidence_info_eval.json）
    --selects で選択の規則を選ぶ

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_evidence_info --phase dev
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import torch

from ..context_inference import run_context_sequence
from ..continual_metrics import forgetting_metrics
from ..run_env import env_info
from .run_context_evidence import SELECTS as EVIDENCE_SELECTS
from .run_context_gated_long import DEV_SEEDS, EVAL_SEEDS, FINAL_RESIDUAL, FINAL_SPLIT, native_contexts
from .run_context_split_generalize import family_specs

FAMILIES = ("parity", "flip_m1", "cue_q0.9", "cue_q0.75")
SUPPLEMENT = (("flip_m1", "reset"),)  # 補足: 12.6.140節で低下の原因を特定した組
MECHS = {"split": FINAL_SPLIT, "reset": FINAL_RESIDUAL}
CUE_NS = (1, 2, 4, 8, 16)
LS = (0, 250, 500)
FIX2 = EVIDENCE_SELECTS["evidence_fix2"]  # 12.6.135節の確定値（ε = 0.02、α = 0.001）
# 選択の規則。修正は開発用シードの結果を見て追加する（上限3回）
SELECTS: dict[str, dict] = {
    "batch": {"cue_select": "batch"},
    "evidence_fix2": FIX2,
    "info": {**FIX2, "cue_weight": "info"},
    "active": {**FIX2, "active_pool": 4},
    "active_info": {**FIX2, "cue_weight": "info", "active_pool": 4},
    # 修正1: 開発用シードで info が evidence_fix2 を下回った（補足の作り直しの手がかり8個で平均 −0.017）。最終モデルでは
    # 上書きされた本来の文脈が自分のタスクでも正確度 0.5〜0.7 しかなく、一様な事前分布ではその雑音の予測でどの手がかりも
    # 不一致が大きく見え（反転領域の外 0.40 対 中 0.50）、重みが区別しない。重みを観測直前の事後分布で測り、
    # 先頭の候補の間の不一致にする
    "info_fix1": {**FIX2, "cue_weight": "info_post"},
    "active_info_fix1": {**FIX2, "cue_weight": "info_post", "active_pool": 4},
}
BASE_SELECTS = ("batch", "evidence_fix2", "info", "active", "active_info")
EVAL_SELECTS = BASE_SELECTS
SELECTION_TRIALS = 50


def jobs(selects=BASE_SELECTS) -> list[tuple]:
    combos = [(fam, "split") for fam in FAMILIES] + list(SUPPLEMENT)
    return [(fam, mech, L, n, sel) for fam, mech in combos for L in LS for n in CUE_NS for sel in selects]


def run_one(job: tuple) -> dict:
    fam, mech, L, cue_n, sel, seed = job
    torch.set_num_threads(1)
    specs, _ = family_specs(fam)
    torch.manual_seed(seed)
    r = run_context_sequence("inferred_gated", specs, 32, torch.Generator().manual_seed(seed + 50_000),
                             torch.Generator().manual_seed(seed + 90_000), transition_steps=L, steps=500,
                             diagnostics=True, cue_n=cue_n, selection_trials=SELECTION_TRIALS, **MECHS[mech],
                             **SELECTS[sel])
    d, T = r["diag"], len(specs)
    nat = native_contexts(r["assign"], T)
    acc = d["ctx_acc"][-1]  # [タスク][文脈]（used の順。used は 0, 1, 2, … なので文脈番号と一致）
    ch, cu = r["selection_choices"], r["selection_cues"]
    n_tr = T * SELECTION_TRIALS
    m = forgetting_metrics(r["R"], chance=0.5)
    row = {"seed": seed, "n_contexts": r["n_contexts"], "ACC": m["ACC"], "native": nat,
           "final_selection_correct": sum(d["chosen"][-1][j] == nat[j] for j in range(T)) / T,
           "selection_acc": sum(c == nat[j] for j in range(T) for c in ch[j]) / n_tr,
           "cues_used": sum(sum(x) for x in cu) / n_tr,
           "expected_acc": sum(2 * acc[j][c] - 1 for j in range(T) for c in ch[j]) / n_tr,
           "native_acc": sum(2 * acc[j][nat[j]] - 1 for j in range(T)) / T}
    if "selection_info" in r:
        row["cue_info"] = sum(sum(x) for x in r["selection_info"]) / n_tr
    return {"key": f"{fam}/{mech}_L{L}/cue{cue_n}/{sel}", "row": row}


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
    out_path = args.out or Path(f"results/context_inference/evidence_info_{args.phase}.json")
    all_jobs = [(*cfg, s) for cfg in jobs(args.selects) for s in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(run_one, all_jobs, chunksize=1)
    out: dict = {"環境": env_info(),
                 "説明": f"手がかりの情報量による重み付けと能動的な手がかりの選択（12.6.143〜144節）。phase={args.phase}",
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
