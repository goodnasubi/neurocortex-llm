"""テスト時の文脈選択を証拠の蓄積にする（12.6.135〜136節）。

従来の選択（cue_select="batch"。cue_n 個の手がかりの平均損失が最小の文脈を1回で選ぶ）と、証拠の蓄積
（cue_select="evidence"。手がかりごとに文脈ごとの対数尤度を積み上げ、事後確率が 1 − α に達するか手がかりが
尽きた時点で決める）を、手がかり 1/2/4/8/16 個で比べる。切り分けとして lapse なし（ε = 0）の証拠の蓄積も測る。
機構は 12.6.129節の分離可塑性（FINAL_SPLIT）と 12.6.121節の確定機構（FINAL_RESIDUAL）。
課題はパリティ・似たタスク m = 1・入力分布が異なる課題 q = 0.9、0.75。L = 0/250/500（500ステップ）。

    --phase dev（開発用シード, evidence_dev.json）/ eval（評価用シードで一度だけ, evidence_eval.json）
    --selects で選択の規則を選ぶ（修正の確認は --selects evidence_fix1 --out .../evidence_fix1_dev.json）

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_evidence --phase dev
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
from .run_context_gated_long import DEV_SEEDS, EVAL_SEEDS, FINAL_RESIDUAL, FINAL_SPLIT, native_contexts
from .run_context_split_generalize import family_specs

FAMILIES = ("parity", "flip_m1", "cue_q0.9", "cue_q0.75")
MECHS = (("split", FINAL_SPLIT), ("reset", FINAL_RESIDUAL))
CUE_NS = (1, 2, 4, 8, 16)
LS = (0, 250, 500)
# 選択の規則。修正は開発用シードの結果を見て追加する（上限3回）
SELECTS: dict[str, dict] = {
    "batch": {"cue_select": "batch"},
    "evidence": {"cue_select": "evidence", "evidence_lapse": 0.1, "evidence_alpha": 0.01},
    "evidence_nolapse": {"cue_select": "evidence", "evidence_lapse": 0.0, "evidence_alpha": 0.01},
    # 修正1: 開発用シードで、打ち切りの誤り（モデルの事後確率の過信）で選択精度が従来を下回った（最悪 −0.034）ため、
    # 閾値を α = 0.001 に厳しくする
    "evidence_fix1": {"cue_select": "evidence", "evidence_lapse": 0.1, "evidence_alpha": 0.001},
}
BASE_SELECTS = ("batch", "evidence", "evidence_nolapse")
SELECTION_TRIALS = 50


def jobs(selects=BASE_SELECTS) -> list[tuple]:
    return [(fam, mech, sc, L, n, sel) for fam in FAMILIES for mech, sc in MECHS for L in LS for n in CUE_NS
            for sel in selects]


def run_one(job: tuple) -> dict:
    fam, mech, sc, L, cue_n, sel, seed = job
    torch.set_num_threads(1)
    specs, _ = family_specs(fam)
    torch.manual_seed(seed)
    r = run_context_sequence("inferred_gated", specs, 32, torch.Generator().manual_seed(seed + 50_000),
                             torch.Generator().manual_seed(seed + 90_000), transition_steps=L, steps=500,
                             diagnostics=True, cue_n=cue_n, selection_trials=SELECTION_TRIALS, **sc, **SELECTS[sel])
    d, T = r["diag"], len(specs)
    nat = native_contexts(r["assign"], T)
    acc = d["ctx_acc"][-1]  # [タスク][文脈]（used の順。used は 0, 1, 2, … なので文脈番号と一致）
    ch, cu = r["selection_choices"], r["selection_cues"]
    n_tr = T * SELECTION_TRIALS
    row = {"seed": seed, "R": r["R"], "n_contexts": r["n_contexts"], **forgetting_metrics(r["R"], chance=0.5),
           "native": nat,
           "final_selection_correct": sum(d["chosen"][-1][j] == nat[j] for j in range(T)) / T,
           "final_cues_used": sum(d["cues_used"][-1]) / T,
           "selection_acc": sum(c == nat[j] for j in range(T) for c in ch[j]) / n_tr,
           "cues_used": sum(sum(x) for x in cu) / n_tr,
           "expected_acc": sum(2 * acc[j][c] - 1 for j in range(T) for c in ch[j]) / n_tr,
           "native_acc": sum(2 * acc[j][nat[j]] - 1 for j in range(T)) / T}
    return {"key": f"{fam}/{mech}_L{L}/cue{cue_n}/{sel}", "row": row}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("dev", "eval"), default="dev")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--selects", nargs="+", choices=list(SELECTS), default=list(BASE_SELECTS))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    seeds = args.seeds or (EVAL_SEEDS if args.phase == "eval" else DEV_SEEDS)
    out_path = args.out or Path(f"results/context_inference/evidence_{args.phase}.json")
    all_jobs = [(*cfg, s) for cfg in jobs(args.selects) for s in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(run_one, all_jobs)
    out: dict = {"環境": env_info(), "説明": f"テスト時の文脈選択を証拠の蓄積にする（12.6.135〜136節）。phase={args.phase}",
                 "seeds": seeds, "selects": {k: SELECTS[k] for k in args.selects}, "selection_trials": SELECTION_TRIALS, "runs": {}}
    keys = ("ACC", "BWT", "LA", "final_selection_correct", "final_cues_used", "selection_acc", "cues_used",
            "expected_acc", "native_acc")
    for key in dict.fromkeys(x["key"] for x in results):
        rows = [x["row"] for x in results if x["key"] == key]
        summ = {k: {"mean": float(torch.tensor([float(x[k]) for x in rows]).mean()),
                    "std": float(torch.tensor([float(x[k]) for x in rows]).std(unbiased=False))} for k in keys}
        out["runs"][key] = {"summary": summ, "per_seed": rows}
        print(key, " ".join(f"{k}={v['mean']:.3f}" for k, v in summ.items()), flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
