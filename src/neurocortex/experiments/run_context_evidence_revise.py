"""文脈の信頼性による閾値と、決定後の乗り換え（change of mind）（12.6.139〜140節）。

従来の選択（batch）、12.6.135節の証拠の蓄積（evidence_fix2）、(a) 文脈の信頼性から lapse・閾値を決める形（rel）、
(b) 決定後も手がかりを観測し続けて乗り換える形（com）、(a)+(b)（rel_com）を、手がかり 1/2/4/8/16 個で比べる。
機構は 12.6.129節の分離可塑性（FINAL_SPLIT）。課題はパリティ・似たタスク m = 1・入力分布が異なる課題 q = 0.9、
0.75（L = 0/250/500、500ステップ）。補足として似たタスク m = 1 の作り直し（FINAL_RESIDUAL）も測る。

    --phase dev（開発用シード, evidence_revise_dev.json）/ eval（評価用シードで一度だけ, evidence_revise_eval.json）
    --selects で選択の規則を選ぶ（修正の確認は --selects rel_fix1 --out .../evidence_revise_fix1_dev.json）

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_evidence_revise --phase dev
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
SUPPLEMENT = (("flip_m1", "reset"),)  # 補足: 12.6.136節で過信の打ち切りが起きた組
MECHS = {"split": FINAL_SPLIT, "reset": FINAL_RESIDUAL}
CUE_NS = (1, 2, 4, 8, 16)
LS = (0, 250, 500)
FIX2 = EVIDENCE_SELECTS["evidence_fix2"]  # 12.6.135節の確定値（ε = 0.02、α = 0.001）
# 選択の規則。修正は開発用シードの結果を見て追加する（上限3回）
SELECTS: dict[str, dict] = {
    "batch": {"cue_select": "batch"},
    "evidence_fix2": FIX2,
    "rel": {**FIX2, "evidence_reliability": True},
    "com": {**FIX2, "evidence_com": True, "com_alpha": 0.1, "com_window": 0},
    "rel_com": {**FIX2, "evidence_reliability": True, "evidence_com": True, "com_alpha": 0.1, "com_window": 0},
    # 修正1: 開発用シードで rel が evidence_fix2 を下回った（手がかり2個で平均 −0.004、作り直しの L = 250 で −0.044）。
    # 文脈ごとの ε_c が選択を信頼性の低い文脈に偏らせることと、ゲート後の正確度で信頼性を高めに見積もる
    # （作り直しで 0.97〜1.0、実際は 0.80〜0.83）ことが原因なので、信頼性は閾値だけに使い、ゲート前の正確度で測る
    "rel_fix1": {**FIX2, "evidence_reliability": True, "reliability_lapse": False, "reliability_source": "pre"},
    "rel_com_fix1": {**FIX2, "evidence_reliability": True, "reliability_lapse": False, "reliability_source": "pre",
                     "evidence_com": True, "com_alpha": 0.1, "com_window": 0},
    # 修正2: 修正1でも作り直しの L = 0 が改善しなかった。上書きされた本来の文脈（新しいサンプルで正確度 0.57〜0.66）の
    # バッファの整合度が 1.0 のまま（リプレイがバッファのサンプルそのものを学習し続けるため）なので、リプレイに使わない
    # 保留サンプル（文脈ごとに 50 個）の整合度で忘却を測る
    "rel_fix2": {**FIX2, "evidence_reliability": True, "reliability_lapse": False, "reliability_source": "pre",
                 "reliability_holdout": 50},
    "rel_com_fix2": {**FIX2, "evidence_reliability": True, "reliability_lapse": False, "reliability_source": "pre",
                     "reliability_holdout": 50, "evidence_com": True, "com_alpha": 0.1, "com_window": 0},
}
BASE_SELECTS = ("batch", "evidence_fix2", "rel", "com", "rel_com")
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
    obs = r.get("selection_observed", cu)
    first = r.get("selection_first", ch)
    sw = r.get("selection_switches", [[0] * len(x) for x in ch])
    n_tr = T * SELECTION_TRIALS
    pairs = [(first[j][i], ch[j][i], nat[j]) for j in range(T) for i in range(SELECTION_TRIALS)]
    changed = [(f, c, n) for f, c, n in pairs if f != c]
    m = forgetting_metrics(r["R"], chance=0.5)
    row = {"seed": seed, "n_contexts": r["n_contexts"], "ACC": m["ACC"], "native": nat,
           "final_selection_correct": sum(d["chosen"][-1][j] == nat[j] for j in range(T)) / T,
           "selection_acc": sum(c == nat[j] for j in range(T) for c in ch[j]) / n_tr,
           "first_selection_acc": sum(f == n for f, _, n in pairs) / n_tr,
           "cues_used": sum(sum(x) for x in cu) / n_tr,
           "cues_observed": sum(sum(x) for x in obs) / n_tr,
           "switch_rate": sum(sum(1 for s in x if s > 0) for x in sw) / n_tr,
           "changed_rate": len(changed) / n_tr,
           "changed_to_native": sum(c == n for _, c, n in changed),
           "changed_from_native": sum(f == n for f, _, n in changed),
           "n_changed": len(changed),
           "expected_acc": sum(2 * acc[j][c] - 1 for j in range(T) for c in ch[j]) / n_tr,
           "native_acc": sum(2 * acc[j][nat[j]] - 1 for j in range(T)) / T}
    if "reliability" in d and d["reliability"] and d["reliability"][-1]:
        rel = d["reliability"][-1]
        row["native_reliability"] = sum(rel.get(nat[j], 1.0) for j in range(T)) / T
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
    out_path = args.out or Path(f"results/context_inference/evidence_revise_{args.phase}.json")
    all_jobs = [(*cfg, s) for cfg in jobs(args.selects) for s in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(run_one, all_jobs, chunksize=1)
    out: dict = {"環境": env_info(),
                 "説明": f"文脈の信頼性による閾値と決定後の乗り換え（12.6.139〜140節）。phase={args.phase}",
                 "seeds": seeds, "selects": {k: SELECTS[k] for k in args.selects},
                 "selection_trials": SELECTION_TRIALS, "runs": {}}
    keys = ("ACC", "final_selection_correct", "selection_acc", "first_selection_acc", "cues_used", "cues_observed",
            "switch_rate", "changed_rate", "expected_acc", "native_acc", "native_reliability")
    for key in dict.fromkeys(x["key"] for x in results):
        rows = [x["row"] for x in results if x["key"] == key]
        summ = {k: {"mean": float(torch.tensor([float(x[k]) for x in rows]).mean()),
                    "std": float(torch.tensor([float(x[k]) for x in rows]).std(unbiased=False))}
                for k in keys if all(k in x for x in rows)}
        summ["changed_to_native"] = sum(x["changed_to_native"] for x in rows)
        summ["changed_from_native"] = sum(x["changed_from_native"] for x in rows)
        summ["n_changed"] = sum(x["n_changed"] for x in rows)
        out["runs"][key] = {"summary": summ, "per_seed": rows}
        print(key, " ".join(f"{k}={v['mean']:.3f}" for k, v in summ.items() if isinstance(v, dict)), flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
