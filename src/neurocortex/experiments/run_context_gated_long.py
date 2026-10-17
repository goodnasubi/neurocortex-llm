"""ゲート付き文脈推定の長い移行期間での限界（12.6.116節）の原因の切り分けと修正。

予備実験（--phase prelim、results/context_inference/gated_long_dev.json）:
    仮説「L = 500 では新タスクが多数派になるのが後半で、新しい文脈を学習する時間が足りない」を、学習ステップを
    500 → 1000 に延ばして検証する。新しい文脈の作成時点を記録する。
切り分け（--phase diag、results/context_inference/gated_long_diag.json）:
    診断（文脈ごとの正確度・選択した文脈・バッファと学習の出どころ）つきで、統制（oracle、L = 0）と比べる。
修正（--phase fix、results/context_inference/gated_long_fix_dev.json）:
    store_consistent（文脈の規則と整合するサンプルだけを保存）を L = 0〜500 で、修正前と比べる。
評価（--phase eval、results/context_inference/gated_long_eval.json）:
    確定後に評価用シード 0〜4 で一度だけ測る。

開発用シード 100〜104 のみを --phase prelim / diag / fix で使う。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_gated_long --phase diag
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import torch

from ..basal_ganglia_core import ParitySpec
from ..context_inference import assignment_purity, run_context_sequence
from ..continual_metrics import forgetting_metrics
from ..run_env import env_info

PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]
DEV_SEEDS = [100, 101, 102, 103, 104]
EVAL_SEEDS = [0, 1, 2, 3, 4]

# (名前, 条件, L, steps, (store_consistent, train_consistent))
NONE, STORE, BOTH = (False, False), (True, False), (True, True)
FINAL = STORE  # 開発用シードで確定した機構（12.6.116節、修正1）
PHASES = {
    "prelim": [("gated", "inferred_gated", 250, 500, NONE), ("gated", "inferred_gated", 500, 500, NONE),
               ("gated", "inferred_gated", 500, 1000, NONE)],
    "diag": [("gated", "inferred_gated", 500, 500, NONE), ("gated", "inferred_gated", 500, 1000, NONE),
             ("gated", "inferred_gated", 0, 1000, NONE), ("oracle", "oracle", 500, 1000, NONE)],
    "fix": [(n, "inferred_gated", L, s, sc)
            for sc, n in ((NONE, "gated"), (STORE, "gated_store"), (BOTH, "gated_store_train"))
            for L, s in ((0, 500), (100, 500), (250, 500), (500, 500), (500, 1000))],
    "eval": [(n, "inferred_gated", L, s, sc) for sc, n in ((NONE, "gated"), (FINAL, "gated_final"))
             for L, s in ((0, 500), (100, 500), (250, 500), (500, 500), (500, 1000))],
}


def creation_steps(assign: list[tuple[int, int]]) -> list[int]:
    """新しい文脈が初めて使われた時点を、そのタスク内のステップ番号で返す（タスク0を除く）。"""
    seen, out, start = {assign[0][1]}, [], {}
    for i, (k, c) in enumerate(assign):
        start.setdefault(k, i)
        if c not in seen:
            seen.add(c)
            out.append(i - start[k])
    return out


def native_contexts(assign: list[tuple[int, int]], n_tasks: int) -> list[int]:
    """各タスクの「本来の文脈」。タスク0は最初の文脈、タスク k ≥ 1 はタスク k の期間に新しく作られた文脈
    （作られなければ最頻の文脈）。移行期間が長いと作成が遅れ、最頻の文脈は前のタスクの文脈になるため。"""
    seen, native = set(), {}
    for k, c in assign:
        if c not in seen:
            seen.add(c)
            native.setdefault(k, c)
    out = []
    for k in range(n_tasks):
        ctxs = [c for t, c in assign if t == k]
        out.append(native.get(k, max(set(ctxs), key=ctxs.count)))
    return out


def buffer_contamination(buffer_src: dict, native: list[int]) -> float:
    """バッファのサンプルのうち、その文脈が本来のタスク以外から来たものの割合（ラベルが一致するものも含む）。"""
    owner = {c: k for k, c in enumerate(native)}
    total = sum(sum(v.values()) for v in buffer_src.values())
    foreign = sum(n for c, v in buffer_src.items() for t, n in v.items() if t != owner.get(c, t))
    return foreign / total if total else 0.0


def buffer_conflict(buf: dict, native: list[int], specs: list) -> float:
    """バッファのサンプルのうち、ラベルがその文脈の本来のタスクの規則と食い違うものの割合。"""
    owner = {c: k for k, c in enumerate(native)}
    if buf["labels"].numel() == 0:
        return 0.0
    bad = 0
    for i in range(buf["labels"].shape[0]):
        a, b = specs[owner.get(int(buf["ctx"][i]), int(buf["src"][i]))].relevant
        bad += int(((buf["bits"][i, a] > 0) ^ (buf["bits"][i, b] > 0)).long() != buf["labels"][i])
    return bad / buf["labels"].shape[0]


def run_one(job: tuple) -> dict:
    name, cond, L, steps, sc, seed = job
    torch.set_num_threads(1)
    specs = [ParitySpec(n_inputs=8, relevant=p) for p in PAIRS]
    torch.manual_seed(seed)
    r = run_context_sequence(cond, specs, 32, torch.Generator().manual_seed(seed + 50_000),
                             torch.Generator().manual_seed(seed + 90_000), transition_steps=L, steps=steps,
                             store_consistent=sc[0], train_consistent=sc[1], diagnostics=True)
    d = r["diag"]
    row = {"seed": seed, "R": r["R"], "n_contexts": r["n_contexts"], **forgetting_metrics(r["R"], chance=0.5),
           "final_chosen": d["chosen"][-1], "final_ctx_acc": d["ctx_acc"][-1],
           "buffer_src": {str(c): {str(t): n for t, n in v.items()} for c, v in d["buffer_src"].items()},
           "trained_src": d["trained_src"]}
    if cond != "oracle":
        T = len(specs)
        # 本来の文脈で評価した場合の最終正確度（生の値）と、テスト時の文脈選択の正しさ
        nat = native_contexts(r["assign"], T)
        row.update(assignment_purity(r["assign"], T), created_at=creation_steps(r["assign"]), native=nat,
                   buffer_contamination=buffer_contamination(d["buffer_src"], nat),
                   buffer_conflict=buffer_conflict(d["buffer"], nat, specs),
                   final_acc_native_ctx=sum(d["ctx_acc"][-1][j][nat[j]] for j in range(T)) / T,
                   final_selection_correct=sum(d["chosen"][-1][j] == nat[j] for j in range(T)) / T)
    return {"key": f"{name}_L{L}_steps{steps}", "row": row}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=list(PHASES), default="diag")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    seeds = args.seeds or (EVAL_SEEDS if args.phase == "eval" else DEV_SEEDS)
    out_name = {"prelim": "gated_long_dev", "diag": "gated_long_diag", "fix": "gated_long_fix_dev",
                "eval": "gated_long_eval"}[args.phase]
    out_path = args.out or Path(f"results/context_inference/{out_name}.json")
    jobs = [(*cfg, s) for cfg in PHASES[args.phase] for s in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(run_one, jobs)
    out: dict = {"環境": env_info(), "説明": f"長い移行期間での限界の原因の切り分けと修正（12.6.116節）。phase={args.phase}",
                 "seeds": seeds, "runs": {}}
    keys = ("ACC", "BWT", "LA", "n_contexts", "distinct", "buffer_contamination", "buffer_conflict", "final_acc_native_ctx",
            "final_selection_correct")
    for key in dict.fromkeys(x["key"] for x in results):
        rows = [x["row"] for x in results if x["key"] == key]
        summ = {k: {"mean": float(torch.tensor([float(x[k]) for x in rows]).mean()),
                    "std": float(torch.tensor([float(x[k]) for x in rows]).std(unbiased=False))}
                for k in keys if k in rows[0]}
        out["runs"][key] = {"summary": summ, "per_seed": rows}
        print(key, " ".join(f"{k}={v['mean']:.3f}±{v['std']:.3f}" for k, v in summ.items()), flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
