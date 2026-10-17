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

12.6.121節（L = 500 が L = 250 に届かない残りの差）:
    --phase residual_diag（results/context_inference/gated_residual_diag.json）… 本当の出どころでサンプルを除く
        反実仮想（diag_oracle_filter）と、新しい文脈の作成時に最適化器を作り直す条件を、修正1（12.6.116節）と比べる。
    --phase residual_fix（gated_residual_fix_dev.json）… reset_opt_on_new（修正1）と、それに train_consistent を
        加えたもの（修正2）を L = 0〜500 で 12.6.116節の機構と比べる。
    --phase residual_eval（gated_residual_eval.json）… 確定後に評価用シードで一度だけ測る。

12.6.125節（最適化器の作り直しの代わりに、驚きに応じた可塑性の調節 plasticity_cap）:
    --phase neuromod_fix（gated_neuromod_fix_dev.json）… 12.6.121節の確定機構（作り直し＋学習時の整合フィルタ）、
        作り直しなし、作り直しの代わりに plasticity_cap を L = 0〜500 で比べる。
    --phase neuromod_sgd（gated_neuromod_sgd_dev.json）… Adam 以外（モーメントなし SGD）でも作成直後の過大な更新が
        起きるか、作り直し・plasticity_cap が効くかを確かめる。
    --phase neuromod_fix2 / neuromod_fix3（gated_neuromod_fix2_dev.json / gated_neuromod_fix3_dev.json）… 修正2
        （ū に縮める前の値を使う）と修正3（ρ = 2.0）。
    --phase neuromod_ctrl（gated_neuromod_ctrl_dev.json）… 統制。作り直し・調節なしで学習率だけを下げる。
    --phase neuromod_eval（gated_neuromod_eval.json）… 確定後に評価用シードで一度だけ測る。
    --phase neuromod_eval_ctrl（gated_neuromod_eval_ctrl.json）… 統制（学習率 0.01）を評価用シードで一度だけ測る。

開発用シード 100〜104 のみを --phase prelim / diag / fix / residual_diag / residual_fix / neuromod_fix〜fix3 /
neuromod_sgd / neuromod_ctrl で使う。

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

# (名前, 条件, L, steps, run_context_sequence への追加の引数)
NONE = {}
STORE = {"store_consistent": True}
BOTH = {"store_consistent": True, "train_consistent": True}
FINAL = STORE  # 開発用シードで確定した機構（12.6.116節、修正1）
RESET = {"store_consistent": True, "reset_opt_on_new": True}  # 12.6.121節の修正1
RESET_TRAIN = {**RESET, "train_consistent": True}  # 12.6.121節の修正2
FINAL_RESIDUAL = RESET_TRAIN  # 開発用シードで確定した機構（12.6.121節、修正2）
TRAIN = {"store_consistent": True, "train_consistent": True}  # 作り直しなし
CAP = {**TRAIN, "plasticity_cap": 1.5}  # 12.6.125節の修正1（作り直しの代わり）
CAP_RAW = {**CAP, "plasticity_ref": "raw"}  # 12.6.125節の修正2
CAP2 = {**TRAIN, "plasticity_cap": 2.0}  # 12.6.125節の修正3
FINAL_NEUROMOD = CAP2  # 開発用シードで確定した機構（12.6.125節、修正3）
SGD_LR = 0.5
SGD = {"optimizer": "sgd", "lr": SGD_LR}
LS = ((0, 500), (100, 500), (250, 500), (500, 500), (500, 1000))
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
    "residual_diag": [(n, "inferred_gated", L, s, sc)
                      for sc, n in ((STORE, "gated_store"), ({**STORE, "diag_oracle_filter": "new"}, "oracle_filter_new"),
                                    ({**STORE, "diag_oracle_filter": "old"}, "oracle_filter_old"),
                                    ({**STORE, "diag_oracle_filter": "both"}, "oracle_filter_both"),
                                    (RESET, "gated_store_reset"),
                                    ({**RESET, "diag_oracle_filter": "new"}, "reset_oracle_filter_new"),
                                    ({**RESET, "diag_oracle_filter": "old"}, "reset_oracle_filter_old"),
                                    ({**RESET, "diag_oracle_filter": "both"}, "reset_oracle_filter_both"))
                      for L, s in ((250, 500), (500, 500), (500, 1000))],
    "residual_fix": [(n, "inferred_gated", L, s, sc)
                     for sc, n in ((STORE, "gated_store"), (RESET, "gated_store_reset"),
                                   (RESET_TRAIN, "gated_store_reset_train")) for L, s in LS],
    "residual_eval": [(n, "inferred_gated", L, s, sc)
                      for sc, n in ((STORE, "gated_store"), (FINAL_RESIDUAL, "gated_store_reset_train")) for L, s in LS],
    "neuromod_fix": [(n, "inferred_gated", L, s, sc)
                     for sc, n in ((FINAL_RESIDUAL, "gated_store_reset_train"), (TRAIN, "gated_store_train"),
                                   (CAP, "gated_store_cap_train")) for L, s in LS],
    "neuromod_fix2": [(n, "inferred_gated", L, s, sc) for sc, n in ((CAP_RAW, "gated_store_capraw_train"),) for L, s in LS],
    "neuromod_fix3": [(n, "inferred_gated", L, s, sc) for sc, n in ((CAP2, "gated_store_cap2_train"),) for L, s in LS],
    "neuromod_sgd": [(n, "inferred_gated", L, s, sc)
                     for sc, n in (({**TRAIN, **SGD}, "sgd_store_train"),
                                   ({**FINAL_RESIDUAL, **SGD}, "sgd_store_reset_train"),
                                   ({**FINAL_NEUROMOD, **SGD}, "sgd_store_cap2_train")) for L, s in LS],
    # 統制: 可塑性の調節の効果が、全体の学習率を下げただけの効果かどうか（作り直し・調節なし）
    "neuromod_ctrl": [(n, "inferred_gated", L, s, sc)
                      for sc, n in (({**TRAIN, "lr": 0.02}, "gated_store_train_lr0.02"),
                                    ({**TRAIN, "lr": 0.01}, "gated_store_train_lr0.01")) for L, s in LS],
    # 統制（学習率だけを下げる）を評価用シードで一度だけ測る。neuromod_eval の結果を見る前に追加を決めた
    "neuromod_eval_ctrl": [(n, "inferred_gated", L, s, sc)
                           for sc, n in (({**TRAIN, "lr": 0.01}, "gated_store_train_lr0.01"),) for L, s in LS],
    "neuromod_eval": [(n, "inferred_gated", L, s, sc)
                      for sc, n in ((FINAL_RESIDUAL, "gated_store_reset_train"), (FINAL_NEUROMOD, "gated_store_cap2_train"))
                      for L, s in LS],
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


def creation_dynamics(assign: list[tuple[int, int]], trace: list[dict], upd: list[tuple], native: list[int],
                      window: int = 20, after: int = 50) -> dict:
    """新しい文脈の作成（タスク k ≥ 1）の前後で、前のタスクの本来の文脈での正確度の落ち込みと、作成直前・直後
    window ステップの勾配ノルム・更新ノルムの平均（12.6.121節）。trace は作成時点を挟む 10 ステップ刻み。"""
    created = {}
    seen = {assign[0][1]}
    start = {}
    for i, (k, c) in enumerate(assign):
        start.setdefault(k, i)
        if c not in seen:
            seen.add(c)
            created[k] = i - start[k]
    drops, g_pre, g_post, u_pre, u_post = [], [], [], [], []
    for k, s0 in created.items():
        if k == 0:
            continue
        pre = [e for e in trace if e["task"] == k and e["step"] < s0]
        post = [e for e in trace if e["task"] == k and s0 + after - 10 <= e["step"] <= s0 + after]
        if pre and post:
            c = native[k - 1]
            drops.append(pre[-1]["acc"][k - 1][c] - post[-1]["acc"][k - 1][c])
        before = [u for u in upd if u[0] == k and s0 - window <= u[1] < s0]
        aft = [u for u in upd if u[0] == k and s0 <= u[1] < s0 + window]
        if before and aft:
            g_pre.append(sum(u[3] for u in before) / len(before))
            g_post.append(sum(u[3] for u in aft) / len(aft))
            u_pre.append(sum(u[4] for u in before) / len(before))
            u_post.append(sum(u[4] for u in aft) / len(aft))
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    return {"prev_drop_at_creation": mean(drops), "grad_pre": mean(g_pre), "grad_post": mean(g_post),
            "upd_pre": mean(u_pre), "upd_post": mean(u_post)}


def run_one(job: tuple) -> dict:
    name, cond, L, steps, sc, seed = job
    torch.set_num_threads(1)
    specs = [ParitySpec(n_inputs=8, relevant=p) for p in PAIRS]
    torch.manual_seed(seed)
    r = run_context_sequence(cond, specs, 32, torch.Generator().manual_seed(seed + 50_000),
                             torch.Generator().manual_seed(seed + 90_000), transition_steps=L, steps=steps,
                             diagnostics=True, trace_every=10, **sc)
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
        row.update(creation_dynamics(r["assign"], d["trace"], d["upd"], nat))
    return {"key": f"{name}_L{L}_steps{steps}", "row": row}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=list(PHASES), default="diag")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    seeds = args.seeds or (EVAL_SEEDS if args.phase in ("eval", "residual_eval", "neuromod_eval", "neuromod_eval_ctrl")
                          else DEV_SEEDS)
    out_name = {"prelim": "gated_long_dev", "diag": "gated_long_diag", "fix": "gated_long_fix_dev",
                "eval": "gated_long_eval", "residual_diag": "gated_residual_diag",
                "residual_fix": "gated_residual_fix_dev", "residual_eval": "gated_residual_eval",
                "neuromod_fix": "gated_neuromod_fix_dev", "neuromod_fix2": "gated_neuromod_fix2_dev",
                "neuromod_fix3": "gated_neuromod_fix3_dev",
                "neuromod_sgd": "gated_neuromod_sgd_dev",
                "neuromod_ctrl": "gated_neuromod_ctrl_dev",
                "neuromod_eval": "gated_neuromod_eval", "neuromod_eval_ctrl": "gated_neuromod_eval_ctrl"}[args.phase]
    out_path = args.out or Path(f"results/context_inference/{out_name}.json")
    jobs = [(*cfg, s) for cfg in PHASES[args.phase] for s in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(run_one, jobs)
    sec = ("12.6.121節" if args.phase.startswith("residual") else
           "12.6.125節" if args.phase.startswith("neuromod") else "12.6.116節")
    out: dict = {"環境": env_info(), "説明": f"長い移行期間での限界の原因の切り分けと修正（{sec}）。phase={args.phase}",
                 "seeds": seeds, "runs": {}}
    keys = ("ACC", "BWT", "LA", "n_contexts", "distinct", "buffer_contamination", "buffer_conflict", "final_acc_native_ctx",
            "final_selection_correct", "prev_drop_at_creation", "grad_pre", "grad_post", "upd_pre", "upd_post")
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
