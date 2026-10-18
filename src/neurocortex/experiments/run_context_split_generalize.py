"""文脈固有と共有で可塑性を分ける機構（12.6.129〜130節）の汎化の確認と、学習率への依存の改善（12.6.131〜132節）。

汎化（12.6.131節。設定は 12.6.129節の確定値のまま変えない）:
    12.6.121節の確定機構（store_consistent + reset_opt_on_new + train_consistent）と分離可塑性（split_window = 100,
    split_ctx_gain = 2.0, split_shared_gain = 0.2）を、これまでの課題で比べる。参考に、作り直しも分離もしない
    もの（store_consistent + train_consistent）と oracle も測る。
      flip_m{1..4} … 似たタスク（FlipSpec, 12.6.108節。14入力、不一致率 2^-m）
      cue_q{0.9,0.75} … 入力分布が異なる課題（CueSpec, 12.6.111節。16入力）
      cuen{1,2,4,8} … 手がかりの数を減らした場合（パリティ課題, 12.6.107節。16 は 12.6.130節の結果）
    それぞれ L = 0/100/250/500（500ステップ）と L = 500・1000ステップ。
    --phase gen_dev（開発用シードでの動作確認, gen_dev.json）/ gen_eval（評価用シードで一度だけ, gen_eval.json）

学習率への依存（12.6.132節）: パリティ課題で、土台の学習率 lr = 0.01・0.02 でも悪化しない形を開発用シードで探す。
    --phase lr_fix1〜3（split_lr_fix{1,2,3}_dev.json）/ lr_eval（split_lr_eval.json）

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_split_generalize --phase gen_dev
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import torch

from ..basal_ganglia_core import ParitySpec
from ..context_inference import CueSpec, FlipSpec, assignment_purity, run_context_sequence
from ..continual_metrics import forgetting_metrics
from ..run_env import env_info
from .run_context_gated_long import (DEV_SEEDS, EVAL_SEEDS, FINAL_RESIDUAL, FINAL_SPLIT, LS, NONE, TRAIN,
                                     buffer_contamination, creation_dynamics, creation_steps, native_contexts)
from .run_context_similar_tasks import task_specs as flip_specs
from .run_context_unsupervised import specs_for as cue_specs

PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]


def family_specs(family: str) -> tuple[list, dict]:
    """課題の族の名前から (specs, run_context_sequence への追加の引数) を返す。"""
    if family == "parity":
        return [ParitySpec(n_inputs=8, relevant=p) for p in PAIRS], {}
    if family.startswith("flip_m"):
        return flip_specs(int(family[6:])), {}
    if family.startswith("cue_q"):
        return cue_specs(float(family[5:])), {}
    if family.startswith("cuen"):
        return [ParitySpec(n_inputs=8, relevant=p) for p in PAIRS], {"cue_n": int(family[4:])}
    raise ValueError(family)


def true_labels(spec, bits: torch.Tensor) -> torch.Tensor:
    """make_batch と同じ規則で、入力 bits に対する spec の正解を返す。"""
    a, b = spec.relevant
    lab = (bits[:, a] > 0) ^ (bits[:, b] > 0)
    if isinstance(spec, FlipSpec) and spec.flip:
        lab = lab ^ (bits[:, list(spec.flip)] > 0).all(dim=1)
    return lab.long()


def buffer_conflict(buf: dict, native: list[int], specs: list) -> float:
    """バッファのうち、ラベルがその文脈の本来のタスクの規則と食い違うものの割合（FlipSpec にも対応）。"""
    if buf["labels"].numel() == 0:
        return 0.0
    owner = {c: k for k, c in enumerate(native)}
    bad = 0
    for i in range(buf["labels"].shape[0]):
        spec = specs[owner.get(int(buf["ctx"][i]), int(buf["src"][i]))]
        bad += int(true_labels(spec, buf["bits"][i:i + 1])[0] != buf["labels"][i])
    return bad / buf["labels"].shape[0]


MECHS = (("oracle", "oracle", NONE), ("train", "inferred_gated", TRAIN), ("reset", "inferred_gated", FINAL_RESIDUAL),
         ("split", "inferred_gated", FINAL_SPLIT))
GEN_FAMILIES = ("flip_m1", "flip_m2", "flip_m3", "flip_m4", "cue_q0.9", "cue_q0.75", "cuen1", "cuen2", "cuen4", "cuen8")


def gen_jobs() -> list[tuple]:
    out = []
    for fam in GEN_FAMILIES:
        for name, cond, sc in MECHS:
            if cond == "oracle" and fam.startswith("cuen"):
                continue  # oracle は手がかりを使わない
            out += [(fam, name, cond, L, s, sc) for L, s in LS]
    return out


# 学習率への依存（12.6.132節）。修正は開発用シードの結果を見て追加する（上限3回。修正1で条件を満たし確定）
LR_FIXES: dict[int, dict] = {
    1: {**FINAL_SPLIT, "split_shared_lr_floor": 0.01},
}
LRS = (0.01, 0.02)


def lr_jobs(fix: dict | None) -> list[tuple]:
    out = []
    for lr in LRS:
        cfgs = [("train", {**TRAIN, "lr": lr}), ("split", {**FINAL_SPLIT, "lr": lr})] if fix is None else []
        if fix is not None:
            cfgs.append(("splitfix", {**fix, "lr": lr}))
        out += [("parity", f"{n}_lr{lr}", "inferred_gated", L, s, sc) for n, sc in cfgs for L, s in LS]
    return out


def lr_eval_jobs() -> list[tuple]:
    fx = LR_FIXES[max(LR_FIXES)] if LR_FIXES else FINAL_SPLIT
    out = []
    for lr in LRS:
        for n, sc in (("train", {**TRAIN, "lr": lr}), ("split", {**FINAL_SPLIT, "lr": lr}), ("splitfix", {**fx, "lr": lr})):
            out += [("parity", f"{n}_lr{lr}", "inferred_gated", L, s, sc) for L, s in LS]
    # lr = 0.05 では下限が効かず確定機構と同じになることの確認
    out += [("parity", "splitfix_lr0.05", "inferred_gated", L, s, fx) for L, s in LS]
    return out


PHASES = {"gen_dev": gen_jobs, "gen_eval": gen_jobs, "lr_eval": lr_eval_jobs, "lr_base_dev": lambda: lr_jobs(None),
          **{f"lr_fix{i}": (lambda f=f: lr_jobs(f)) for i, f in LR_FIXES.items()}}
OUT_NAMES = {"gen_dev": "split_gen_dev", "gen_eval": "split_gen_eval", "lr_eval": "split_lr_eval",
             "lr_base_dev": "split_lr_base_dev", **{f"lr_fix{i}": f"split_lr_fix{i}_dev" for i in LR_FIXES}}


def run_one(job: tuple) -> dict:
    fam, name, cond, L, steps, sc, seed = job
    torch.set_num_threads(1)
    specs, extra = family_specs(fam)
    torch.manual_seed(seed)
    r = run_context_sequence(cond, specs, 32, torch.Generator().manual_seed(seed + 50_000),
                             torch.Generator().manual_seed(seed + 90_000), transition_steps=L, steps=steps,
                             diagnostics=True, trace_every=10, **extra, **sc)
    d = r["diag"]
    row = {"seed": seed, "R": r["R"], "n_contexts": r["n_contexts"], **forgetting_metrics(r["R"], chance=0.5)}
    if cond != "oracle":
        T = len(specs)
        nat = native_contexts(r["assign"], T)
        row.update(assignment_purity(r["assign"], T), created_at=creation_steps(r["assign"]), native=nat,
                   buffer_contamination=buffer_contamination(d["buffer_src"], nat),
                   buffer_conflict=buffer_conflict(d["buffer"], nat, specs),
                   final_acc_native_ctx=sum(d["ctx_acc"][-1][j][nat[j]] for j in range(T)) / T,
                   final_selection_correct=sum(d["chosen"][-1][j] == nat[j] for j in range(T)) / T)
        row.update(creation_dynamics(r["assign"], d["trace"], d["upd"], nat))
    return {"key": f"{fam}/{name}_L{L}_steps{steps}", "row": row}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=list(PHASES), default="gen_dev")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    seeds = args.seeds or (EVAL_SEEDS if args.phase.endswith("_eval") else DEV_SEEDS)
    out_path = args.out or Path(f"results/context_inference/{OUT_NAMES[args.phase]}.json")
    jobs = [(*cfg, s) for cfg in PHASES[args.phase]() for s in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(run_one, jobs)
    sec = "12.6.131節" if args.phase.startswith("gen") else "12.6.132節"
    out: dict = {"環境": env_info(), "説明": f"分離可塑性の汎化と学習率への依存（{sec}）。phase={args.phase}",
                 "seeds": seeds, "runs": {}}
    keys = ("ACC", "BWT", "LA", "n_contexts", "distinct", "buffer_conflict", "final_acc_native_ctx",
            "final_selection_correct", "prev_drop_at_creation", "upd_post")
    for key in dict.fromkeys(x["key"] for x in results):
        rows = [x["row"] for x in results if x["key"] == key]
        summ = {k: {"mean": float(torch.tensor([float(x[k]) for x in rows]).nanmean()),
                    "std": float(torch.tensor([float(x[k]) for x in rows]).std(unbiased=False))}
                for k in keys if k in rows[0]}
        out["runs"][key] = {"summary": summ, "per_seed": rows}
        print(key, " ".join(f"{k}={v['mean']:.3f}±{v['std']:.3f}" for k, v in summ.items()), flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
