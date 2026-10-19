"""テスト時の再固定化、隠れ層の表現での近さによる読み出し、保留サンプルの正確さによる調停（12.6.159〜160節）。

12.6.156節では、上書きされた本来の文脈の一般化は、個別の痕跡のラベルの読み出しでは取り戻せなかった（14 ビットの課題で
+0.002〜+0.006）。ここでは選択の後の予測を、(1) 選んだ文脈の痕跡で最終モデルの複製を短く学び直した予測（再固定化）、
(2) 痕跡を入力のビットではなく隠れ層の表現（嗅内皮質の表現にあたる）の近さで検索する読み出し、(3) 保留サンプルで
測った正確さで param・mix・switch を選ぶ調停、にして期待 ACC を比べる。

    --phase dev（開発用シード）/ eval（評価用シードで一度だけ）
    結果は results/context_inference/reconsolidation_{phase}.json

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_reconsolidation --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_context_reconsolidation --analyze <結果ファイル>
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import torch

from ..context_inference import EPISODIC_EXTRA_READOUTS, EPISODIC_READOUTS, RECON_READOUTS, run_context_sequence
from ..continual_metrics import forgetting_metrics
from ..run_env import env_info
from .run_context_episodic_readout import EPI_KW, FAMILIES, LS, MECHS_USED, SELECTS, specs_for
from .run_context_evidence_info import MECHS, SELECTION_TRIALS
from .run_context_gated_long import DEV_SEEDS, EVAL_SEEDS, native_contexts
from .run_context_replay_verify import _mean

CUE_N = 16
SELECT = "gen_replay"  # 12.6.151節の設定（P = 4、バッファから候補を取る）
READOUTS = ("param", "mix_all", "switch_all", "mix_all200", *EPISODIC_EXTRA_READOUTS, *RECON_READOUTS)
RECON_KW = {"episodic_extra": True, "recon_steps": 100, "recon_lr": 0.01, "recon_batch": 32}
CONFIGS: dict[str, dict] = {"base": RECON_KW}
# 修正は開発用シードの結果を見て追加する（上限3回）


def jobs(configs=("base",), families=FAMILIES) -> list[tuple]:
    return [(f, m, L, cfg) for m in MECHS_USED for f in families for L in LS for cfg in configs]


def run_job(job: tuple, recon_kw: dict) -> dict:
    """job = (課題, 機構, L, 設定の名前, シード)。"""
    fam, mech, L, cfg, seed = job
    torch.set_num_threads(1)
    specs = specs_for(fam)
    torch.manual_seed(seed)
    r = run_context_sequence("inferred_gated", specs, 32, torch.Generator().manual_seed(seed + 50_000),
                             torch.Generator().manual_seed(seed + 90_000), transition_steps=L, steps=500,
                             diagnostics=True, cue_n=CUE_N, selection_trials=SELECTION_TRIALS, **MECHS[mech],
                             **SELECTS[SELECT], **EPI_KW, **recon_kw)
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
           "cover_nat": _mean([d["episodic_cover"][j][nat[j]] for j in range(T)])}
    for k in READOUTS:
        if k not in ea[0][0]:
            continue
        row[f"exp_{k}"] = sum(2 * ea[j][c][k] - 1 for j in range(T) for c in ch[j]) / n_tr
        row[f"nat_{k}"] = sum(2 * ea[j][nat[j]][k] - 1 for j in range(T)) / T
    if "episodic_arb" in d:
        # 調停で記憶側（mix・switch）を選んだ割合（選んだ文脈について）
        for kind in ("bits", "rep"):
            row[f"arb_mem_{kind}"] = sum(d["episodic_arb"][j][c][kind] != "param"
                                         for j in range(T) for c in ch[j]) / n_tr
    if "recon_side" in d:
        # 本来の文脈 c で学び直した複製が、他のタスク j（本来の文脈が c でない）の本来の文脈での正確度に及ぼす変化
        for name, side in d["recon_side"].items():
            vals = [side[c][j][nat[j]] for c in sorted(set(nat)) for j in range(T) if nat[j] != c]
            row[f"side_{name}"] = _mean(vals)
    return {"key": f"{fam}/{mech}_L{L}/{cfg}", "row": row}


def run_one(job: tuple, table: dict) -> dict:
    return run_job(job, table[job[3]])


SUMMARY_KEYS = ("ACC", "selection_acc", "expected_acc", "nat_fresh", "nat_own_buf", "nat_holdout", "cover_nat",
                *(f"exp_{k}" for k in READOUTS), *(f"nat_{k}" for k in READOUTS), "arb_mem_bits", "arb_mem_rep",
                "side_recon_all", "side_recon_inter")


def _m(runs: dict, key: str, k: str) -> float | None:
    s = runs.get(key, {}).get("summary", {}).get(k)
    return s["mean"] if s else None


def analyze(paths: list[Path], cfg: str = "base") -> dict:
    """組ごとに gap（12.6.151節の上書きの大きさ）と、各読み出しの期待 ACC の param との差を求め、判定に使う平均を返す。"""
    runs: dict = {}
    for p in paths:
        runs.update(json.loads(p.read_text(encoding="utf-8"))["runs"])
    rows = {}
    for key in runs:
        fam, ml, c = key.split("/")
        if c != cfg:
            continue
        own, fresh = _m(runs, key, "nat_own_buf"), _m(runs, key, "nat_fresh")
        r = {"gap": own - fresh if own is not None and fresh is not None else None,
             "sel": _m(runs, key, "selection_acc"), "cover_nat": _m(runs, key, "cover_nat"),
             "exp_param": _m(runs, key, "exp_param"), "nat_param": _m(runs, key, "nat_param"),
             "arb_mem_bits": _m(runs, key, "arb_mem_bits"), "arb_mem_rep": _m(runs, key, "arb_mem_rep"),
             "side_recon_all": _m(runs, key, "side_recon_all"), "side_recon_inter": _m(runs, key, "side_recon_inter")}
        for k in READOUTS[1:]:
            e = _m(runs, key, f"exp_{k}")
            r[f"d_{k}"] = e - r["exp_param"] if e is not None else None
            n = _m(runs, key, f"nat_{k}")
            r[f"dnat_{k}"] = n - r["nat_param"] if n is not None else None
        rows[f"{fam}/{ml}"] = r

    def group(pred) -> dict:
        g = {k: v for k, v in rows.items() if pred(k, v)}
        out = {"n": len(g), "combos": sorted(g)}
        for k in [f"d_{x}" for x in READOUTS[1:]] + ["side_recon_all", "side_recon_inter"]:
            vals = [v[k] for v in g.values() if v[k] is not None]
            out[k] = _mean(vals)
            out[f"min_{k}"] = min(vals) if vals else None
        return out

    over = lambda k, v: v["gap"] is not None and v["gap"] >= 0.10  # noqa: E731
    return {"config": cfg, "rows": rows,
            "over_14bit": group(lambda k, v: over(k, v) and k.startswith(("flip_m1", "flip7_m1"))),
            "flip8_m1_reset": group(lambda k, v: k.startswith("flip8_m1/reset")),
            "over": group(over),
            "not_over": group(lambda k, v: v["gap"] is not None and v["gap"] <= 0.03),
            "all": group(lambda k, v: True)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", type=Path, nargs="+", help="結果ファイルを集計して判定に使う量を表示する")
    ap.add_argument("--config", default="base", help="--analyze で集計する設定")
    ap.add_argument("--configs", nargs="+", default=["base"])
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--phase", choices=("dev", "eval"), default="dev")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.analyze:
        print(json.dumps(analyze(args.analyze, args.config), ensure_ascii=False, indent=1))
        return
    seeds = args.seeds or (EVAL_SEEDS if args.phase == "eval" else DEV_SEEDS)
    out_path = args.out or Path(f"results/context_inference/reconsolidation_{args.phase}.json")
    all_jobs = [(*cfg, s) for cfg in jobs(tuple(args.configs), tuple(args.families)) for s in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(partial(run_one, table=CONFIGS), all_jobs, chunksize=1)
    out: dict = {"環境": env_info(),
                 "説明": f"テスト時の再固定化・隠れ層の表現での読み出し・調停（12.6.159〜160節）。phase={args.phase}",
                 "seeds": seeds, "select": {SELECT: SELECTS[SELECT]}, "episodic": EPI_KW,
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
        print(key, " ".join(f"{k}={v['mean']:.3f}" for k, v in summ.items() if k.startswith(("sel", "exp_", "side"))),
              flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
