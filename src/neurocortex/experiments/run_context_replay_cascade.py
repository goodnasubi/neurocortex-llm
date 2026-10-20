"""シナプスの固定化（多時間尺度の隠れ変数の鎖。Benna & Fusi 2016 のカスケード）で、加えた痕跡の固定化と新しい写像の
共有の重みの上の競合を変え、取り戻した一般化と新しいタスクの学習を両立できるか（12.6.233〜234節）。

12.6.163〜170・225〜232節では、リプレイの配分（量・時期・中身・睡眠の段階）を変えても、keep と ach_c を結ぶ一般化と
保護の交換の曲線の上を動くだけだった。ここではリプレイの配分を変えず、共有の重みの可塑性そのものを多時間尺度にする:
    base_cas      … 加えた痕跡なし（base）＋カスケード（カスケードだけの効果）
    keep_cas      … keep（加えた痕跡を一様にリプレイ）＋カスケード
    ach_c_cas     … ach_c（連続的な新しさの信号で固定化を配分）＋カスケード（主）
    ach_c_cas_all … ach_c＋カスケードを文脈の one-hot の列にもかける（範囲の対照）
base・keep・ach_c は 12.6.167〜168節の結果ファイル（同じシード・同じ実装で再現する）と対にして使う。

    --phase dev（開発用シード 100〜104）/ eval（評価用シード 0〜4 で一度だけ）
    結果は results/context_inference/replay_cascade_{phase}.json

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_replay_cascade --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_context_replay_cascade --analyze <結果ファイル> <12.6.167〜168節の結果ファイル>
"""

from __future__ import annotations

import argparse
import json
import tempfile
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import torch

from ..context_inference import run_context_sequence
from ..run_env import env_info
from .run_context_episodic_readout import FAMILIES, LS, MECHS_USED, SELECTS, specs_for
from .run_context_evidence_info import MECHS, SELECTION_TRIALS
from .run_context_gated_long import native_contexts
from .run_context_replay_novelty import SUMMARY_KEYS
from .run_context_replay_split import CONFIGS as SPLIT_CONFIGS
from .run_context_replay_split import _criteria
from .run_context_replay_split import analyze as _split_analyze
from .run_context_writeback import CUE_N, NEW_TASK, SELECT, _exp

DEV_SEEDS = list(range(100, 105))
EVAL_SEEDS = list(range(0, 5))
# カスケードの設定（計測前に決めた。m = 4 段、C_k = 2^{k−1}、g_{k,k+1} = 0.05·4^{−(k−1)}。一定の勾配を 500 ステップ
# 続けたとき、見える重みは可塑性だけの場合の約 22% 動き、その後の 500 ステップで約 12% に戻る）
CAS = {"cascade_levels": 4, "cascade_g": 0.05}
CONFIGS: dict[str, dict] = {
    "base": SPLIT_CONFIGS["base"],
    "keep": SPLIT_CONFIGS["keep"],
    "ach_c": SPLIT_CONFIGS["ach_c"],
    "base_cas": {**SPLIT_CONFIGS["base"], **CAS},
    "keep_cas": {**SPLIT_CONFIGS["keep"], **CAS},
    "ach_c_cas": {**SPLIT_CONFIGS["ach_c"], **CAS},
    "ach_c_cas_all": {**SPLIT_CONFIGS["ach_c"], **CAS, "cascade_scope": "all"},
}
REUSED = ("base", "keep", "ach_c")  # 12.6.167〜168節の結果ファイルから読む
NEW = ("base_cas", "keep_cas", "ach_c_cas", "ach_c_cas_all")
# 修正（12.6.233節。上限3回）は開発用シードの結果を見て追加する
FIXES: tuple[str, ...] = ()
FIX_CANDIDATES: tuple[str, ...] = ()
FRONTIER = ("keep_cas", "ach_c_cas", "ach_c_cas_all")  # 交換の曲線からの残差を測る条件
RULES = ("keep", "ach_c", *NEW, *FIXES)


def jobs(configs=NEW, families=FAMILIES) -> list[tuple]:
    return [(f, m, L, cfg) for m in MECHS_USED for f in families for L in LS for cfg in configs]


def run_job(job: tuple, kw: dict) -> dict:
    """job = (課題, 機構, L, 設定の名前, シード)。12.6.231節の run_job と同じ量に、カスケードの診断を加える。"""
    fam, mech, L, cfg, seed = job
    torch.set_num_threads(1)
    specs = specs_for(fam)
    T0 = len(specs)
    specs = specs + [NEW_TASK[fam]]
    T = len(specs)
    torch.manual_seed(seed)
    r = run_context_sequence("inferred_gated", specs, 32, torch.Generator().manual_seed(seed + 50_000),
                             torch.Generator().manual_seed(seed + 90_000), transition_steps=L, steps=500,
                             diagnostics=True, cue_n=CUE_N, selection_trials=SELECTION_TRIALS, **MECHS[mech],
                             **SELECTS[SELECT], writeback_at=T0, **kw)
    d = r["diag"]
    nat = native_contexts(r["assign"], T)
    ch = r["selection_choices"]
    acc = d["ctx_acc"][-1]
    orig = range(T0)
    w = d["writeback"]
    ci = [w["used"].index(nat[j]) for j in orig]
    mix = d.get("replay_mix", {}).get(T0, {})
    row = {"seed": seed, "n_contexts": r["n_contexts"], "native": nat,
           "selection_acc": sum(c == nat[j] for j in orig for c in ch[j]) / (T0 * SELECTION_TRIALS),
           "exp_end": _exp(acc, ch, orig), "end_nat": [2 * acc[j][nat[j]] - 1 for j in orig],
           "pre_nat": [2 * w["pre"]["fresh"][j][ci[j]] - 1 for j in orig],
           "gap": [None if w["pre"]["own_buf"][j][ci[j]] is None else w["pre"]["own_buf"][j][ci[j]] - w["pre"]["fresh"][j][ci[j]]
                   for j in orig],
           "new_nat": 2 * acc[T0][nat[T0]] - 1, "new_exp": _exp(acc, ch, [T0]), "new_ctx": nat[T0] not in w["used"],
           "mix_own": mix.get("own"), "mix_kept": mix.get("kept"), "mix_size": mix.get("size"),
           "mix_fam": mix.get("fam"), "sleep_steps": mix.get("sleep"),
           "cas_gap": d.get("cascade", {}).get(T0, {}).get("gap"), "cas_move": d.get("cascade", {}).get(T0, {}).get("move")}
    return {"key": f"{fam}/{mech}_L{L}/{cfg}", "row": row}


def run_one(job: tuple, table: dict) -> dict:
    return run_job(job, table[job[3]])


def analyze(paths: list[Path], rules=RULES) -> dict:
    """12.6.169節の集計（同じシードの base との差、組の分類も同じ）に、12.6.233節の判定を付ける。
    新しい条件を測ったシードに限るので、10シードの結果ファイルと対にしても、ここで測ったシードだけを使う。"""
    seeds: set[int] = set()
    tmp = []
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        if any(k.endswith(NEW + FIXES) for k in d["runs"]):
            seeds.update(d["seeds"])
        tmp.append(d)
    with tempfile.TemporaryDirectory() as td:
        fs = []
        for i, d in enumerate(tmp):
            if seeds:
                for v in d["runs"].values():
                    v["per_seed"] = [x for x in v["per_seed"] if x["seed"] in seeds]
            f = Path(td) / f"{i}.json"
            f.write_text(json.dumps({"runs": d["runs"]}), encoding="utf-8")
            fs.append(f)
        res = _split_analyze(fs, tuple(r for r in rules if any(k.endswith("/" + r) for d in tmp for k in d["runs"])))
    res["seeds"] = sorted(seeds)
    res["verdict"] = verdict(res)
    return res


def frontier_residual(o: dict, a: dict, w: str) -> float | None:
    """keep と ach_c を結ぶ直線（横軸 = 30組の new の平均、縦軸 = 14 ビットの上書きありの組の end）からの、条件 w の
    end の残差（12.6.226節の残る課題 (2)。正なら曲線の外側）。"""
    keys = ("end_keep", "end_ach_c", f"end_{w}")
    if any(o.get(k) is None for k in keys) or any(a.get(f"new_{r}") is None for r in ("keep", "ach_c", w)):
        return None
    dn = a["new_keep"] - a["new_ach_c"]
    if abs(dn) < 1e-9:
        return None
    line = o["end_ach_c"] + (o["end_keep"] - o["end_ach_c"]) * (a[f"new_{w}"] - a["new_ach_c"]) / dn
    return o[f"end_{w}"] - line


def verdict(res: dict) -> dict:
    """12.6.233節で計測前に固定した判定条件。"""
    o, n, a = res["over_14bit"], res["not_over"], res["all"]
    out: dict = {}
    for w in ("keep", "ach_c", *NEW, *FIXES):
        if o.get(f"end_{w}") is None:
            continue
        v = _criteria(o, n, a, w)
        if w not in ("keep", "ach_c") and o.get("end_ach_c") is not None:
            # D4: ach_c からの上積み（一般化 +0.02 以上、かつ大きく損なった回が増えない）
            v["D4"] = bool(o[f"end_{w}"] - o["end_ach_c"] >= 0.02 and a[f"bad_{w}"] <= a["bad_ach_c"])
        r = frontier_residual(o, a, w) if w in (*FRONTIER, *FIXES) else None
        if r is not None:
            v["residual"] = r
            v["F"] = bool(r >= 0.02)  # 交換の曲線より +0.02 以上外側
        out[w] = v
    out["H"] = out.get("ach_c_cas", {}).get("success")
    out["F"] = any(out.get(w, {}).get("F") for w in FRONTIER) if any(w in out for w in FRONTIER) else None
    if "base_cas" in out:
        # B: カスケードだけ（加えた痕跡なし）で新しいタスクを損なわない（30組の new の平均 −0.02 以上、−0.10 未満の回が
        # base の 1% 以下）
        out["B"] = bool(a["new_base_cas"] >= -0.02 and a["bad_frac_base_cas"] <= 0.01)
    out["H_fix"] = any(out.get(w, {}).get("success") for w in FIX_CANDIDATES) if FIX_CANDIDATES else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", type=Path, nargs="+", help="結果ファイル（12.6.167〜168節のものも）を集計して判定する")
    ap.add_argument("--configs", nargs="+", default=[*NEW, *FIXES])
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
    out_path = args.out or Path(f"results/context_inference/replay_cascade_{args.phase}.json")
    all_jobs = [(*cfg, s) for cfg in jobs(tuple(args.configs), tuple(args.families)) for s in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(partial(run_one, table=CONFIGS), all_jobs, chunksize=1)
    out: dict = {"環境": env_info(),
                 "説明": f"シナプスの固定化のカスケード（12.6.233〜234節）。phase={args.phase}",
                 "seeds": seeds, "select": {SELECT: SELECTS[SELECT]}, "new_task": {k: repr(v) for k, v in NEW_TASK.items()},
                 "configs": {k: CONFIGS[k] for k in args.configs}, "selection_trials": SELECTION_TRIALS,
                 "runs": {}}
    for key in dict.fromkeys(x["key"] for x in results):
        rows = [x["row"] for x in results if x["key"] == key]
        summ = {}
        for k in (*SUMMARY_KEYS, "cas_gap", "cas_move"):
            vals = [float(x[k]) for x in rows if x.get(k) is not None]
            if vals:
                t = torch.tensor(vals)
                summ[k] = {"mean": float(t.mean()), "std": float(t.std(unbiased=False)), "n": len(vals)}
        out["runs"][key] = {"summary": summ, "per_seed": rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out_path, len(results), "runs", flush=True)


if __name__ == "__main__":
    main()
