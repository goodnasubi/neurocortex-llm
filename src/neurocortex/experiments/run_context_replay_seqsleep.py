"""覚醒の後の NREM/REM の睡眠（Singh, Norman & Schapiro 2022 の順序と分業）で、取り戻した一般化と新しいタスクの学習を
両立できるか（12.6.231〜232節）。

12.6.167〜168節の nrem・nrem2 は、REM に似たステップで保存したラベル（海馬の痕跡）を学び、新しい文脈の写像を守らず、
新しいタスクを大きく損なった。Singh et al. では、覚醒中に新皮質が記憶を一度学び、REM では新皮質が自分の持つ記憶を
（海馬なしに）再生して保つ。ここでは覚醒中は ach_c（加えた痕跡を新しい入力と交互に学ぶ）とし、睡眠の段階を:
    ach_c_seq      … NREM = 海馬の痕跡（新しい文脈の痕跡と加えた痕跡、保存したラベル）、REM = 新皮質の自己生成
                     （新しい文脈を含む全ての文脈の痕跡を入力し、睡眠の始めの自分の出力の分布を標的にする）
    ach_c_seq_lab  … REM の標的を保存したラベルにする（自己生成の対照）
    ach_c_seq_rem  … 睡眠の全ステップを REM（自己生成）にする（NREM の対照）
とする。base・keep・ach_c・ach_c_nrem2 は 12.6.167〜168節の結果ファイル（同じシード・同じ実装で再現する）と対にして使う。

    --phase dev（開発用シード 100〜104）/ eval（評価用シード 0〜4 で一度だけ）
    結果は results/context_inference/replay_seqsleep_{phase}.json

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_replay_seqsleep --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_context_replay_seqsleep --analyze <結果ファイル> <12.6.167〜168節の結果ファイル>
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
from .run_context_replay_novelty import CONFIGS as NOVELTY_CONFIGS
from .run_context_replay_novelty import SUMMARY_KEYS
from .run_context_replay_split import CONFIGS as SPLIT_CONFIGS
from .run_context_replay_split import _criteria
from .run_context_replay_split import analyze as _split_analyze
from .run_context_writeback import CUE_N, NEW_TASK, SELECT, _exp

DEV_SEEDS = list(range(100, 105))
EVAL_SEEDS = list(range(0, 5))
CONFIGS: dict[str, dict] = {
    "base": SPLIT_CONFIGS["base"],
    "keep": SPLIT_CONFIGS["keep"],
    "ach_c": SPLIT_CONFIGS["ach_c"],
    "ach_c_nrem2": NOVELTY_CONFIGS["ach_c_nrem2"],
    **{r: {**NOVELTY_CONFIGS["ach_c_nrem2"], "replay_rule": r} for r in ("ach_c_seq", "ach_c_seq_lab", "ach_c_seq_rem")},
}
REUSED = ("base", "keep", "ach_c", "ach_c_nrem2")  # 12.6.167〜168節の結果ファイルから読む
NEW = ("ach_c_seq", "ach_c_seq_lab", "ach_c_seq_rem")
# 修正（12.6.231節。上限3回）。修正1: NREM の海馬の痕跡を新しい入力なしに学ぶステップが新しいタスクを損なった
# （開発用 ach_c_seq の −0.10 未満 16/150、REM だけの ach_c_seq_rem は 2/150）。NREM と自己生成の再生を交互にせず、
# 同じバッチで学ぶ（リップルが徐波・紡錘波に入れ子になる）
CONFIGS["ach_c_seq_mix"] = {**CONFIGS["ach_c_seq"], "replay_rule": "ach_c_seq_mix"}
FIXES: tuple[str, ...] = ("ach_c_seq_mix",)
# 確定の規則（修正1の計測の前に決めた）: 開発用シードで D1〜D3 を満たせば H_fix の候補にする。判定の基準は変えない
FIX_CANDIDATES: tuple[str, ...] = ()
RULES = ("keep", "ach_c", "ach_c_nrem2", *NEW, *FIXES)


def jobs(configs=NEW, families=FAMILIES) -> list[tuple]:
    return [(f, m, L, cfg) for m in MECHS_USED for f in families for L in LS for cfg in configs]


def run_job(job: tuple, kw: dict) -> dict:
    """job = (課題, 機構, L, 設定の名前, シード)。12.6.227節の run_job と同じ量に、睡眠のステップ数を加える。"""
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
           "mix_fam": mix.get("fam"), "sleep_steps": mix.get("sleep")}
    return {"key": f"{fam}/{mech}_L{L}/{cfg}", "row": row}


def run_one(job: tuple, table: dict) -> dict:
    return run_job(job, table[job[3]])


def analyze(paths: list[Path], rules=RULES) -> dict:
    """12.6.169節の集計（同じシードの base との差、組の分類も同じ）に、12.6.231節の判定を付ける。
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


def verdict(res: dict) -> dict:
    """12.6.231節で計測前に固定した判定条件。"""
    o, n, a = res["over_14bit"], res["not_over"], res["all"]
    out: dict = {}
    for w in ("ach_c", "ach_c_nrem2", *NEW, *FIXES):
        if o.get(f"end_{w}") is None:
            continue
        v = _criteria(o, n, a, w)
        if w != "ach_c" and o.get("end_ach_c") is not None:
            # D4: ach_c からの上積み（一般化 +0.02 以上、かつ大きく損なった回が増えない）
            v["D4"] = bool(o[f"end_{w}"] - o["end_ach_c"] >= 0.02 and a[f"bad_{w}"] <= a["bad_ach_c"])
        out[w] = v
    s = "ach_c_seq"
    if s in out:
        if o.get("end_ach_c_seq_lab") is not None:
            # R: REM の自己生成の標的が新しいタスクを守る（保存したラベルの REM より大きく損なった回が半分以下、
            # かつ最悪の組が +0.05 以上よい）
            out[s]["R"] = bool(a[f"bad_{s}"] <= a["bad_ach_c_seq_lab"] / 2
                               and a[f"min_new_{s}"] - a["min_new_ach_c_seq_lab"] >= 0.05)
        if o.get("end_ach_c_seq_rem") is not None:
            # N: NREM の海馬の痕跡が一般化を足す（REM だけの睡眠より +0.02 以上）
            out[s]["N"] = bool(o[f"end_{s}"] - o["end_ach_c_seq_rem"] >= 0.02)
        if o.get("end_ach_c_nrem2") is not None:
            # O: 12.6.167節の ach_c_nrem2 より大きく損なった回が半分以下
            out[s]["O"] = bool(a[f"bad_{s}"] <= a["bad_ach_c_nrem2"] / 2)
    out["H"] = out.get(s, {}).get("success") if s in out else None
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
    out_path = args.out or Path(f"results/context_inference/replay_seqsleep_{args.phase}.json")
    all_jobs = [(*cfg, s) for cfg in jobs(tuple(args.configs), tuple(args.families)) for s in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(partial(run_one, table=CONFIGS), all_jobs, chunksize=1)
    out: dict = {"環境": env_info(),
                 "説明": f"覚醒の後の NREM/REM の睡眠（12.6.231〜232節）。phase={args.phase}",
                 "seeds": seeds, "select": {SELECT: SELECTS[SELECT]}, "new_task": {k: repr(v) for k, v in NEW_TASK.items()},
                 "configs": {k: CONFIGS[k] for k in args.configs}, "selection_trials": SELECTION_TRIALS,
                 "runs": {}}
    for key in dict.fromkeys(x["key"] for x in results):
        rows = [x["row"] for x in results if x["key"] == key]
        summ = {}
        for k in (*SUMMARY_KEYS, "sleep_steps"):
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
