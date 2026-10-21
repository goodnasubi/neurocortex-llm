"""類似度に基づく選択的リプレイで、取り戻した一般化と新しいタスクの学習を両立できるか（12.6.227〜228節）。

12.6.167〜168節の ach_c は、なじみの度合い f に応じて加えた痕跡（再活性化した保留サンプル）の固定化を増やすが、
一般化（上書きありの組の end）は keep より小さく、固定化を増やすほど新しいタスクが削られる関係は変わらなかった。
ここでは、固定化に回す痕跡の数（round(f·batch)）は ach_c のまま変えず、どの痕跡を引くかだけを、今のバッチとの
表現の類似度で重みづけする（McClelland, McNaughton & Lampinen 2020 の類似度で重みづけしたインターリーブ学習）。
    ach_c_hid      … 隠れ層の表現が今の入力に似た痕跡を優先する（主仮説。干渉を受けやすい痕跡を選んで固定化する）
    ach_c_hid_neg  … 似ていない痕跡を優先する（向きの対照。新しい写像と競合しにくい痕跡を選ぶ）
    ach_c_hid_shuf … 類似度を並べ替える（重みの分布は同じで、類似度との対応だけを壊す。S の対照）
    ach_c_inp      … 類似度を入力のビットの一致率で測る（表現ではなく入力の重なり）
base・keep・ach_c は 12.6.167〜168節の結果ファイル（同じシード・同じ実装で再現する）と対にして使う。

    --phase dev（開発用シード 100〜104）/ eval（評価用シード 0〜4 で一度だけ）
    結果は results/context_inference/replay_similarity_{phase}.json

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_replay_similarity --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_context_replay_similarity --analyze <結果ファイル> <12.6.167〜168節の結果ファイル>
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
BETA = 2.0  # 重みの鋭さ exp(β·z)（計測前に固定。z が +1 の痕跡は −1 の痕跡の e^4 ≈ 55 倍引かれやすい）
CONFIGS: dict[str, dict] = {
    "base": SPLIT_CONFIGS["base"],
    "keep": SPLIT_CONFIGS["keep"],
    "ach_c": SPLIT_CONFIGS["ach_c"],
    **{f"ach_c_{s}": {**SPLIT_CONFIGS["ach_c"], "replay_sim": s, "replay_sim_beta": BETA}
       for s in ("hid", "hid_neg", "hid_shuf", "inp")},
}
REUSED = ("base", "keep", "ach_c")  # 12.6.167〜168節の結果ファイルから読む
NEW = ("ach_c_hid", "ach_c_hid_neg", "ach_c_hid_shuf", "ach_c_inp")
# 修正（12.6.227節。上限3回）。修正1: 痕跡ごとの重みは固定化の痕跡を少数に集中させ、並べ替えでも一般化が下がった
# （開発用 ach_c +0.086、hid +0.066、shuf +0.072）。SWIL の元の形（文脈＝クラス単位の類似度、文脈の中は一様）にする
CONFIGS.update({f"ach_c_{s}": {**SPLIT_CONFIGS["ach_c"], "replay_sim": s, "replay_sim_beta": BETA}
                for s in ("hid_ctx", "hid_ctx_shuf")})
FIXES: tuple[str, ...] = ("ach_c_hid_ctx", "ach_c_hid_ctx_shuf")
# 確定の規則（修正1の計測の前に決めた）: 開発用シードで D1〜D3 と S（ach_c_hid_ctx_shuf との比較）を満たせば
# H_fix の候補にする。判定の基準は変えない
# 修正1は開発用シードで D1・D2(c)・S が不合格（end +0.069、並べ替え +0.079）。初期設定と修正1で確定し、H_fix の候補はない
FIX_CANDIDATES: tuple[str, ...] = ()
SHUF_OF = {"ach_c_hid_ctx": "ach_c_hid_ctx_shuf"}  # S の対照（既定は ach_c_hid_shuf）
RULES = ("keep", "ach_c", *NEW, *FIXES)


def jobs(configs=NEW, families=FAMILIES) -> list[tuple]:
    return [(f, m, L, cfg) for m in MECHS_USED for f in families for L in LS for cfg in configs]


def run_job(job: tuple, kw: dict) -> dict:
    """job = (課題, 機構, L, 設定の名前, シード)。12.6.219節の run_job と同じ量に、引いた痕跡の類似度を加える。"""
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
           "mix_fam": mix.get("fam"), "mix_sim": mix.get("sim")}
    return {"key": f"{fam}/{mech}_L{L}/{cfg}", "row": row}


def run_one(job: tuple, table: dict) -> dict:
    return run_job(job, table[job[3]])


def analyze(paths: list[Path], rules=RULES) -> dict:
    """12.6.169節の集計（同じシードの base との差、組の分類も同じ）に、12.6.227節の判定を付ける。
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
    """12.6.227節で計測前に固定した判定条件。"""
    o, n, a = res["over_14bit"], res["not_over"], res["all"]
    out: dict = {}
    for w in ("ach_c", *NEW, *FIXES):
        if o.get(f"end_{w}") is None:
            continue
        v = _criteria(o, n, a, w)
        if w != "ach_c" and o.get("end_ach_c") is not None:
            # D4: ach_c からの上積み（一般化 +0.02 以上、かつ大きく損なった回が増えない）
            v["D4"] = bool(o[f"end_{w}"] - o["end_ach_c"] >= 0.02 and a[f"bad_{w}"] <= a["bad_ach_c"])
        sh = SHUF_OF.get(w, "ach_c_hid_shuf")
        if not w.endswith("_shuf") and o.get(f"end_{sh}") is not None:
            # S: 類似度との対応が効く（並べ替えより一般化 +0.02 以上、かつ大きく損なった回が増えない）
            v["S"] = bool(o[f"end_{w}"] - o[f"end_{sh}"] >= 0.02 and a[f"bad_{w}"] <= a[f"bad_{sh}"])
        out[w] = v
    m = "ach_c_hid"
    out["H"] = bool(out[m]["success"] and out[m].get("S")) if m in out else None
    out["H_neg"] = bool(out["ach_c_hid_neg"]["success"] and out["ach_c_hid_neg"].get("S")) \
        if "ach_c_hid_neg" in out else None
    out["H_fix"] = any(out.get(w, {}).get("success") and out[w].get("S") for w in FIX_CANDIDATES) \
        if FIX_CANDIDATES else None
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
    out_path = args.out or Path(f"results/context_inference/replay_similarity_{args.phase}.json")
    all_jobs = [(*cfg, s) for cfg in jobs(tuple(args.configs), tuple(args.families)) for s in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(partial(run_one, table=CONFIGS), all_jobs, chunksize=1)
    out: dict = {"環境": env_info(),
                 "説明": f"類似度に基づく選択的リプレイ（12.6.227〜228節）。phase={args.phase}",
                 "seeds": seeds, "select": {SELECT: SELECTS[SELECT]}, "new_task": {k: repr(v) for k, v in NEW_TASK.items()},
                 "beta": BETA, "configs": {k: CONFIGS[k] for k in args.configs}, "selection_trials": SELECTION_TRIALS,
                 "runs": {}}
    for key in dict.fromkeys(x["key"] for x in results):
        rows = [x["row"] for x in results if x["key"] == key]
        summ = {}
        for k in (*SUMMARY_KEYS, "mix_sim"):
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
