"""小脳の合成勾配で、時間方向の逆伝播を短く切った再帰型の皮質の学習を助ける（12.6.259〜260節）。

言語の組 R・G の全言語の流れを並行して読む GRU を、逆伝播の窓 K（主 2、副 1）で 5000 回学び、窓の終わりに小脳の
予測 ĝ(h_K) を注入すると（cb）、切ったまま（tbptt）より良くなり、切らない参照（long、過去 64 文字まで逆伝播）との
差を埋めるかを、小脳の対照（cb_shuf）と信頼度の利得（cb_gain）と比べる（realtext_cerebellum_tbptt）。

    --phase dev（開発用シード 100〜104）／ --phase eval（評価用シード 0〜4、一度だけ）
    結果は results/context_lm/realtext_cerebellum_tbptt_{phase}.json、集計と判定は --analyze <結果ファイル>…

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_realtext_cerebellum_tbptt --phase dev
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import torch

from ..realtext_cerebellum_tbptt import CONDS, TBPTTConfig, train_tbptt
from ..run_env import env_info
from .run_context_realtext import (DEFAULT_CORPUS, LANG_CODES, N_SPEAKER, N_TRAIN, SETS, fetch_corpus, load_sets,
                                   split_docs)
from .run_realtext_forgetting import _mean, _t

DEV_SEEDS = tuple(range(100, 105))
EVAL_SEEDS = tuple(range(5))
KS = (2, 1)  # 主の判定は K = 2、K = 1 は副（同じ基準で参考に判定する）
STEPS = 5000
# 判定条件（12.6.259節で計測前に固定。CE は nats、テストの最終の CE の言語の平均）
V_GAP = 0.02  # V: tbptt − long ≧ 0.02（切断が害になる）
RECOVER = 0.25  # H1: (tbptt − cb) / (tbptt − long) ≧ 0.25
T_MIN = 2.0


def run_one(set_name: str, k: int, cond: str, seed: int, steps: int = STEPS, corpus: Path = DEFAULT_CORPUS) -> dict:
    torch.set_num_threads(1)
    vocab, docs = load_sets(corpus)
    langs = SETS[set_name]
    data = [split_docs(docs[lang], vocab, N_TRAIN, N_SPEAKER, lang) for lang in langs]
    out = train_tbptt(data, len(vocab) + 1, TBPTTConfig(cond=cond, k=k, steps=steps),
                      torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed + 2024))
    out.update({"set": set_name, "k": k, "cond": cond, "seed": seed, "langs": [LANG_CODES[x] for x in langs]})
    return out


def _paired(d: list[float], least: float = 0.0) -> dict:
    m = _mean(d) if d else None
    return {"差": m, "t": _t(d), "正の回": sum(x > 0 for x in d), "n": len(d),
            "合格": bool(d) and _t(d) >= T_MIN and m >= least}


def analyze_k(runs: list[dict]) -> dict:
    """1つの K の判定（runs はその K のものだけ）。"""
    by: dict = {}
    for r in runs:
        by.setdefault(r["cond"], {})[(r["set"], r["seed"])] = r
    sets = sorted({kk[0] for c in by.values() for kk in c})
    keys = ("final_ce_mean", "auc", "final_loss", "cb_cos_true", "cb_cos_true_late", "cb_cos_boot", "gain_biased",
            "gain_unbiased", "sg_norm")
    table: dict = {}
    for c in CONDS:
        if c not in by:
            continue
        for s in sets + ["全体"]:
            sel = [v for kk, v in by[c].items() if s == "全体" or kk[0] == s]
            if sel:
                table[f"{c}/{s}"] = {"n": len(sel)} | {
                    k: (_mean([v[k] for v in sel]) if all(v.get(k) is not None for v in sel) else None) for k in keys}

    def diff(a: str, b: str, k: str = "final_ce_mean", kb: str | None = None) -> list[float]:
        if a not in by or b not in by:
            return []
        ks = sorted(kk for kk in by[a] if kk in by[b])
        return [by[a][kk][k] - by[b][kk][kb or k] for kk in ks]

    j: dict = {}
    if {"long", "tbptt"} <= by.keys():
        j["V"] = _paired(diff("tbptt", "long"), V_GAP)
    if {"long", "tbptt", "cb"} <= by.keys():
        j["H1"] = _paired(diff("tbptt", "cb"))
        gap = _mean(diff("tbptt", "long"))
        j["H1"]["回復の割合"] = _mean(diff("tbptt", "cb")) / gap if gap else None
        j["H1"]["合格"] = j["H1"]["合格"] and j["H1"]["回復の割合"] is not None and j["H1"]["回復の割合"] >= RECOVER
    if {"cb", "cb_shuf"} <= by.keys():
        j["H2"] = _paired(diff("cb_shuf", "cb"))
        a = _paired(diff("cb_shuf", "cb_shuf", "gain_biased", "gain_unbiased"))
        b = _paired(diff("cb", "cb_shuf", "gain_unbiased"))
        j["H3"] = {"偏りの除去（cb_shuf の biased − unbiased）": a, "識別（unbiased の cb − cb_shuf）": b,
                   "合格": a["合格"] and b["合格"]}
    if {"tbptt", "cb_gain"} <= by.keys():
        j["H4"] = _paired(diff("tbptt", "cb_gain"))
    ref: dict = {}
    for a, b in (("cb", "long"), ("cb_gain", "cb"), ("cb_shuf", "tbptt"), ("cb_gain", "long")):
        if {a, b} <= by.keys():
            ref[f"{a} − {b} の CE"] = _paired(diff(a, b))
            ref[f"{a} − {b} の AUC"] = _paired(diff(a, b, "auc"))
    if {"tbptt", "cb"} <= by.keys():
        ref["tbptt − cb の AUC"] = _paired(diff("tbptt", "cb", "auc"))
    j["参考"] = ref
    return {"表": table, "判定": j}


def analyze(runs: list[dict]) -> dict:
    return {f"K={k}": analyze_k([r for r in runs if r["k"] == k]) for k in sorted({r["k"] for r in runs}, reverse=True)}


def jobs_for(phase: str, sets: tuple[str, ...], ks: tuple[int, ...], conds: list[str] | None = None) -> list[tuple]:
    seeds = DEV_SEEDS if phase == "dev" else EVAL_SEEDS
    return [(s, k, c, seed) for s in sets for k in ks for c in (conds or list(CONDS)) for seed in seeds]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("dev", "eval"), default="dev")
    ap.add_argument("--conds", nargs="+", default=None, choices=list(CONDS))
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--ks", nargs="+", type=int, default=list(KS))
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--analyze", type=Path, nargs="+", default=None)
    args = ap.parse_args()
    if args.analyze:
        runs = [r for p in args.analyze for r in json.loads(p.read_text(encoding="utf-8"))["runs"]]
        print(json.dumps(analyze(runs), ensure_ascii=False, indent=1))
        return
    corpus = fetch_corpus(args.corpus)
    jobs = jobs_for(args.phase, tuple(args.sets), tuple(args.ks), args.conds)
    jobs.sort(key=lambda j: j[2] != "long")  # 長い条件（long）から先に配る
    with Pool(args.workers) as pool:
        runs = pool.starmap(run_one, [(*j, args.steps, corpus) for j in jobs], chunksize=1)
    out = {"環境": env_info(),
           "設定": {"corpus": "Europarl（NLTK europarl_raw の標本）",
                  "sets": {k: [LANG_CODES[x] for x in v] for k, v in SETS.items()},
                  "split": {"train_days": N_TRAIN, "speaker_days": N_SPEAKER},
                  "model": TBPTTConfig().__dict__ | {"steps": args.steps}, "ks": args.ks, "phase": args.phase,
                  "conds": list(CONDS), "jobs": [list(j) for j in jobs]},
           "runs": runs}
    out["集計"] = analyze(runs)
    path = args.out or Path("results/context_lm") / f"realtext_cerebellum_tbptt_{args.phase}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v["判定"] for k, v in out["集計"].items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
