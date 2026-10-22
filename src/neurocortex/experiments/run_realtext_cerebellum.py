"""小脳の合成勾配で、皮質の遠いフィードバックがまれにしか届かない下の層を学ばせる（12.6.257〜258節）。

言語の組 R・G の全言語を混ぜて 3000 ステップ学び（言語の番号は与える）、下の皮質に本物の誤差の勾配が 10 ステップに
1度だけ届く条件で、小脳の予測（cb）がその間を埋めて学習を助けるかを、bp（毎ステップ届く）・frozen（届かない）・
sparse（届いたときだけ）と、小脳の対照（cb_shuf・cb_state）と比べる（realtext_cerebellum）。

    --phase dev（開発用シード 100〜104）／ --phase eval（評価用シード 0〜4、一度だけ）
    結果は results/context_lm/realtext_cerebellum_{phase}.json、集計と判定は --analyze <結果ファイル>…

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_realtext_cerebellum --phase dev
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import torch

from ..realtext_cerebellum import CONDS, CBConfig, train_cerebellum
from ..run_env import env_info
from .run_context_realtext import (DEFAULT_CORPUS, LANG_CODES, N_SPEAKER, N_TRAIN, SETS, fetch_corpus, load_sets,
                                   split_docs)
from .run_realtext_forgetting import _mean, _t

DEV_SEEDS = tuple(range(100, 105))
EVAL_SEEDS = tuple(range(5))
# 判定条件（12.6.257節で計測前に固定。CE は nats、テストの最終の CE の言語の平均）
V_GAP = 0.03  # V: frozen − bp ≧ 0.03（下の皮質の学習が効く）
V2_GAP = 0.02  # V2: sparse − bp ≧ 0.02（フィードバックがまれだと損をする）
RECOVER = 0.25  # H1: (sparse − cb) / (sparse − bp) ≧ 0.25
T_MIN = 2.0


def run_one(set_name: str, cond: str, seed: int, steps: int = 3000, corpus: Path = DEFAULT_CORPUS,
            fix: int = 0) -> dict:
    torch.set_num_threads(1)
    vocab, docs = load_sets(corpus)
    v = len(vocab) + 1
    langs = SETS[set_name]
    data = [split_docs(docs[lang], vocab, N_TRAIN, N_SPEAKER, lang) for lang in langs]
    out = train_cerebellum(data, v, CBConfig(cond=cond, steps=steps, reliability=fix >= 1), torch.Generator().manual_seed(seed),
                           torch.Generator().manual_seed(seed + 2024))
    out.update({"set": set_name, "cond": cond, "seed": seed, "fix": fix, "langs": [LANG_CODES[x] for x in langs]})
    return out


def analyze(runs: list[dict]) -> dict:
    by: dict = {}
    for r in runs:
        by.setdefault(r["cond"], {})[(r["set"], r["seed"])] = r
    sets = sorted({k[0] for c in by.values() for k in c})
    keys = ("final_ce_mean", "auc", "final_loss", "cb_cos", "cb_cos_late", "gain_mean")
    table: dict = {}
    for c in CONDS:
        if c not in by:
            continue
        for s in sets + ["全体"]:
            sel = [v for k, v in by[c].items() if s == "全体" or k[0] == s]
            if sel:
                table[f"{c}/{s}"] = {"n": len(sel)} | {
                    k: (_mean([v[k] for v in sel]) if all(v.get(k) is not None for v in sel) else None) for k in keys}

    def diff(a: str, b: str, k: str = "final_ce_mean") -> list[float]:
        if a not in by or b not in by:
            return []
        ks = sorted(kk for kk in by[a] if kk in by[b])
        return [by[a][kk][k] - by[b][kk][k] for kk in ks]

    def paired(d: list[float], least: float = 0.0) -> dict:
        m = _mean(d) if d else None
        return {"差": m, "t": _t(d), "正の回": sum(x > 0 for x in d), "n": len(d),
                "合格": bool(d) and _t(d) >= T_MIN and m >= least}

    j: dict = {}
    if {"bp", "frozen"} <= by.keys():
        j["V"] = paired(diff("frozen", "bp"), V_GAP)
    if {"bp", "sparse"} <= by.keys():
        j["V2"] = paired(diff("sparse", "bp"), V2_GAP)
    if {"bp", "sparse", "cb"} <= by.keys():
        j["H1"] = paired(diff("sparse", "cb"))
        gap = _mean(diff("sparse", "bp"))
        j["H1"]["回復の割合"] = _mean(diff("sparse", "cb")) / gap if gap else None
        j["H1"]["合格"] = j["H1"]["合格"] and j["H1"]["回復の割合"] is not None and j["H1"]["回復の割合"] >= RECOVER
    if {"cb", "cb_shuf"} <= by.keys():
        j["H2"] = paired(diff("cb_shuf", "cb"))
    if {"cb", "cb_state"} <= by.keys():
        j["H3"] = paired(diff("cb_state", "cb"))
        j["H3"]["余弦の差（cb − cb_state）"] = paired(diff("cb", "cb_state", "cb_cos"))
        j["H3"]["合格"] = j["H3"]["合格"] and j["H3"]["余弦の差（cb − cb_state）"]["合格"]
    ref: dict = {}
    for a, b in (("sparse", "cb"), ("frozen", "cb"), ("cb_shuf", "cb"), ("sparse", "frozen"), ("sparse10", "bp"),
                 ("sparse", "sparse10")):
        if {a, b} <= by.keys():
            ref[f"{a} − {b} の CE"] = paired(diff(a, b))
            ref[f"{a} − {b} の AUC"] = paired(diff(a, b, "auc"))
    j["参考"] = ref
    return {"表": table, "判定": j}


def jobs_for(phase: str, sets: tuple[str, ...], conds: list[str] | None = None) -> list[tuple]:
    seeds = DEV_SEEDS if phase == "dev" else EVAL_SEEDS
    return [(s, c, seed) for s in sets for c in (conds or list(CONDS)) for seed in seeds]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("dev", "eval"), default="dev")
    ap.add_argument("--conds", nargs="+", default=None, choices=list(CONDS))
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--fix", type=int, default=0, help="12.6.258節の修正の番号（1: 小脳の信号を信頼度の利得で重みづける）")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--analyze", type=Path, nargs="+", default=None)
    args = ap.parse_args()
    if args.analyze:
        runs = [r for p in args.analyze for r in json.loads(p.read_text(encoding="utf-8"))["runs"]]
        print(json.dumps(analyze(runs), ensure_ascii=False, indent=1))
        return
    corpus = fetch_corpus(args.corpus)
    jobs = jobs_for(args.phase, tuple(args.sets), args.conds)
    # 長い条件（小脳つき）から先に配って、最後に1本だけ残る待ちを減らす
    jobs.sort(key=lambda j: not j[1].startswith("cb"))
    with Pool(args.workers) as pool:
        runs = pool.starmap(run_one, [(*j, args.steps, corpus, args.fix) for j in jobs], chunksize=1)
    out = {"環境": env_info(),
           "設定": {"corpus": "Europarl（NLTK europarl_raw の標本）",
                  "sets": {k: [LANG_CODES[x] for x in v] for k, v in SETS.items()},
                  "split": {"train_days": N_TRAIN, "speaker_days": N_SPEAKER},
                  "model": CBConfig(reliability=args.fix >= 1).__dict__ | {"steps": args.steps}, "phase": args.phase,
                  "fix": args.fix,
                  "conds": list(CONDS), "jobs": [list(j) for j in jobs]},
           "runs": runs}
    out["集計"] = analyze(runs)
    path = args.out or Path("results/context_lm") / (
        f"realtext_cerebellum_{args.phase}.json" if not args.fix else f"realtext_cerebellum_fix{args.fix}_{args.phase}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["集計"]["判定"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
