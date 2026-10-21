"""継続学習の忘却指標を実言語モデル（Europarl の多言語を順に学ぶ文字 Transformer）で測る（12.6.229〜230節）。

言語の組 R（ロマンス語4つ）・G（ゲルマン語5つ＋フィンランド語）を、12.6.213節と同じ順（シード + 31 の並べ替え）で
1言語 1500 ステップずつ学ぶ。規則は naive・replay（海馬のリプレイ）・ewc（シナプスの固定化）・replay_ewc・joint（参照）。

    --phase dev（開発用シード 100〜104。ewc の λ を格子 EWC_GRID から ACC で選ぶ）
    --phase dev2（開発用シードで、選んだ λ の replay_ewc）
    --phase eval（評価用シード 0〜4 で一度だけ。--lam に dev で選んだ λ）
    結果は results/context_lm/realtext_forgetting_{phase}.json、集計と判定は --analyze <結果ファイル>…

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_realtext_forgetting --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_realtext_forgetting --analyze results/context_lm/realtext_forgetting_dev.json
"""

from __future__ import annotations

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import torch

from ..realtext_forgetting import ForgetConfig, train_forgetting
from ..run_env import env_info
from .run_context_realtext import (DEFAULT_CORPUS, LANG_CODES, N_SPEAKER, N_TRAIN, SETS, fetch_corpus, load_sets,
                                   split_docs)

DEV_SEEDS = tuple(range(100, 105))
EVAL_SEEDS = tuple(range(5))
EWC_GRID = (0.003, 0.03, 0.3)  # 12.6.229節で計測前に固定（罰則の大きさの見積もりから）
# 判定条件（12.6.229節で計測前に固定。値は −CE（nats）の指標）
FORGET_MIN = 0.05  # V: naive の BWT ≦ −0.05
HALF = 0.5  # H1: |BWT_replay| ≦ 0.5·|BWT_naive|
LA_COST = 0.01  # H3: LA の低下 ≧ 0.01
T_MIN = 2.0


def run_one(set_name: str, rule: str, seed: int, lam: float = 0.0, steps: int = 1500,
            corpus: Path = DEFAULT_CORPUS) -> dict:
    torch.set_num_threads(1)
    vocab, docs = load_sets(corpus)
    v = len(vocab) + 1
    langs = SETS[set_name]
    data = [split_docs(docs[lang], vocab, N_TRAIN, N_SPEAKER, lang) for lang in langs]
    order = torch.randperm(len(langs), generator=torch.Generator().manual_seed(seed + 31)).tolist()
    cfg = ForgetConfig(rule=rule, steps_per_lang=steps, ewc_lambda=lam)
    out = train_forgetting(data, v, cfg, order, torch.Generator().manual_seed(seed),
                           torch.Generator().manual_seed(seed + 2024))
    out.update({"set": set_name, "rule": rule, "seed": seed, "lam": lam,
                "order_langs": [LANG_CODES[langs[i]] for i in order]})
    return out


def _cond(r: dict) -> str:
    return r["rule"] if r["rule"] not in ("ewc", "replay_ewc") else f"{r['rule']}@{r['lam']:g}"


def _mean(x: list[float]) -> float:
    return sum(x) / len(x)


def _t(d: list[float]) -> float:
    n = len(d)
    if n < 2:
        return 0.0
    m = _mean(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    return m / (sd / math.sqrt(n)) if sd > 0 else (math.inf if m > 0 else -math.inf if m < 0 else 0.0)


def analyze(runs: list[dict]) -> dict:
    """条件 × 言語の組の指標の平均と、12.6.229節の判定（同じシード・組の対の差）。"""
    by: dict = {}
    for r in runs:
        by.setdefault(_cond(r), {})[(r["set"], r["seed"])] = r
    table: dict = {}
    for c, rs in sorted(by.items()):
        for s in sorted({k[0] for k in rs}):
            sel = [r for k, r in rs.items() if k[0] == s]
            table[f"{c}/{s}"] = {f"{m}_{key}": _mean([r[f"metrics_{m}"][key] for r in sel])
                                 for m in ("nce", "acc") for key in sel[0][f"metrics_{m}"]} | {"n": len(sel)}

    def diff(a: str, b: str, key: str, s: str | None = None) -> list[float]:
        if a not in by or b not in by:
            return []
        ks = sorted(k for k in by[a] if k in by[b] and (s is None or k[0] == s))
        return [by[a][k]["metrics_nce"][key] - by[b][k]["metrics_nce"][key] for k in ks]

    def val(c: str, key: str, s: str | None = None) -> list[float]:
        return [r["metrics_nce"][key] for k, r in sorted(by.get(c, {}).items()) if s is None or k[0] == s]

    ewc = sorted((c for c in by if c.startswith("ewc@")), key=lambda c: -_mean(val(c, "ACC")))
    rew = sorted((c for c in by if c.startswith("replay_ewc@")), key=lambda c: -_mean(val(c, "ACC")))
    j: dict = {"ewc_best": ewc[0] if ewc else None, "replay_ewc_best": rew[0] if rew else None}
    sets = sorted({k[0] for c in by.values() for k in c})
    if "naive" in by:
        j["V"] = {s: {"BWT_naive": _mean(val("naive", "BWT", s)),
                      "合格": _mean(val("naive", "BWT", s)) <= -FORGET_MIN} for s in sets}
    if "naive" in by and "replay" in by:
        h1 = {}
        for s in sets:
            d = diff("replay", "naive", "BWT", s)
            bn, br = _mean(val("naive", "BWT", s)), _mean(val("replay", "BWT", s))
            h1[s] = {"差": _mean(d), "正の回": sum(x > 0 for x in d), "n": len(d), "比": br / bn if bn else None,
                     "合格": len(d) > 0 and all(x > 0 for x in d) and abs(br) <= HALF * abs(bn)}
        h1["合格"] = all(h1[s]["合格"] for s in sets)
        j["H1"] = h1
        d = diff("replay", "naive", "ACC")
        j["H2"] = {"差": _mean(d), "t": _t(d), "正の回": sum(x > 0 for x in d), "n": len(d), "合格": _t(d) >= T_MIN}
        d = diff("replay", "naive", "LA")
        j["H3"] = {"差": _mean(d), "t": _t(d), "合格": _mean(d) <= -LA_COST and _t(d) <= -T_MIN}
        if "joint" in by:
            jn, na, rp = _mean(val("joint", "ACC")), _mean(val("naive", "ACC")), _mean(val("replay", "ACC"))
            j["closure"] = {"joint": jn, "naive": na, "replay": rp,
                            "割合": (rp - na) / (jn - na) if jn != na else None}
            if ewc:
                j["closure"]["ewc"] = (_mean(val(ewc[0], "ACC")) - na) / (jn - na) if jn != na else None
    if "replay" in by and ewc:
        d = diff("replay", ewc[0], "ACC")
        j["H4"] = {"比較": ewc[0], "差": _mean(d), "t": _t(d), "正の回": sum(x > 0 for x in d), "n": len(d),
                   "合格": _t(d) >= T_MIN}
    if "replay" in by and rew:
        d = diff(rew[0], "replay", "ACC")
        j["H5"] = {"比較": rew[0], "差": _mean(d), "t": _t(d), "正の回": sum(x > 0 for x in d), "n": len(d),
                   "合格": _t(d) >= T_MIN}
    return {"表": table, "判定": j}


def jobs_for(phase: str, sets: tuple[str, ...], lam: float | None, conds: list[str] | None = None) -> list[tuple]:
    if conds:  # 「規則:λ」の並び（開発用シードでの追加の計測。12.6.230節の修正1）
        pairs = [(c.split(":")[0], float(c.split(":")[1]) if ":" in c else 0.0) for c in conds]
        return [(s, rule, seed, x) for s in sets for rule, x in pairs for seed in DEV_SEEDS]
    if phase == "dev":
        conds = [("naive", 0.0), ("replay", 0.0)] + [("ewc", x) for x in EWC_GRID]
        seeds = DEV_SEEDS
    elif phase == "dev2":
        conds, seeds = [("replay_ewc", lam)], DEV_SEEDS
    else:
        conds = [("naive", 0.0), ("replay", 0.0), ("ewc", lam), ("replay_ewc", lam), ("joint", 0.0)]
        seeds = EVAL_SEEDS
    return [(s, rule, seed, x) for s in sets for rule, x in conds for seed in seeds]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("dev", "dev2", "eval"), default="dev")
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--conds", nargs="+", default=None)
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--analyze", type=Path, nargs="+", default=None)
    args = ap.parse_args()
    if args.analyze:
        runs = [r for p in args.analyze for r in json.loads(p.read_text(encoding="utf-8"))["runs"]]
        print(json.dumps(analyze(runs), ensure_ascii=False, indent=1))
        return
    if args.phase != "dev" and args.lam is None and not args.conds:
        raise SystemExit("--lam が要る（dev で選んだ λ）")
    corpus = fetch_corpus(args.corpus)
    jobs = jobs_for(args.phase, tuple(args.sets), args.lam, args.conds)
    with Pool(args.workers) as pool:
        runs = pool.starmap(run_one, [(*j, args.steps, corpus) for j in jobs])
    out = {"環境": env_info(),
           "設定": {"corpus": "Europarl（NLTK europarl_raw の標本）", "sets": {k: [LANG_CODES[x] for x in v]
                                                                      for k, v in SETS.items()},
                  "split": {"train_days": N_TRAIN, "speaker_days": N_SPEAKER},
                  "model": ForgetConfig().__dict__ | {"steps_per_lang": args.steps}, "phase": args.phase,
                  "ewc_grid": EWC_GRID, "lam": args.lam, "jobs": [list(j) for j in jobs]},
           "runs": runs}
    out["集計"] = analyze(runs)
    path = args.out or Path("results/context_lm") / f"realtext_forgetting_{args.phase}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["集計"]["判定"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
