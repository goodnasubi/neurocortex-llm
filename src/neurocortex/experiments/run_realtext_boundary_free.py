"""言語の境界を与えない継続学習の忘却指標（12.6.245〜246節）。

Europarl の言語の組 R・G を 12.6.229節と同じ順で流し（1言語 1500 ステップ）、学習の間も言語の境界を与えずに
文脈を推定して割り当てる（realtext_boundary_free）。条件は naive_tf・replay_tf（境界を推定）と naive_bd・replay_bd
（境界を与える対照、それ以外は同じ）。評価はどれも言語の番号を与えない推定の読み出し。文脈を持たない共有の1ヘッドの
対照（replay_sh）は 12.6.240節の結果ファイル（同じシード・同じ評価の断片）から読む。

    --phase dev（開発用シード 100〜104）／ --phase eval（評価用シード 0〜4、一度だけ）
    結果は results/context_lm/realtext_boundary_free_{phase}.json、集計と判定は --analyze <結果ファイル>…

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_realtext_boundary_free --phase dev
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
from statistics import median

import torch

from ..realtext_boundary_free import BoundaryConfig, train_boundary_free
from ..run_env import env_info
from .run_context_realtext import (DEFAULT_CORPUS, LANG_CODES, N_SPEAKER, N_TRAIN, SETS, fetch_corpus, load_sets,
                                   split_docs)
from .run_realtext_forgetting import _mean, _t

DEV_SEEDS = tuple(range(100, 105))
EVAL_SEEDS = tuple(range(5))
# 条件: 名前 → (規則, 境界)
CONDS = {"naive_tf": ("naive", "infer"), "replay_tf": ("replay", "infer"),
         "naive_bd": ("naive", "given"), "replay_bd": ("replay", "given")}
SHARED = {"dev": Path("results/context_lm/realtext_taskfree_dev.json"),
          "eval": Path("results/context_lm/realtext_taskfree_eval.json")}  # 12.6.240節の replay_sh
# 判定条件（12.6.245節で開発用シードの計測の前に固定。値は −CE（nats）の指標、推定の読み出し）
ARI_MIN = 0.9  # S1: 割り当てと言語の ARI の平均 ≧ 0.9（naive_tf・replay_tf それぞれ）
DETECT_STEPS = 100  # S2: 境界の 90% 以上を 100 ステップ以内に検出
DETECT_FRAC = 0.9
FORGET_MIN = 0.05  # V: naive_tf の BWT ≦ −0.05（両方の組）
HALF = 0.5  # H1: |BWT_replay_tf| ≦ 0.5·|BWT_naive_tf|
BD_COST = 0.05  # H3: replay_tf の ACC − replay_bd の ACC ≧ −0.05
T_MIN = 2.0


def run_one(set_name: str, cond: str, seed: int, steps: int = 1500, corpus: Path = DEFAULT_CORPUS,
            delta: float | None = None) -> dict:
    torch.set_num_threads(1)
    vocab, docs = load_sets(corpus)
    v = len(vocab) + 1
    langs = SETS[set_name]
    data = [split_docs(docs[lang], vocab, N_TRAIN, N_SPEAKER, lang) for lang in langs]
    order = torch.randperm(len(langs), generator=torch.Generator().manual_seed(seed + 31)).tolist()
    rule, boundary = CONDS[cond]
    cfg = BoundaryConfig(rule=rule, boundary=boundary, steps_per_lang=steps)
    if delta is not None:
        cfg.delta = delta
    out = train_boundary_free(data, v, cfg, order, torch.Generator().manual_seed(seed),
                              torch.Generator().manual_seed(seed + 2024))
    out.update({"set": set_name, "cond": cond, "rule": rule, "boundary": boundary, "seed": seed,
                "delta": cfg.delta, "order_langs": [LANG_CODES[langs[i]] for i in order]})
    return out


def load_shared(paths: list[Path]) -> dict:
    """12.6.240節の replay_sh・naive_sh の −CE の指標（(組, シード) → 指標）。"""
    out: dict = {}
    for p in paths:
        if p.exists():
            for r in json.loads(p.read_text(encoding="utf-8"))["runs"]:
                if r.get("shared_head"):
                    out.setdefault(r["cond"], {})[(r["set"], r["seed"])] = r["metrics_nce"]
    return out


def analyze(runs: list[dict], shared: dict | None = None) -> dict:
    """条件 × 組の指標の平均と、12.6.245節の判定（同じシード・組の対の差）。"""
    by: dict = {}
    for r in runs:
        by.setdefault(r["cond"], {})[(r["set"], r["seed"])] = r
    sets = sorted({k[0] for c in by.values() for k in c})
    table: dict = {}
    for c, rs in sorted(by.items()):
        for s in sets:
            sel = [r for k, r in rs.items() if k[0] == s]
            if not sel:
                continue
            row = {"n": len(sel)} | {k: _mean([r["metrics_nce"][k] for r in sel]) for k in sel[0]["metrics_nce"]}
            row |= {"id_last": _mean([_mean(r["R_id"][-1]) for r in sel]),
                    "n_ctx": _mean([r["seg"]["n_ctx"] for r in sel]),
                    "ARI": _mean([r["seg"]["ARI"] for r in sel]),
                    "delay_median": median([d for r in sel for d in r["seg"]["delays"]])}
            table[f"{c}/{s}"] = row

    def val(c: str, key: str, s: str | None = None) -> list[float]:
        return [r["metrics_nce"][key] for k, r in sorted(by.get(c, {}).items()) if s is None or k[0] == s]

    def diff(a: str, b: str, key: str, s: str | None = None) -> list[float]:
        if a not in by or b not in by:
            return []
        ks = sorted(k for k in by[a] if k in by[b] and (s is None or k[0] == s))
        return [by[a][k]["metrics_nce"][key] - by[b][k]["metrics_nce"][key] for k in ks]

    def paired(d: list[float]) -> dict:
        return {"差": _mean(d) if d else None, "t": _t(d), "正の回": sum(x > 0 for x in d), "n": len(d),
                "合格": bool(d) and _t(d) >= T_MIN}

    j: dict = {}
    tf = [c for c in ("naive_tf", "replay_tf") if c in by]
    if tf:
        j["S1"] = {c: {"ARI": _mean([r["seg"]["ARI"] for r in by[c].values()]),
                       "最小": min(r["seg"]["ARI"] for r in by[c].values())} for c in tf}
        j["S1"]["合格"] = all(j["S1"][c]["ARI"] >= ARI_MIN for c in tf)
        s2 = {}
        for c in tf:
            ds = [d for r in by[c].values() for d in r["seg"]["delays"]]
            s2[c] = {"中央値": median(ds), "検出の割合": sum(d <= DETECT_STEPS for d in ds) / len(ds),
                     "見逃し": sum(r["seg"]["missed"] for r in by[c].values()),
                     "余分な文脈": _mean([r["seg"]["n_ctx"] - len(r["order"]) for r in by[c].values()])}
        s2["合格"] = all(s2[c]["検出の割合"] >= DETECT_FRAC for c in tf)
        j["S2"] = s2
    if "naive_tf" in by:
        j["V"] = {s: {"BWT_naive_tf": _mean(val("naive_tf", "BWT", s)),
                      "合格": _mean(val("naive_tf", "BWT", s)) <= -FORGET_MIN} for s in sets}
        j["V"]["合格"] = all(j["V"][s]["合格"] for s in sets)
    if "naive_tf" in by and "replay_tf" in by:
        h1 = {}
        for s in sets:
            d = diff("replay_tf", "naive_tf", "BWT", s)
            bn, br = _mean(val("naive_tf", "BWT", s)), _mean(val("replay_tf", "BWT", s))
            h1[s] = {"差": _mean(d), "正の回": sum(x > 0 for x in d), "n": len(d), "比": br / bn if bn else None,
                     "合格": len(d) > 0 and all(x > 0 for x in d) and abs(br) <= HALF * abs(bn)}
        h1["合格"] = all(h1[s]["合格"] for s in sets)
        j["H1"] = h1
        j["H2"] = paired(diff("replay_tf", "naive_tf", "ACC"))
    if "replay_tf" in by and "replay_bd" in by:
        d = diff("replay_tf", "replay_bd", "ACC")
        j["H3"] = {"差": _mean(d), "最小": min(d), "合格": _mean(d) >= -BD_COST}
    if shared and "replay_sh" in shared and "replay_tf" in by:
        ks = sorted(k for k in by["replay_tf"] if k in shared["replay_sh"])
        j["H4"] = paired([by["replay_tf"][k]["metrics_nce"]["ACC"] - shared["replay_sh"][k]["ACC"] for k in ks])
    ref: dict = {}
    if "naive_tf" in by and "naive_bd" in by:
        ref["naive_tf − naive_bd"] = {k: _mean(diff("naive_tf", "naive_bd", k)) for k in ("ACC", "BWT", "LA")}
    if "replay_tf" in by and "replay_bd" in by:
        ref["replay_tf − replay_bd"] = {k: _mean(diff("replay_tf", "replay_bd", k)) for k in ("ACC", "BWT", "LA")}
        same = [by["replay_tf"][k]["seg"]["runs"] == by["replay_bd"][k]["seg"]["runs"]
                for k in by["replay_tf"] if k in by["replay_bd"]]
        ref["replay: 割り当てが境界を与えた系と一致した回"] = f"{sum(same)}/{len(same)}"
    if shared and "naive_sh" in shared and "naive_tf" in by:
        ks = sorted(k for k in by["naive_tf"] if k in shared["naive_sh"])
        ref["naive_tf − naive_sh（BWT）"] = paired([by["naive_tf"][k]["metrics_nce"]["BWT"]
                                                  - shared["naive_sh"][k]["BWT"] for k in ks])
    for c in tf:  # 驚きの余裕（記録は 10 ステップごと）: 境界の直後の最小と、言語の中の最大
        bnd, within = [], []
        for r in by[c].values():
            every = len(r["trace"]) // len(r["order"])
            for i, (s_, _) in enumerate(r["trace"]):
                if s_ is not None:
                    (bnd if i % every == 0 else within).append(s_)
        ref[f"{c}: 驚き（境界の最小・言語の中の最大）"] = [min(bnd), max(within)] if bnd and within else None
    j["参考"] = ref
    return {"表": table, "判定": j}


def jobs_for(phase: str, sets: tuple[str, ...], conds: list[str] | None = None,
             seeds: list[int] | None = None) -> list[tuple]:
    seeds = seeds or (DEV_SEEDS if phase == "dev" else EVAL_SEEDS)
    return [(s, c, seed) for s in sets for c in (conds or list(CONDS)) for seed in seeds]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("dev", "eval", "pre"), default="dev")
    ap.add_argument("--conds", nargs="+", default=None, choices=list(CONDS))
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--analyze", type=Path, nargs="+", default=None)
    args = ap.parse_args()
    shared = load_shared(list(SHARED.values()))
    if args.analyze:
        runs = [r for p in args.analyze for r in json.loads(p.read_text(encoding="utf-8"))["runs"]]
        print(json.dumps(analyze(runs, shared), ensure_ascii=False, indent=1))
        return
    if args.phase == "pre" and not args.seeds:
        raise SystemExit("pre には --seeds が要る（予備の確認用、評価にも開発にも使わないシード）")
    corpus = fetch_corpus(args.corpus)
    jobs = jobs_for(args.phase, tuple(args.sets), args.conds, args.seeds)
    with Pool(args.workers) as pool:
        runs = pool.starmap(run_one, [(*j, args.steps, corpus, args.delta) for j in jobs], chunksize=1)
    out = {"環境": env_info(),
           "設定": {"corpus": "Europarl（NLTK europarl_raw の標本）",
                  "sets": {k: [LANG_CODES[x] for x in v] for k, v in SETS.items()},
                  "split": {"train_days": N_TRAIN, "speaker_days": N_SPEAKER},
                  "model": BoundaryConfig().__dict__ | {"steps_per_lang": args.steps, "delta_arg": args.delta},
                  "phase": args.phase, "conds": {k: list(v) for k, v in CONDS.items()},
                  "jobs": [list(j) for j in jobs]},
           "runs": runs}
    out["集計"] = analyze(runs, shared)
    path = args.out or Path("results/context_lm") / f"realtext_boundary_free_{args.phase}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["集計"]["判定"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
