"""境界の推定が難しい設定（戻ってくる言語・1断片ずつの割り当て）での継続学習の忘却指標（12.6.249〜250節）。

Europarl の言語の組 R・G を、各言語 3 区間（1区間 500 ステップ、1言語の総ステップは 12.6.245節と同じ 1500）の
戻ってくる流れで学ぶ（realtext_hard_boundary）。条件は規則（naive・replay）× 割り当て（ev: 1断片ずつ、驚きを
漏れのある積分で集めて推定、frag: 1断片ずつそのまま推定、batch: バッチごとに推定、id: 言語の番号を与える、
blk: 区間ごとに新しい文脈）の 7 つ（CONDS）。評価はどれも言語の番号を与えない
推定の読み出し。

    --phase dev（開発用シード 100〜104）／ --phase eval（評価用シード 0〜4、一度だけ）／ --phase pre（予備の確認）
    結果は results/context_lm/realtext_hard_boundary_{phase}.json、集計と判定は --analyze <結果ファイル>…

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_realtext_hard_boundary --phase dev
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
from statistics import median

import torch

from ..realtext_hard_boundary import HardConfig, train_hard
from ..run_env import env_info
from .run_context_realtext import (DEFAULT_CORPUS, LANG_CODES, N_SPEAKER, N_TRAIN, SETS, fetch_corpus, load_sets,
                                   split_docs)
from .run_realtext_forgetting import _mean, _t

DEV_SEEDS = tuple(range(100, 105))
EVAL_SEEDS = tuple(range(5))
# 条件: 名前 → (規則, 割り当て)
# naive_frag・naive_batch・naive_blk は計算の都合で除く（12.6.249節）。replay_frag は予備の確認で分けすぎた規則の参照
CONDS = {"naive_ev": ("naive", "ev"), "replay_ev": ("replay", "ev"), "replay_frag": ("replay", "frag"),
         "replay_batch": ("replay", "batch"), "naive_id": ("naive", "id"), "replay_id": ("replay", "id"),
         "replay_blk": ("replay", "blk")}
# 判定条件（12.6.249節で開発用シードの計測の前に固定。値は −CE（nats）の指標、推定の読み出し）
FORGET_MIN = 0.05  # V: naive_id の FGT ≦ −0.05（両方の組）
ARI_MIN = 0.9  # A1: replay_ev の ARI の平均 ≧ 0.9
REACT_MIN = 0.9  # A2: replay_ev・replay_batch の再活性化の割合の平均 ≧ 0.9
ID_COST = 0.05  # H2: replay_ev − replay_id の ACC の平均 ≧ −0.05
T_MIN = 2.0


def run_one(set_name: str, cond: str, seed: int, steps: int = 1500, corpus: Path = DEFAULT_CORPUS,
            delta: float | None = None) -> dict:
    torch.set_num_threads(1)
    vocab, docs = load_sets(corpus)
    v = len(vocab) + 1
    langs = SETS[set_name]
    data = [split_docs(docs[lang], vocab, N_TRAIN, N_SPEAKER, lang) for lang in langs]
    order = torch.randperm(len(langs), generator=torch.Generator().manual_seed(seed + 31)).tolist()
    rule, asg = CONDS[cond]
    cfg = HardConfig(rule=rule, assign=asg, steps_per_lang=steps)
    if delta is not None:
        cfg.delta = delta
    out = train_hard(data, v, cfg, order, torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed + 2024))
    out.update({"set": set_name, "cond": cond, "rule": rule, "assign": asg, "seed": seed, "delta": cfg.delta,
                "order_langs": [LANG_CODES[langs[i]] for i in order],
                "blocks_langs": [LANG_CODES[langs[i]] for i in out["blocks"]]})
    return out


def analyze(runs: list[dict]) -> dict:
    """条件 × 組の指標の平均と、12.6.249節の判定（同じシード・組の対の差）。"""
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
            row = {"n": len(sel)} | {k: _mean([r["metrics"][k] for r in sel]) for k in sel[0]["metrics"]}
            row |= {"id_last": _mean([_mean(r["R_id"][-1]) for r in sel]),
                    "n_ctx": _mean([r["seg"]["n_ctx"] for r in sel]),
                    "ARI": _mean([r["seg"]["ARI"] for r in sel]),
                    "merged": _mean([r["seg"]["merged"] for r in sel]),
                    "purity": _mean([r["seg"]["purity"] for r in sel]),
                    "react": _mean([r["seg"]["react"] for r in sel]),
                    "split": _mean([r["seg"]["split"] for r in sel]),
                    "delay_median": median([d for r in sel for d in r["seg"]["delays"]])}
            table[f"{c}/{s}"] = row

    def val(c: str, key: str, s: str | None = None) -> list[float]:
        return [r["metrics"][key] for k, r in sorted(by.get(c, {}).items()) if s is None or k[0] == s]

    def diff(a: str, b: str, key: str, s: str | None = None) -> list[float]:
        if a not in by or b not in by:
            return []
        ks = sorted(k for k in by[a] if k in by[b] and (s is None or k[0] == s))
        return [by[a][k]["metrics"][key] - by[b][k]["metrics"][key] for k in ks]

    def paired(d: list[float]) -> dict:
        return {"差": _mean(d) if d else None, "t": _t(d), "正の回": sum(x > 0 for x in d), "n": len(d),
                "合格": bool(d) and _t(d) >= T_MIN}

    j: dict = {}
    if "naive_id" in by:
        j["V"] = {s: {"FGT_naive_id": _mean(val("naive_id", "FGT", s)),
                      "合格": _mean(val("naive_id", "FGT", s)) <= -FORGET_MIN} for s in sets}
        j["V"]["合格"] = all(j["V"][s]["合格"] for s in sets)
    if "replay_ev" in by:
        a = [r["seg"]["ARI"] for r in by["replay_ev"].values()]
        j["A1"] = {"ARI": _mean(a), "最小": min(a), "合格": _mean(a) >= ARI_MIN}
    ra = [c for c in ("replay_ev", "replay_batch") if c in by]
    if ra:
        j["A2"] = {c: _mean([r["seg"]["react"] for r in by[c].values()]) for c in ra}
        j["A2"]["合格"] = all(j["A2"][c] >= REACT_MIN for c in ra)
    if "replay_ev" in by and "naive_ev" in by:
        j["H1"] = paired(diff("replay_ev", "naive_ev", "ACC"))
    if "replay_ev" in by and "replay_id" in by:
        d = diff("replay_ev", "replay_id", "ACC")
        j["H2"] = {"差": _mean(d), "最小": min(d), "t": _t(d), "合格": _mean(d) >= -ID_COST}
    if "replay_id" in by and "replay_blk" in by:
        j["H3"] = paired(diff("replay_id", "replay_blk", "ACC"))
    ref: dict = {}
    for a, b in (("replay_ev", "replay_frag"), ("replay_ev", "replay_batch"), ("replay_batch", "replay_id"),
                 ("naive_ev", "naive_id"), ("replay_frag", "replay_id"), ("replay_ev", "replay_blk")):
        if a in by and b in by:
            ref[f"{a} − {b}"] = {k: paired(diff(a, b, k)) for k in ("ACC", "FGT", "AVG", "LA1")}
    for c in ("naive_ev", "replay_ev", "replay_frag", "replay_batch"):
        if c in by:
            w = [x for r in by[c].values() for x in r["surprise"]["within"]]
            sw = [x for r in by[c].values() for x in r["surprise"]["switch"]]
            q = sorted(w)
            ref[f"{c}: 驚き"] = {"言語の中の 99% 点": q[int(0.99 * (len(q) - 1))] if q else None,
                                 "言語の中で δ を超える割合": sum(x > by[c][next(iter(by[c]))]["delta"] for x in w)
                                 / len(w) if w else None,
                                 "切り替えの最小": min(sw) if sw else None,
                                 "切り替えの中央値": median(sw) if sw else None}
    # 割り当ての誤りと忘却の関係（参考）: 推定の条件の回ごとに、ARI と id との ACC の差
    for tf, ref_c in (("naive_ev", "naive_id"), ("replay_ev", "replay_id"), ("replay_frag", "replay_id")):
        if tf in by and ref_c in by:
            ks = sorted(k for k in by[tf] if k in by[ref_c])
            ref[f"{tf}: (ARI, 純度, n_ctx, ACC − {ref_c})"] = [
                [round(by[tf][k]["seg"]["ARI"], 3), round(by[tf][k]["seg"]["purity"], 3), by[tf][k]["seg"]["n_ctx"],
                 round(by[tf][k]["metrics"]["ACC"] - by[ref_c][k]["metrics"]["ACC"], 3)] for k in ks]
    j["参考"] = ref
    return {"表": table, "判定": j}


def jobs_for(phase: str, sets: tuple[str, ...], conds: list[str] | None = None,
             seeds: list[int] | None = None) -> list[tuple]:
    seeds = seeds or (DEV_SEEDS if phase == "dev" else EVAL_SEEDS)
    return [(s, c, seed) for seed in seeds for s in sets for c in (conds or list(CONDS))]


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
    if args.analyze:
        runs = [r for p in args.analyze for r in json.loads(p.read_text(encoding="utf-8"))["runs"]]
        print(json.dumps(analyze(runs), ensure_ascii=False, indent=1))
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
                  "model": HardConfig().__dict__ | {"steps_per_lang": args.steps, "delta_arg": args.delta},
                  "phase": args.phase, "conds": {k: list(v) for k, v in CONDS.items()},
                  "jobs": [list(j) for j in jobs]},
           "runs": runs}
    out["集計"] = analyze(runs)
    path = args.out or Path("results/context_lm") / f"realtext_hard_boundary_{args.phase}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["集計"]["判定"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
