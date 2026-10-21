"""タスクの番号を与えない形での継続学習の忘却指標（12.6.239〜240節）。

12.6.229〜230節は、言語ごとのヘッドを言語の番号で選ぶ形（タスクの番号を与える形）だけを測った。脳は番号を受け取らない。
ここでは同じ学習（Europarl の言語の組 R・G、同じ順、1言語 1500 ステップ）の後に、
- oracle: 言語の番号でヘッドを選ぶ（12.6.230節と同じ値になる）
- infer: それまでに学んだ言語の文脈の事後分布（断片の前の文字から逐次に求める）でヘッドの予測を混ぜる
の両方で正確度行列を測る。文脈の系を持たない対照として、1つの共有のヘッドのモデル（naive_sh・replay_sh）も学ぶ。

    --phase dev（開発用シード 100〜104）／ --phase eval（評価用シード 0〜4、一度だけ）
    結果は results/context_lm/realtext_taskfree_{phase}.json、集計と判定は --analyze <結果ファイル>…

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_realtext_forgetting_taskfree --phase dev
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import torch

from ..realtext_forgetting import ForgetConfig, train_forgetting
from ..run_env import env_info
from .run_context_realtext import (DEFAULT_CORPUS, LANG_CODES, N_SPEAKER, N_TRAIN, SETS, fetch_corpus, load_sets,
                                   split_docs)
from .run_realtext_forgetting import _mean, _t

DEV_SEEDS = tuple(range(100, 105))
EVAL_SEEDS = tuple(range(5))
LAM = 0.3  # 12.6.230節で開発用シードから選んだ λ（ここでは選び直さない）
# 条件: 名前 → (規則, λ, 共有のヘッド)
CONDS = {"naive": ("naive", 0.0, False), "replay": ("replay", 0.0, False), "ewc": ("ewc", LAM, False),
         "joint": ("joint", 0.0, False), "naive_sh": ("naive", 0.0, True), "replay_sh": ("replay", 0.0, True)}
# 判定条件（12.6.239節で計測前に固定。値は −CE（nats）の指標）
FORGET_MIN = 0.05  # V: naive（推定）の BWT ≦ −0.05
HALF = 0.5  # H1: |BWT_replay| ≦ 0.5·|BWT_naive|（推定）
INFER_COST = 0.10  # H3: replay の ACC（推定）− ACC（oracle）≧ −0.10
T_MIN = 2.0


def run_one(set_name: str, cond: str, seed: int, steps: int = 1500, corpus: Path = DEFAULT_CORPUS) -> dict:
    torch.set_num_threads(1)
    vocab, docs = load_sets(corpus)
    v = len(vocab) + 1
    langs = SETS[set_name]
    data = [split_docs(docs[lang], vocab, N_TRAIN, N_SPEAKER, lang) for lang in langs]
    order = torch.randperm(len(langs), generator=torch.Generator().manual_seed(seed + 31)).tolist()
    rule, lam, sh = CONDS[cond]
    cfg = ForgetConfig(rule=rule, steps_per_lang=steps, ewc_lambda=lam, shared_head=sh)
    out = train_forgetting(data, v, cfg, order, torch.Generator().manual_seed(seed),
                           torch.Generator().manual_seed(seed + 2024))
    out.update({"set": set_name, "cond": cond, "rule": rule, "seed": seed, "lam": lam, "shared_head": sh,
                "order_langs": [LANG_CODES[langs[i]] for i in order]})
    return out


def metrics(r: dict, mode: str) -> dict:
    """mode は oracle・infer。共有のヘッドは番号を使わないので、どちらの mode でも同じ値（推定の行）。"""
    if mode == "infer" and not r["shared_head"]:
        return r["metrics_nce_inf"]
    return r["metrics_nce"]


def analyze(runs: list[dict]) -> dict:
    """条件 × 組 × 読み出しの指標の平均と、12.6.239節の判定（同じシード・組の対の差）。"""
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
            row = {"n": len(sel)}
            for mode in ("oracle", "infer"):
                for key in metrics(sel[0], mode):
                    row[f"{mode}_{key}"] = _mean([metrics(r, mode)[key] for r in sel])
            if not sel[0]["shared_head"]:  # 最後の行の文脈の同定の正確度（全言語の平均）
                row["id_last"] = _mean([_mean(r["R_id_inf"][-1]) for r in sel])
            table[f"{c}/{s}"] = row

    def val(c: str, key: str, mode: str = "infer", s: str | None = None) -> list[float]:
        return [metrics(r, mode)[key] for k, r in sorted(by.get(c, {}).items()) if s is None or k[0] == s]

    def diff(a: str, b: str, key: str, ma: str = "infer", mb: str = "infer", s: str | None = None) -> list[float]:
        if a not in by or b not in by:
            return []
        ks = sorted(k for k in by[a] if k in by[b] and (s is None or k[0] == s))
        return [metrics(by[a][k], ma)[key] - metrics(by[b][k], mb)[key] for k in ks]

    def paired(d: list[float]) -> dict:
        return {"差": _mean(d) if d else None, "t": _t(d), "正の回": sum(x > 0 for x in d), "n": len(d),
                "合格": bool(d) and _t(d) >= T_MIN}

    j: dict = {}
    if "naive" in by:
        j["V"] = {s: {"BWT_naive": _mean(val("naive", "BWT", s=s)),
                      "合格": _mean(val("naive", "BWT", s=s)) <= -FORGET_MIN} for s in sets}
        j["V"]["合格"] = all(j["V"][s]["合格"] for s in sets)
    if "naive" in by and "replay" in by:
        h1 = {}
        for s in sets:
            d = diff("replay", "naive", "BWT", s=s)
            bn, br = _mean(val("naive", "BWT", s=s)), _mean(val("replay", "BWT", s=s))
            h1[s] = {"差": _mean(d), "正の回": sum(x > 0 for x in d), "n": len(d), "比": br / bn if bn else None,
                     "合格": len(d) > 0 and all(x > 0 for x in d) and abs(br) <= HALF * abs(bn)}
        h1["合格"] = all(h1[s]["合格"] for s in sets)
        j["H1"] = h1
        j["H2"] = paired(diff("replay", "naive", "ACC"))
    if "replay" in by:
        d = diff("replay", "replay", "ACC", "infer", "oracle")
        j["H3"] = {"差": _mean(d), "最小": min(d), "合格": _mean(d) >= -INFER_COST}
    if "replay" in by and "ewc" in by:
        j["H4"] = paired(diff("replay", "ewc", "ACC"))
    if "replay" in by and "replay_sh" in by:
        j["H5"] = paired(diff("replay", "replay_sh", "ACC"))
    if "naive" in by and "naive_sh" in by:
        j["H6"] = paired(diff("naive", "naive_sh", "BWT"))
    # 参考（判定に含めない）
    ref: dict = {}
    for c in ("naive", "replay", "ewc"):
        if c in by:
            ref[f"{c}: 推定 − oracle"] = {k: _mean(diff(c, c, k, "infer", "oracle")) for k in ("ACC", "BWT", "LA")}
    if "joint" in by and "naive" in by and "replay" in by:
        jn, na, rp = _mean(val("joint", "ACC")), _mean(val("naive", "ACC")), _mean(val("replay", "ACC"))
        ref["joint までの差を埋めた割合（推定）"] = (rp - na) / (jn - na) if jn != na else None
    if "replay_sh" in by and "naive_sh" in by:
        ref["共有: replay_sh − naive_sh"] = {k: _mean(diff("replay_sh", "naive_sh", k)) for k in ("ACC", "BWT", "LA")}
    if "naive" in by and "naive_sh" in by:
        ref["naive − naive_sh（ACC・LA）"] = {k: _mean(diff("naive", "naive_sh", k)) for k in ("ACC", "LA")}
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
    corpus = fetch_corpus(args.corpus)
    jobs = jobs_for(args.phase, tuple(args.sets), args.conds)
    with Pool(args.workers) as pool:
        runs = pool.starmap(run_one, [(*j, args.steps, corpus) for j in jobs], chunksize=1)
    out = {"環境": env_info(),
           "設定": {"corpus": "Europarl（NLTK europarl_raw の標本）",
                  "sets": {k: [LANG_CODES[x] for x in v] for k, v in SETS.items()},
                  "split": {"train_days": N_TRAIN, "speaker_days": N_SPEAKER},
                  "model": ForgetConfig().__dict__ | {"steps_per_lang": args.steps}, "phase": args.phase,
                  "conds": {k: list(v) for k, v in CONDS.items()}, "jobs": [list(j) for j in jobs]},
           "runs": runs}
    out["集計"] = analyze(runs)
    path = args.out or Path("results/context_lm") / f"realtext_taskfree_{args.phase}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["集計"]["判定"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
