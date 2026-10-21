"""胴体とヘッドのずれの診断（12.6.253〜254節）。

12.6.230・240節と同じ学習（Europarl の言語の組 R・G、同じ順、1言語 1500 ステップ、naive・replay・ewc λ = 0.3）の
各言語の学習の前後のスナップショットで、胴体を凍結して読み出しだけを学び直すプローブを行い、忘却 F を
胴体の情報の喪失 L と読み出しのずれ M に分ける（realtext_head_probe）。

    --phase dev（開発用シード 100〜104）／ --phase eval（評価用シード 0〜4、一度だけ）
    結果は results/context_lm/realtext_head_probe_{phase}.json、集計と判定は --analyze <結果ファイル>…

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_realtext_head_probe --phase dev
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import torch

from ..realtext_forgetting import ForgetConfig
from ..realtext_head_probe import PROBE_BATCH, PROBE_LR, PROBE_STEPS, train_and_probe
from ..run_env import env_info
from .run_context_realtext import (DEFAULT_CORPUS, LANG_CODES, N_SPEAKER, N_TRAIN, SETS, fetch_corpus, load_sets,
                                   split_docs)
from .run_realtext_forgetting import _mean, _t

DEV_SEEDS = tuple(range(100, 105))
EVAL_SEEDS = tuple(range(5))
LAM = 0.3  # 12.6.230節で開発用シードから選んだ λ（選び直さない）
CONDS = {"naive": ("naive", 0.0), "replay": ("replay", 0.0), "ewc": ("ewc", LAM)}
# 判定条件（12.6.253節で計測前に固定。CE は nats）
FORGET_MIN = 0.05  # V: naive の古い言語の F の平均 ≧ 0.05（各組）
DIAG_GAIN_MAX = 0.05  # V2: naive の diag でのプローブの利得 CE_diag − CE*_diag ≦ 0.05
CONV_MAX = 0.02  # V2: final のプローブの CE(300) − CE(600) ≦ 0.02
M_FRAC = 0.2  # H1: M / F ≧ 0.2
I_MIN = 0.5  # H2: naive の保持の指標 I ≧ 0.5
T_MIN = 2.0


def run_one(set_name: str, cond: str, seed: int, steps: int = 1500, corpus: Path = DEFAULT_CORPUS) -> dict:
    torch.set_num_threads(1)
    vocab, docs = load_sets(corpus)
    v = len(vocab) + 1
    langs = SETS[set_name]
    data = [split_docs(docs[lang], vocab, N_TRAIN, N_SPEAKER, lang) for lang in langs]
    order = torch.randperm(len(langs), generator=torch.Generator().manual_seed(seed + 31)).tolist()
    rule, lam = CONDS[cond]
    cfg = ForgetConfig(rule=rule, steps_per_lang=steps, ewc_lambda=lam)
    out = train_and_probe(data, v, cfg, order, torch.Generator().manual_seed(seed),
                          torch.Generator().manual_seed(seed + 2024), torch.Generator().manual_seed(seed + 777))
    for k in ("R_acc", "R_acc_inf", "R_ce_inf", "R_id_inf", "metrics_acc_inf", "metrics_nce_inf"):
        out.pop(k, None)  # 12.6.240節で測った値（容量を抑える）
    out.update({"set": set_name, "cond": cond, "rule": rule, "seed": seed, "lam": lam,
                "order_langs": [LANG_CODES[langs[i]] for i in order]})
    return out


def run_summary(r: dict) -> dict:
    """1回の学習の古い言語（最後に学んだ言語を除く）の F・L・M・I・CKA などの平均。"""
    old = [x for x in r["probe"]["langs"] if x["pos"] < len(r["probe"]["langs"]) - 1]
    s = {k: _mean([x[k] for x in old]) for k in ("F", "L", "M", "gain_diag", "gain_final", "learned",
                                                  "cka_diag_final")}
    i_vals = [x["I"] for x in old if x["I"] is not None]
    s["I"] = _mean(i_vals) if i_vals else None
    s["conv_final"] = _mean([x["probe_curve"]["final"][-2] - x["probe_curve"]["final"][-1] for x in old])
    return s


def analyze(runs: list[dict]) -> dict:
    by: dict = {}
    for r in runs:
        by.setdefault(r["cond"], {})[(r["set"], r["seed"])] = run_summary(r) | {"_run": r}
    sets = sorted({k[0] for c in by.values() for k in c})
    keys = ("F", "L", "M", "I", "gain_diag", "gain_final", "learned", "cka_diag_final", "conv_final")
    table: dict = {}
    for c, rs in sorted(by.items()):
        for s in sets:
            sel = [v for k, v in rs.items() if k[0] == s]
            if sel:
                table[f"{c}/{s}"] = {"n": len(sel)} | {k: _mean([v[k] for v in sel if v[k] is not None]) for k in keys}
        allv = list(rs.values())
        table[f"{c}/全体"] = {"n": len(allv)} | {k: _mean([v[k] for v in allv if v[k] is not None]) for k in keys}

    def vals(c: str, k: str, s: str | None = None) -> list[float]:
        return [v[k] for kk, v in sorted(by.get(c, {}).items()) if (s is None or kk[0] == s) and v[k] is not None]

    def diff(a: str, b: str, k: str) -> list[float]:
        if a not in by or b not in by:
            return []
        ks = sorted(kk for kk in by[a] if kk in by[b])
        return [by[a][kk][k] - by[b][kk][k] for kk in ks if by[a][kk][k] is not None and by[b][kk][k] is not None]

    def paired(d: list[float]) -> dict:
        return {"差": _mean(d) if d else None, "t": _t(d), "正の回": sum(x > 0 for x in d), "n": len(d),
                "合格": bool(d) and _t(d) >= T_MIN}

    j: dict = {}
    if "naive" in by:
        j["V"] = {s: {"F": _mean(vals("naive", "F", s)), "合格": _mean(vals("naive", "F", s)) >= FORGET_MIN}
                  for s in sets}
        j["V"]["合格"] = all(j["V"][s]["合格"] for s in sets)
        g, cv = _mean(vals("naive", "gain_diag")), _mean(vals("naive", "conv_final"))
        j["V2"] = {"gain_diag": g, "conv_final": cv, "合格": g <= DIAG_GAIN_MAX and cv <= CONV_MAX}
        m, f = vals("naive", "M"), vals("naive", "F")
        j["H1"] = paired(m) | {"M/F": _mean(m) / _mean(f) if _mean(f) else None}
        j["H1"]["合格"] = j["H1"]["合格"] and (j["H1"]["M/F"] or 0) >= M_FRAC
        iv = vals("naive", "I")
        j["H2"] = {"I": _mean(iv), "各組": {s: _mean(vals("naive", "I", s)) for s in sets},
                   "合格": _mean(iv) >= I_MIN}
    if "naive" in by and "replay" in by:
        j["H3"] = paired(diff("replay", "naive", "I"))
        j["H4"] = paired(diff("naive", "replay", "M"))
    ref: dict = {}
    for c in by:
        ref[f"{c}: M/F"] = _mean(vals(c, "M")) / _mean(vals(c, "F")) if _mean(vals(c, "F")) else None
    if "ewc" in by and "replay" in by:
        ref["ewc − replay の I"] = paired(diff("ewc", "replay", "I"))
        ref["ewc − replay の L"] = paired(diff("ewc", "replay", "L"))
    # 学習の位置ごと（naive・replay、最初に学んだ言語から）
    for c in ("naive", "replay"):
        if c in by:
            per: dict = {}
            for v in by[c].values():
                for x in v["_run"]["probe"]["langs"]:
                    per.setdefault(f"{v['_run']['set']}/{x['pos']}", []).append(x)
            ref[f"{c}: 位置ごと"] = {p: {k: _mean([x[k] for x in xs if x[k] is not None])
                                       for k in ("F", "L", "M", "I", "cka_diag_final")}
                                   for p, xs in sorted(per.items())}
    j["参考"] = ref
    for rs in by.values():
        for v in rs.values():
            v.pop("_run")
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
                  "probe": {"steps": list(PROBE_STEPS), "batch": PROBE_BATCH, "lr": PROBE_LR},
                  "conds": {k: list(v) for k, v in CONDS.items()}, "jobs": [list(j) for j in jobs]},
           "runs": runs}
    out["集計"] = analyze(runs)
    path = args.out or Path("results/context_lm") / f"realtext_head_probe_{args.phase}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["集計"]["判定"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
