"""リプレイの効果を「読み出しの追従」と「胴体の情報の保護」に因果的に分ける（12.6.255〜256節）。

12.6.253〜254節の学習（Europarl の言語の組 R・G、同じ順、1言語 1500 ステップ）と読み出しのプローブで、
naive・replay に、リプレイの断片の勾配を過去の言語の読み出しにだけ流す replay_ro と、胴体にだけ流す replay_body を
加え、忘却 F を胴体の情報の喪失 L と読み出しのずれ M に分ける（realtext_head_probe）。

    --phase dev（開発用シード 100〜104）／ --phase eval（評価用シード 0〜4、一度だけ）
    結果は results/context_lm/replay_readout_only_{phase}.json、集計と判定は --analyze <結果ファイル>…

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_replay_readout_only --phase dev
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
from .run_realtext_head_probe import DEV_SEEDS, EVAL_SEEDS, run_summary

CONDS = ("naive", "replay", "replay_ro", "replay_body")
# 判定条件（12.6.255節で計測前に固定。CE は nats）
RO_M_FRAC = 0.2  # H1: replay_ro の M ≦ 0.2 × naive の M（読み出しの追従だけでずれの 8 割以上を消す）
T_MIN = 2.0


def run_one(set_name: str, cond: str, seed: int, steps: int = 1500, corpus: Path = DEFAULT_CORPUS) -> dict:
    torch.set_num_threads(1)
    vocab, docs = load_sets(corpus)
    v = len(vocab) + 1
    langs = SETS[set_name]
    data = [split_docs(docs[lang], vocab, N_TRAIN, N_SPEAKER, lang) for lang in langs]
    order = torch.randperm(len(langs), generator=torch.Generator().manual_seed(seed + 31)).tolist()
    cfg = ForgetConfig(rule=cond, steps_per_lang=steps)
    out = train_and_probe(data, v, cfg, order, torch.Generator().manual_seed(seed),
                          torch.Generator().manual_seed(seed + 2024), torch.Generator().manual_seed(seed + 777))
    for k in ("R_acc", "R_acc_inf", "R_ce_inf", "R_id_inf", "metrics_acc_inf", "metrics_nce_inf"):
        out.pop(k, None)
    out.update({"set": set_name, "cond": cond, "rule": cond, "seed": seed,
                "order_langs": [LANG_CODES[langs[i]] for i in order]})
    return out


def analyze(runs: list[dict]) -> dict:
    by: dict = {}
    for r in runs:
        by.setdefault(r["cond"], {})[(r["set"], r["seed"])] = run_summary(r) | {
            "ACC": r["metrics_nce"]["ACC"], "LA": r["metrics_nce"]["LA"], "_run": r}
    sets = sorted({k[0] for c in by.values() for k in c})
    keys = ("F", "L", "M", "I", "learned", "cka_diag_final", "gain_diag", "conv_final", "ACC", "LA")
    table: dict = {}
    for c in CONDS:
        if c not in by:
            continue
        for s in sets + ["全体"]:
            sel = [v for k, v in by[c].items() if s == "全体" or k[0] == s]
            if sel:
                table[f"{c}/{s}"] = {"n": len(sel)} | {k: _mean([v[k] for v in sel if v[k] is not None]) for k in keys}

    def vals(c: str, k: str) -> list[float]:
        return [v[k] for _, v in sorted(by.get(c, {}).items()) if v[k] is not None]

    def diff(a: str, b: str, k: str) -> list[float]:
        if a not in by or b not in by:
            return []
        ks = sorted(kk for kk in by[a] if kk in by[b])
        return [by[a][kk][k] - by[b][kk][k] for kk in ks if by[a][kk][k] is not None and by[b][kk][k] is not None]

    def paired(d: list[float]) -> dict:
        return {"差": _mean(d) if d else None, "t": _t(d), "正の回": sum(x > 0 for x in d), "n": len(d),
                "合格": bool(d) and _t(d) >= T_MIN}

    j: dict = {}
    if all(c in by for c in CONDS):
        j["H1"] = paired(diff("naive", "replay_ro", "M"))
        j["H1"]["ro_M/naive_M"] = _mean(vals("replay_ro", "M")) / _mean(vals("naive", "M"))
        j["H1"]["合格"] = j["H1"]["合格"] and j["H1"]["ro_M/naive_M"] <= RO_M_FRAC
        j["H2"] = paired(diff("replay", "replay_ro", "I"))
        j["H3"] = paired(diff("replay_body", "naive", "I"))
        j["H4"] = paired(diff("naive", "replay_body", "M"))
        ref: dict = {}
        fn, fr = _mean(vals("naive", "F")), _mean(vals("replay", "F"))
        for c in ("replay_ro", "replay_body"):
            ref[f"{c}: F の減りの replay に対する割合"] = (fn - _mean(vals(c, "F"))) / (fn - fr) if fn != fr else None
            ref[f"{c}: M/F"] = _mean(vals(c, "M")) / _mean(vals(c, "F")) if _mean(vals(c, "F")) else None
        ref["replay_ro − naive の I"] = paired(diff("replay_ro", "naive", "I"))
        ref["replay_ro − naive の L"] = paired(diff("replay_ro", "naive", "L"))
        ref["replay_body − replay_ro の F"] = paired(diff("replay_body", "replay_ro", "F"))
        ref["replay − replay_body の I"] = paired(diff("replay", "replay_body", "I"))
        ref["replay_ro − replay の M"] = paired(diff("replay_ro", "replay", "M"))
        ref["replay_body − naive の CKA"] = paired(diff("replay_body", "naive", "cka_diag_final"))
        ref["replay_ro − naive の LA"] = paired(diff("replay_ro", "naive", "LA"))
        ref["replay_body − naive の LA"] = paired(diff("replay_body", "naive", "LA"))
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
                  "conds": list(CONDS), "jobs": [list(j) for j in jobs]},
           "runs": runs}
    out["集計"] = analyze(runs)
    path = args.out or Path("results/context_lm") / f"replay_readout_only_{args.phase}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["集計"]["判定"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
