"""記憶の痕跡の数を文脈の間で偏らせ、CA1 比較器のエピソードごとの寄与を実テキストで測る（12.6.215〜216節）。

12.6.214節では、比較器のゲート（λ = 1 − δ）は λ をエピソードの間で並べ替えた対照と区別できなかった（t = 0.35）。
常に補完しても害がなく、補完が害になるエピソードがほとんどないためと解釈した。ここでは、学習（12.6.213節の修正1の
設定、context_realtext）は変えずに、想起に使う記憶の痕跡の数を言語の間で偏らせる（よく経験した言語 1つは 2000 個、
他は 500 個）。CA3 のアトラクタが優勢な言語へ引き寄せられ、少数の言語では補完が害になり、食い違い δ のエピソード間の
分布が広がると予想する。同じ学習済みのモデルで、均等な記憶（12.6.214節と同じ）と偏った記憶を比べる。

    --phase dev（開発用シード 100〜104）/ eval（評価用シード 0〜4 で一度だけ）
    結果は results/context_lm/context_imbalance_{phase}.json、集計と判定は --analyze <結果ファイル>

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_imbalance --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_context_imbalance --analyze results/context_lm/context_imbalance_dev.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from multiprocessing import Pool
from pathlib import Path

import torch

from ..context_realtext import (LANG_CODES, REAL_RULES, NgramSpeaker, RealConfig, evaluate_realtext, split_docs,
                                train_realtext)
from ..run_env import env_info
from .run_context_cmp import _t
from .run_context_lm import _mean
from .run_context_realtext import (DEFAULT_CORPUS, EPISODES, FIXES, N_SPEAKER, N_TRAIN, POOL, REGIMES, SETS,
                                   SPEAKER_ORDER, URL, VOCAB_LANGS, fetch_corpus, load_sets)

DEV_SEEDS = tuple(range(100, 105))
EVAL_SEEDS = tuple(range(5))
CUE_NS = (1, 2, 4, 8, 16)
# 偏った記憶（12.6.215節で計測前に固定）: 優勢な言語（シードごとの乱数で1つ）は 2000 個、他は 500 個（4 : 1）
N_DOM, N_MINOR = 2000, 500
# 開発用シードでの修正（12.6.215節の開発の経過）。0 は初期設定
FIXES_IMB: dict[int, dict] = {0: {}}
# 判定条件（12.6.215節で計測前に固定）
T_MIN, NO_HARM, SD_FRAC = 2.0, -0.005, 0.75
NS3 = ("2", "4", "8")


def dominant_of(seed: int, n_l: int) -> int:
    return int(torch.randint(0, n_l, (1,), generator=torch.Generator().manual_seed(seed + 57)))


def run_one(args: tuple, steps: int = 1500, episodes: int = EPISODES, cue_ns=CUE_NS, corpus: Path = DEFAULT_CORPUS,
            fix: int = 0) -> dict:
    set_name, regime, seed = args
    t0 = time.time()
    torch.set_num_threads(1)
    vocab, docs = load_sets(corpus)
    v = len(vocab) + 1
    langs = SETS[set_name]
    data = [split_docs(docs[lang], vocab, N_TRAIN, N_SPEAKER, lang) for lang in langs]
    speakers = [NgramSpeaker(d.speaker, v, SPEAKER_ORDER) for d in data]
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(langs), generator=torch.Generator().manual_seed(seed + 31)).tolist()
    # 学習は 12.6.214節（修正1）と同じ
    cfg = RealConfig(regime=regime, steps_per_lang=steps, order=order, **FIXES[1])
    trained = train_realtext(data, v, cfg, g)
    dom = dominant_of(seed, len(langs))
    keep = [N_DOM if a == dom else N_MINOR for a in range(len(langs))]
    over = FIXES_IMB[fix]
    out = {"set": set_name, "regime": regime, "seed": seed, "order": [LANG_CODES[langs[i]] for i in order],
           "langs": [LANG_CODES[x] for x in langs], "dominant": dom, "keep": keep,
           "final_loss": trained["final_loss"]}
    state = g.get_state()
    for key, mk in (("bal", None), ("imb", keep)):
        g.set_state(state)  # 2つの記憶で同じ乱数の流れから始める
        ev = evaluate_realtext(data, speakers, trained, cfg, v, tuple(cue_ns), g, episodes=episodes, pool=POOL,
                               mem_keep=mk, detail=True, **over)
        ev.pop("native_per_lang", None)
        out[key] = ev
    out["seconds"] = time.time() - t0
    return out


def _run_mean(r: dict, mem: str, a: str, b: str | None, ns=NS3, langs: list[int] | None = None) -> float:
    """1回の acc(a) − acc(b)（ns の平均。langs を与えればその言語だけの平均）。"""
    def acc(rule: str, n: str) -> float:
        c = r[mem]["cues"][n]
        return _mean([c["acc_lang"][rule][j] for j in langs]) if langs is not None else c["acc"][rule]
    return _mean([acc(a, n) - (acc(b, n) if b else 0.0) for n in ns])


def _stat(d: list[float]) -> dict:
    return {"平均": _mean(d), "t": _t(d) if len(d) > 1 else float("nan"), "回数": len(d),
            "正の回": f"{sum(x > 0 for x in d)}/{len(d)}"}


def analyze(runs: list[dict]) -> dict:
    """12.6.215節の H9 の判定。1回 = 言語の組 × 学習の形 × シード。全 4 組をまとめて判定する。"""
    cmp, shuf, ca3, tcm = ("active_replay_cmp", "active_replay_cmp_shuf", "active_replay_ca3", "active_replay_tcm")
    minor = {id(r): [j for j in range(len(r["langs"])) if j != r["dominant"]] for r in runs}
    # 操作の確認 H9-D: (a) 少数の言語で補完が害になる、(b) λ_cmp の言語の中の標準偏差が均等な記憶より大きい
    harm = [_run_mean(r, "imb", ca3, tcm, ("2", "4"), minor[id(r)]) for r in runs]
    harm_bal = [_run_mean(r, "bal", ca3, tcm, ("2", "4"), minor[id(r)]) for r in runs]
    sd = {mem: [_mean([r[mem]["cues"][n]["lam_sd"][cmp] for n in NS3]) for r in runs] for mem in ("bal", "imb")}
    wider = sum(b > a for a, b in zip(sd["bal"], sd["imb"]))
    d_ok = _mean(harm) < 0 and wider >= SD_FRAC * len(runs)
    v: dict = {"H9-D": {"少数の言語の acc(ca3) − acc(tcm)（2・4）": _stat(harm), "参考_均等な記憶": _stat(harm_bal),
                        "λ_cmp の言語の中の SD": {"均等": _mean(sd["bal"]), "偏り": _mean(sd["imb"]),
                                              "偏りが大きい回": f"{wider}/{len(runs)}"},
                        "合格": d_ok}}
    g_imb = [_run_mean(r, "imb", cmp, shuf) for r in runs]
    g_bal = [_run_mean(r, "bal", cmp, shuf) for r in runs]
    s_g = _stat(g_imb)
    v["H9-G"] = {"偏り: cmp − cmp_shuf（2・4・8）": s_g, "合格": s_g["平均"] > 0 and s_g["t"] > T_MIN}
    s_i = _stat([a - b for a, b in zip(g_imb, g_bal)])
    v["H9-I"] = {"(cmp − cmp_shuf)_偏り − (同)_均等": s_i, "参考_均等の cmp − cmp_shuf": _stat(g_bal),
                 "合格": s_i["平均"] > 0 and s_i["t"] > T_MIN}
    s_k = _stat([_run_mean(r, "imb", cmp, ca3) for r in runs])
    v["H9-K"] = {"偏り: cmp − ca3（2・4・8）": s_k, "参考_均等": _stat([_run_mean(r, "bal", cmp, ca3) for r in runs]),
                 "合格": s_k["平均"] > 0 and s_k["t"] > T_MIN}
    s_s = {n: _mean([_run_mean(r, "imb", cmp, tcm, (n,)) for r in runs]) for n in NS3}
    v["H9-S"] = {"偏り: cmp − tcm（手がかりごと）": s_s, "合格": all(x >= NO_HARM for x in s_s.values())}
    # 参考（判定に含めない）
    ref: dict = {}
    for mem in ("bal", "imb"):
        ref[mem] = {
            "gain（active との差、2・4・8 の平均）": {
                a: _mean([_run_mean(r, mem, rule, "active") for r in runs])
                for a, rule in (("tcm", tcm), ("ca3", ca3), ("ach", "active_replay_ach"), ("cmp", cmp),
                                ("cmp_shuf", shuf), ("own", "active_replay_own"), ("holdout_cmp", "active_holdout_cmp"))},
            "ach − ach_shuf": _stat([_run_mean(r, mem, "active_replay_ach", "active_replay_ach_shuf") for r in runs]),
            "少数の言語の cmp − cmp_shuf": _stat([_run_mean(r, mem, cmp, shuf, NS3, minor[id(r)]) for r in runs]),
            "優勢な言語の cmp − cmp_shuf": _stat([_run_mean(r, mem, cmp, shuf, NS3, [r["dominant"]]) for r in runs]),
            "純度（手がかり2個、cmp・ca3・tcm）": {a: _mean([r[mem]["cues"]["2"]["purity"][rule] for r in runs])
                                       for a, rule in (("cmp", cmp), ("ca3", ca3), ("tcm", tcm))},
            "λ_cmp の平均": {n: _mean([r[mem]["cues"][n]["lambda"][cmp] for r in runs]) for n in ("1", "2", "4", "8", "16")}}
    groups: dict = {}
    for r in runs:
        groups.setdefault(f"{r['set']}-{r['regime']}", []).append(r)
    ref["組ごとの cmp − cmp_shuf（偏り）"] = {nm: _stat([_run_mean(r, "imb", cmp, shuf) for r in rs])
                                         for nm, rs in sorted(groups.items())}
    ref["組ごとの cmp − ca3（偏り）"] = {nm: _stat([_run_mean(r, "imb", cmp, ca3) for r in rs])
                                    for nm, rs in sorted(groups.items())}
    ref["組ごとの gap（CE_holdout − CE_own_buf）"] = {
        nm: _mean([r["bal"]["native"]["holdout"] - r["bal"]["native"]["own_buf"] for r in rs])
        for nm, rs in sorted(groups.items())}
    v["参考"] = ref
    status = {k: v[k]["合格"] for k in ("H9-D", "H9-G", "H9-I", "H9-K", "H9-S")}
    v["合否"] = status
    v["条件を作れた・エピソードごとの寄与"] = status["H9-D"] and status["H9-G"]
    v["分布の広さが寄与を生んだ"] = v["条件を作れた・エピソードごとの寄与"] and status["H9-I"]
    v["偏った記憶ではゲートが要る"] = status["H9-K"] and status["H9-S"]
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", type=Path, nargs="+", help="結果ファイルを集計して判定する")
    ap.add_argument("--phase", choices=("dev", "eval"), default="dev")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--regimes", nargs="+", default=list(REGIMES))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--fix", type=int, default=0, help="12.6.215節の修正の番号")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.analyze:
        runs = [r for p in args.analyze for r in json.loads(p.read_text(encoding="utf-8"))["runs"]]
        print(json.dumps(analyze(runs), ensure_ascii=False, indent=1))
        return
    corpus = fetch_corpus(args.corpus)
    vocab, docs = load_sets(corpus)
    digest = {LANG_CODES[lang]: hashlib.sha256("\n".join(docs[lang]).encode("utf-8")).hexdigest()[:16]
              for lang in VOCAB_LANGS}
    seeds = args.seeds or (DEV_SEEDS if args.phase == "dev" else EVAL_SEEDS)
    jobs = [(s_, reg, seed) for s_ in args.sets for reg in args.regimes for seed in seeds]
    with Pool(args.workers) as pool:
        runs = pool.starmap(run_one, [(j, args.steps, EPISODES, CUE_NS, corpus, args.fix) for j in jobs])
    out = {"環境": env_info(),
           "設定": {"corpus": "Europarl（NLTK europarl_raw の標本）", "url": URL, "sha256_16": digest,
                  "vocab": vocab, "sets": {k: [LANG_CODES[x] for x in v] for k, v in SETS.items()},
                  "model": RealConfig(**FIXES[1]).__dict__ | {"steps_per_lang": args.steps},
                  "memory": {"dominant": N_DOM, "minor": N_MINOR}, "fix": args.fix, "fix_over": FIXES_IMB[args.fix],
                  "episodes": EPISODES, "pool": POOL, "cue_ns": CUE_NS, "rules": REAL_RULES, "seeds": list(seeds)},
           "runs": runs}
    out["判定"] = analyze(runs)
    tag = f"_fix{args.fix}" if args.fix else ""
    path = args.out or Path("results/context_lm") / f"context_imbalance{tag}_{args.phase}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["判定"]["合否"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
