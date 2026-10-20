"""比較器ゲートの CA3 補完を、実テキストの多言語の混合と自己回帰の文字 Transformer に移す（12.6.213〜214節）。

12.6.191〜192節の比較器のゲート（λ = 1 − δ）は、合成言語の上書きのある組で診断の利得の 93% を取り戻した。ここでは
同じ規則を、Europarl の実テキスト（言語の組 R: 近縁のロマンス語4つ、G: ゲルマン語5つ＋フィンランド語）と、文脈ごとの
ヘッドを持つ因果的な文字 Transformer に移す（context_realtext）。比べるのは補完なし（active_replay_tcm）、常に補完
（active_replay_ca3）、λ = u（active_replay_ach）、比較器（active_replay_cmp、主）、比較器の λ の並べ替え
（active_replay_cmp_shuf）。統制は保留サンプル（active_holdout_cmp）、診断は本来の文脈のバッファ（active_replay_own）。

    --phase dev（開発用シード 100〜104）/ eval（評価用シード 0〜4 で一度だけ）
    結果は results/context_lm/context_realtext_{phase}.json、集計と判定は --analyze <結果ファイル>
    コーパスは --corpus（既定 ~/.cache/neurocortex/europarl_raw）。なければ NLTK のデータの配布元から取得する。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_realtext --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_context_realtext --analyze results/context_lm/context_realtext_dev.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from multiprocessing import Pool
from pathlib import Path

import torch

from ..context_realtext import (LANG_CODES, REAL_RULES, NgramSpeaker, RealConfig, build_vocab, evaluate_realtext,
                                read_europarl, split_docs, train_realtext)
from ..run_env import env_info
from .run_context_cmp import _t
from .run_context_lm import _mean

URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/europarl_raw.zip"
DEFAULT_CORPUS = Path.home() / ".cache" / "neurocortex" / "europarl_raw"
# 言語の組（12.6.213節で計測前に固定）。R は近縁（合成の族 B にあたる重なりの大きい組）、G は近縁の対（da・sv、
# de・nl）と離れた言語（en、fi）の混ざった組
SETS = {"R": ("spanish", "portuguese", "italian", "french"),
        "G": ("english", "german", "dutch", "danish", "swedish", "finnish")}
VOCAB_LANGS = ("danish", "dutch", "english", "finnish", "french", "german", "italian", "portuguese", "spanish",
               "swedish")  # 共有の文字の集合はこの10言語から（ギリシャ語は文字が違うので除く）
N_TRAIN, N_SPEAKER = 6, 3  # 日ごとの文書: 先頭6日を LM の学習、次の3日を話し手、最後の1日をテスト
REGIMES = ("joint", "seq")
DEV_SEEDS = tuple(range(100, 105))
EVAL_SEEDS = tuple(range(5))
CUE_NS = (1, 2, 4, 8, 16)
NS = tuple(str(n) for n in CUE_NS)
EPISODES, POOL = 100, 4
SPEAKER_ORDER = 5
# 開発用シードでの修正（12.6.213節の開発の経過）。0 は初期設定
FIXES: dict[int, dict] = {0: {}, 1: {"replay_window": True}}

# 判定条件（12.6.213節で計測前に固定。値は 12.6.191節と同じ）
GAP_OVER, GAP_NONE = 0.05, 0.02  # 上書きあり: CE_holdout − CE_own_buf ≧ 0.05、上書きなし: ≦ 0.02（nats）
NO_HARM, BAD_RUN, BAD_FRAC = -0.005, -0.02, 0.10
CAPTURE, OWN_MIN, NONE_ABS, T_MIN = 0.5, 0.01, 0.01, 2.0


def fetch_corpus(root: Path) -> Path:
    """europarl_raw がなければ取得して展開する（root はその親のディレクトリに europarl_raw/ を作る）。"""
    if (root / "english").is_dir():
        return root
    root.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=300) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for info in z.infolist():
            target = (root.parent / info.filename).resolve()
            if root.parent.resolve() not in target.parents:  # 展開先の外へ書かない
                raise ValueError(f"不正なパス: {info.filename}")
            z.extract(info, root.parent)
    return root


def load_sets(root: Path) -> tuple[str, dict]:
    """共有の文字の集合と、言語ごとの日ごとの文書。"""
    docs = {lang: read_europarl(root, lang) for lang in VOCAB_LANGS}
    return build_vocab([d for lang in VOCAB_LANGS for d in docs[lang]]), docs


def run_one(args: tuple, steps: int = 1500, episodes: int = EPISODES, cue_ns=CUE_NS, corpus: Path = DEFAULT_CORPUS,
            cfg_over: dict | None = None) -> dict:
    set_name, regime, seed = args
    torch.set_num_threads(1)
    vocab, docs = load_sets(corpus)
    v = len(vocab) + 1
    langs = SETS[set_name]
    data = [split_docs(docs[lang], vocab, N_TRAIN, N_SPEAKER, lang) for lang in langs]
    speakers = [NgramSpeaker(d.speaker, v, SPEAKER_ORDER) for d in data]
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(langs), generator=torch.Generator().manual_seed(seed + 31)).tolist()
    cfg = RealConfig(regime=regime, steps_per_lang=steps, order=order, **(cfg_over or {}))
    trained = train_realtext(data, v, cfg, g)
    ev = evaluate_realtext(data, speakers, trained, cfg, v, tuple(cue_ns), g, episodes=episodes, pool=POOL)
    ev.update({"set": set_name, "regime": regime, "seed": seed, "order": [LANG_CODES[langs[i]] for i in order],
               "langs": [LANG_CODES[x] for x in langs], "final_loss": trained["final_loss"]})
    return ev


def _name(r: dict) -> str:
    return f"{r['set']}-{r['regime']}"


def analyze(runs: list[dict]) -> dict:
    """12.6.213節の組の分類と H8 の判定。組 = 言語の組 × 学習の形、1回 = 組 × シード。"""
    groups: dict[str, list[dict]] = {}
    for r in runs:
        groups.setdefault(_name(r), []).append(r)
    rules = {"tcm": "active_replay_tcm", "ca3": "active_replay_ca3", "ach": "active_replay_ach",
             "ach_shuf": "active_replay_ach_shuf", "cmp": "active_replay_cmp", "cmp_shuf": "active_replay_cmp_shuf",
             "holdout_cmp": "active_holdout_cmp", "own": "active_replay_own", "replay": "active_replay",
             "uniform": "uniform"}
    g: dict = {}
    for nm, rs in sorted(groups.items()):
        nat = {key: _mean([r["native"][key] for r in rs]) for key in rs[0]["native"]}
        gap = nat["holdout"] - nat["own_buf"]
        cls = "上書きあり" if gap >= GAP_OVER else ("上書きなし" if gap <= GAP_NONE else "中間")
        row = {"n_seeds": len(rs), "gap": gap, "分類": cls, "native": nat,
               "ce_oracle": _mean([r["ce_oracle"] for r in rs]),
               "ce_wrong": _mean([_mean([x for i, row_ in enumerate(r["ce_matrix"]) for jj, x in enumerate(row_)
                                         if i != jj]) for r in rs]),
               "speaker_ce": _mean([_mean(r["speaker_ce"]) for r in rs]), "acc": {}, "gain": {}, "purity": {},
               "lambda": {}}
        for n in NS:
            row["acc"][n] = {rule: _mean([r["cues"][n]["acc"][rule] for r in rs]) for rule in rs[0]["cues"][n]["acc"]}
            row["gain"][n] = {a: row["acc"][n][rule] - row["acc"][n]["active"] for a, rule in rules.items()}
            row["purity"][n] = {a: _mean([r["cues"][n]["purity"][rule] for r in rs]) for a, rule in rules.items()
                                if rule in rs[0]["cues"][n]["purity"]}
            row["lambda"][n] = {a: _mean([r["cues"][n]["lambda"][rule] for r in rs]) for a, rule in rules.items()
                                if rule in rs[0]["cues"][n]["lambda"]}
        g[nm] = row
    over = [nm for nm in g if g[nm]["分類"] == "上書きあり"]
    none = [nm for nm in g if g[nm]["分類"] == "上書きなし"]
    over_runs = [r for r in runs if _name(r) in over]
    none_runs = [r for r in runs if _name(r) in none]

    def avg(nms: list[str], n: str, a: str) -> float:
        return _mean([g[nm]["gain"][n][a] for nm in nms])

    def paired(rs: list[dict], a: str, b: str, ns: tuple[str, ...]) -> dict:
        d = [_mean([r["cues"][n]["acc"][a] - r["cues"][n]["acc"][b] for n in ns]) for r in rs]
        return {"平均": _mean(d), "t": _t(d) if len(d) > 1 else float("nan"), "回数": len(d)}

    v: dict = {"上書きあり": over, "上書きなし": none}
    if not over:
        for key in ("H8-T1", "H8-T2", "H8-C", "H8-K", "H8-G"):
            v[key] = "検証できず（上書きありの組がない）"
    else:
        bad = {n: sum(r["cues"][n]["acc"]["active_replay_cmp"] - r["cues"][n]["acc"]["active"] < BAD_RUN
                      for r in over_runs) for n in ("8", "16")}
        m = {n: avg(over, n, "cmp") for n in ("2", "4", "8", "16")}
        v["H8-T1"] = {"gain_cmp の平均": m, "−0.02 を下回る回": {n: f"{b}/{len(over_runs)}" for n, b in bad.items()},
                      "合格": all(x >= NO_HARM for x in m.values())
                      and all(b <= BAD_FRAC * len(over_runs) for b in bad.values())}
        a3 = {a: _mean([avg(over, n, a) for n in ("2", "4", "8")]) for a in rules}
        if a3["own"] < OWN_MIN:
            t2: object = "検証できず（診断の利得が 0.01 未満）"
        else:
            t2 = a3["cmp"] >= CAPTURE * a3["own"] and a3["holdout_cmp"] <= a3["cmp"] / 2
        v["H8-T2"] = {"gain（2・4・8 の平均）": a3,
                      "取り戻した割合": {a: a3[a] / a3["own"] if a3["own"] else float("nan")
                                    for a in ("tcm", "ca3", "ach", "ach_shuf", "cmp", "cmp_shuf", "replay")},
                      "合格": t2}
        d_c = _mean([avg(over, n, "cmp") - avg(over, n, "tcm") for n in ("2", "4")])
        v["H8-C"] = {"gain_cmp − gain_tcm（2・4 の平均）": d_c, "合格": d_c > 0}
        d_k = {a: a3["cmp"] - a3[a] for a in ("ca3", "ach")}
        v["H8-K"] = {"gain_cmp − gain_x（2・4・8 の平均）": d_k, "合格": all(x >= 0 for x in d_k.values())}
        p = paired(over_runs, "active_replay_cmp", "active_replay_cmp_shuf", ("2", "4", "8"))
        v["H8-G"] = {"cmp − cmp_shuf（1回ごと、2・4・8）": p, "合格": p["平均"] > 0 and p["t"] > T_MIN,
                     "参考_ach − ach_shuf": paired(over_runs, "active_replay_ach", "active_replay_ach_shuf",
                                                   ("2", "4", "8"))}
    if none:
        m_none = {n: avg(none, n, "cmp") for n in ("8", "16")}
        v["H8b"] = {"上書きなしの gain_cmp": m_none,
                    "参考": {a: {n: avg(none, n, a) for n in ("8", "16")} for a in ("tcm", "ca3", "ach", "cmp_shuf")},
                    "合格": all(abs(x) <= NONE_ABS for x in m_none.values())}
        v["参考_上書きなしの cmp − cmp_shuf"] = paired(none_runs, "active_replay_cmp", "active_replay_cmp_shuf",
                                                    ("8", "16"))
    else:
        v["H8b"] = "検証できず（上書きなしの組がない）"
    status = {key: (v[key] if isinstance(v[key], str) else v[key]["合格"]) for key in
              ("H8-T1", "H8-T2", "H8b", "H8-C", "H8-K", "H8-G")}
    v["合否"] = status
    v["利得が移った"] = all(status[k] is True for k in ("H8-T1", "H8-T2", "H8b"))
    v["比較器が最良のまま"] = v["利得が移った"] and all(status[k] is True for k in ("H8-C", "H8-K"))
    v["エピソードごとの寄与"] = status["H8-G"] is True
    return {"組": g, "判定": v}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", type=Path, nargs="+", help="結果ファイルを集計して判定する")
    ap.add_argument("--phase", choices=("dev", "eval"), default="dev")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--regimes", nargs="+", default=list(REGIMES))
    ap.add_argument("--reuse", type=Path, help="修正が変えない組（joint）の回をこの結果ファイルから再利用する")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--fix", type=int, default=0, help="12.6.213節の修正の番号（1: リプレイを問う窓にする）")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.analyze:
        runs = [r for p in args.analyze for r in json.loads(p.read_text(encoding="utf-8"))["runs"]]
        print(json.dumps(analyze(runs)["判定"], ensure_ascii=False, indent=1))
        return
    corpus = fetch_corpus(args.corpus)
    vocab, docs = load_sets(corpus)
    digest = {LANG_CODES[lang]: hashlib.sha256("\n".join(docs[lang]).encode("utf-8")).hexdigest()[:16]
              for lang in VOCAB_LANGS}
    seeds = args.seeds or (DEV_SEEDS if args.phase == "dev" else EVAL_SEEDS)
    jobs = [(s_, reg, seed) for s_ in args.sets for reg in args.regimes for seed in seeds]
    with Pool(args.workers) as pool:
        runs = pool.starmap(run_one, [(j, args.steps, EPISODES, CUE_NS, corpus, FIXES[args.fix]) for j in jobs])
    if args.reuse:  # 修正1は seq のリプレイだけを変えるので、joint の回は同じ値になる（テストで確認）
        old = json.loads(args.reuse.read_text(encoding="utf-8"))["runs"]
        runs += [r for r in old if r["regime"] not in args.regimes and r["seed"] in seeds]
    for r in runs:
        r.pop("native_per_lang", None)
    out = {"環境": env_info(),
           "設定": {"corpus": "Europarl（NLTK europarl_raw の標本）", "url": URL, "sha256_16": digest,
                  "vocab": vocab, "sets": {k: [LANG_CODES[x] for x in v] for k, v in SETS.items()},
                  "split": {"train_days": N_TRAIN, "speaker_days": N_SPEAKER}, "speaker_order": SPEAKER_ORDER,
                  "model": RealConfig(**FIXES[args.fix]).__dict__ | {"steps_per_lang": args.steps}, "fix": args.fix,
                  "cmp": "λ = 1 − δ、δ = 1 − cos(s_S, t̂)/cos(s_1, t̂)", "episodes": EPISODES, "pool": POOL,
                  "cue_ns": CUE_NS, "rules": REAL_RULES, "seeds": list(seeds)},
           "runs": runs}
    out["集計"] = analyze(runs)
    tag = f"_fix{args.fix}" if args.fix else ""
    path = args.out or Path("results/context_lm") / f"context_realtext{tag}_{args.phase}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["集計"]["判定"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
