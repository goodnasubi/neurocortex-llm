"""可塑性のブレーキの範囲と時期、臨界期の階層的な順序（12.6.153節の設計）の主実験。

閾値の閉じ方は (開いている期間, 閉じる期間) = (0.25, 0.25) で、閉じ始めと同時に θ の積分を止める（12.6.149節の frz）。
規模 E=16・32 の1層、η は 12.6.145節で選んだもの（E=16: 0.3、E=32: 1）。ブレーキの倍率は閉じ終わりで β = 0.1。
(A) 範囲: ブレーキを閾値と同じ時期に、ルータ（R）・GRU（G）・埋め込み（M）・専門家（X）の組合せに掛ける。
(B) 順序: 4つの群すべてに、閉じ終わる時期をずらして掛ける（階層的 = 下位から M → G → R → X、逆順 = X → R → G → M、
    同時 = 4つの時期の平均で一斉に）。各群の閉じる期間は 0.125·T。
(C) CE を下げる割り当て（`moe_hier.oracle_assignment`）と均等な割り当てのずれ。選択専用（閉じない）の E=8（η=0.01）・
    16（0.3）・32（1）と、(A)(B) の全条件で学習の終わりに測る。

測定・H1〜H7 の判定は `run_moe_scale.py`、時系列の補助は `run_moe_critical.py`、偏りの戻りは `run_moe_brake.py` の関数を使う。

使い方（シード×規模ごとに別プロセスで出して結合する）:
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_hier --seeds 100 --scales 32,1 \
        --out results/moe_hier/parts/dev_s100_E32L1.json
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_hier --merge results/moe_hier/parts/dev_*.json \
        --out results/moe_hier/dev.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from ..data import CharCorpus, synthetic_corpus
from ..hub_load import make_windows
from ..moe_critical import absorption_metrics
from ..moe_hier import HierBrakeLM, oracle_assignment, train_hier
from ..run_env import env_info
from .run_homeostatic_moe import ms
from .run_moe_brake import CP_ETA, bias_return, series_mean
from .run_moe_critical import BASE, judge_key, slim
from .run_moe_scale import run_one

TH_SCHED = (0.25, 0.25)  # 閾値の閉じ方（12.6.149節の H11 の組と同じ。固定化期間 50%）
BETA = 0.1
# (A) 範囲。閾値と同じ時期（TH_SCHED）に掛ける群
SCOPES = {"none": (), "R": ("router",), "RG": ("router", "rnn"), "RGM": ("router", "rnn", "embed"),
          "GM": ("rnn", "embed"), "X": ("experts",), "RGMX": ("router", "rnn", "embed", "experts")}
# (B) 順序。群ごとの閉じ終わる時期（総ステップ数の割合）。閉じる期間はどれも ORDER_RAMP
ORDER_RAMP = 0.125
ORDER_ENDS = (0.25, 0.375, 0.5, 0.625)
ORDERS = {"hier": dict(zip(("embed", "rnn", "router", "experts"), ORDER_ENDS)),
          "rev": dict(zip(("experts", "router", "rnn", "embed"), ORDER_ENDS)),
          "sim": {g: statistics.mean(ORDER_ENDS) for g in ("embed", "rnn", "router", "experts")}}
SEL_ETA = {"E8_L1": 0.01, **CP_ETA}  # (C) の選択専用（閉じない）
# 条件の名前の接頭辞 → 閉じ終わりの倍率 β。"" は主条件（β=0.1）、"0" は修正1で足した β=0（完全に止める）
PREFIX_BETA = {"": BETA, "0": 0.0}


def scope_sched(scope: str) -> dict:
    return {g: TH_SCHED for g in SCOPES[scope]}


def order_sched(order: str) -> dict:
    return {g: (end - ORDER_RAMP, ORDER_RAMP) for g, end in ORDERS[order].items()}


def conditions(e: int, l: int) -> dict[str, dict]:
    name = f"E{e}_L{l}"
    c: dict[str, dict] = {}
    eta = SEL_ETA.get(name)
    if name != "E8_L1":
        c.update({"aux0": {"kind": "plain", "coef": 0.0}, "aux0.05": {"kind": "plain", "coef": 0.05},
                  "gate0.1": {"kind": "gate", "eta": 0.1}})
    if eta is None:
        return c
    c[f"sel{eta:g}"] = {"kind": "select", "eta": eta}
    if name == "E8_L1":
        return c
    th = {"kind": "select", "eta": eta, "open": TH_SCHED[0], "ramp": TH_SCHED[1], "freeze": True}
    for pre, beta in PREFIX_BETA.items():
        c.update({f"sc{pre}_{s}": {**th, "sched": scope_sched(s), "beta": beta} for s in SCOPES})
        c.update({f"ord{pre}_{o}": {**th, "sched": order_sched(o), "beta": beta} for o in ORDERS})
    return c


def run_cond(seed: int, e: int, l: int, key: str, cond: dict, corpus: CharCorpus, args: argparse.Namespace) -> dict:
    probe = make_windows(corpus.val, args.seq_len, args.n_windows)[0][:args.n_probe]

    def post(model, x, y, out):
        if l == 1:
            out["oracle"] = oracle_assignment(model, x, y)
        if model.kind != "plain":
            out["absorb"] = absorption_metrics(model, x, y)
            model.adoption()
            out["series"] = model.series

    kw = {"total_steps": args.train_steps, "open_frac": cond.get("open"), "ramp_frac": cond.get("ramp", 0.0),
          "probe": probe, "log_every": args.log_every, "vec_every": 10 ** 9, "freeze": cond.get("freeze", False),
          "lag": args.lag, "brake_sched": cond.get("sched"), "beta": cond.get("beta", args.beta)}
    r = run_one(seed, e, l, cond, corpus, args, HierBrakeLM, args.train_steps, post=post, train_fn=train_hier, **kw)
    return slim(r, False)


def summarize_key(sub: list[dict], key: str, cond: dict, total: int) -> dict:
    rs = [r[key] for r in sub]
    d = {"gini": ms([r["gini"] for r in rs]), "gini_max": max(r["gini"] for r in rs),
         "n_used": ms([r["n_used"] for r in rs]), "all_used_seeds": sum(r["all_used"] for r in rs),
         "base_ce": ms([r["lesion"]["zero"]["base_ce"] for r in rs]),
         "ratio_of_means": statistics.mean(r["lesion"]["zero"]["d_target"] for r in rs)
         / max(statistics.mean(r["lesion"]["zero"]["d_random_mean"] for r in rs), 1e-4),
         "gate_mean": ms([r["gate_mean"] for r in rs])}
    if "oracle" in rs[0]:
        d["oracle"] = {m: ms([r["oracle"][m] for r in rs if r["oracle"][m] is not None]) for m in rs[0]["oracle"]}
    if "threshold_off" in rs[0]:
        d["gini_off"] = ms([r["threshold_off"]["gini"] for r in rs])
        d["absorb"] = {m: ms([r["absorb"][m] for r in rs]) for m in ("agree", "d_ce_off", "d_ce_off_norm", "contrib")}
        d["series"] = series_mean(sub, key)
    if cond.get("open") is not None:
        br = [bias_return(r, cond, total) for r in rs]
        d["return"] = {k: ms([b[k] for b in br]) for k in ("gini_close", "gini_end", "return")}
    return d


def returns(sub: list[dict], key: str, cond: dict, total: int) -> list[float]:
    return [bias_return(r[key], cond, total)["return"] for r in sub]


def paired_less(a: list[float], b: list[float]) -> int:
    return sum(x < y for x, y in zip(a, b))


def judge_scale(sub: list[dict], conds: dict, args: argparse.Namespace, pre: str = "") -> dict:
    """H16（範囲）・H17（順序）の判定。12.6.153節の合格条件。pre は β の接頭辞（`PREFIX_BETA`）。"""
    T, n = args.train_steps, len(sub)
    full = {k: k.replace(f"sc{pre}_", "sc_", 1).replace(f"ord{pre}_", "ord_", 1)
            for k in conds if k.startswith((f"sc{pre}_", f"ord{pre}_"))}
    sub = [{**r, **{v: r[k] for k, v in full.items()}} for r in sub]
    conds = {**conds, **{v: conds[k] for k, v in full.items()}}
    ret = {v: returns(sub, v, conds[v], T) for v in full.values()}
    gini = {k: [r[k]["gini"] for r in sub] for k in ret}
    m = {k: statistics.mean(v) for k, v in ret.items()}
    v: dict = {"戻り_平均": m, "戻り_シード別": ret,
               "ジニ_平均": {k: statistics.mean(gini[k]) for k in ret}}
    v["H16a_入力の安定化が要る"] = bool(m["sc_RGM"] <= 0.5 * m["sc_R"] and paired_less(ret["sc_RGM"], ret["sc_R"]) >= n - 1)
    v["H16b_入力だけでルータ以上に戻りを防ぐ"] = bool(m["sc_GM"] <= m["sc_R"]
                                            and paired_less(ret["sc_GM"], ret["sc_R"]) >= n - 1)
    v["H16c_専門家のブレーキは戻りに効かない"] = bool(abs(m["sc_X"] - m["sc_none"]) <= 0.02)
    v["参考_GRUだけで足りる_RGとRGMの差≦0.01"] = bool(m["sc_RG"] - m["sc_RGM"] <= 0.01)
    gh, gr, gs = gini["ord_hier"], gini["ord_rev"], gini["ord_sim"]
    v["H17a_階層的な順序で学習の終わりの偏りが小さい"] = bool(
        statistics.mean(gh) < statistics.mean(gr) and paired_less(gh, gr) >= n - 1
        and statistics.mean(gh) < statistics.mean(gs) and paired_less(gh, gs) >= n - 1)
    v["H17b_階層的な順序で戻りが小さい_≦0.02"] = bool(m["ord_hier"] <= 0.02)
    jk = {k: judge_key(sub, k, args) for k in ret}
    for k, j in jk.items():
        j["取り込み_H1〜H3かつジニ≦0.3"] = bool(j["H1_崩壊を防ぐ"] and j["H2_ハブの非対称性を消す"]
                                         and j["H3_性能を損なわない"] and j["ジニ"] <= 0.3)
    v["条件ごとのH1〜H3"] = {k: {kk: j[kk] for kk in ("H1_崩壊を防ぐ", "H2_ハブの非対称性を消す", "H3_性能を損なわない",
                                                  "ジニ", "取り込み_H1〜H3かつジニ≦0.3")} for k, j in jk.items()}
    v["H17c_階層的な順序で取り込み"] = jk["ord_hier"]["取り込み_H1〜H3かつジニ≦0.3"]
    return v


def judge_h15(subs: dict[str, list[dict]]) -> dict:
    """H15: CE を下げる割り当てのジニ（選択専用・閉じない・学習の終わり）が E=32 で E=8 より大きく（平均と、同じシードで
    5本中4本以上）、閾値がずらしたトークンのうち閾値なしの選択が CE 最小だった割合（CE に逆らったずらし）も E=32 で大きい。"""
    if "E8_L1" not in subs or "E32_L1" not in subs:
        return {}
    k8, k32 = f"sel{SEL_ETA['E8_L1']:g}", f"sel{SEL_ETA['E32_L1']:g}"
    out = {}
    for f, name in (("oracle_gini", "H15a_CEを下げる割り当ての偏りがE32で大きい"),
                    ("shift_cost", "H15b_CEに逆らったずらしがE32で多い")):
        a8 = [r[k8]["oracle"][f] for r in subs["E8_L1"]]
        a32 = [r[k32]["oracle"][f] for r in subs["E32_L1"]]
        out[f] = {"E8": a8, "E32": a32}
        out[name] = bool(statistics.mean(a32) > statistics.mean(a8) and paired_less(a8, a32) >= len(a8) - 1)
    for n, sub in subs.items():
        k = f"sel{SEL_ETA[n]:g}" if n in SEL_ETA else None
        if k and k in sub[0]:
            out[f"{n}_{k}"] = {f: statistics.mean(r[k]["oracle"][f] for r in sub)
                               for f in ("oracle_gini", "shift_cost", "shift_benefit", "agree_on", "agree_off",
                                         "regret_on", "regret_off")}
    return out


def analyze(subs: dict[str, list[dict]], args: argparse.Namespace) -> tuple[dict, dict]:
    summ, verdict = {}, {}
    for n, sub in subs.items():
        e, l = map(int, n[1:].split("_L"))
        conds = conditions(e, l)
        summ[n] = {k: summarize_key(sub, k, conds.get(k, {}), args.train_steps) for k in sub[0]}
        for pre in PREFIX_BETA:
            ks = [k for k in conds if k.startswith((f"sc{pre}_", f"ord{pre}_"))]
            if ks and all(k in sub[0] for k in (*ks, *BASE)):
                verdict.setdefault(n, {})[f"β={PREFIX_BETA[pre]:g}"] = judge_scale(sub, conds, args, pre)
    verdict["H15"] = judge_h15(subs)
    return summ, verdict


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    p.add_argument("--scales", type=str, nargs="+", default=["8,1", "16,1", "32,1"], help="E,L の列")
    p.add_argument("--only", type=str, nargs="*", default=None, help="この条件のキーだけ測る（部分ファイル）")
    p.add_argument("--beta", type=float, default=BETA)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-sentences", type=int, default=3000)
    p.add_argument("--d", type=int, default=48)
    p.add_argument("--train-steps", type=int, default=600)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--n-windows", type=int, default=64)
    p.add_argument("--n-probe", type=int, default=16)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--lag", type=int, default=100)
    p.add_argument("--k-num", type=int, default=1)
    p.add_argument("--k-den", type=int, default=4)
    p.add_argument("--n-random", type=int, default=30)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", type=str, default="results/moe_hier/dev.json")
    p.add_argument("--merge", type=str, nargs="+", default=None)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    if not args.merge:
        parts = []
        for seed in args.seeds:
            corpus = CharCorpus(synthetic_corpus(n_sentences=args.n_sentences, seed=seed))
            for sc in args.scales:
                e, l = map(int, sc.split(","))
                runs = {}
                for key, cond in conditions(e, l).items():
                    if args.only and key not in args.only:
                        continue
                    z = runs[key] = run_cond(seed, e, l, key, cond, corpus, args)
                    a, o = z.get("absorb", {}), z.get("oracle", {})
                    extra = f" off_gini={z['threshold_off']['gini']:.3f} agree={a['agree']:.3f}" if a else ""
                    br = bias_return(z, cond, args.train_steps) if cond.get("open") is not None else None
                    extra += f" close={br['gini_close']:.3f} ret={br['return']:+.3f}" if br else ""
                    extra += f" oracle_gini={o['oracle_gini']:.3f}" if o else ""
                    print(f"seed={seed} E{e}_L{l}/{key} gini={z['gini']:.3f} used={z['n_used']:.1f} "
                          f"base={z['lesion']['zero']['base_ce']:.4f} ratio={z['lesion']['zero']['ratio']:.2f} "
                          f"gate={z['gate_mean']:.3f}{extra}", flush=True)
                parts.append({"環境": env_info(), "seed": seed, "scale": f"E{e}_L{l}", "runs": runs})
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"環境": env_info(), "設定": vars(args), "部分": parts}, ensure_ascii=False))
        return
    parts = [pt for f in args.merge for pt in json.loads(Path(f).read_text())["部分"]]
    merged: dict = {}
    for pt in parts:
        mm = merged.setdefault((pt["seed"], pt["scale"]), {"seed": pt["seed"], "scale": pt["scale"], "runs": {},
                                                          "環境": []})
        mm["runs"].update(pt["runs"])
        mm["環境"].append(pt["環境"])
    parts = list(merged.values())
    names = sorted(dict.fromkeys(pt["scale"] for pt in parts), key=lambda n: tuple(map(int, n[1:].split("_L"))))
    subs = {n: [pt["runs"] for pt in sorted((pt for pt in parts if pt["scale"] == n), key=lambda pt: pt["seed"])]
            for n in names}
    summ, verdict = analyze(subs, args)
    keep = {"sel0.01", "sel0.3", "sel1", *(f"{a}{p}_{b}" for p in PREFIX_BETA for a, b in
                                         (("sc", "none"), ("sc", "R"), ("sc", "RGM"), ("ord", "hier"), ("ord", "rev"),
                                          ("ord", "sim")))}
    for sub in subs.values():  # 結果ファイルを数 MB に収めるため、シード別の時系列は主な条件だけ残す
        for runs in sub:
            for k in list(runs):
                if k not in keep and "series" in runs[k]:
                    runs[k] = {kk: vv for kk, vv in runs[k].items() if kk != "series"}
    out = {"環境": env_info(), "設定": vars(args),
           "結合元の環境": sorted({json.dumps(e, sort_keys=True) for pt in parts for e in pt["環境"]}),
           "要約": summ, "判定": verdict,
           "シード別": {n: {pt["seed"]: pt["runs"] for pt in parts if pt["scale"] == n} for n in names}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False))
    print(json.dumps(verdict, ensure_ascii=False, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
