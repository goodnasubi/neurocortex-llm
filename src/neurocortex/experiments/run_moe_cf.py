"""選ばれなかった専門家の反実仮想の学習信号でルータを学習する（12.6.157節の設計）の主実験。

(1) 1層・E=16（η=0.3）・E=32（η=1）の選択専用（閉じない）に、ルータの教師つきの補助損失（`moe_cf.CounterfactualLM`）を
    足す。教師は cf（CE を下げる割り当て）・cfs（反実仮想の結果のソフトな教師）・dist（閾値ありの割り当ての蒸留）・
    shuf（cf の教師をトークンで並べ替えた対照）、λ = 0.1・1。閾値を外したときの偏り・ΔCE_off・選択の一致率を測る。
    参考に、閾値なし（補助損失なし）に cf だけを足した条件 p_cf1。
(2) 臨界期型（12.6.149節の frz と同じ閉じ方 (0.25, 0.25)、θ の積分を止める、ブレーキなし）に教師を足し、閉じた後の
    学習の終わりに H1〜H3 とジニ ≦ 0.3 を満たすか（12.6.150節の H12 の続き）。
(3) 余力の課題: 2層・E=32（η=0.3、12.6.141節）で、閉じる順序（階層的・逆順・同時、β=0.1・0）を比べる（12.6.154節の
    残課題2）。条件の作り方は `run_moe_hier.py` と同じ。

測定・H1〜H3 の判定は `run_moe_scale.py`、H7* は `run_moe_critical.py`、要約と戻りは `run_moe_hier.py` の関数を使う。

使い方（シード×規模ごとに別プロセスで出して結合する）:
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_cf --seeds 100 --scales 32,1 \
        --out results/moe_cf/parts/dev_s100_E32L1.json
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_cf --merge results/moe_cf/parts/dev_*.json \
        --ref results/moe_hier/dev_rev1.json --out results/moe_cf/dev.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from ..data import CharCorpus, synthetic_corpus
from ..hub_load import make_windows
from ..moe_cf import CounterfactualLM, train_cf
from ..moe_critical import absorption_metrics
from ..moe_hier import oracle_assignment
from ..run_env import env_info
from .run_homeostatic_moe import ms
from .run_moe_brake import bias_return
from .run_moe_critical import BASE, absorb_summary, judge_key, judge_star, slim
from .run_moe_hier import ORDERS, TH_SCHED, order_sched, paired_less, summarize_key
from .run_moe_scale import run_one

ETA = {"E16_L1": 0.3, "E32_L1": 1.0, "E32_L2": 0.3}  # 12.6.145節（1層）・12.6.141節（2層）で選んだ η
LAMS = (0.1, 1.0)
TEACH = ("cf", "cfs", "dist", "shuf")
CP_TEACH = ("cf", "cfs", "dist")
TAU = 0.05
ORDER_BETAS = {"": 0.1, "0": 0.0}  # (3) の β（接頭辞は run_moe_hier と同じ）


def conditions(e: int, l: int) -> dict[str, dict]:
    name = f"E{e}_L{l}"
    eta = ETA.get(name)
    if eta is None:
        return {}
    c: dict[str, dict] = {"aux0": {"kind": "plain", "coef": 0.0}, "aux0.05": {"kind": "plain", "coef": 0.05},
                          "gate0.1": {"kind": "gate", "eta": 0.1}}
    sel = {"kind": "select", "eta": eta}
    c[f"sel{eta:g}"] = sel
    if l == 1:
        c.update({f"{t}{lam:g}": {**sel, "teacher": t, "lam": lam} for t in TEACH for lam in LAMS})
        c["p_cf1"] = {"kind": "plain", "coef": 0.0, "teacher": "cf", "lam": 1.0}
        cp = {**sel, "open": TH_SCHED[0], "ramp": TH_SCHED[1], "freeze": True}
        c["cp"] = cp
        c.update({f"cp_{t}{lam:g}": {**cp, "teacher": t, "lam": lam} for t in CP_TEACH for lam in LAMS})
    else:
        th = {**sel, "open": TH_SCHED[0], "ramp": TH_SCHED[1], "freeze": True}
        for pre, beta in ORDER_BETAS.items():
            c.update({f"ord{pre}_{o}": {**th, "sched": order_sched(o), "beta": beta} for o in ORDERS})
    return c


def run_cond(seed: int, e: int, l: int, key: str, cond: dict, corpus: CharCorpus, args: argparse.Namespace) -> dict:
    probe = make_windows(corpus.val, args.seq_len, args.n_windows)[0][:args.n_probe]

    def post(model, x, y, out):
        if l == 1:
            out["oracle"] = oracle_assignment(model, x, y)
        if model.kind != "plain":
            out["absorb"] = absorption_metrics(model, x, y)
            out["series"] = model.series
        ts = model.teacher_stats
        if ts:  # 教師と閾値なしのルータの一致率（学習中のバッチ、log_every ステップごと）
            out["teacher_agree"] = [round(ts[i]["agree"], 4) for i in range(0, len(ts), args.log_every)]
            out["teacher_agree_last"] = statistics.mean(t["agree"] for t in ts[-args.log_every * 5:])

    kw = {"total_steps": args.train_steps, "open_frac": cond.get("open"), "ramp_frac": cond.get("ramp", 0.0),
          "probe": probe, "log_every": args.log_every, "vec_every": 10 ** 9, "freeze": cond.get("freeze", False),
          "brake_sched": cond.get("sched"), "beta": cond.get("beta", 0.1), "teacher": cond.get("teacher"),
          "lam": cond.get("lam", 0.0), "tau": args.tau, "shuffle_seed": seed + 1000}
    r = run_one(seed, e, l, cond, corpus, args, CounterfactualLM, args.train_steps, post=post, train_fn=train_cf, **kw)
    return slim(r, False)


def summarize(sub: list[dict], key: str, cond: dict, total: int) -> dict:
    d = summarize_key(sub, key, cond, total)
    rs = [r[key] for r in sub]
    if "teacher_agree_last" in rs[0]:
        d["teacher_agree_last"] = ms([r["teacher_agree_last"] for r in rs])
    if "absorb" in rs[0]:
        d["d_ce_off"] = ms([r["threshold_off"]["d_ce"] for r in rs])
    return d


def select_lam(summ: dict, teacher: str, prefix: str = "") -> float:
    """12.6.157節の規則: H3（検証 CE ≦ 補助損失 0.05 + 0.01）を満たす λ のうち、閾値を外したときのジニの平均が最小のもの。
    満たすものがなければ検証 CE が最小のもの。"""
    ref = summ["aux0.05"]["base_ce"]["mean"] + 0.01
    keys = {lam: f"{prefix}{teacher}{lam:g}" for lam in LAMS}
    ok = [lam for lam, k in keys.items() if summ[k]["base_ce"]["mean"] <= ref]
    if ok:
        return min(ok, key=lambda lam: (summ[keys[lam]]["gini_off"]["mean"], lam))
    return min(LAMS, key=lambda lam: summ[keys[lam]]["base_ce"]["mean"])


def _vals(sub, key, f):
    return [f(r[key]) for r in sub]


def judge_scale(sub: list[dict], summ: dict, lams: dict[str, float], args: argparse.Namespace,
                ref_norm: float | None) -> dict:
    """H18・H19 の判定（12.6.157節の合格条件）。lams: 教師 → 選んだ λ。"""
    n = len(sub)
    sel = next(k for k in sub[0] if k.startswith("sel"))
    g_off = lambda r: r["threshold_off"]["gini"]  # noqa: E731
    d_off = lambda r: r["threshold_off"]["d_ce"]  # noqa: E731
    agree = lambda r: r["absorb"]["agree"]  # noqa: E731
    v: dict = {"選んだλ": lams}

    def lower(key, base, f):
        a, b = _vals(sub, key, f), _vals(sub, base, f)
        return bool(statistics.mean(a) < statistics.mean(b) and paired_less(a, b) >= n - 1), a, b

    def decrease(key):
        okg, a, b = lower(key, sel, g_off)
        x, y = _vals(sub, key, d_off), _vals(sub, sel, d_off)
        okd = bool(statistics.mean(x) <= statistics.mean(y) / 2 and paired_less(x, y) >= n - 1)
        return {"外したときのジニ": [statistics.mean(a), statistics.mean(b)], "ΔCE_off": [statistics.mean(x),
                                                                                    statistics.mean(y)],
                "ジニが下がる": okg, "ΔCE_offが半分以下": okd, "合格": okg and okd}

    for t in TEACH:
        k = f"{t}{lams[t]:g}"
        v[k] = {"偏りとΔCE_offの低下": decrease(k), "H7*": judge_star(absorb_summary(sub, k), ref_norm)}
        v[k]["H1〜H3"] = {kk: vv for kk, vv in judge_key(sub, k, args).items()
                         if kk in ("H1_崩壊を防ぐ", "H2_ハブの非対称性を消す", "H3_性能を損なわない", "ジニ")}
    v[sel] = {"H7*": judge_star(absorb_summary(sub, sel), ref_norm)}
    kc, ks = f"cf{lams['cf']:g}", f"shuf{lams['cf']:g}"
    v["H18a_反実仮想の信号で偏りとΔCE_offが下がる"] = v[kc]["偏りとΔCE_offの低下"]["合格"]
    v["H18a'_ソフトな教師"] = v[f"cfs{lams['cfs']:g}"]["偏りとΔCE_offの低下"]["合格"]
    v["H18b_反実仮想の信号でH7*"] = v[kc]["H7*"]["H7*"]
    okg = lower(kc, ks, g_off)[0]
    oka = lower(ks, kc, agree)[0]
    v["H18c_トークンの情報が要る_cfがshufより偏らず一致率が高い"] = bool(okg and oka)
    v["H19_蒸留で偏りとΔCE_offが下がる"] = v[f"dist{lams['dist']:g}"]["偏りとΔCE_offの低下"]["合格"]
    cps = {}
    for k in ["cp", *(f"cp_{t}{lams[t]:g}" for t in CP_TEACH)]:
        j = judge_key(sub, k, args)
        cps[k] = {kk: j[kk] for kk in ("H1_崩壊を防ぐ", "H2_ハブの非対称性を消す", "H3_性能を損なわない", "ジニ")}
        cps[k]["取り込み_H1〜H3かつジニ≦0.3"] = bool(j["H1_崩壊を防ぐ"] and j["H2_ハブの非対称性を消す"]
                                               and j["H3_性能を損なわない"] and j["ジニ"] <= 0.3)
    v["閉じる条件"] = cps
    v["H18d_閉じた後の取り込み_cp_cf"] = cps[f"cp_cf{lams['cf']:g}"]["取り込み_H1〜H3かつジニ≦0.3"]
    v["参考_閾値なしにcfだけ_ジニ"] = summ["p_cf1"]["gini"]["mean"]
    return v


def judge_order(sub: list[dict], args: argparse.Namespace, pre: str) -> dict:
    """H20: 2層で逆順の学習の終わりのジニが階層的・同時のどちらよりも小さい（平均と、同じシードで5本中4本以上）。"""
    n = len(sub)
    g = {o: [r[f"ord{pre}_{o}"]["gini"] for r in sub] for o in ORDERS}
    ce = {o: statistics.mean(r[f"ord{pre}_{o}"]["lesion"]["zero"]["base_ce"] for r in sub) for o in ORDERS}
    ok = all(statistics.mean(g["rev"]) < statistics.mean(g[o]) and paired_less(g["rev"], g[o]) >= n - 1
             for o in ("hier", "sim"))
    jk = {o: judge_key(sub, f"ord{pre}_{o}", args) for o in ORDERS}
    return {"ジニ": {o: statistics.mean(v) for o, v in g.items()}, "ジニ_シード別": g, "検証CE": ce,
            "H20_逆順で偏りが小さい": bool(ok),
            "H1〜H3": {o: {kk: j[kk] for kk in ("H1_崩壊を防ぐ", "H2_ハブの非対称性を消す", "H3_性能を損なわない")}
                      for o, j in jk.items()}}


def ref_norm_from(path: str | None) -> float | None:
    """H7* の基準: 同じシードの組での E=8・L=1・η=0.01（選択専用）の正規化 ΔCE_off の平均（12.6.153節の結果ファイル）。"""
    if not path:
        return None
    s = json.loads(Path(path).read_text())["要約"]["E8_L1"]["sel0.01"]
    return s["absorb"]["d_ce_off_norm"]["mean"]


def analyze(subs: dict[str, list[dict]], args: argparse.Namespace) -> tuple[dict, dict]:
    summ, verdict = {}, {}
    ref = ref_norm_from(args.ref)
    for n, sub in subs.items():
        e, l = map(int, n[1:].split("_L"))
        conds = conditions(e, l)
        summ[n] = {k: summarize(sub, k, conds.get(k, {}), args.train_steps) for k in sub[0]}
        if not all(k in sub[0] for k in conds):
            continue
        if l == 1:
            lams = {t.partition("@")[0]: v for t, v in (args.lam or []) if t.partition("@")[2] in ("", n)}
            for t in CP_TEACH:
                lams.setdefault(t, select_lam(summ[n], t))
            lams["shuf"] = lams["cf"]  # 対照は cf と同じ λ
            verdict[n] = judge_scale(sub, summ[n], lams, args, ref)
            verdict[n]["H7*の正規化ΔCE_offの基準"] = ref
        else:
            verdict[n] = {f"β={b:g}": judge_order(sub, args, pre) for pre, b in ORDER_BETAS.items()}
            for pre in ORDER_BETAS:
                for o in ORDERS:
                    k = f"ord{pre}_{o}"
                    br = [bias_return(r[k], conds[k], args.train_steps)["return"] for r in sub]
                    verdict[n][f"β={ORDER_BETAS[pre]:g}"].setdefault("戻り", {})[o] = statistics.mean(br)
    return summ, verdict


def parse_lam(s: str) -> tuple[str, float]:
    t, v = s.split("=")
    return t, float(v)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    p.add_argument("--scales", type=str, nargs="+", default=["16,1", "32,1", "32,2"], help="E,L の列")
    p.add_argument("--only", type=str, nargs="*", default=None, help="この条件のキーだけ測る（部分ファイル）")
    p.add_argument("--lam", type=parse_lam, nargs="*", default=None,
                   help="評価用シード: 開発用シードで選んだ λ（例 cf@E32_L1=1）。規模を省くと全規模。省略時は規則で選ぶ")
    p.add_argument("--ref", type=str, default=None, help="H7* の基準（E=8 の正規化 ΔCE_off）を読む結果ファイル")
    p.add_argument("--tau", type=float, default=TAU)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-sentences", type=int, default=3000)
    p.add_argument("--d", type=int, default=48)
    p.add_argument("--train-steps", type=int, default=600)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--n-windows", type=int, default=64)
    p.add_argument("--n-probe", type=int, default=16)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--k-num", type=int, default=1)
    p.add_argument("--k-den", type=int, default=4)
    p.add_argument("--n-random", type=int, default=30)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", type=str, default="results/moe_cf/dev.json")
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
                    extra = (f" off_gini={z['threshold_off']['gini']:.3f} off_dce={z['threshold_off']['d_ce']:+.4f}"
                             f" agree={a['agree']:.3f}") if a else ""
                    br = bias_return(z, cond, args.train_steps) if cond.get("open") is not None else None
                    extra += f" close={br['gini_close']:.3f} ret={br['return']:+.3f}" if br else ""
                    extra += f" oracle_gini={o['oracle_gini']:.3f}" if o else ""
                    extra += f" t_agree={z['teacher_agree_last']:.3f}" if "teacher_agree_last" in z else ""
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
    keep = {"sel0.3", "sel1", "cf0.1", "cf1", "dist0.1", "dist1", "shuf1", "cp", "cp_cf1", "ord0_rev", "ord0_sim",
            "ord0_hier", "ord_rev", "ord_sim", "ord_hier"}
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
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, default=str))
    print(json.dumps(verdict, ensure_ascii=False, indent=1, default=str)[:8000])


if __name__ == "__main__":
    main()
