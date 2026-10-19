"""反実仮想の結果を一部の専門家だけで推定する教師と、価値を予測するルータ（12.6.165節の設計）の主実験。

1層・E=16（η=0.3）・E=32（η=1）の選択専用（閉じない）と臨界期型に、`moe_cfe.EstimatedCFLM` の教師を足す。
(a) 一部の専門家だけを評価する: top1・top3（選ばれた専門家＋閾値なしのルータの上位 k 個）、rand1・rand3（選ばれた専門家＋
    無作為な m 個）。λ は 12.6.158節の cf と同じ（E=16: 1、E=32: 0.1）。
(b) 価値の予測: val0（選ばれた専門家の実際の結果だけ）・val1（＋無作為な1個）・valall（全専門家）、λ = 0.1・1。
(c) 余力: cfg（全専門家をゲート 1 で評価した CE の argmin、λ は cf と同じ）。
比較: 教師なしの sel、全専門家の cf（12.6.158節）、補助損失 0.05。閉じる条件: cp と cp_{cf,top1,rand1,cfg,val0,val1}。
費用として、教師が評価した専門家の数（トークンあたり）・教師の計算時間・学習全体の時間を測る。

測定・判定は `run_moe_cf.py`（要約・λ の選択・H7* の基準）、`run_moe_critical.py`・`run_moe_scale.py` の関数を使う。

使い方（シード×規模ごとに別プロセスで出して結合する）:
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_cfe --seeds 100 --scales 32,1 \\
        --out results/moe_cfe/parts/dev_s100_E32L1.json
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_cfe --merge results/moe_cfe/parts/dev_*.json \\
        --ref results/moe_hier/dev_rev1.json --out results/moe_cfe/dev.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from ..data import CharCorpus, synthetic_corpus
from ..hub_load import make_windows
from ..moe_cfe import EstimatedCFLM
from ..moe_cf import train_cf
from ..moe_critical import absorption_metrics
from ..moe_hier import oracle_assignment
from ..run_env import env_info
from .run_homeostatic_moe import ms
from .run_moe_brake import bias_return
from .run_moe_cf import ref_norm_from, summarize
from .run_moe_critical import absorb_summary, judge_key, judge_star, slim
from .run_moe_hier import TH_SCHED, paired_less
from .run_moe_scale import run_one

ETA = {"E16_L1": 0.3, "E32_L1": 1.0}  # 12.6.145節で選んだ η
LAM_CF = {"E16_L1": 1.0, "E32_L1": 0.1}  # 12.6.158節で選んだ cf の λ
LAMS = (0.1, 1.0)
PARTIAL = ("top1", "top3", "rand1", "rand3")
VALUES = ("val0", "val1", "valall", "dval1", "dval3", "dvalall")  # dval* は修正1（実際の結果との差の予測）
VALUES_T1 = ("dval1t1", "dvalallt1")  # 修正2: 差の予測を τ=1 で（キーの名前。教師は dval1・dvalall）
CP_FIXED = ("cf", "top1", "rand1", "cfg")
CP_VAL = ("val0", "val1", "dval1")
TAU = 0.05


def conditions(e: int, l: int) -> dict[str, dict]:
    name = f"E{e}_L{l}"
    eta = ETA.get(name)
    if eta is None:
        return {}
    lam = LAM_CF[name]
    sel = {"kind": "select", "eta": eta}
    c: dict[str, dict] = {"aux0": {"kind": "plain", "coef": 0.0}, "aux0.05": {"kind": "plain", "coef": 0.05},
                          "gate0.1": {"kind": "gate", "eta": 0.1}, f"sel{eta:g}": sel}  # H1〜H3 の判定の基準
    c.update({t: {**sel, "teacher": t, "lam": lam} for t in ("cf", *PARTIAL, "cfg")})
    c.update({f"{t}_{lv:g}": {**sel, "teacher": t, "lam": lv} for t in VALUES for lv in LAMS})
    # 修正2: 差の予測を CE の自然な単位（τ=1）で
    c.update({f"{t}t1_{lv:g}": {**sel, "teacher": t, "lam": lv, "tau": 1.0} for t in ("dval1", "dvalall") for lv in LAMS})
    cp = {**sel, "open": TH_SCHED[0], "ramp": TH_SCHED[1], "freeze": True}
    c["cp"] = cp
    c.update({f"cp_{t}": {**cp, "teacher": t, "lam": lam} for t in CP_FIXED})
    c.update({f"cp_{t}_{lv:g}": {**cp, "teacher": t, "lam": lv} for t in CP_VAL for lv in LAMS})
    return c


def run_cond(seed: int, e: int, l: int, key: str, cond: dict, corpus: CharCorpus, args: argparse.Namespace) -> dict:
    probe = make_windows(corpus.val, args.seq_len, args.n_windows)[0][:args.n_probe]
    t_train = {}

    def train_timed(model, *a):
        t0 = time.perf_counter()
        r = train_cf(model, *a)
        t_train["sec"] = time.perf_counter() - t0
        return r

    def post(model, x, y, out):
        out["oracle"] = oracle_assignment(model, x, y)
        if model.kind != "plain":
            out["absorb"] = absorption_metrics(model, x, y)
            out["series"] = model.series
        ts = model.teacher_stats
        if ts:
            out["teacher_agree_last"] = statistics.mean(t["agree"] for t in ts[-args.log_every * 5:])
        c = model.cost
        out["cost"] = {"evals_per_token": c["evals"] / max(c["tokens"], 1), "teacher_sec": c["sec"],
                       "train_sec": t_train["sec"]}

    kw = {"total_steps": args.train_steps, "open_frac": cond.get("open"), "ramp_frac": cond.get("ramp", 0.0),
          "probe": probe, "log_every": args.log_every, "vec_every": 10 ** 9, "freeze": cond.get("freeze", False),
          "teacher": cond.get("teacher"), "lam": cond.get("lam", 0.0), "tau": cond.get("tau", args.tau),
          "shuffle_seed": seed + 1000, "sample_seed": seed + 2000}
    r = run_one(seed, e, l, cond, corpus, args, EstimatedCFLM, args.train_steps, post=post, train_fn=train_timed, **kw)
    return slim(r, False)


def summarize_cfe(sub: list[dict], key: str, cond: dict, total: int) -> dict:
    d = summarize(sub, key, cond, total)
    rs = [r[key] for r in sub]
    d["cost"] = {k: ms([r["cost"][k] for r in rs]) for k in ("evals_per_token", "teacher_sec", "train_sec")}
    return d


def select_val_lam(summ: dict, t: str) -> float:
    """12.6.157節と同じ規則: H3（検証 CE ≦ 補助損失 0.05 + 0.01）を満たす λ のうち外したときのジニの平均が最小。
    満たすものがなければ検証 CE が最小のもの。"""
    ref = summ["aux0.05"]["base_ce"]["mean"] + 0.01
    keys = {lv: f"{t}_{lv:g}" for lv in LAMS}
    ok = [lv for lv, k in keys.items() if summ[k]["base_ce"]["mean"] <= ref]
    if ok:
        return min(ok, key=lambda lv: (summ[keys[lv]]["gini_off"]["mean"], lv))
    return min(LAMS, key=lambda lv: summ[keys[lv]]["base_ce"]["mean"])


def judge_retain(sub: list[dict], summ: dict, key: str, sel: str, args: argparse.Namespace) -> dict:
    """H21・H22 の基準（12.6.165節）: (i) 外したときのジニが sel より小さい、(ii) 選択の一致率が sel より大きい
    （それぞれ平均と、同じシードで5本中4本以上）、(iii) cf の改善の半分以上を保つ（両指標の平均で）、(iv) H3。"""
    n = len(sub)
    g = lambda k: [r[k]["threshold_off"]["gini"] for r in sub]  # noqa: E731
    a = lambda k: [r[k]["absorb"]["agree"] for r in sub]  # noqa: E731
    m = statistics.mean
    gx, gs, gc = g(key), g(sel), g("cf")
    ax, as_, ac = a(key), a(sel), a("cf")
    ok_g = bool(m(gx) < m(gs) and paired_less(gx, gs) >= n - 1)
    ok_a = bool(m(ax) > m(as_) and paired_less(as_, ax) >= n - 1)
    rg = (m(gs) - m(gx)) / (m(gs) - m(gc)) if m(gs) != m(gc) else float("nan")
    ra = (m(ax) - m(as_)) / (m(ac) - m(as_)) if m(ac) != m(as_) else float("nan")
    ok_r = bool(rg >= 0.5 and ra >= 0.5)
    h3 = bool(summ[key]["base_ce"]["mean"] <= summ["aux0.05"]["base_ce"]["mean"] + 0.01)
    return {"外したときのジニ": [m(gx), m(gs), m(gc)], "一致率": [m(ax), m(as_), m(ac)],
            "保持率_ジニ": rg, "保持率_一致率": ra, "(i)ジニが下がる": ok_g, "(ii)一致率が上がる": ok_a,
            "(iii)cfの改善の半分以上": ok_r, "(iv)H3": h3, "合格": ok_g and ok_a and ok_r and h3,
            "評価数_トークンあたり": summ[key]["cost"]["evals_per_token"]["mean"]}


def closed_ok(sub: list[dict], key: str, args: argparse.Namespace) -> dict:
    j = judge_key(sub, key, args)
    d = {kk: j[kk] for kk in ("H1_崩壊を防ぐ", "H2_ハブの非対称性を消す", "H3_性能を損なわない", "ジニ")}
    d["取り込み_H1〜H3かつジニ≦0.3"] = bool(j["H1_崩壊を防ぐ"] and j["H2_ハブの非対称性を消す"]
                                         and j["H3_性能を損なわない"] and j["ジニ"] <= 0.3)
    return d


def judge_scale(sub: list[dict], summ: dict, lams: dict[str, float], args: argparse.Namespace,
                ref_norm: float | None) -> dict:
    n = len(sub)
    sel = next(k for k in sub[0] if k.startswith("sel"))
    keys = {t: t for t in (*PARTIAL, "cfg")} | {t: f"{t}_{lams[t]:g}" for t in (*VALUES, *VALUES_T1)}
    v: dict = {"選んだλ_価値の予測": lams, "条件のキー": keys}
    for t, k in keys.items():
        v[k] = {"保持": judge_retain(sub, summ, k, sel, args), "H7*": judge_star(absorb_summary(sub, k), ref_norm)}
    for k in ("cf", sel):
        v[k] = {"H7*": judge_star(absorb_summary(sub, k), ref_norm)}
    v["H21_一部の評価で取り込みが保たれる"] = {t: v[t]["保持"]["合格"] for t in PARTIAL}
    v["H21b_一部の評価でH7*"] = {t: v[t]["H7*"]["H7*"] for t in PARTIAL}
    v["H22_価値の予測で取り込みが保たれる_val1"] = v[keys["val1"]]["保持"]["合格"]
    v["H22_修正1_差の予測_dval1"] = v[keys["dval1"]]["保持"]["合格"]
    v["参考_dval3"] = v[keys["dval3"]]["保持"]["合格"]
    v["参考_dvalall"] = v[keys["dvalall"]]["保持"]["合格"]
    v["H22_修正2_差の予測τ1_dval1t1"] = v[keys["dval1t1"]]["保持"]["合格"]
    v["参考_dvalallt1"] = v[keys["dvalallt1"]]["保持"]["合格"]
    v["H22'_実際の結果だけ_val0"] = v[keys["val0"]]["保持"]["合格"]
    v["参考_valall"] = v[keys["valall"]]["保持"]["合格"]
    v["H22b_価値の予測でH7*"] = {t: v[keys[t]]["H7*"]["H7*"] for t in VALUES}
    cps = {k: closed_ok(sub, k, args) for k in ["cp", *(f"cp_{t}" for t in CP_FIXED),
                                                 *(f"cp_{t}_{lams[t]:g}" for t in CP_VAL)]}
    gcp = [r["cp"]["gini"] for r in sub]
    for k, d in cps.items():
        gk = [r[k]["gini"] for r in sub]
        d["cpより偏りが小さい"] = bool(statistics.mean(gk) < statistics.mean(gcp) and paired_less(gk, gcp) >= n - 1)
    v["閉じる条件"] = cps
    v["H23_閉じた後の取り込み"] = {k: cps[k]["取り込み_H1〜H3かつジニ≦0.3"] for k in cps if k != "cp"}
    agc, agf = [r["cfg"]["absorb"]["agree"] for r in sub], [r["cf"]["absorb"]["agree"] for r in sub]
    v["H24_ゲートに依らない教師"] = {
        "一致率": [statistics.mean(agc), statistics.mean(agf)],
        "一致率≧0.85": bool(statistics.mean(agc) >= 0.85),
        "cfより一致率が高い": bool(statistics.mean(agc) > statistics.mean(agf) and paired_less(agf, agc) >= n - 1),
        "H7*": v["cfg"]["H7*"]["H7*"]}
    v["H24_ゲートに依らない教師"]["合格"] = all(v["H24_ゲートに依らない教師"][k]
                                        for k in ("一致率≧0.85", "cfより一致率が高い", "H7*"))
    v["費用"] = {k: {m: summ[k]["cost"][m]["mean"] for m in ("evals_per_token", "teacher_sec", "train_sec")}
               for k in (sel, "cf", *keys.values())}
    return v


def analyze(subs: dict[str, list[dict]], args: argparse.Namespace) -> tuple[dict, dict]:
    summ, verdict = {}, {}
    ref = ref_norm_from(args.ref)
    for n, sub in subs.items():
        e, l = map(int, n[1:].split("_L"))
        conds = conditions(e, l)
        summ[n] = {k: summarize_cfe(sub, k, conds.get(k, {}), args.train_steps) for k in sub[0]}
        if not all(k in sub[0] for k in conds):
            continue
        lams = {t.partition("@")[0]: lv for t, lv in (args.lam or []) if t.partition("@")[2] in ("", n)}
        for t in (*VALUES, *VALUES_T1):
            lams.setdefault(t, select_val_lam(summ[n], t))
        verdict[n] = judge_scale(sub, summ[n], lams, args, ref)
        verdict[n]["H7*の正規化ΔCE_offの基準"] = ref
        verdict[n]["戻り"] = {k: statistics.mean(bias_return(r[k], conds[k], args.train_steps)["return"] for r in sub)
                            for k in conds if k.startswith("cp")}
    return summ, verdict


def parse_lam(s: str) -> tuple[str, float]:
    t, v = s.split("=")
    return t, float(v)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    p.add_argument("--scales", type=str, nargs="+", default=["16,1", "32,1"], help="E,L の列")
    p.add_argument("--only", type=str, nargs="*", default=None, help="この条件のキーだけ測る（部分ファイル）")
    p.add_argument("--lam", type=parse_lam, nargs="*", default=None,
                   help="評価用シード: 開発用シードで選んだ価値の予測の λ（例 val1@E32_L1=1）。省略時は規則で選ぶ")
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
    p.add_argument("--out", type=str, default="results/moe_cfe/dev.json")
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
                    a, c = z.get("absorb", {}), z["cost"]
                    extra = (f" off_gini={z['threshold_off']['gini']:.3f} off_dce={z['threshold_off']['d_ce']:+.4f}"
                             f" agree={a['agree']:.3f}") if a else ""
                    br = bias_return(z, cond, args.train_steps) if cond.get("open") is not None else None
                    extra += f" close={br['gini_close']:.3f} ret={br['return']:+.3f}" if br else ""
                    extra += f" t_agree={z['teacher_agree_last']:.3f}" if "teacher_agree_last" in z else ""
                    print(f"seed={seed} E{e}_L{l}/{key} gini={z['gini']:.3f} used={z['n_used']:.1f} "
                          f"base={z['lesion']['zero']['base_ce']:.4f} ratio={z['lesion']['zero']['ratio']:.2f} "
                          f"gate={z['gate_mean']:.3f}{extra} evals={c['evals_per_token']:.2f} "
                          f"t_sec={c['teacher_sec']:.1f} train_sec={c['train_sec']:.1f}", flush=True)
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
    keep = {"sel0.3", "sel1", "cf", "top1", "rand1", "val1_0.1", "val1_1", "cfg", "cp", "cp_cf", "cp_rand1"}
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
    print(json.dumps(verdict, ensure_ascii=False, indent=1, default=str)[:12000])


if __name__ == "__main__":
    main()
