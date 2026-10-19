"""教師の割り当ての恒常性の閾値と、臨界期を閉じる前の使われない専門家の再学習（12.6.171節の設計）の主実験。

1層・E=16（η=0.3）・E=32（η=1）の選択専用（閉じない）と臨界期型（閉じ方 (0.25, 0.25)、閉じ始めと同時に θ の積分を
止める）に、12.6.166節の部分評価の教師 rand3（選ばれた専門家＋無作為な3個、λ は cf と同じ E=16: 1、E=32: 0.1）を
足し、`moe_react.ReactCFLM` の手当てを重ねる:
(a) ht: 教師の割り当てに恒常性の閾値を掛ける。 (b) re: 閉じ終わるまで使われない専門家を再学習させる。 (a)+(b): htre。
比較: 教師なしの sel・cp、rand3・cp_rand3、参考に教師なしの再学習 cp_re。補助損失 0.05（H3 の基準）・補助損失なし・
ゲート込み η=0.1（H1〜H3 の判定の基準）。

測定・判定は `run_moe_cfe.py`（要約・閉じる条件の判定）、`run_moe_cf.py`（H7* の基準）、`run_moe_critical.py` の関数を使う。

使い方（シード×規模ごとに別プロセスで出して結合する）:
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_react --seeds 100 --scales 32,1 \\
        --out results/moe_react/parts/dev_s100_E32L1.json
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_react --merge results/moe_react/parts/dev_*.json \\
        --ref results/moe_hier/dev_rev1.json --out results/moe_react/dev.json
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
from ..moe_critical import absorption_metrics
from ..moe_hier import oracle_assignment
from ..moe_react import ReactCFLM, train_react
from ..run_env import env_info
from .run_homeostatic_moe import ms
from .run_moe_brake import bias_return, close_step
from .run_moe_cf import ref_norm_from
from .run_moe_cfe import closed_ok, summarize_cfe
from .run_moe_critical import absorb_summary, judge_star, slim
from .run_moe_hier import TH_SCHED, paired_less
from .run_moe_scale import run_one

ETA = {"E16_L1": 0.3, "E32_L1": 1.0}  # 12.6.145節で選んだ η
LAM_CF = {"E16_L1": 1.0, "E32_L1": 0.1}  # 12.6.158節で選んだ cf の λ（12.6.166節の rand3 も同じ）
TEACHER = "rand3"
TAU = 0.05
VARIANTS = {"": {}, "_ht": {"teacher_homeo": True}, "_re": {"react": True},
            "_htre": {"teacher_homeo": True, "react": True}}
LAST = 100  # 教師の統計を平均する学習の終わりのステップ数


def conditions(e: int, l: int) -> dict[str, dict]:
    name = f"E{e}_L{l}"
    eta = ETA.get(name)
    if eta is None:
        return {}
    t = {"teacher": TEACHER, "lam": LAM_CF[name]}
    sel = {"kind": "select", "eta": eta}
    c: dict[str, dict] = {"aux0": {"kind": "plain", "coef": 0.0}, "aux0.05": {"kind": "plain", "coef": 0.05},
                          "gate0.1": {"kind": "gate", "eta": 0.1}, f"sel{eta:g}": sel}
    c[TEACHER] = {**sel, **t}
    c[f"{TEACHER}_ht"] = {**sel, **t, **VARIANTS["_ht"]}
    cp = {**sel, "open": TH_SCHED[0], "ramp": TH_SCHED[1], "freeze": True}
    c["cp"] = cp
    c.update({f"cp_{TEACHER}{v}": {**cp, **t, **kw} for v, kw in VARIANTS.items()})
    c["cp_re"] = {**cp, "react": True}  # 参考: 教師なしの再学習
    return c


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def run_cond(seed: int, e: int, l: int, key: str, cond: dict, corpus: CharCorpus, args: argparse.Namespace) -> dict:
    probe = make_windows(corpus.val, args.seq_len, args.n_windows)[0][:args.n_probe]
    t_train = {}

    def train_timed(model, *a):
        t0 = time.perf_counter()
        r = train_react(model, *a)
        t_train["sec"] = time.perf_counter() - t0
        return r

    def post(model, x, y, out):
        out["oracle"] = oracle_assignment(model, x, y)
        if model.kind != "plain":
            out["absorb"] = absorption_metrics(model, x, y)
            out["series"] = model.series
        ts = model.teacher_stats
        if ts:
            out["teacher_agree_last"] = _mean(t["agree"] for t in ts[-LAST:])
            out["teacher_gini_last"] = _mean(t.get("t_gini") for t in ts[-LAST:])
            out["teacher_gini_raw_last"] = _mean(t.get("t_gini_raw") for t in ts[-LAST:])
        rs = model.react_stats
        if rs:
            ce_step = close_step(cond, args.train_steps)
            act = [r["dormant"] for r in rs if r["active"]]
            out["react"] = {"dormant_mean_active": _mean(act), "dormant_max_active": max(act) if act else None,
                            "dormant_at_close": next((r["dormant"] for r in rs if r["step"] == ce_step), None),
                            "steps_with_dormant": sum(d > 0 for d in act)}
        c = model.cost
        n_tok = max(c["tokens"], 1) if c["tokens"] else args.train_steps * 32 * args.seq_len
        out["cost"] = {"evals_per_token": c["evals"] / max(c["tokens"], 1), "teacher_sec": c["sec"],
                       "train_sec": t_train["sec"], "react_evals_per_token": c["react_evals"] / n_tok}
        out["teacher_threshold_range"] = float(model.teacher_threshold.max() - model.teacher_threshold.min())

    kw = {"total_steps": args.train_steps, "open_frac": cond.get("open"), "ramp_frac": cond.get("ramp", 0.0),
          "probe": probe, "log_every": args.log_every, "vec_every": 10 ** 9, "freeze": cond.get("freeze", False),
          "teacher": cond.get("teacher"), "lam": cond.get("lam", 0.0), "tau": args.tau,
          "shuffle_seed": seed + 1000, "sample_seed": seed + 2000,
          "teacher_homeo": cond.get("teacher_homeo", False), "react": cond.get("react", False),
          "react_rho": args.rho, "react_mu": args.mu}
    r = run_one(seed, e, l, cond, corpus, args, ReactCFLM, args.train_steps, post=post, train_fn=train_timed, **kw)
    return slim(r, False)


def summarize_react(sub: list[dict], key: str, cond: dict, total: int) -> dict:
    d = summarize_cfe(sub, key, cond, total)
    rs = [r[key] for r in sub]
    d["cost"]["react_evals_per_token"] = ms([r["cost"]["react_evals_per_token"] for r in rs])
    for m in ("teacher_gini_last", "teacher_gini_raw_last"):
        if m in rs[0]:
            d[m] = ms([r[m] for r in rs])
    if "react" in rs[0]:
        d["react"] = {m: ms([r["react"][m] if r["react"][m] is not None else float("nan") for r in rs])
                      for m in rs[0]["react"]}
    return d


def _less(a: list[float], b: list[float]) -> bool:
    """a が b より小さい（平均と、同じシードで n−1 本以上）。"""
    return bool(statistics.mean(a) < statistics.mean(b) and paired_less(a, b) >= len(a) - 1)


def judge_scale(sub: list[dict], summ: dict, conds: dict, args: argparse.Namespace, ref_norm: float | None) -> dict:
    """12.6.171節の合格条件。"""
    sel = next(k for k in sub[0] if k.startswith("sel"))
    base, ht = TEACHER, f"{TEACHER}_ht"
    v: dict = {}
    tg = lambda k: [r[k]["teacher_gini_last"] for r in sub]  # noqa: E731
    ag = lambda k: [r[k]["absorb"]["agree"] for r in sub]  # noqa: E731
    go = lambda k: [r[k]["threshold_off"]["gini"] for r in sub]  # noqa: E731
    gi = lambda k: [r[k]["gini"] for r in sub]  # noqa: E731
    m = statistics.mean
    v["P1_教師の閾値で教師の偏りが下がる"] = {"教師のジニ": [m(tg(ht)), m(tg(base))],
                                 "閾値を掛ける前の教師のジニ": [m([r[ht]["teacher_gini_raw_last"] for r in sub]),
                                                    m([r[base]["teacher_gini_raw_last"] for r in sub])],
                                 "合格": _less(tg(ht), tg(base))}
    h3 = summ[ht]["base_ce"]["mean"] <= summ["aux0.05"]["base_ce"]["mean"] + 0.01
    v["H25_教師の閾値で一致率が上がる"] = {
        "一致率": [m(ag(ht)), m(ag(base)), m(ag(sel))], "外したときのジニ": [m(go(ht)), m(go(base)), m(go(sel))],
        "rand3より一致率が高い": _less(ag(base), ag(ht)), "一致率≧0.85": bool(m(ag(ht)) >= 0.85), "H3": bool(h3)}
    v["H25_教師の閾値で一致率が上がる"]["合格"] = all(v["H25_教師の閾値で一致率が上がる"][k]
                                          for k in ("rand3より一致率が高い", "一致率≧0.85", "H3"))
    v["H25b_H7*"] = {k: judge_star(absorb_summary(sub, k), ref_norm) for k in (sel, base, ht)}
    cps = {k: closed_ok(sub, k, args) for k in ["cp", "cp_re", *(f"cp_{TEACHER}{s}" for s in VARIANTS)]}
    for k, d in cps.items():
        d["使用専門家の平均"] = m([r[k]["n_used"] for r in sub])
        d["全専門家が使われたシード数"] = sum(r[k]["all_used"] for r in sub)
    v["閉じる条件"] = cps
    b = f"cp_{TEACHER}"
    ok = lambda k: cps[k]["取り込み_H1〜H3かつジニ≦0.3"]  # noqa: E731
    v["H26_教師の閾値で閉じた後の取り込み"] = {"合格": ok(f"{b}_ht"), "副_cp_rand3より偏りが小さい": _less(gi(f"{b}_ht"), gi(b))}
    v["H27_再学習で閉じた後の取り込み"] = {
        "合格": ok(f"{b}_re"), "副_cp_rand3より偏りが小さい": _less(gi(f"{b}_re"), gi(b)),
        "副_全専門家が使われたシードが増える": cps[f"{b}_re"]["全専門家が使われたシード数"] > cps[b]["全専門家が使われたシード数"]}
    v["H28_組み合わせ"] = {"合格": bool(ok(f"{b}_htre") and m(gi(f"{b}_htre")) < m(gi(f"{b}_ht"))
                                     and m(gi(f"{b}_htre")) < m(gi(f"{b}_re"))),
                         "取り込み": ok(f"{b}_htre"), "ジニ": {k: m(gi(k)) for k in (b, f"{b}_ht", f"{b}_re", f"{b}_htre")}}
    v["H29_参考_教師なしの再学習"] = {"取り込み": ok("cp_re"), "cpより偏りが小さい": _less(gi("cp_re"), gi("cp"))}
    v["戻り"] = {k: m(bias_return(r[k], conds[k], args.train_steps)["return"] for r in sub)
               for k in cps}
    v["費用"] = {k: {mm: summ[k]["cost"][mm]["mean"] for mm in ("evals_per_token", "react_evals_per_token",
                                                             "teacher_sec", "train_sec")}
               for k in (sel, base, ht, *cps)}
    return v


def analyze(subs: dict[str, list[dict]], args: argparse.Namespace) -> tuple[dict, dict]:
    summ, verdict = {}, {}
    ref = ref_norm_from(args.ref)
    for n, sub in subs.items():
        e, l = map(int, n[1:].split("_L"))
        conds = conditions(e, l)
        summ[n] = {k: summarize_react(sub, k, conds.get(k, {}), args.train_steps) for k in sub[0]}
        if not all(k in sub[0] for k in conds):
            continue
        verdict[n] = judge_scale(sub, summ[n], conds, args, ref)
        verdict[n]["H7*の正規化ΔCE_offの基準"] = ref
    return summ, verdict


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    p.add_argument("--scales", type=str, nargs="+", default=["16,1", "32,1"], help="E,L の列")
    p.add_argument("--only", type=str, nargs="*", default=None, help="この条件のキーだけ測る（部分ファイル）")
    p.add_argument("--ref", type=str, default=None, help="H7* の基準（E=8 の正規化 ΔCE_off）を読む結果ファイル")
    p.add_argument("--tau", type=float, default=TAU)
    p.add_argument("--rho", type=float, default=0.25, help="使われない専門家の判定: 閾値なしの負荷 < ρ/E")
    p.add_argument("--mu", type=float, default=1.0, help="再学習の損失の係数")
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
    p.add_argument("--out", type=str, default="results/moe_react/dev.json")
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
                    if "teacher_gini_last" in z:
                        extra += (f" t_agree={z['teacher_agree_last']:.3f} t_gini={z['teacher_gini_last']:.3f}"
                                  f" t_gini_raw={z['teacher_gini_raw_last']:.3f}")
                    if "react" in z:
                        extra += f" dormant={z['react']['dormant_mean_active']} at_close={z['react']['dormant_at_close']}"
                    print(f"seed={seed} E{e}_L{l}/{key} gini={z['gini']:.3f} used={z['n_used']:.1f} "
                          f"base={z['lesion']['zero']['base_ce']:.4f} ratio={z['lesion']['zero']['ratio']:.2f} "
                          f"gate={z['gate_mean']:.3f}{extra} evals={c['evals_per_token']:.2f} "
                          f"react_evals={c['react_evals_per_token']:.2f} train_sec={c['train_sec']:.1f}", flush=True)
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
    keep = {"cp", f"cp_{TEACHER}", f"cp_{TEACHER}_ht", f"cp_{TEACHER}_re", f"cp_{TEACHER}_htre"}
    for sub in subs.values():  # 結果ファイルを数 MB に収めるため、シード別の時系列は閉じる条件だけ残す
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
