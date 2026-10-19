"""ルータの学習を局所の活動 × 反実仮想の教える信号の三要素則だけにする（12.6.181節の設計）の主実験。

1層・E=32（η=1）の臨界期型（閉じ方 (0.25, 0.25)、閉じ始めと同時に θ の積分を止める）に部分評価の教師 rand3
（λ=0.1）を足した cp_rand3 を土台にする（12.6.177節と同じ）。`moe_tf.ThreeFactorLM` で、学習の初めからゲートの
LM の勾配をルータに送らない（tf_rand3・ソフトな tf_srand3・全専門家の tf_cf。教師なしの対照 tf_none）。
閉じ始めから切る cl_gate（12.6.177節の d_gate）も参考に比べる。分解は `run_moe_consol.decompose`。

使い方（シードごとに別プロセスで出して結合する）:
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_tf --seeds 100 --out results/moe_tf/parts/dev_s100.json
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_tf --merge results/moe_tf/parts/dev_*.json \\
        --out results/moe_tf/dev.json
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
from ..moe_consol import cross_assignment, train_consol
from ..moe_critical import absorption_metrics
from ..moe_tf import ThreeFactorLM
from ..run_env import env_info
from .run_moe_brake import bias_return
from .run_moe_cfe import closed_ok
from .run_moe_consol import ETA, LAM_CF, _less, decompose, summarize_consol
from .run_moe_critical import slim
from .run_moe_hier import TH_SCHED, paired_less
from .run_moe_scale import run_one

BASE = "cp_rand3"
# tf_rand3e は修正1（評価のゲートをそろえる）、tf_pg・tf_rand3pg は修正2（選んだ行動の適格性 × 反実仮想の RPE）
TF = ("tf_rand3", "tf_srand3", "tf_cf", "tf_rand3e", "tf_pg", "tf_rand3pg")
DROP = 0.03  # A の低下の基準（12.6.177節の原因の基準と同じ、事前に固定）
CE_TOL = 0.005  # 学習を損なわない許容（検証 CE、事前に固定）


def conditions(e: int, l: int) -> dict[str, dict]:
    name = f"E{e}_L{l}"
    eta = ETA.get(name)
    if eta is None or l != 1:
        return {}
    cp = {"kind": "select", "eta": eta, "open": TH_SCHED[0], "ramp": TH_SCHED[1], "freeze": True}
    lam = LAM_CF[name]
    r3 = {"teacher": "rand3", "lam": lam}
    return {"aux0": {"kind": "plain", "coef": 0.0}, "aux0.05": {"kind": "plain", "coef": 0.05},
            "gate0.1": {"kind": "gate", "eta": 0.1}, "cp": cp, BASE: {**cp, **r3},
            "cl_gate": {**cp, **r3, "gate_detach": True},
            "tf_rand3": {**cp, **r3, "tf": True}, "tf_srand3": {**cp, **r3, "tf": True, "soft": True},
            "tf_cf": {**cp, "teacher": "cf", "lam": lam, "tf": True}, "tf_none": {**cp, "tf": True},
            # 修正1（12.6.181節）: 評価したすべての専門家を選ばれた専門家のゲートで評価する教師（三要素則だけ・LM の勾配あり）
            "tf_rand3e": {**cp, **r3, "tf": True, "eq_gate": True}, "cp_rand3e": {**cp, **r3, "eq_gate": True},
            # 修正2（12.6.181節）: 選んだ専門家の適格性 × 反実仮想の報酬予測誤差の三要素則（だけ・教師に足す）
            "tf_pg": {**cp, **r3, "tf": True, "pg": "only"}, "tf_rand3pg": {**cp, **r3, "tf": True, "pg": "add"}}


def run_cond(seed: int, e: int, l: int, key: str, cond: dict, corpus: CharCorpus, args: argparse.Namespace) -> dict:
    probe = make_windows(corpus.val, args.seq_len, args.n_windows)[0][:args.n_probe]
    t_train = {}

    def train_timed(model, *a):
        t0 = time.perf_counter()
        r = train_consol(model, *a)
        t_train["sec"] = time.perf_counter() - t0
        return r

    def post(model, x, y, out):
        out["train_sec"] = t_train["sec"]
        if model.kind != "plain":
            out["absorb"] = absorption_metrics(model, x, y)
            out["series"] = model.series
            out["cross"] = cross_assignment(model, x, args.rho)
            ts = model.teacher_stats
            out["teacher_agree_last"] = (statistics.mean(t["agree"] for t in ts[-50:]) if ts else None)

    kw = {"total_steps": args.train_steps, "open_frac": cond.get("open"), "ramp_frac": cond.get("ramp", 0.0),
          "probe": probe, "log_every": args.log_every, "vec_every": 10 ** 9, "freeze": cond.get("freeze", False),
          "teacher": cond.get("teacher"), "lam": cond.get("lam", 0.0), "tau": args.tau,
          "shuffle_seed": seed + 1000, "sample_seed": seed + 2000,
          "gate_detach": cond.get("gate_detach", False), "tf": cond.get("tf", False), "soft": cond.get("soft", False),
          "eq_gate": cond.get("eq_gate", False), "pg": cond.get("pg")}
    r = run_one(seed, e, l, cond, corpus, args, ThreeFactorLM, args.train_steps, post=post, train_fn=train_timed, **kw)
    return slim(r, False)


def a_d(r: dict) -> tuple[float, float]:
    """シードごとの A（取り込みの不足）と D（漂流）。"""
    c = r["cross"]
    oo = c["gini"]["open"]["open"]
    return oo - c["on_open"], c["gini"]["end"]["end"] - oo


def judge_scale(sub: list[dict], conds: dict, args: argparse.Namespace) -> dict:
    """12.6.181節の合格条件。"""
    m = statistics.mean
    n = len(sub)
    gi = lambda k: [r[k]["gini"] for r in sub]  # noqa: E731
    ce = lambda k: [r[k]["lesion"]["zero"]["base_ce"] for r in sub]  # noqa: E731
    av = lambda k: [a_d(r[k])[0] for r in sub]  # noqa: E731
    dv = lambda k: [a_d(r[k])[1] for r in sub]  # noqa: E731
    cpk = [k for k in conds if conds[k].get("open") is not None]
    v: dict = {"D0_分解": {k: decompose(sub, k) for k in cpk}}
    cps = {k: closed_ok(sub, k, args) for k in cpk}
    for k, d in cps.items():
        d["使用専門家の平均"] = m([r[k]["n_used"] for r in sub])
        d["全専門家が使われたシード数"] = sum(r[k]["all_used"] for r in sub)
        d["検証CE"] = m(ce(k))
        d["ゲートの平均"] = m([r[k]["gate_mean"] for r in sub])
        d["閾値なしの一致率"] = m([r[k]["absorb"]["agree"] for r in sub])
    v["閉じる条件"] = cps
    v["H34_閉じる前の取り込みAが下がる"] = {
        k: {"A": [m(av(k)), m(av(BASE))], "低下": m(av(BASE)) - m(av(k)), "小さいシード数": paired_less(av(k), av(BASE)),
            "合格": bool(m(av(BASE)) - m(av(k)) >= DROP and paired_less(av(k), av(BASE)) >= n - 1)}
        for k in (*TF, "cl_gate", "cp_rand3e")}
    v["H34_閉じる前の取り込みAが下がる"]["主_tf_rand3"] = v["H34_閉じる前の取り込みAが下がる"]["tf_rand3"]["合格"]
    v["H35_閉じた後の取り込み"] = {k: cps[k]["取り込み_H1〜H3かつジニ≦0.3"] for k in TF}
    v["H35_閉じた後の取り込み"]["合格"] = any(v["H35_閉じた後の取り込み"].values())
    v["H36_偏りを減らし学習を損なわない"] = {
        k: {"ジニ": [m(gi(k)), m(gi(BASE))], "偏りが小さい": _less(gi(k), gi(BASE)),
            "検証CE": [m(ce(k)), m(ce(BASE))], "CEが許容内": bool(m(ce(k)) <= m(ce(BASE)) + CE_TOL),
            "合格": bool(_less(gi(k), gi(BASE)) and m(ce(k)) <= m(ce(BASE)) + CE_TOL)}
        for k in (*TF, "cl_gate", "cp_rand3e")}
    v["H37_漂流Dが小さい"] = {k: {"D": [m(dv(k)), m(dv(BASE))], "合格": _less(dv(k), dv(BASE))}
                         for k in (*TF, "cl_gate", "cp_rand3e")}
    v["H38_教える信号の寄与_tf_rand3がtf_noneより偏らない"] = {
        "ジニ": [m(gi("tf_rand3")), m(gi("tf_none"))], "合格": _less(gi("tf_rand3"), gi("tf_none"))}
    v["修正1_評価のゲートをそろえた教師"] = {
        "H35_tf_rand3e": cps["tf_rand3e"]["取り込み_H1〜H3かつジニ≦0.3"],
        "tf_rand3eがtf_rand3より偏らない": _less(gi("tf_rand3e"), gi("tf_rand3")),
        "tf_rand3eのAがtf_rand3より小さい": _less(av("tf_rand3e"), av("tf_rand3")),
        "cp_rand3e_H35": cps["cp_rand3e"]["取り込み_H1〜H3かつジニ≦0.3"]}
    v["修正2_選んだ行動の適格性と反実仮想のRPE"] = {
        k: {"H35": cps[k]["取り込み_H1〜H3かつジニ≦0.3"], "tf_rand3より偏らない": _less(gi(k), gi("tf_rand3")),
            "Aがtf_rand3より小さい": _less(av(k), av("tf_rand3"))} for k in ("tf_pg", "tf_rand3pg")}
    v["戻り_プローブ"] = {k: m(bias_return(r[k], conds[k], args.train_steps)["return"] for r in sub) for k in cpk}
    v["教師との一致率_最後の50ステップ"] = {
        k: m([r[k]["teacher_agree_last"] for r in sub]) for k in cpk if sub[0][k]["teacher_agree_last"] is not None}
    return v


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    p.add_argument("--scales", type=str, nargs="+", default=["32,1"], help="E,L の列")
    p.add_argument("--only", type=str, nargs="*", default=None, help="この条件のキーだけ測る（部分ファイル）")
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--rho", type=float, default=0.25, help="使われない専門家: 閾値なしの負荷 < ρ/E")
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
    p.add_argument("--out", type=str, default="results/moe_tf/dev.json")
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
                    extra = ""
                    if z.get("cross"):
                        g = z["cross"]["gini"]
                        a, d = a_d(z)
                        extra = (f" on_open={z['cross']['on_open']:.3f} off_open={g['open']['open']:.3f}"
                                 f" A={a:.3f} D={d:.3f} agree_open={z['cross']['agree_open']:.3f}"
                                 f" agree_end={z['absorb']['agree']:.3f} dormant_end={z['cross']['dormant_end']}")
                    print(f"seed={seed} E{e}_L{l}/{key} gini={z['gini']:.3f} used={z['n_used']:.1f} "
                          f"base={z['lesion']['zero']['base_ce']:.4f} ratio={z['lesion']['zero']['ratio']:.2f} "
                          f"gate={z['gate_mean']:.3f}{extra} train_sec={z['train_sec']:.1f}", flush=True)
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
    summ, verdict = {}, {}
    for n, sub in subs.items():
        e, l = map(int, n[1:].split("_L"))
        conds = conditions(e, l)
        summ[n] = {k: summarize_consol(sub, k) for k in sub[0]}
        if all(k in sub[0] for k in conds):
            verdict[n] = judge_scale(sub, conds, args)
    keep = {"cp", BASE, *TF, "cp_rand3e"}
    for sub in subs.values():  # 結果ファイルを数 MB に収めるため、シード別の時系列は土台と三要素則の条件だけ残す
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
    print(json.dumps(verdict, ensure_ascii=False, indent=1, default=str)[:15000])


if __name__ == "__main__":
    main()
