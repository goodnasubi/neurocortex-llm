"""臨界期の間と閉じた後でルータの学習則を切り替える（12.6.185節の設計）の主実験。

1層・E=32（η=1）の臨界期型（閉じ方 (0.25, 0.25)）に部分評価の教師 rand3（λ=0.1）を足した cp_rand3 を土台にする
（12.6.181節と同じ）。`moe_sw.SwitchLM` で、開いている期間はゲートの LM の勾配で学び、閉じ始めから（sw_srand3・
sw_r3s）または閉じ終わりから（sw_srand3_late）教師の三要素則だけで学ぶ。12.6.182節の tf_rand3・tf_srand3・cl_gate と、
ソフトな教師にゲートの勾配を残した cp_srand3 を比べる。分解は `run_moe_consol.decompose`。

使い方（シードごとに別プロセスで出して結合する）:
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_sw --seeds 100 --out results/moe_sw/parts/dev_s100.json
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_sw --merge results/moe_sw/parts/dev_*.json \\
        --out results/moe_sw/dev.json
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
from ..moe_sw import SwitchLM
from ..run_env import env_info
from .run_moe_cfe import closed_ok
from .run_moe_consol import ETA, LAM_CF, _less, decompose, summarize_consol
from .run_moe_critical import slim
from .run_moe_hier import TH_SCHED, paired_less
from .run_moe_scale import run_one
from .run_moe_tf import a_d

BASE = "cp_rand3"
MAIN = "sw_srand3"
SW = ("sw_srand3", "sw_r3s", "sw_srand3_late")
A_TOL = 0.01  # A が土台より悪くならない許容（事前に固定）
CE_TOL = 0.005  # 学習を損なわない許容（検証 CE、12.6.181節と同じ）


def conditions(e: int, l: int) -> dict[str, dict]:
    name = f"E{e}_L{l}"
    eta = ETA.get(name)
    if eta is None or l != 1:
        return {}
    cp = {"kind": "select", "eta": eta, "open": TH_SCHED[0], "ramp": TH_SCHED[1], "freeze": True}
    r3 = {**cp, "teacher": "rand3", "lam": LAM_CF[name]}
    return {"aux0": {"kind": "plain", "coef": 0.0}, "aux0.05": {"kind": "plain", "coef": 0.05},
            "gate0.1": {"kind": "gate", "eta": 0.1}, "cp": cp, BASE: r3, "cl_gate": {**r3, "gate_detach": True},
            "tf_rand3": {**r3, "tf": True}, "tf_srand3": {**r3, "tf": True, "soft": True},
            "cp_srand3": {**r3, "soft": True},
            "sw_srand3": {**r3, "soft": True, "switch": "closing"},
            "sw_r3s": {**r3, "soft_after": True, "switch": "closing"},
            "sw_srand3_late": {**r3, "soft": True, "switch": "closed"}}


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
        if model.kind == "plain":
            return
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
          "switch": cond.get("switch"), "soft_after": cond.get("soft_after", False)}
    r = run_one(seed, e, l, cond, corpus, args, SwitchLM, args.train_steps, post=post, train_fn=train_timed, **kw)
    return slim(r, False)


def judge_scale(sub: list[dict], conds: dict, args: argparse.Namespace) -> dict:
    """12.6.185節の合格条件。"""
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

    def h39(k: str) -> dict:
        return {"A": [m(av(k)), m(av("tf_srand3")), m(av(BASE))],
                "tf_srand3より小さいシード数": paired_less(av(k), av("tf_srand3")),
                "tf_srand3より小さい": _less(av(k), av("tf_srand3")),
                "土台+0.01以下": bool(m(av(k)) <= m(av(BASE)) + A_TOL),
                "合格": bool(_less(av(k), av("tf_srand3")) and m(av(k)) <= m(av(BASE)) + A_TOL)}

    def h41(k: str) -> dict:
        return {"ジニ": [m(gi(k)), m(gi(BASE)), m(gi("tf_srand3"))],
                "土台より偏らない": _less(gi(k), gi(BASE)), "tf_srand3より偏らない": _less(gi(k), gi("tf_srand3")),
                "検証CE": [m(ce(k)), m(ce(BASE))], "CEが許容内": bool(m(ce(k)) <= m(ce(BASE)) + CE_TOL),
                "合格": bool(_less(gi(k), gi(BASE)) and _less(gi(k), gi("tf_srand3"))
                           and m(ce(k)) <= m(ce(BASE)) + CE_TOL)}

    v["H39_閉じる前の取り込みA"] = {k: h39(k) for k in SW}
    v["H39_閉じる前の取り込みA"]["主_sw_srand3"] = v["H39_閉じる前の取り込みA"][MAIN]["合格"]
    v["H40_閉じた後の取り込み"] = {k: cps[k]["取り込み_H1〜H3かつジニ≦0.3"] for k in SW}
    v["H40_閉じた後の取り込み"]["主_sw_srand3"] = v["H40_閉じた後の取り込み"][MAIN]
    v["H41_両者より偏らず学習を損なわない"] = {k: h41(k) for k in SW}
    v["H41_両者より偏らず学習を損なわない"]["主_sw_srand3"] = v["H41_両者より偏らず学習を損なわない"][MAIN]["合格"]
    v["H42_漂流Dが土台より小さい"] = {k: {"D": [m(dv(k)), m(dv(BASE))], "合格": _less(dv(k), dv(BASE))} for k in SW}
    v["H43_閉じた後の教師はソフトがハードより偏らない_sw_r3s対cl_gate"] = {
        "ジニ": [m(gi("sw_r3s")), m(gi("cl_gate"))], "合格": _less(gi("sw_r3s"), gi("cl_gate"))}
    v["参考_切り替えの時期_late対sw_srand3"] = {
        "ジニ": [m(gi("sw_srand3_late")), m(gi(MAIN))], "lateが偏らない": _less(gi("sw_srand3_late"), gi(MAIN)),
        "D": [m(dv("sw_srand3_late")), m(dv(MAIN))]}
    v["参考_cp_srand3のA"] = {"A": [m(av("cp_srand3")), m(av(BASE))], "ジニ": [m(gi("cp_srand3")), m(gi(BASE))]}
    v["教師との一致率_最後の50ステップ"] = {
        k: m([r[k]["teacher_agree_last"] for r in sub]) for k in cpk if sub[0][k]["teacher_agree_last"] is not None}
    v["n"] = n
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
    p.add_argument("--out", type=str, default="results/moe_sw/dev.json")
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
                                 f" dormant_end={z['cross']['dormant_end']}")
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
    keep = {BASE, "tf_srand3", *SW}
    for sub in subs.values():  # 結果ファイルを数 MB に収めるため、シード別の時系列は土台と比べる条件だけ残す
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
