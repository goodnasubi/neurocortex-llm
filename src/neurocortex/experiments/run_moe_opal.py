"""専門家の選択を OpAL 型の G（Go）・N（NoGo）の2系統で表すルータ（12.6.207節の設計）の主実験。

1層・E=32（η=1）の臨界期型（閉じ方 (0.25, 0.25)）に部分評価の教師 rand3 を足した cp_rand3 を土台にする（12.6.185節と
同じ）。`moe_opal.OpALLM` で、ルータを G − N で表し、反実仮想の報酬予測誤差を第三因子とする逆向きのヘッブ則だけで
学ぶ（opal）。対照は G だけ（N を固定、opal_g）、切り替え（sw_srand3）、三要素則だけのソフトな教師（tf_srand3。評価する
専門家と乱数の消費が opal と同じ）。分解は `run_moe_consol.decompose`、A・D は `run_moe_tf.a_d`。

使い方（シードごとに別プロセスで出して結合する）:
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_opal --seeds 100 --out results/moe_opal/parts/dev_s100.json
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_opal --merge results/moe_opal/parts/dev_*.json \\
        --out results/moe_opal/dev.json
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
from ..moe_opal import OpALLM
from ..run_env import env_info
from .run_moe_cfe import closed_ok
from .run_moe_consol import ETA, LAM_CF, _less, decompose, summarize_consol
from .run_moe_critical import slim
from .run_moe_hier import TH_SCHED, paired_less
from .run_moe_scale import run_one
from .run_moe_tf import a_d

BASE = "cp_rand3"
MAIN = "opal"
OPALS = ("opal", "opal_g")
REF = ("opal_cf", "opal_cfc")  # 設計時の δ（修正前）と修正3 の δ（参考。12.6.207節の修正の記録）
SWK = "sw_srand3"
A_TOL = 0.01  # A が土台より悪くならない許容（12.6.185節と同じ、事前に固定）
ON_TOL = 0.05  # 臨界期の割り当て（閾値あり on_open）が土台より偏らない許容（A が小さいのが閾値の失敗によらないこと）
CE_TOL = 0.005  # 学習を損なわない許容（検証 CE、12.6.181節と同じ）
# 12.6.186節の補償関係（閉じた後にゲートの勾配を切った5条件 × 評価用シード 5本 = 25 本の回帰。results/moe_sw/eval.json）
LINE = (0.0856, -0.707)  # D ≈ 切片 + 傾き·A
CRITIC = {"opal_delta": "critic", "critic_lr": 1.0}  # 確定した設計（修正2）


def conditions(e: int, l: int) -> dict[str, dict]:
    name = f"E{e}_L{l}"
    eta = ETA.get(name)
    if eta is None or l != 1:
        return {}
    cp = {"kind": "select", "eta": eta, "open": TH_SCHED[0], "ramp": TH_SCHED[1], "freeze": True}
    r3 = {**cp, "teacher": "rand3", "lam": LAM_CF[name]}
    return {"aux0": {"kind": "plain", "coef": 0.0}, "aux0.05": {"kind": "plain", "coef": 0.05},
            "gate0.1": {"kind": "gate", "eta": 0.1}, BASE: r3,
            "tf_srand3": {**r3, "tf": True, "soft": True},
            SWK: {**r3, "soft": True, "switch": "closing"},
            "opal": {**r3, "tf": True, "opal": "gn", **CRITIC}, "opal_g": {**r3, "tf": True, "opal": "g", **CRITIC},
            "opal_cf": {**r3, "tf": True, "opal": "gn", "opal_delta": "cf", "critic_lr": None},
            "opal_cfc": {**r3, "tf": True, "opal": "gn", "opal_delta": "cfcritic", "critic_lr": 1.0}}


def resid(a: float, d: float) -> float:
    """12.6.186節の補償関係の直線からの D のずれ（負なら、同じ A の下で直線より漂流が小さい）。"""
    return d - (LINE[0] + LINE[1] * a)


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
        if getattr(model, "opal", None):
            os_ = model.opal_stats
            out["opal_stats"] = {k: [round(o[k], 5) for o in os_[::20]] for k in os_[0]}

    kw = {"total_steps": args.train_steps, "open_frac": cond.get("open"), "ramp_frac": cond.get("ramp", 0.0),
          "probe": probe, "log_every": args.log_every, "vec_every": 10 ** 9, "freeze": cond.get("freeze", False),
          "teacher": cond.get("teacher"), "lam": cond.get("lam", 0.0), "tau": args.tau,
          "shuffle_seed": seed + 1000, "sample_seed": seed + 2000,
          "gate_detach": cond.get("gate_detach", False), "tf": cond.get("tf", False), "soft": cond.get("soft", False),
          "switch": cond.get("switch"), "soft_after": cond.get("soft_after", False),
          "opal": cond.get("opal"), "opal_lr": args.opal_lr, "opal_init": args.opal_init, "opal_seed": seed + 3000,
          "opal_act": args.opal_act, "opal_beta": args.opal_beta,
          "opal_delta": cond.get("opal_delta", args.opal_delta),
          "critic_lr": cond.get("critic_lr", None if args.critic_lr < 0 else args.critic_lr)}
    r = run_one(seed, e, l, cond, corpus, args, OpALLM, args.train_steps, post=post, train_fn=train_timed, **kw)
    return slim(r, False)


def judge_scale(sub: list[dict], conds: dict, args: argparse.Namespace) -> dict:
    """12.6.207節の合格条件。"""
    m = statistics.mean
    n = len(sub)
    gi = lambda k: [r[k]["gini"] for r in sub]  # noqa: E731
    ce = lambda k: [r[k]["lesion"]["zero"]["base_ce"] for r in sub]  # noqa: E731
    av = lambda k: [a_d(r[k])[0] for r in sub]  # noqa: E731
    dv = lambda k: [a_d(r[k])[1] for r in sub]  # noqa: E731
    rv = lambda k: [resid(*a_d(r[k])) for r in sub]  # noqa: E731
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
    v["A_D_残差"] = {k: {"A": m(av(k)), "D": m(dv(k)), "残差": m(rv(k)), "ジニ": m(gi(k))} for k in cpk}

    on = lambda k: [r[k]["cross"]["on_open"] for r in sub]  # noqa: E731

    def h44(k: str) -> dict:
        return {"A": [m(av(k)), m(av(BASE))], "on_open": [m(on(k)), m(on(BASE))],
                "合格": bool(m(av(k)) <= m(av(BASE)) + A_TOL and m(on(k)) <= m(on(BASE)) + ON_TOL)}

    def h45(k: str) -> dict:
        return {"D": [m(dv(k)), m(dv(BASE))], "土台より小さいシード数": paired_less(dv(k), dv(BASE)),
                "合格": bool(_less(dv(k), dv(BASE)) and m(dv(k)) <= 0.0)}

    def h46(k: str) -> dict:
        return {"残差": [m(rv(k)), m(rv(SWK))], "sw_srand3より小さいシード数": paired_less(rv(k), rv(SWK)),
                "合格": bool(m(rv(k)) < 0.0 and _less(rv(k), rv(SWK)))}

    def h48(k: str) -> dict:
        return {"ジニ": [m(gi(k)), m(gi(BASE)), m(gi(SWK))],
                "土台より偏らない": _less(gi(k), gi(BASE)), "sw_srand3より偏らない": _less(gi(k), gi(SWK)),
                "検証CE": [m(ce(k)), m(ce(BASE))], "CEが許容内": bool(m(ce(k)) <= m(ce(BASE)) + CE_TOL),
                "合格": bool(_less(gi(k), gi(BASE)) and _less(gi(k), gi(SWK)) and m(ce(k)) <= m(ce(BASE)) + CE_TOL)}

    meths = (*OPALS, *REF, SWK, "tf_srand3")
    v["H44_閉じる前の取り込みA"] = {k: h44(k) for k in meths}
    v["H45_閉じた後の漂流D"] = {k: h45(k) for k in meths}
    both = {k: bool(v["H44_閉じる前の取り込みA"][k]["合格"] and v["H45_閉じた後の漂流D"][k]["合格"]) for k in meths}
    v["主_AとDを同時に満たす_H44かつH45"] = both
    v["H46_補償関係の直線より下"] = {k: h46(k) for k in (*OPALS, *REF, "tf_srand3")}
    v["H47_対立（N）が要る"] = {
        "opalは同時に満たしopal_gは満たさない": bool(both["opal"] and not both["opal_g"]),
        "ジニ_opal対opal_g": [m(gi("opal")), m(gi("opal_g"))], "opalがopal_gより偏らない": _less(gi("opal"), gi("opal_g")),
        "合格": bool(both["opal"] and not both["opal_g"])}
    v["H48_偏りを減らし学習を損なわない"] = {k: h48(k) for k in (*OPALS, *REF)}
    v["H49_閉じた後の取り込み_H1〜H3かつジニ≦0.3"] = {k: cps[k]["取り込み_H1〜H3かつジニ≦0.3"] for k in (*OPALS, *REF)}
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
    p.add_argument("--opal-lr", type=float, default=0.1, help="OpAL のヘッブ則の学習率 α")
    p.add_argument("--opal-init", type=float, default=None,
                   help="G・N の重みの初期値の標準偏差（既定は初期の対数の散らばりを線形ルータにそろえる値）")
    p.add_argument("--opal-act", type=str, default="sigmoid", help="G・N の活動（sigmoid・softplus）")
    p.add_argument("--opal-beta", type=float, default=4.0, help="選択の強さ β·(G − N) の β")
    p.add_argument("--opal-delta", type=str, default="cfcritic",
                   help="δ の作り方（cf: 評価した結果の平均との差＝設計時、critic: 選んだ専門家とクリティック＝修正1・2、"
                        "cfcritic: 評価した全専門家とクリティック＝修正3）")
    p.add_argument("--critic-lr", type=float, default=1.0,
                   help="クリティックの正規化 LMS のステップ（修正2。負なら行為者と同じ α＝修正1）")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", type=str, default="results/moe_opal/dev.json")
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
    keep = {BASE, SWK, *OPALS, *REF}
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
