"""閉じた後に偏りが戻る原因の切り分けと、ルータのシナプス固定化（12.6.177節の設計）の主実験。

1層・E=32（η=1）の臨界期型（閉じ方 (0.25, 0.25)、閉じ始めと同時に θ の積分を止める）に、12.6.166節の部分評価の
教師 rand3（λ=0.1）を足した cp_rand3 を土台にする（教師なしの cp も比べる）。`moe_consol.ConsolLM` で:
(1) 切り分け: 閉じ始めから学習の終わりまで、ルータ（d_router）・前段（埋め込みと GRU、d_front）・専門家（d_experts）・
    ルータと前段（d_both、12.6.154節の β=0 に当たる陽性対照）の更新を止める。ゲートからルータへの勾配を切る（d_gate）。
    教師を外す（d_teacher）。教師なしの cp にも同じ切り分け（教師に関するものを除く）を掛ける。
(2) 固定化: ルータに一様な安定化（c_pnn）・重要度に応じたメタ可塑性（c_meta）・タグ付けと捕獲（c_stc）・
    Benna & Fusi のカスケード（c_bf）を掛ける。
比較: 補助損失なし・0.05・ゲート込み η=0.1（H1〜H3 の判定の基準）。

使い方（シード×規模ごとに別プロセスで出して結合する）:
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_consol --seeds 100 --scales 32,1 \\
        --out results/moe_consol/parts/dev_s100.json
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_consol --merge results/moe_consol/parts/dev_*.json \\
        --out results/moe_consol/dev.json
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
from ..moe_consol import ConsolLM, cross_assignment, train_consol
from ..moe_critical import absorption_metrics
from ..run_env import env_info
from .run_homeostatic_moe import ms
from .run_moe_brake import bias_return
from .run_moe_cfe import closed_ok
from .run_moe_critical import slim
from .run_moe_hier import TH_SCHED, paired_less
from .run_moe_scale import run_one

ETA = {"E16_L1": 0.3, "E32_L1": 1.0}  # 12.6.145節で選んだ η
LAM_CF = {"E16_L1": 1.0, "E32_L1": 0.1}  # 12.6.158節の cf の λ（12.6.166・172節の rand3 も同じ）
TEACHER = "rand3"
DIAG = {"d_router": {"fix": ("router",)}, "d_front": {"fix": ("front",)}, "d_experts": {"fix": ("experts",)},
        "d_both": {"fix": ("router", "front")}, "d_gate": {"gate_detach": True}}
DIAG_T = {"d_teacher": {"teacher_off": True}}
CONSOL = {"c_pnn": {"consol": "pnn"}, "c_meta": {"consol": "meta"}, "c_stc": {"consol": "stc"},
          "c_bf": {"consol": "bf"}}
MODEL_FLAGS = ("fix", "gate_detach", "teacher_off", "consol")
DROP = 0.03  # 原因とみなす学習の終わりのジニの低下（事前に固定）
CE_TOL = 0.005  # 固定化が学習を損なわない許容（検証 CE、事前に固定）


def conditions(e: int, l: int) -> dict[str, dict]:
    name = f"E{e}_L{l}"
    eta = ETA.get(name)
    if eta is None or l != 1:
        return {}
    cp = {"kind": "select", "eta": eta, "open": TH_SCHED[0], "ramp": TH_SCHED[1], "freeze": True}
    t = {"teacher": TEACHER, "lam": LAM_CF[name]}
    c: dict[str, dict] = {"aux0": {"kind": "plain", "coef": 0.0}, "aux0.05": {"kind": "plain", "coef": 0.05},
                          "gate0.1": {"kind": "gate", "eta": 0.1}, "cp": cp, f"cp_{TEACHER}": {**cp, **t}}
    c.update({k: {**cp, **t, **kw} for k, kw in {**DIAG, **DIAG_T}.items()})
    c.update({f"cp_{k}": {**cp, **kw} for k, kw in DIAG.items()})
    c.update({k: {**cp, **t, **kw} for k, kw in CONSOL.items()})
    return c


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
            out["plasticity_mean"] = model.plasticity_mean()

    kw = {"total_steps": args.train_steps, "open_frac": cond.get("open"), "ramp_frac": cond.get("ramp", 0.0),
          "probe": probe, "log_every": args.log_every, "vec_every": 10 ** 9, "freeze": cond.get("freeze", False),
          "teacher": cond.get("teacher"), "lam": cond.get("lam", 0.0), "tau": args.tau,
          "shuffle_seed": seed + 1000, "sample_seed": seed + 2000,
          "fix": tuple(cond.get("fix", ())), "gate_detach": cond.get("gate_detach", False),
          "teacher_off": cond.get("teacher_off", False), "consol": cond.get("consol")}
    r = run_one(seed, e, l, cond, corpus, args, ConsolLM, args.train_steps, post=post, train_fn=train_timed, **kw)
    return slim(r, False)


def _less(a: list[float], b: list[float]) -> bool:
    """a が b より小さい（平均と、同じシードで n−1 本以上）。"""
    return bool(statistics.mean(a) < statistics.mean(b) and paired_less(a, b) >= len(a) - 1)


def decompose(sub: list[dict], key: str) -> dict:
    """D0: 臨界期に作った割り当て（開いている期間の終わりの閾値あり）から学習の終わりまでの偏りの増加を、
    取り込みの不足 A（同じ時点の閾値なし − 閾値あり）と、その後の漂流 D（終わり − 開いている期間の終わりの閾値なし）に分け、
    D をルータの重みの変化（R_end・F_open）と前段の変化（R_open・F_end）の寄与で見る。"""
    m = statistics.mean
    cr = [r[key]["cross"] for r in sub]
    g = lambda a, b: [c["gini"][a][b] for c in cr]  # noqa: E731
    on = [c["on_open"] for c in cr]
    oo, ee = g("open", "open"), g("end", "end")
    d = {"臨界期の割り当て_on_open": m(on), "開いている期間の終わりの閾値なし": m(oo), "閉じ終わり": m(g("close", "close")),
         "学習の終わり": m(ee), "A_取り込みの不足": m(oo) - m(on), "D_漂流": m(ee) - m(oo),
         "ルータだけ変えた_R_end_F_open": m(g("end", "open")) - m(oo),
         "前段だけ変えた_R_open_F_end": m(g("open", "end")) - m(oo),
         "閉じる間の漂流_close−open": m(g("close", "close")) - m(oo),
         "閉じた後の漂流_end−close": m(ee) - m(g("close", "close")),
         "開いている期間の終わりの一致率": m([c["agree_open"] for c in cr]),
         "使われない専門家_終わり": m([c["dormant_end"] for c in cr]),
         "使われない専門家_開いている期間の終わりの閾値なし": m([c["dormant_open_off"] for c in cr]),
         "ゲート": {t: m([c["gate"][t] for c in cr]) for t in ("open", "close", "end")},
         "専門家の重みの相対変化": m([c["expert_change"] for c in cr])}
    fr = [c["dormant_end_from_open"] for c in cr if c["dormant_end_from_open"] is not None]
    d["終わりに使われない専門家のうち開いている期間の終わりにすでに使われていない割合"] = m(fr) if fr else None
    d["Aが漂流以上"] = bool(d["A_取り込みの不足"] >= d["D_漂流"])
    return d


def judge_scale(sub: list[dict], conds: dict, args: argparse.Namespace) -> dict:
    """12.6.177節の合格条件。"""
    m = statistics.mean
    gi = lambda k: [r[k]["gini"] for r in sub]  # noqa: E731
    ce = lambda k: [r[k]["lesion"]["zero"]["base_ce"] for r in sub]  # noqa: E731
    base = f"cp_{TEACHER}"
    v: dict = {"D0_分解": {k: decompose(sub, k) for k in ("cp", base)}}
    cps = {k: closed_ok(sub, k, args) for k in conds if conds[k].get("open") is not None}
    for k, d in cps.items():
        d["使用専門家の平均"] = m([r[k]["n_used"] for r in sub])
        d["全専門家が使われたシード数"] = sum(r[k]["all_used"] for r in sub)
        d["検証CE"] = m(ce(k))
    v["閉じる条件"] = cps

    def cause(k: str, ref: str) -> dict:
        drop = m(gi(ref)) - m(gi(k))
        return {"ジニ": [m(gi(k)), m(gi(ref))], "低下": drop, "低いシード数": paired_less(gi(k), gi(ref)),
                "原因": bool(drop >= DROP and paired_less(gi(k), gi(ref)) >= len(sub) - 1),
                "検証CE": [m(ce(k)), m(ce(ref))]}

    v["D1〜D6_切り分け_cp_rand3"] = {k: cause(k, base) for k in (*DIAG, *DIAG_T)}
    v["切り分け_参考_cp"] = {f"cp_{k}": cause(f"cp_{k}", "cp") for k in DIAG}
    v["H30_固定化で閉じた後の取り込み"] = {k: cps[k]["取り込み_H1〜H3かつジニ≦0.3"] for k in CONSOL}
    v["H30_固定化で閉じた後の取り込み"]["合格"] = any(v["H30_固定化で閉じた後の取り込み"].values())
    v["H31_偏りを減らし学習を損なわない"] = {
        k: {"偏りが小さい": _less(gi(k), gi(base)), "CEが許容内": bool(m(ce(k)) <= m(ce(base)) + CE_TOL),
            "合格": bool(_less(gi(k), gi(base)) and m(ce(k)) <= m(ce(base)) + CE_TOL)} for k in CONSOL}
    v["H32_重要度に応じた固定化が一様な安定化より偏りを減らす"] = {
        k: _less(gi(k), gi("c_pnn")) for k in ("c_meta", "c_stc")}
    v["H32_重要度に応じた固定化が一様な安定化より偏りを減らす"]["合格"] = any(
        v["H32_重要度に応じた固定化が一様な安定化より偏りを減らす"].values())
    full = m(gi(base)) - m(gi("d_router"))
    v["H33_ルータの完全な固定の効果の半分以上"] = {
        "ルータの完全な固定による低下": full,
        **{k: {"低下": m(gi(base)) - m(gi(k)),
               "合格": bool(full > 0 and m(gi(base)) - m(gi(k)) >= 0.5 * full)} for k in CONSOL}}
    v["戻り_プローブ"] = {k: m(bias_return(r[k], conds[k], args.train_steps)["return"] for r in sub) for k in cps}
    v["可塑性の平均"] = {k: m([r[k]["plasticity_mean"] for r in sub]) if sub[0][k]["plasticity_mean"] is not None
                     else None for k in CONSOL}
    return v


def summarize_consol(sub: list[dict], key: str) -> dict:
    rs = [r[key] for r in sub]
    d = {mm: ms([r[mm] for r in rs]) for mm in ("gini", "n_used", "gate_mean", "train_sec")}
    d["base_ce"] = ms([r["lesion"]["zero"]["base_ce"] for r in rs])
    d["ratio"] = ms([r["lesion"]["zero"]["ratio"] for r in rs])
    d["all_used_seeds"] = sum(r["all_used"] for r in rs)
    if "absorb" in rs[0]:
        d["agree"] = ms([r["absorb"]["agree"] for r in rs])
    return d


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
    p.add_argument("--out", type=str, default="results/moe_consol/dev.json")
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
                    if "cross" in z and z["cross"]:
                        g = z["cross"]["gini"]
                        extra = (f" on_open={z['cross']['on_open']:.3f} off_open={g['open']['open']:.3f}"
                                 f" R_end_F_open={g['end']['open']:.3f} R_open_F_end={g['open']['end']:.3f}"
                                 f" close={g['close']['close']:.3f} end={g['end']['end']:.3f}"
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
    keep = {"cp", f"cp_{TEACHER}", "c_pnn", "c_meta", "c_stc", "c_bf"}
    for sub in subs.values():  # 結果ファイルを数 MB に収めるため、シード別の時系列は土台と固定化の条件だけ残す
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
