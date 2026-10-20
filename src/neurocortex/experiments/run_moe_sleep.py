"""臨界期を閉じる直前の睡眠に似た再生による、閾値の割り当てのルータのシナプスへの転写（12.6.221節の設計）の主実験。

土台は 12.6.177節と同じ 1層・E=32（η=1）の臨界期型に部分評価の教師 rand3（λ=0.1）を足した cp_rand3（閉じ方
(0.25, 0.25)、閉じ始めと同時に θ の積分を止める）。睡眠は閉じ始めに一度だけ入れ、睡眠の後は閾値を一度に外す
（閉じ方 (0.25, 0)。睡眠の間に臨界期が閉じる）。`moe_sleep.SleepLM` で:
- abrupt: 睡眠なしで一度に閉じる（睡眠の条件と同じ閉じ方・同じデータの順の対照）。
- sleep_w: 睡眠でルータの重み W だけを学ぶ（主）。sleep_wb: W と偏り b。sleep_perm: θ を並べ替えた目標（W だけ）。
- fold: 閉じる間に下がった分の θ をルータの偏りに移す（閉じ方 (0.25, 0.25)。内在的な興奮性の恒久化、参考）。
- 教師なしの cp に abrupt・sleep_w（参考）。
比較: 補助損失なし・0.05・ゲート込み η=0.1（H1〜H3 の判定の基準）。

使い方（シードごとに別プロセスで出して結合する）:
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_sleep --seeds 100 --out results/moe_sleep/parts/dev_s100.json
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_sleep --merge results/moe_sleep/parts/dev_*.json \\
        --out results/moe_sleep/dev.json
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
from ..moe_sleep import SleepLM, slept_assignment
from ..run_env import env_info
from .run_homeostatic_moe import ms
from .run_moe_cfe import closed_ok
from .run_moe_critical import slim
from .run_moe_hier import TH_SCHED, paired_less
from .run_moe_scale import run_one

ETA = 1.0  # 12.6.145節で選んだ E=32 の η
LAM_CF = 0.1  # 12.6.158節の cf の λ（12.6.166・172・177節の rand3 も同じ）
TEACHER = "rand3"
BASE = f"cp_{TEACHER}"
DROP = 0.03  # 意味のある偏りの低下（12.6.177節と同じ、事前に固定）
CE_TOL = 0.005  # 学習を損なわない許容（検証 CE、12.6.177節と同じ）


def conditions() -> dict[str, dict]:
    cp = {"kind": "select", "eta": ETA, "open": TH_SCHED[0], "ramp": TH_SCHED[1], "freeze": True}
    ab = {**cp, "ramp": 0.0}
    t = {"teacher": TEACHER, "lam": LAM_CF}
    return {"aux0": {"kind": "plain", "coef": 0.0}, "aux0.05": {"kind": "plain", "coef": 0.05},
            "gate0.1": {"kind": "gate", "eta": 0.1}, "cp": cp, BASE: {**cp, **t},
            "abrupt": {**ab, **t}, "sleep_w": {**ab, **t, "sleep": "w"}, "sleep_wb": {**ab, **t, "sleep": "wb"},
            "sleep_perm": {**ab, **t, "sleep": "perm"}, "fold": {**cp, **t, "fold": True},
            "cp_abrupt": ab, "cp_sleep_w": {**ab, "sleep": "w"}}


def run_cond(seed: int, key: str, cond: dict, corpus: CharCorpus, args: argparse.Namespace) -> dict:
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
            out["slept"] = slept_assignment(model, x)
            out["sleep_stats"] = model.sleep_stats

    kw = {"total_steps": args.train_steps, "open_frac": cond.get("open"), "ramp_frac": cond.get("ramp", 0.0),
          "probe": probe, "log_every": args.log_every, "vec_every": 10 ** 9, "freeze": cond.get("freeze", False),
          "teacher": cond.get("teacher"), "lam": cond.get("lam", 0.0), "tau": args.tau,
          "shuffle_seed": seed + 1000, "sample_seed": seed + 2000,
          "sleep": cond.get("sleep"), "sleep_buffer": args.sleep_buffer, "sleep_steps": args.sleep_steps,
          "sleep_lr": args.sleep_lr, "sleep_seed": seed + 3000, "fold": cond.get("fold", False)}
    r = run_one(seed, 32, 1, cond, corpus, args, SleepLM, args.train_steps, post=post, train_fn=train_timed, **kw)
    return slim(r, False)


def _less(a: list[float], b: list[float]) -> bool:
    """a が b より小さい（平均と、同じシードで n−1 本以上）。"""
    return bool(statistics.mean(a) < statistics.mean(b) and paired_less(a, b) >= len(a) - 1)


def decompose(sub: list[dict], key: str) -> dict:
    """12.6.177節の D0 の分解（取り込みの不足 A と漂流 D）。一度に閉じる条件には閉じ終わりの点がないので省く。
    睡眠の条件では、睡眠の直後の閾値なしの割り当て（R_slept, F_open）と、それに残る取り込みの不足 A_slept を足す。"""
    m = statistics.mean
    cr = [r[key]["cross"] for r in sub]
    g = lambda a, b: [c["gini"][a][b] for c in cr]  # noqa: E731
    on, oo, ee = [c["on_open"] for c in cr], g("open", "open"), g("end", "end")
    d = {"臨界期の割り当て_on_open": m(on), "開いている期間の終わりの閾値なし": m(oo), "学習の終わり": m(ee),
         "A_取り込みの不足": m(oo) - m(on), "D_漂流": m(ee) - m(oo),
         "使われない専門家_終わり": m([c["dormant_end"] for c in cr])}
    if "close" in cr[0]["gini"]:
        d["閉じ終わり"] = m(g("close", "close"))
    sl = [r[key].get("slept") or {} for r in sub]
    if all(sl):
        gs = [s["gini_slept_off"] for s in sl]
        d.update({"睡眠の直後の閾値なし": m(gs), "A_slept_睡眠の後に残る取り込みの不足": m(gs) - m(on),
                  "A_slept_シード別": [a - b for a, b in zip(gs, on)],
                  "睡眠の直後の閾値ありとの一致率": m([s["agree_with_on_open"] for s in sl]),
                  "睡眠から終わりまでの漂流": m(ee) - m(gs)})
        st = [r[key]["sleep_stats"] for r in sub]
        d["睡眠の統計"] = {k: m([s[k] for s in st]) for k in st[0]}
    return d


def judge(sub: list[dict], args: argparse.Namespace) -> dict:
    """12.6.221節の合格条件。"""
    m = statistics.mean
    conds = conditions()
    gi = lambda k: [r[k]["gini"] for r in sub]  # noqa: E731
    ce = lambda k: [r[k]["lesion"]["zero"]["base_ce"] for r in sub]  # noqa: E731
    v: dict = {"分解": {k: decompose(sub, k) for k in conds if conds[k].get("open") is not None}}
    cps = {k: closed_ok(sub, k, args) for k in conds if conds[k].get("open") is not None}
    for k, d in cps.items():
        d["使用専門家の平均"] = m([r[k]["n_used"] for r in sub])
        d["全専門家が使われたシード数"] = sum(r[k]["all_used"] for r in sub)
        d["検証CE"] = m(ce(k))
    v["閉じる条件"] = cps
    a_base = v["分解"][BASE]["A_取り込みの不足"]
    a_sl = v["分解"]["sleep_w"]["A_slept_睡眠の後に残る取り込みの不足"]
    lower = {}
    for k, ref in (("sleep_w", BASE), ("sleep_w", "abrupt"), ("sleep_w", "sleep_perm"),
                   ("cp_sleep_w", "cp"), ("cp_sleep_w", "cp_abrupt"), ("sleep_wb", BASE), ("fold", BASE)):
        drop = m(gi(ref)) - m(gi(k))
        lower[f"{k}_対_{ref}"] = {"ジニ": [m(gi(k)), m(gi(ref))], "低下": drop, "低いシード数": paired_less(gi(k), gi(ref)),
                                 "合格": bool(drop >= DROP and paired_less(gi(k), gi(ref)) >= len(sub) - 1)}
    v["偏りの低下"] = lower
    v["H50_主_睡眠で閉じた後の取り込み"] = {"取り込み_H1〜H3かつジニ≦0.3": cps["sleep_w"]["取り込み_H1〜H3かつジニ≦0.3"],
                                   "cp_rand3より0.03以上低い": lower[f"sleep_w_対_{BASE}"]["合格"]}
    v["H50_主_睡眠で閉じた後の取り込み"]["合格"] = all(v["H50_主_睡眠で閉じた後の取り込み"].values())
    v["H51_睡眠で取り込みの不足が減る"] = {
        "A_cp_rand3": a_base, "A_slept": a_sl, "半分以下": bool(a_sl <= 0.5 * a_base),
        "一致率": v["分解"]["sleep_w"]["睡眠の直後の閾値ありとの一致率"],
        "一致率≧0.9": bool(v["分解"]["sleep_w"]["睡眠の直後の閾値ありとの一致率"] >= 0.9)}
    v["H51_睡眠で取り込みの不足が減る"]["合格"] = bool(v["H51_睡眠で取り込みの不足が減る"]["半分以下"]
                                         and v["H51_睡眠で取り込みの不足が減る"]["一致率≧0.9"])
    v["H52_閾値の内容による"] = {"対_abrupt": lower["sleep_w_対_abrupt"]["合格"],
                           "対_sleep_perm": lower["sleep_w_対_sleep_perm"]["合格"]}
    v["H52_閾値の内容による"]["合格"] = all(v["H52_閾値の内容による"].values())
    v["H53_学習を損なわない"] = {"検証CE": [m(ce("sleep_w")), m(ce(BASE))],
                          "合格": bool(m(ce("sleep_w")) <= m(ce(BASE)) + CE_TOL)}
    v["参考_シナプスと内在的な興奮性"] = {k: {"ジニ": m(gi(k)), "取り込み": cps[k]["取り込み_H1〜H3かつジニ≦0.3"],
                                   "検証CE": m(ce(k))} for k in ("sleep_w", "sleep_wb", "fold")}
    v["参考_教師なし"] = {"cp_sleep_w_対_cp": lower["cp_sleep_w_対_cp"], "cp_sleep_w_対_cp_abrupt":
                     lower["cp_sleep_w_対_cp_abrupt"]}
    return v


def summarize(sub: list[dict], key: str) -> dict:
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
    p.add_argument("--sleep-buffer", type=int, default=16, help="再生する覚醒中の学習の入力の数（バッチ）")
    p.add_argument("--sleep-steps", type=int, default=200)
    p.add_argument("--sleep-lr", type=float, default=3e-3)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", type=str, default="results/moe_sleep/dev.json")
    p.add_argument("--merge", type=str, nargs="+", default=None)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    if not args.merge:
        parts = []
        for seed in args.seeds:
            corpus = CharCorpus(synthetic_corpus(n_sentences=args.n_sentences, seed=seed))
            runs = {}
            for key, cond in conditions().items():
                if args.only and key not in args.only:
                    continue
                z = runs[key] = run_cond(seed, key, cond, corpus, args)
                extra = ""
                if z.get("cross"):
                    g = z["cross"]["gini"]
                    extra = (f" on_open={z['cross']['on_open']:.3f} off_open={g['open']['open']:.3f}"
                             f" end={g['end']['end']:.3f} dormant_end={z['cross']['dormant_end']}")
                if z.get("slept"):
                    extra += (f" slept_off={z['slept']['gini_slept_off']:.3f}"
                              f" slept_agree={z['slept']['agree_with_on_open']:.3f}")
                print(f"seed={seed} {key} gini={z['gini']:.3f} used={z['n_used']:.1f} "
                      f"base={z['lesion']['zero']['base_ce']:.4f} ratio={z['lesion']['zero']['ratio']:.2f} "
                      f"gate={z['gate_mean']:.3f}{extra} train_sec={z['train_sec']:.1f}", flush=True)
            parts.append({"環境": env_info(), "seed": seed, "runs": runs})
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"環境": env_info(), "設定": vars(args), "部分": parts}, ensure_ascii=False))
        return
    parts = [pt for f in args.merge for pt in json.loads(Path(f).read_text())["部分"]]
    merged: dict = {}
    for pt in parts:
        mm = merged.setdefault(pt["seed"], {"seed": pt["seed"], "runs": {}, "環境": []})
        mm["runs"].update(pt["runs"])
        mm["環境"].append(pt["環境"])
    parts = sorted(merged.values(), key=lambda pt: pt["seed"])
    sub = [pt["runs"] for pt in parts]
    summ = {k: summarize(sub, k) for k in sub[0]}
    verdict = judge(sub, args) if all(k in sub[0] for k in conditions()) else {}
    keep = {"cp", BASE, "abrupt", "sleep_w"}
    for runs in sub:  # 結果ファイルを数 MB に収めるため、シード別の時系列は主な条件だけ残す
        for k in list(runs):
            if k not in keep and "series" in runs[k]:
                runs[k] = {kk: vv for kk, vv in runs[k].items() if kk != "series"}
    out = {"環境": env_info(), "設定": vars(args),
           "結合元の環境": sorted({json.dumps(e, sort_keys=True) for pt in parts for e in pt["環境"]}),
           "要約": summ, "判定": verdict, "シード別": {pt["seed"]: pt["runs"] for pt in parts}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, default=str))
    print(json.dumps(verdict, ensure_ascii=False, indent=1, default=str)[:15000])


if __name__ == "__main__":
    main()
