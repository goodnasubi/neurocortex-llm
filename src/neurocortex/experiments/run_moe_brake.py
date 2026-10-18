"""臨界期を閉じた後の可塑性のブレーキと、E=32 の取り込みが止まる原因の切り分け（12.6.149節の設計）の主実験。

(1) 臨界期型（12.6.145節）の閉じ方 × 閉じた後の手当て（θ の積分を止める・ルータの学習率のブレーキ）。規模 E=16・32 の1層、
    η は 12.6.145節で選んだもの（E=16: 0.3、E=32: 1）。開発用シードで規則（`select_pb`）により規模ごとに1つ選ぶ。
(2) E=32・選択専用 η=1（閉じない）で、ルータの表現力（2層ルータ）・恒常性の時定数 α・d を1つずつ変える。
(3) トークンごとの取り込み（`moe_brake.adoption_stats`）の時系列。閾値ありの条件すべてで記録する。

測定・H1〜H7 の判定は `run_moe_scale.py`、H7* と時系列の補助は `run_moe_critical.py` の関数を使う。

使い方（シード×規模ごとに別プロセスで出して結合する）:
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_brake --seeds 100 --scales 32,1 --grid \
        --out results/moe_brake/parts/dev_s100_E32L1.json
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_brake --merge results/moe_brake/parts/dev_*.json \
        --out results/moe_brake/dev.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from ..data import CharCorpus, synthetic_corpus
from ..hub_load import make_windows
from ..moe_brake import BrakedCriticalLM, train_braked
from ..moe_critical import absorption_metrics
from ..run_env import env_info
from .run_homeostatic_moe import ms
from .run_moe_critical import BASE, absorb_summary, at_step, judge_key, judge_star, slim
from .run_moe_scale import run_one

# 閉じ方（開いている期間, 閉じる期間）。固定化期間 0.70・0.50・0.25・0
SCHEDULES = ((0.25, 0.05), (0.25, 0.25), (0.5, 0.25), (0.5, 0.5))
# 閉じた後の手当て: frz = 閉じ始めと同時に θ の積分を止める、b<β> = ルータの学習率の倍率 β + (1−β)·s(t)
VARIANTS = ("none", "frz", "b0.1", "b0", "frz_b0.1", "frz_b0")
CP_ETA = {"E16_L1": 0.3, "E32_L1": 1.0}  # 12.6.145節で開発用シードから選んだ臨界期型の η
H11_PAIR = ((0.25, 0.25), "frz_b0.1", "none")  # H11 の固定の比較（固定化期間 50%）
DISSECT = {"sel1_a0.01": {"alpha": 0.01}, "sel1_a0.2": {"alpha": 0.2}, "sel1_mlp96": {"router_hidden": 96}}
D_BIG = 96


def pb_key(eta: float, o: float, r: float, v: str) -> str:
    return f"pb{eta:g}_o{o:g}_r{r:g}_{v}"


def parse_pb(key: str) -> dict:
    eta, o, r, v = key[2:].split("_", 3)
    c = {"kind": "select", "eta": float(eta), "open": float(o[1:]), "ramp": float(r[1:]), "variant": v}
    c["freeze"] = v.startswith("frz")
    b = v.split("b")[-1] if "b" in v.replace("frz", "") else None
    c["brake"] = float(b) if b is not None else None
    return c


def conditions(args: argparse.Namespace, e: int, l: int) -> dict[str, dict]:
    name = f"E{e}_L{l}"
    c: dict[str, dict] = {}
    if name == "E8_L1":  # (3) の基準（12.6.134節で取り込みを認めた条件）
        c["sel0.01"] = {"kind": "select", "eta": 0.01}
        return c
    c.update({"aux0": {"kind": "plain", "coef": 0.0}, "aux0.05": {"kind": "plain", "coef": 0.05},
              "gate0.1": {"kind": "gate", "eta": 0.1}})
    eta = CP_ETA.get(name)
    if eta is not None:
        c[f"sel{eta:g}"] = {"kind": "select", "eta": eta}
    if args.grid and eta is not None:
        c.update({pb_key(eta, o, r, v): parse_pb(pb_key(eta, o, r, v)) for o, r in SCHEDULES for v in VARIANTS})
    for kv in args.pb or []:  # 評価用シード: 開発用シードで選んだもの・H11 の比較・12.6.145節の選択
        sc, key = kv.split("=")
        if sc == f"{e},{l}":
            c[key] = parse_pb(key)
    if name == "E32_L1" and args.dissect:
        c.update({k: {"kind": "select", "eta": 1.0, **v} for k, v in DISSECT.items()})
        c.update({f"d{D_BIG}/{k}": {**c[k], "d": D_BIG} for k in (*BASE, "sel1")})
    return c


def close_step(cond: dict, total: int) -> int | None:
    if cond.get("open") is None:
        return None
    return round(min(1.0, cond["open"] + cond["ramp"]) * total)


def run_cond(seed: int, e: int, l: int, key: str, cond: dict, corpus: CharCorpus, args: argparse.Namespace) -> dict:
    a = argparse.Namespace(**{**vars(args), "d": cond.get("d", args.d), "alpha": cond.get("alpha", args.alpha)})
    probe = make_windows(corpus.val, args.seq_len, args.n_windows)[0][:args.n_probe]

    def post(model, x, y, out):
        if model.kind != "plain":
            out["absorb"] = absorption_metrics(model, x, y)
            model.adoption()
            out["series"] = model.series

    kw = {"total_steps": args.train_steps, "open_frac": cond.get("open"), "ramp_frac": cond.get("ramp", 0.0),
          "probe": probe, "log_every": args.log_every, "vec_every": 10 ** 9, "freeze": cond.get("freeze", False),
          "brake": cond.get("brake"), "router_hidden": cond.get("router_hidden", 0), "lag": args.lag}
    r = run_one(seed, e, l, cond, corpus, a, BrakedCriticalLM, args.train_steps, post=post, train_fn=train_braked, **kw)
    return slim(r, False)


def series_mean(sub: list[dict], key: str) -> list[dict]:
    """シード平均の時系列（数値だけ。None は除いて平均）。"""
    ss = [r[key].get("series") or [] for r in sub]
    if not ss[0]:
        return []
    out = []
    for pts in zip(*ss):
        p = {}
        for k in pts[0]:
            vals = [q[k] for q in pts if isinstance(q.get(k), (int, float)) and not isinstance(q.get(k), bool)]
            if vals:
                p[k] = round(statistics.mean(vals), 4)
        out.append(p)
    return out


def window_mean(series: list[dict], field: str, lo: int, hi: int) -> float | None:
    v = [p[field] for p in series if lo <= p["step"] <= hi and p.get(field) is not None]
    return statistics.mean(v) if v else None


def bias_return(r: dict, cond: dict, total: int) -> dict | None:
    """閉じ終わり（s = 0 になった時点）から学習の終わりまでのプローブのジニの変化（偏りの戻り）。"""
    t = close_step(cond, total)
    if t is None or not r.get("series"):
        return None
    g_close, g_end = at_step(r["series"], t)["gini_on"], at_step(r["series"], total)["gini_on"]
    return {"close_step": t, "gini_close": g_close, "gini_end": g_end, "return": g_end - g_close}


def adoption_summary(sub: list[dict], key: str, total: int) -> dict:
    """(3) の要約: 学習の T/4〜3T/4 の点で平均した、終わりのルータによる取り込み（adopt_T・keep_T・R_T）と Δ 後の取り込み。"""
    lo, hi = total // 4, 3 * total // 4
    per = {f: [window_mean(r[key]["series"], f, lo, hi) for r in sub]
           for f in ("adopt_T", "keep_T", "chance_T", "R_T", "adopt_lag", "keep_lag", "R_lag", "shift")}
    out = {}
    for f, v in per.items():
        vals = [x for x in v if x is not None]
        out[f] = {"mean": statistics.mean(vals) if vals else None, "per_seed": v}
    return out


def select_pb(verdicts: dict[str, dict]) -> tuple[str | None, str]:
    """12.6.145節の選び方を手当ての格子に広げたもの。合格（H1〜H3 かつジニ ≦ 0.3）のうちシード最大のジニが最小。同じなら
    固定化期間が長い方、次に手当てが弱い方（VARIANTS の順）。合格がなければ H3 を満たす中で同じ順の最小を「基準を満たさない」として返す。"""
    def rank(k):
        p = parse_pb(k)
        return (verdicts[k]["ジニ_シード最大"], p["open"] + p["ramp"], VARIANTS.index(p["variant"]))

    ok = [k for k, v in verdicts.items() if v["臨界期型の合格_H1〜H3かつジニ≦0.3"]]
    if ok:
        return min(ok, key=rank), "合格した候補のうちシード最大のジニが最小"
    h3 = [k for k, v in verdicts.items() if v["H3_性能を損なわない"]] or list(verdicts)
    if not h3:
        return None, "候補なし"
    return min(h3, key=rank), "合格した候補がなく、H3 を満たす中でシード最大のジニが最小（基準を満たさない）"


def judge_h11(sub: list[dict], total: int) -> dict | None:
    """H11: 固定化期間 50% の閉じ方で、θ の積分を止めてブレーキ β=0.1 をかけると、閉じ終わりからの偏りの戻りが 0.02 以下で、
    学習の終わりのジニ（全検証窓）が手当てなしより小さいシードが5本中4本以上。"""
    (o, r), v_on, v_off = H11_PAIR
    keys = [k for k in sub[0] if k.startswith("pb") and parse_pb(k)["open"] == o and parse_pb(k)["ramp"] == r]
    kon = next((k for k in keys if parse_pb(k)["variant"] == v_on), None)
    koff = next((k for k in keys if parse_pb(k)["variant"] == v_off), None)
    if kon is None or koff is None:
        return None
    ron = [bias_return(x[kon], parse_pb(kon), total)["return"] for x in sub]
    roff = [bias_return(x[koff], parse_pb(koff), total)["return"] for x in sub]
    lower = sum(x[kon]["gini"] < x[koff]["gini"] for x in sub)
    return {"手当てあり": kon, "手当てなし": koff, "戻り_手当てあり": statistics.mean(ron), "戻り_手当てなし": statistics.mean(roff),
            "戻り_手当てあり_シード別": ron, "戻り_手当てなし_シード別": roff,
            "ジニ_手当てあり": statistics.mean(x[kon]["gini"] for x in sub),
            "ジニ_手当てなし": statistics.mean(x[koff]["gini"] for x in sub), "手当てありが小さいシード数": lower,
            "H11_偏りの戻りを防ぐ": statistics.mean(ron) <= 0.02 and lower >= len(sub) - 1}


def judge_dissect(sub: list[dict], ref_norm: float | None, args: argparse.Namespace) -> dict:
    """H13: 要因を1つ変えたとき、閾値を外したときのジニ（学習の終わり）の平均が基準（sel1）より 0.05 以上下がり、5本中4本以上で
    下がり、かつ一致率 A の平均が 0.03 以上上がれば「寄与する」。H7* と H1〜H3 を満たせば「解消する」。"""
    base = "sel1"
    if base not in sub[0]:
        return {}
    g0 = [r[base]["threshold_off"]["gini"] for r in sub]
    a0 = [r[base]["absorb"]["agree"] for r in sub]
    out = {}
    for k in [base, *DISSECT, f"d{D_BIG}/sel1"]:
        if k not in sub[0]:
            continue
        g = [r[k]["threshold_off"]["gini"] for r in sub]
        a = [r[k]["absorb"]["agree"] for r in sub]
        dg, da = statistics.mean(g) - statistics.mean(g0), statistics.mean(a) - statistics.mean(a0)
        pre = f"d{D_BIG}/" if k.startswith(f"d{D_BIG}/") else ""
        v = judge_key(sub, k, args, pre)
        star = judge_star(absorb_summary(sub, k), ref_norm)
        out[k] = {"外したときのジニ": statistics.mean(g), "一致率": statistics.mean(a), "Δジニ": dg, "Δ一致率": da,
                  "下がったシード数": sum(x < y for x, y in zip(g, g0)), "H7*": star,
                  "H13_寄与する": k != base and dg <= -0.05 and sum(x < y for x, y in zip(g, g0)) >= len(sub) - 1
                  and da >= 0.03,
                  "H13_解消する": bool(star["H7*"] and v["H1_崩壊を防ぐ"] and v["H2_ハブの非対称性を消す"]
                                     and v["H3_性能を損なわない"])}
    return out


def judge_h14(subs: dict[str, list[dict]], total: int) -> dict:
    """H14: (a) E=8・η=0.01 で R_T（T/4〜3T/4 の平均）の5シード平均 ≧ 3 かつ全シードで > 1。
    (b) E=32・η=1 の adopt_T が E=8・η=0.01 より小さい（平均と、同じシードで5本中4本以上）。"""
    out = {}
    if "E8_L1" in subs:
        a8 = adoption_summary(subs["E8_L1"], "sel0.01", total)
        out["E8_sel0.01"] = a8
        rs = [x for x in a8["R_T"]["per_seed"] if x is not None]
        out["H14a_トークンごとの取り込みがある"] = bool(rs and statistics.mean(rs) >= 3 and all(x > 1 for x in rs))
        if "E32_L1" in subs and "sel1" in subs["E32_L1"][0]:
            a32 = adoption_summary(subs["E32_L1"], "sel1", total)
            out["E32_sel1"] = a32
            p8, p32 = a8["adopt_T"]["per_seed"], a32["adopt_T"]["per_seed"]
            out["H14b_E32の取り込みが小さい"] = bool(a32["adopt_T"]["mean"] < a8["adopt_T"]["mean"]
                                               and sum(x < y for x, y in zip(p32, p8)) >= len(p8) - 1)
    return out


def summarize_key(sub: list[dict], key: str, total: int) -> dict:
    rs = [r[key] for r in sub]
    d = {"gini": ms([r["gini"] for r in rs]), "gini_max": max(r["gini"] for r in rs),
         "n_used": ms([r["n_used"] for r in rs]), "all_used_seeds": sum(r["all_used"] for r in rs),
         "base_ce": ms([r["lesion"]["zero"]["base_ce"] for r in rs]),
         "ratio_of_means": statistics.mean(r["lesion"]["zero"]["d_target"] for r in rs)
         / max(statistics.mean(r["lesion"]["zero"]["d_random_mean"] for r in rs), 1e-4),
         "gate_mean": ms([r["gate_mean"] for r in rs])}
    if "threshold_off" in rs[0]:
        d["gini_off"] = ms([r["threshold_off"]["gini"] for r in rs])
        d["absorb"] = {m: ms([r["absorb"][m] for r in rs]) for m in ("agree", "d_ce_off", "d_ce_off_norm", "contrib")}
        if "agree_state" in rs[0]["absorb"]:
            d["absorb"]["agree_state"] = ms([r["absorb"]["agree_state"] for r in rs])
        d["series"] = series_mean(sub, key)
        d["adoption"] = adoption_summary(sub, key, total)
    if key.startswith("pb"):
        br = [bias_return(r, parse_pb(key), total) for r in rs]
        d["return"] = {k: ms([b[k] for b in br]) for k in ("gini_close", "gini_end", "return")}
    return d


def analyze(subs: dict[str, list[dict]], args: argparse.Namespace) -> tuple[dict, dict]:
    ref = None
    if "E8_L1" in subs:
        ref = absorb_summary(subs["E8_L1"], "sel0.01")["d_ce_off_norm"]["mean"]
    summ, verdict = {}, {"H7*の基準_E8_L1_sel0.01の正規化ΔCE_off": ref}
    for n, sub in subs.items():
        summ[n] = {k: summarize_key(sub, k, args.train_steps) for k in sub[0]}
        if n == "E8_L1":
            continue
        v = {}
        pbs = [k for k in sub[0] if k.startswith("pb")]
        cand = {}
        for k in pbs:
            j = judge_key(sub, k, args)
            j["臨界期型の合格_H1〜H3かつジニ≦0.3"] = bool(j["H1_崩壊を防ぐ"] and j["H2_ハブの非対称性を消す"]
                                                 and j["H3_性能を損なわない"] and j["ジニ"] <= 0.3)
            cand[k] = j
        v["候補"] = cand
        if args.grid and pbs:
            key, why = select_pb(cand)
            v["選択"] = {"キー": key, "選び方": why}
        v["H11"] = judge_h11(sub, args.train_steps)
        if n == "E32_L1":
            v["H13_切り分け"] = judge_dissect(sub, ref, args)
        verdict[n] = v
    verdict["H14"] = judge_h14(subs, args.train_steps)
    return summ, verdict


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    p.add_argument("--scales", type=str, nargs="+", default=["8,1", "16,1", "32,1"], help="E,L の列")
    p.add_argument("--grid", action="store_true", help="閉じ方 × 手当ての格子を測る（開発用シード）")
    p.add_argument("--pb", type=str, nargs="*", default=None, help="評価用シード: 'E,L=キー' の列")
    p.add_argument("--dissect", action="store_true", help="E=32 の切り分け（2層ルータ・α・d）を含める")
    p.add_argument("--only", type=str, nargs="*", default=None, help="この条件のキーだけ測る（部分ファイル）")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-sentences", type=int, default=3000)
    p.add_argument("--d", type=int, default=48)
    p.add_argument("--train-steps", type=int, default=600)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--n-windows", type=int, default=64)
    p.add_argument("--n-probe", type=int, default=16)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--lag", type=int, default=100, help="トークンごとの取り込みの Δ（ステップ）")
    p.add_argument("--k-num", type=int, default=1)
    p.add_argument("--k-den", type=int, default=4)
    p.add_argument("--n-random", type=int, default=30)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", type=str, default="results/moe_brake/dev.json")
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
                for key, cond in conditions(args, e, l).items():
                    if args.only and key not in args.only:
                        continue
                    z = runs[key] = run_cond(seed, e, l, key, cond, corpus, args)
                    a = z.get("absorb", {})
                    extra = (f" off_gini={z['threshold_off']['gini']:.3f} agree={a['agree']:.3f}"
                             f" dce_off={a['d_ce_off']:.4f}") if a else ""
                    br = bias_return(z, cond, args.train_steps) if key.startswith("pb") else None
                    extra += f" close={br['gini_close']:.3f} ret={br['return']:+.3f}" if br else ""
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
        m = merged.setdefault((pt["seed"], pt["scale"]), {"seed": pt["seed"], "scale": pt["scale"], "runs": {}, "環境": []})
        m["runs"].update(pt["runs"])
        m["環境"].append(pt["環境"])
    parts = list(merged.values())
    names = sorted(dict.fromkeys(pt["scale"] for pt in parts), key=lambda n: tuple(map(int, n[1:].split("_L"))))
    subs = {n: [pt["runs"] for pt in sorted((pt for pt in parts if pt["scale"] == n), key=lambda pt: pt["seed"])]
            for n in names}
    summ, verdict = analyze(subs, args)
    for n, sub in subs.items():  # 結果ファイルを数 MB に収めるため、シード別の時系列は主な条件だけ残す
        keep = {"sel0.01", "sel0.3", "sel1", *DISSECT, f"d{D_BIG}/sel1", (verdict.get(n, {}).get("選択") or {}).get("キー")}
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
    print(json.dumps({n: {k: v[k] for k in ("選択", "H11") if k in v} if isinstance(v, dict) else v
                      for n, v in verdict.items() if n != "H14"}, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(verdict.get("H14", {}), ensure_ascii=False, default=str)[:3000])


if __name__ == "__main__":
    main()
