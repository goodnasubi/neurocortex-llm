"""臨界期型の閉じ方とゲートの大きさに左右されない取り込みの指標（12.6.145節の設計）の主実験。

(1) 12.6.142節の条件（補助損失なし・0.05・ゲート込み・選択専用の η 列・(a) 長い学習・(c) η=0.01/0.1）を、閾値を外した
    ときの選択の一致率と ‖g·y‖ で正規化した ΔCE_off で測り直す。
(2) 臨界期型の閉じ方（開いている期間・閉じる速さ・閉じた後の固定化期間）と η の格子から、開発用シードで事前に決めた
    規則（`select_cp`）で規模ごとに1つ選び、評価用シードではそれだけを測る（`--cp 'E,L=キー'`）。
(3) 選択専用・臨界期型の学習中の閾値・ルータの対数・偏り・選択の一致率の時系列（`moe_critical.CriticalPeriodLM`）。

測定・要約・H1〜H7 の判定は `run_moe_scale.py`（12.6.137節）の関数をそのまま使う。

使い方（シード×規模ごとに別プロセスで出して結合する）:
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_critical --seeds 100 --scales 16,1 --grid \
        --out results/moe_critical/parts/dev_s100_E16L1.json
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_critical --merge results/moe_critical/parts/dev_*.json \
        --out results/moe_critical/dev.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from ..data import CharCorpus, synthetic_corpus
from ..hub_load import make_windows
from ..moe_critical import CriticalPeriodLM, absorption_metrics
from ..run_env import env_info
from .run_homeostatic_moe import ms
from .run_moe_scale import judge, run_one, summarize

BASE = ("aux0", "aux0.05", "gate0.1")
# 臨界期型の格子（12.6.145節）。(開いている期間, 閉じる期間) を総ステップ数の割合で。固定化期間 = 1 − 両者の和
SCHEDULES = ((0.25, 0.05), (0.25, 0.25), (0.5, 0.05), (0.5, 0.25), (0.75, 0.05), (0.5, 0.5))
CP_ETAS = (0.03, 0.1, 0.3, 1.0)
ETA_B_142 = {"E8_L1": 0.01, "E16_L1": 0.03, "E32_L1": 1.0, "E32_L2": 0.3}  # 12.6.141節で開発用シードから選んだ η_b
STAR = {"gini_off": 0.3, "agree": 0.85, "norm_x_ref": 2.0}  # 12.6.145節の H7*
VEC_KEYS = ("sel0.01", "ann0.1")  # 時系列のベクトル（θ・Δ対数）を残す条件（η_b と cp はこれに足す）


def cp_key(eta: float, open_frac: float, ramp_frac: float) -> str:
    return f"cp{eta:g}_o{open_frac:g}_r{ramp_frac:g}"


def parse_cp(key: str) -> dict:
    eta, o, r = key[2:].split("_")
    return {"kind": "select", "eta": float(eta), "open": float(o[1:]), "ramp": float(r[1:])}


def conditions(args: argparse.Namespace, n_experts: int, n_layers: int) -> dict[str, dict]:
    name = f"E{n_experts}_L{n_layers}"
    c = {"aux0": {"kind": "plain", "coef": 0.0}, "aux0.05": {"kind": "plain", "coef": 0.05},
         "gate0.1": {"kind": "gate", "eta": 0.1}}
    etas = list(args.sel_etas) + (list(args.extra_etas) if n_experts >= 32 else [])
    c.update({f"sel{v:g}": {"kind": "select", "eta": v} for v in etas})
    c.update({f"ann{v:g}": {"kind": "select", "eta": v, "open": 0.5, "ramp": 0.5} for v in args.ann_etas})
    steps = args.train_steps * n_experts // 8
    if args.long and steps != args.train_steps:  # 12.6.141節 (a)。E=8 は基準と同じ
        for k in (*BASE, "sel0.01"):
            c[f"long/{k}"] = {**c.get(k, {"kind": "select", "eta": 0.01}), "steps": steps}
    if args.grid and f"{n_experts},{n_layers}" in args.grid_scales:
        c.update({cp_key(v, o, r): parse_cp(cp_key(v, o, r)) for v in CP_ETAS for o, r in SCHEDULES})
    for kv in args.cp or []:  # 評価用シード: 開発用シードで選んだ臨界期型の条件だけ
        sc, key = kv.split("=")
        if sc == f"{n_experts},{n_layers}":
            c[key] = parse_cp(key)
    return c


def slim(out: dict, keep_vectors: bool) -> dict:
    """結果ファイルを数 MB に収めるため、負荷・閾値のベクトルと時系列のベクトルを落とす。"""
    out = {k: v for k, v in out.items() if k not in ("load", "threshold")}
    if not keep_vectors and "series" in out:
        out["series"] = [{k: v for k, v in p.items() if k not in ("theta_state", "d_logit")} for p in out["series"]]
    return out


def run_cond(seed: int, e: int, l: int, key: str, cond: dict, corpus: CharCorpus, args: argparse.Namespace) -> dict:
    steps = cond.get("steps", args.train_steps)
    probe = make_windows(corpus.val, args.seq_len, args.n_windows)[0][:args.n_probe]

    def post(model, x, y, out):
        if model.kind != "plain":
            out["absorb"] = absorption_metrics(model, x, y)
            out["series"] = model.series

    kw = {"total_steps": steps, "open_frac": cond.get("open"), "ramp_frac": cond.get("ramp", 0.0),
          "probe": None if key.startswith("long/") else probe, "log_every": args.log_every}
    r = run_one(seed, e, l, cond, corpus, args, CriticalPeriodLM, steps, post=post, **kw)
    vec = key in VEC_KEYS or key.startswith("cp") or key == f"sel{ETA_B_142.get(f'E{e}_L{l}', -1):g}"
    return slim(r, vec)


def judge_key(sub: list[dict], key: str, args: argparse.Namespace, prefix: str = "") -> dict:
    """任意の閾値ありの条件 key を、12.6.137節の judge の「選択専用」の位置に置いて判定する。"""
    csub = [{**{k: r[prefix + k] for k in BASE}, "sel-1": r[key]} for r in sub]
    cs = summarize(csub, list(csub[0]), args.eps)
    v = judge(csub, cs, -1.0, "gate0.1")
    v.pop("eta")
    v["ジニ"] = cs["sel-1"]["gini"]["mean"]
    v["ジニ_シード最大"] = max(r[key]["gini"] for r in sub)
    v["取り込みが起きた_H7かつH1〜H3"] = bool(v["H7_重みが均等な使い方を身につける"] and v["H1_崩壊を防ぐ"]
                                      and v["H2_ハブの非対称性を消す"] and v["H3_性能を損なわない"])
    return v


def absorb_summary(sub: list[dict], key: str) -> dict:
    rs = [r[key]["absorb"] for r in sub]
    d = {m: ms([r[m] for r in rs]) for m in rs[0]}
    d["gini_off"] = ms([r[key]["threshold_off"]["gini"] for r in sub])
    return d


def judge_star(a: dict, ref_norm: float | None) -> dict:
    """H7*（12.6.145節）: 外したときのジニ ≦ 0.3、選択の一致率 ≧ 0.85、正規化した ΔCE_off ≦ 基準（E=8・L=1・η=0.01）の 2 倍。"""
    g, ag, n = a["gini_off"]["mean"], a["agree"]["mean"], a["d_ce_off_norm"]["mean"]
    lim = None if ref_norm is None else STAR["norm_x_ref"] * max(ref_norm, 0.0)
    ok_n = None if lim is None else n <= lim
    return {"外したときのジニ": g, "一致率": ag, "正規化ΔCE_off": n, "正規化ΔCE_offの上限": lim,
            "H7*_ジニ": g <= STAR["gini_off"], "H7*_一致率": ag >= STAR["agree"], "H7*_正規化ΔCE": ok_n,
            "H7*": bool(g <= STAR["gini_off"] and ag >= STAR["agree"] and (ok_n is None or ok_n))}


def series_summary(sub: list[dict], key: str) -> list[dict]:
    """シード平均の時系列（スカラーだけ）。"""
    ss = [r[key].get("series") or [] for r in sub]
    if not ss or not ss[0]:
        return []
    out = []
    for pts in zip(*ss):
        out.append({k: statistics.mean(p[k] for p in pts) for k, v in pts[0].items() if isinstance(v, (int, float))})
    return out


def at_step(series: list[dict], step: int) -> dict:
    return min(series, key=lambda p: abs(p["step"] - step))


def judge_series(sub: list[dict], key: str, total: int) -> dict:
    """H10（12.6.145節）: 取り込みは学習とともに進み（外したときのジニが T/4 から T までに 0.1 以上下がる）、閾値の幅は
    頭打ちになり（T の幅 ≦ T/2 の幅の 1.5 倍）、ルータの対数の変化が −θ と正に相関する（平均 > 0、5本中4本以上で正）。"""
    per = [r[key]["series"] for r in sub]
    g4 = statistics.mean(at_step(s, total // 4)["gini_off"] for s in per)
    gT = statistics.mean(at_step(s, total)["gini_off"] for s in per)
    w2 = statistics.mean(at_step(s, total // 2)["th_range"] for s in per)
    wT = statistics.mean(at_step(s, total)["th_range"] for s in per)
    rhos = [at_step(s, total)["rho"] for s in per]
    return {"外したときのジニ_T/4": g4, "外したときのジニ_T": gT, "閾値の幅_T/2": w2, "閾値の幅_T": wT,
            "rho_T": statistics.mean(rhos), "rho_T_シード別": rhos,
            "H10a_取り込みが学習とともに進む": gT <= g4 - 0.1, "H10b_閾値の幅が頭打ち": wT <= 1.5 * w2,
            "H10c_対数の変化が−θと正に相関": statistics.mean(rhos) > 0 and sum(x > 0 for x in rhos) >= len(rhos) - 1}


def cp_candidate(v: dict) -> bool:
    """臨界期型の候補の合格: H1〜H3 と、学習の終わり（閾値 0）のジニの5シード平均 ≦ 0.3。"""
    return bool(v["H1_崩壊を防ぐ"] and v["H2_ハブの非対称性を消す"] and v["H3_性能を損なわない"] and v["ジニ"] <= 0.3)


def select_cp(verdicts: dict[str, dict]) -> tuple[str | None, str]:
    """12.6.145節の選び方。合格した候補のうち、シード最大のジニが最小のもの（同じなら η が小さい、固定化期間が長い）。
    合格がなければ、H3 を満たす中でシード最大のジニが最小のもの（それもなければ全候補から）を「基準を満たさない」として返す。"""
    def rank(k):
        p = parse_cp(k)
        return (verdicts[k]["ジニ_シード最大"], p["eta"], p["open"] + p["ramp"])

    ok = [k for k, v in verdicts.items() if cp_candidate(v)]
    if ok:
        return min(ok, key=rank), "合格した候補のうちシード最大のジニが最小"
    h3 = [k for k, v in verdicts.items() if v["H3_性能を損なわない"]] or list(verdicts)
    if not h3:
        return None, "候補なし"
    return min(h3, key=rank), "合格した候補がなく、H3 を満たす中でシード最大のジニが最小（基準を満たさない）"


def analyze(sub: list[dict], args: argparse.Namespace, name: str, ref_norm: float | None) -> dict:
    keys = list(sub[0])
    thr = [k for k in keys if k not in BASE and not k.startswith("long/aux") and not k.startswith("long/gate")]
    s = summarize(sub, [k for k in keys if not k.startswith("long/")], args.eps)
    absorb = {k: absorb_summary(sub, k) for k in thr}
    v1 = {}
    for k in thr:
        long = k.startswith("long/")
        v = judge_key(sub, k, args, "long/" if long else "")
        v["H7*"] = judge_star(absorb[k], ref_norm)
        v["取り込み*_H7*かつH1〜H3"] = bool(v["H7*"]["H7*"] and v["H1_崩壊を防ぐ"] and v["H2_ハブの非対称性を消す"]
                                      and v["H3_性能を損なわない"])
        v1[k] = v
    cps = [k for k in keys if k.startswith("cp")]
    v2 = {k: v1[k] for k in cps}
    for k in v2:
        v2[k]["臨界期型の合格_H1〜H3かつジニ≦0.3"] = cp_candidate(v2[k])
    chosen = select_cp(v2) if args.grid and cps else (None, "")
    series = {k: series_summary(sub, k) for k in thr if not k.startswith("long/")}
    v3 = {k: judge_series(sub, k, args.train_steps) for k in thr
          if not k.startswith("long/") and sub[0][k].get("series")}
    return {"要約": {**s, "取り込みの指標": absorb, "時系列": series},
            "判定": {"1_指標の測り直し": v1, "2_臨界期型": v2, "2_選択": {"キー": chosen[0], "選び方": chosen[1]},
                   "3_時系列": v3, "η_b_12.6.141節": ETA_B_142.get(name)}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    p.add_argument("--scales", type=str, nargs="+", default=["8,1", "16,1", "32,1", "32,2"], help="E,L の列")
    p.add_argument("--sel-etas", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3])
    p.add_argument("--extra-etas", type=float, nargs="*", default=[1.0, 3.0], help="E=32 だけに足す η（12.6.141節の修正1）")
    p.add_argument("--ann-etas", type=float, nargs="*", default=[0.01, 0.1], help="12.6.141節 (c) の η")
    p.add_argument("--long", action="store_true", help="12.6.141節 (a) の長い学習を含める")
    p.add_argument("--grid", action="store_true", help="臨界期型の格子を測る（開発用シード）")
    p.add_argument("--grid-scales", type=str, nargs="*", default=["16,1", "32,1"])
    p.add_argument("--cp", type=str, nargs="*", default=None, help="評価用シード: 'E,L=キー' の列（開発用で選んだもの）")
    p.add_argument("--only", type=str, nargs="*", default=None, help="この条件のキーだけ測る（部分ファイル）")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-sentences", type=int, default=3000)
    p.add_argument("--d", type=int, default=48)
    p.add_argument("--train-steps", type=int, default=600)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--n-windows", type=int, default=64)
    p.add_argument("--n-probe", type=int, default=16, help="時系列のプローブの窓の数（検証窓の先頭から）")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--k-num", type=int, default=1)
    p.add_argument("--k-den", type=int, default=4)
    p.add_argument("--n-random", type=int, default=30)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", type=str, default="results/moe_critical/dev.json")
    p.add_argument("--merge", type=str, nargs="+", default=None)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    if args.merge:
        parts = [pt for f in args.merge for pt in json.loads(Path(f).read_text())["部分"]]
    else:
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
                             f" dce_off={a['d_ce_off']:.4f} norm={a['d_ce_off_norm']:.4f}") if a else ""
                    print(f"seed={seed} E{e}_L{l}/{key} gini={z['gini']:.3f} used={z['n_used']:.1f} "
                          f"base={z['lesion']['zero']['base_ce']:.4f} ratio={z['lesion']['zero']['ratio']:.2f} "
                          f"gate={z['gate_mean']:.3f}{extra}", flush=True)
                parts.append({"環境": env_info(), "seed": seed, "scale": f"E{e}_L{l}", "runs": runs})
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"環境": env_info(), "設定": vars(args), "部分": parts}, ensure_ascii=False))
        return
    merged: dict = {}
    for pt in parts:
        m = merged.setdefault((pt["seed"], pt["scale"]), {"seed": pt["seed"], "scale": pt["scale"], "runs": {},
                                                           "環境": []})
        m["runs"].update(pt["runs"])
        m["環境"].append(pt["環境"])
    parts = list(merged.values())
    names = sorted(dict.fromkeys(pt["scale"] for pt in parts), key=lambda n: tuple(map(int, n[1:].split("_L"))))
    subs = {n: [pt["runs"] for pt in sorted((pt for pt in parts if pt["scale"] == n), key=lambda pt: pt["seed"])]
            for n in names}
    ref = None
    if "E8_L1" in subs and "sel0.01" in subs["E8_L1"][0]:
        ref = absorb_summary(subs["E8_L1"], "sel0.01")["d_ce_off_norm"]["mean"]
    out_s, verdict = {}, {}
    for n in names:
        r = analyze(subs[n], args, n, ref)
        out_s[n], verdict[n] = r["要約"], r["判定"]
    out = {"環境": env_info(), "設定": vars(args), "結合元の環境": sorted({json.dumps(e, sort_keys=True)
                                                                for pt in parts for e in pt["環境"]}),
           "H7*の基準_E8_L1_sel0.01の正規化ΔCE_off": ref, "要約": out_s, "判定": verdict,
           "シード別": {n: {pt["seed"]: pt["runs"] for pt in parts if pt["scale"] == n} for n in names}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False))
    print(json.dumps({n: {"2_選択": v["2_選択"]} for n, v in verdict.items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
