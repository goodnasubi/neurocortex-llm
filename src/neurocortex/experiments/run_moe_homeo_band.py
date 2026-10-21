"""ハブを許す目標域つきの乗法的な恒常性スケーリング（12.6.195節の設計）の主実験。

1層・8専門家・トップ1の文字 LM（12.6.124・128・134節と同じ設定）で、
既存のハブスパース MoE（補助損失なし）・補助損失（Switch 型 0.01・0.05）・一点の目標の乗法的な恒常性・
ハブを許す目標域つきの乗法的な恒常性を比べ、負荷集中・性能・ハブの保持・ハブを壊したときの頑健さを測る。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_homeo_band --seeds 100 \
        --out results/moe_homeo_band/dev_rev0_seed100.json
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_homeo_band \
        --merge results/moe_homeo_band/dev_rev0_seed10*.json --out results/moe_homeo_band/dev_rev0.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from ..data import CharCorpus, synthetic_corpus
from ..hub_load import spearman
from ..moe_homeo_band import BandHomeostaticLM, band_violation, single_lesions, topk_share
from ..run_env import env_info
from .run_homeostatic_moe import ms
from .run_moe_scale import run_one

BANDS = {"point": (1.0, 1.0), "band": (0.5, 2.5), "wide": (0.25, 4.0)}


def conditions(args: argparse.Namespace) -> dict[str, dict]:
    c = {f"aux{v:g}": {"kind": "plain", "coef": v, "band": "point"} for v in args.coefs}
    for name in args.bands:
        for eta in args.etas:
            c[f"{name}{eta:g}"] = {"kind": "select", "eta": eta, "band": name}
    return c


def post(k: int, r_lo: float, r_hi: float):
    def f(model, x, y, out):
        load = np.asarray(out["load"][0])
        single = single_lesions(model, x, y)
        out["single_lesion"] = single
        out["spearman_load_single"] = spearman(load, single) if np.ptp(load) > 0 and np.ptp(single) > 0 \
            else float("nan")
        out["topk_share"] = topk_share(load, k)
        out["band_violation"] = band_violation(load, BANDS["band"][0], BANDS["band"][1])
        out["own_band_violation"] = band_violation(load, r_lo, r_hi)
        out["max_share"] = float(load.max() / load.sum())
    return f


METRICS = ("gini", "max_over_mean", "entropy", "n_used", "gate_mean", "shifted_frac", "topk_share",
           "spearman_load_single", "band_violation", "own_band_violation", "max_share")


def summarize(runs: list[dict], keys: list[str]) -> dict:
    s = {}
    for key in keys:
        rs = [r[key] for r in runs]
        d = {m: ms([r[m] for r in rs]) for m in METRICS}
        d["all_used_seeds"] = sum(r["all_used"] for r in rs)
        for mode in ("zero", "reroute"):
            ls = [r["lesion"][mode] for r in rs]
            dt, dr = [l["d_target"] for l in ls], [l["d_random_mean"] for l in ls]
            d[mode] = {"base_ce": ms([l["base_ce"] for l in ls]), "d_target": ms(dt), "d_random": ms(dr),
                       "ratio_of_means": statistics.mean(dt) / max(statistics.mean(dr), 1e-4),
                       "target_gt_random": sum(a > b for a, b in zip(dt, dr)),
                       **{m: ms([l[m] for l in ls]) for m in ("hub_load_frac", "hub_gate_mass", "d_target_per_gate",
                                                              "d_random_per_gate", "d_target_per_load")}}
        if "threshold_off" in rs[0]:
            d["threshold_range"] = ms([r["threshold_range"] for r in rs])
            d["threshold_off"] = {m: ms([r["threshold_off"][m] for r in rs]) for m in ("gini", "n_used", "d_ce")}
        s[key] = d
    return s


def select_eta(s: dict, etas: list[float]) -> float:
    """事前に固定した選び方（12.6.195節）: point と band の両方で、全シードで全専門家が使われ、
    CE ≤ 補助損失0.05 + 0.01 を満たす最小の η。なければ最大の η。"""
    ref = s["aux0.05"]["zero"]["base_ce"]["mean"]
    for e in sorted(etas):
        ok = True
        for b in ("point", "band"):
            d = s[f"{b}{e:g}"]
            ok &= d["all_used_seeds"] == d["gini"]["n"] and d["zero"]["base_ce"]["mean"] <= ref + 0.01
        if ok:
            return e
    return max(etas)


def judge(runs: list[dict], s: dict, eta: float) -> dict:
    """12.6.195節の合格条件。主除去 = 出力の除去。"""
    B, P, N, A = f"band{eta:g}", f"point{eta:g}", "aux0", "aux0.05"
    col = lambda key, f: [f(r[key]) for r in runs]  # noqa: E731
    gB, gN = col(B, lambda z: z["gini"]), col(N, lambda z: z["gini"])
    mB, mN = col(B, lambda z: z["max_over_mean"]), col(N, lambda z: z["max_over_mean"])
    h1 = (statistics.mean(gN) - statistics.mean(gB) >= 0.3 and all(b < a for a, b in zip(gN, gB))
          and statistics.mean(mB) <= 0.6 * statistics.mean(mN) and s[B]["all_used_seeds"] == len(runs))
    ce = {k: s[k]["zero"]["base_ce"]["mean"] for k in (B, P, N, A)}
    h2 = ce[B] <= min(ce[N], ce[A]) + 0.005
    tB, tP = col(B, lambda z: z["topk_share"]), col(P, lambda z: z["topk_share"])
    rho = s[B]["spearman_load_single"]["mean"]
    h3 = statistics.mean(tB) >= 0.40 and rho >= 0.5 and all(b > p for b, p in zip(tB, tP))
    cB, cP = col(B, lambda z: z["lesion"]["zero"]["base_ce"]), col(P, lambda z: z["lesion"]["zero"]["base_ce"])
    h3b = sum(b < p for b, p in zip(cB, cP)) >= 4 and statistics.mean(cB) < statistics.mean(cP)
    dB, dN = col(B, lambda z: z["lesion"]["zero"]["d_target"]), col(N, lambda z: z["lesion"]["zero"]["d_target"])
    h4 = statistics.mean(dB) <= 0.6 * statistics.mean(dN) and all(b < a for a, b in zip(dN, dB))
    pgB = s[B]["zero"]["d_target_per_gate"]["mean"]
    pgN = s[N]["zero"]["d_target_per_gate"]["mean"]
    return {"eta": eta, "H1_負荷集中を下げる": h1, "H2_性能を保つ": h2, "H3_ハブを保つ": h3,
            "H3b_ハブの利点（帯域が一点の目標より CE が低い）": h3b, "H4_ハブを壊したときの頑健さ": h4,
            "参考_ハブ除去のゲートの質量あたり_帯域÷補助損失なし": pgB / pgN,
            "値": {"gini_帯域": statistics.mean(gB), "gini_なし": statistics.mean(gN),
                  "max_over_mean_帯域": statistics.mean(mB), "max_over_mean_なし": statistics.mean(mN),
                  "CE": ce, "topk_share_帯域": statistics.mean(tB), "topk_share_一点": statistics.mean(tP),
                  "spearman_帯域": rho, "dT_帯域": statistics.mean(dB), "dT_なし": statistics.mean(dN)}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--coefs", type=float, nargs="+", default=[0.0, 0.01, 0.05])
    p.add_argument("--bands", type=str, nargs="+", default=["point", "band", "wide"])
    p.add_argument("--etas", type=float, nargs="+", default=[0.01, 0.03, 0.1])
    p.add_argument("--eta-star", type=float, default=None, help="評価用シードでは開発用で選んだ η を渡す")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-sentences", type=int, default=3000)
    p.add_argument("--d", type=int, default=48)
    p.add_argument("--train-steps", type=int, default=600)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--n-windows", type=int, default=64)
    p.add_argument("--k-num", type=int, default=1)
    p.add_argument("--k-den", type=int, default=4)
    p.add_argument("--n-random", type=int, default=30)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", type=str, default="results/moe_homeo_band/dev.json")
    p.add_argument("--merge", type=str, nargs="+", default=None)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    conds = conditions(args)
    E, L = args.experts, args.layers
    k = max(1, E * args.k_num // args.k_den)
    runs, parts = [], []
    if args.merge:
        parts = [json.loads(Path(f).read_text()) for f in args.merge]
        runs = [r for part in parts for r in part["シード別"]]
        args.seeds = [r["seed"] for r in runs]
    for seed in ([] if args.merge else args.seeds):
        corpus = CharCorpus(synthetic_corpus(n_sentences=args.n_sentences, seed=seed))
        r = {"seed": seed}
        for key, cond in conds.items():
            lo, hi = BANDS[cond["band"]]
            z = r[key] = run_one(seed, E, L, cond, corpus, args, model_cls=BandHomeostaticLM,
                                 post=post(k, lo, hi), r_lo=lo, r_hi=hi)
            z["lesion"].pop("zero_gate1")  # ゲート=1 の評価は使えない（12.6.138節）ので残さない
            zl = z["lesion"]["zero"]
            print(f"seed={seed} {key} gini={z['gini']:.3f} mom={z['max_over_mean']:.2f} used={z['n_used']:.0f} "
                  f"top{k}={z['topk_share']:.3f} rho={z['spearman_load_single']:.2f} gate={z['gate_mean']:.3f} "
                  f"base={zl['base_ce']:.4f} dT={zl['d_target']:.3f} dR={zl['d_random_mean']:.3f}",
                  flush=True)
        runs.append(r)
    out_s = summarize(runs, list(conds))
    eta = args.eta_star if args.eta_star is not None else select_eta(out_s, args.etas)
    verdict = judge(runs, out_s, eta)
    verdict["eta_selected_on"] = "引数 --eta-star（開発用シードで選択済み）" if args.eta_star is not None \
        else "このシードでの選択（12.6.195節の選び方）"
    out = {"環境": env_info(), "設定": vars(args),
           **({"結合元の環境": [part["環境"] for part in parts]} if args.merge else {}),
           "要約": out_s, "判定": verdict, "シード別": runs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({"判定": verdict}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
