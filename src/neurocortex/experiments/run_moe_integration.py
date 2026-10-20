"""領域をまたぐ統合が要る課題でのハブの利点（12.6.199節の設計）の主実験。

複数の言語（2つの族）の混合の文字 LM で、統合課題（前半の言語 i の手がかりで後半の言語 j ≠ i の続きが
決まる）と対照課題（j = i）を、12.6.195節の条件（補助損失なし・補助損失 0.01・0.05・一点の目標・
ハブを許す目標域の乗法的な恒常性）で学習し、続き（b2）の CE・ハブの参加係数・ハブ除去の損害を測る。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_integration --seeds 100 \
        --out results/moe_integration/dev_rev0_seed100.json
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_integration \
        --merge results/moe_integration/dev_rev0_seed10*.json --out results/moe_integration/dev_rev0.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from ..data import CharCorpus
from ..hub_load import spearman
from ..moe_homeo_band import BandHomeostaticLM, single_lesions, topk_share
from ..moe_integration import (hub_participation, hub_target_share, integration_corpus, participation, slot_ce,
                               split_like_corpus, windows_like, LANGS)
from ..moe_scale import hub_masks
from ..run_env import env_info
from .run_homeostatic_moe import ms
from .run_moe_homeo_band import BANDS
from .run_moe_scale import run_one

TASKS = ("cross", "within")


def conditions(args: argparse.Namespace) -> dict[str, dict]:
    c = {f"aux{v:g}": {"kind": "plain", "coef": v, "band": "point"} for v in args.coefs}
    for name in args.bands:
        c[f"{name}{args.eta:g}"] = {"kind": "select", "eta": args.eta, "band": name}
    return c


def post(k: int, x_lang: torch.Tensor, y_slot: torch.Tensor):
    def f(model, x, y, out):
        load = np.asarray(out["load"][0])
        choice = model.run(x)["choice"][0].numpy()
        out["topk_share"] = topk_share(load, k)
        single = single_lesions(model, x, y)
        out["spearman_load_single"] = spearman(load, single) if np.ptp(load) > 0 and np.ptp(single) > 0 \
            else float("nan")
        out["slot_ce"] = slot_ce(model, x, y, y_slot)
        p = participation(choice, x_lang.numpy(), model.n_experts, len(LANGS))
        out["participation"] = [None if np.isnan(v) else float(v) for v in p]
        out["p_hub"], out["p_rest"] = hub_participation(load, p, k)
        out["hub_target_share"], out["hub_all_share"] = hub_target_share(choice, y_slot.numpy(), load, k)
        hub, _ = hub_masks(load[None], k, 0, np.random.default_rng(0))
        les = slot_ce(model, x, y, y_slot, disabled=hub, mode="zero")
        out["hub_lesion_d"] = {s: les[s] - out["slot_ce"][s] for s in ("target", "other", "all")}
    return f


def summarize(runs: list[dict], keys: list[str]) -> dict:
    s: dict = {}
    for task in TASKS:
        s[task] = {}
        for key in keys:
            rs = [r[task][key] for r in runs]
            d = {m: ms([z[m] for z in rs]) for m in ("gini", "max_over_mean", "n_used", "gate_mean", "topk_share",
                                                   "spearman_load_single", "p_hub", "p_rest", "hub_target_share",
                                                   "hub_all_share")}
            d["slot_ce"] = {m: ms([z["slot_ce"][m] for z in rs]) for m in rs[0]["slot_ce"]}
            d["hub_lesion_d"] = {m: ms([z["hub_lesion_d"][m] for z in rs]) for m in ("target", "other", "all")}
            d["all_used_seeds"] = sum(z["all_used"] for z in rs)
            s[task][key] = d
    return s


def paired_t(v: list[float]) -> float:
    if len(v) < 2 or statistics.stdev(v) == 0:
        return float("nan")
    return statistics.mean(v) / (statistics.stdev(v) / len(v) ** 0.5)


def judge(runs: list[dict], eta: float) -> dict:
    """12.6.199節の合格条件（評価用シードを見る前に固定）。主指標 = b2 の CE（target）。"""
    B, P, A = f"band{eta:g}", f"point{eta:g}", "aux0.05"
    tce = lambda r, t, k: r[t][k]["slot_ce"]["target"]  # noqa: E731
    d_cross = [tce(r, "cross", B) - tce(r, "cross", P) for r in runs]
    d_within = [tce(r, "within", B) - tce(r, "within", P) for r in runs]
    inter = [a - b for a, b in zip(d_cross, d_within)]
    inter_aux = [(tce(r, "cross", B) - tce(r, "cross", A)) - (tce(r, "within", B) - tce(r, "within", A)) for r in runs]
    gap = {k: statistics.mean([tce(r, "cross", k) - tce(r, "within", k) for r in runs]) for k in runs[0]["cross"]
           if k != "seed"}
    dp = [r["cross"][B]["p_hub"] - r["cross"][B]["p_rest"] for r in runs]
    rel = lambda z: z["hub_lesion_d"]["target"] / max(z["hub_lesion_d"]["other"], 1e-4)  # noqa: E731
    r_cross, r_within = [rel(r["cross"][B]) for r in runs], [rel(r["within"][B]) for r in runs]
    n = len(runs)
    need = n - 1 if n >= 5 else n
    v = all(g > 0.03 for g in gap.values())
    h1 = statistics.mean(inter) <= -0.01 and sum(i < 0 for i in inter) >= need
    h1b = statistics.mean(d_cross) < 0 and sum(d < 0 for d in d_cross) >= need
    h2 = statistics.mean(inter_aux) <= -0.01 and sum(i < 0 for i in inter_aux) >= need
    h3 = statistics.mean(dp) >= 0.1 and sum(d > 0 for d in dp) >= need
    h4 = sum(a > b for a, b in zip(r_cross, r_within)) >= need and statistics.mean(r_cross) > statistics.mean(r_within)
    return {"eta": eta, "V_課題の妥当性（全条件で統合課題の b2 の CE が対照より 0.03 以上高い）": v,
            "H1_交互作用（帯域−一点 の差が統合課題でだけ小さい）": h1, "H1b_統合課題で帯域が一点より低い": h1b,
            "H2_交互作用（帯域−補助損失0.05）": h2, "H3_統合課題でハブの参加係数が高い": h3,
            "H4_ハブ除去の損害が統合課題で b2 に偏る": h4,
            "値": {"interaction_mean": statistics.mean(inter), "interaction_t": paired_t(inter),
                  "interaction_per_seed": inter, "d_cross": d_cross, "d_within": d_within,
                  "interaction_aux_mean": statistics.mean(inter_aux), "interaction_aux_t": paired_t(inter_aux),
                  "interaction_aux_per_seed": inter_aux, "gap_cross_minus_within": gap,
                  "p_hub_minus_rest_cross": dp, "lesion_target_over_other_cross": r_cross,
                  "lesion_target_over_other_within": r_within}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--coefs", type=float, nargs="+", default=[0.0, 0.01, 0.05])
    p.add_argument("--bands", type=str, nargs="+", default=["point", "band"])
    p.add_argument("--eta", type=float, default=0.01, help="12.6.195節の開発用シードで選んだ η*")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-lines", type=int, default=8000)
    p.add_argument("--n-concepts", type=int, default=8)
    p.add_argument("--d", type=int, default=48)
    p.add_argument("--train-steps", type=int, default=600)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--n-windows", type=int, default=256)
    p.add_argument("--k-num", type=int, default=1)
    p.add_argument("--k-den", type=int, default=4)
    p.add_argument("--n-random", type=int, default=30)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", type=str, default="results/moe_integration/dev.json")
    p.add_argument("--merge", type=str, nargs="+", default=None)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    conds = conditions(args)
    E = args.experts
    k = max(1, E * args.k_num // args.k_den)
    runs, parts = [], []
    if args.merge:
        parts = [json.loads(Path(f).read_text()) for f in args.merge]
        runs = [r for part in parts for r in part["シード別"]]
        args.seeds = [r["seed"] for r in runs]
    for seed in ([] if args.merge else args.seeds):
        r = {"seed": seed}
        for task in TASKS:
            text, slots, langs = integration_corpus(task, n_lines=args.n_lines, n_concepts=args.n_concepts, seed=seed)
            corpus = CharCorpus(text)
            x_lang, _ = windows_like(split_like_corpus(langs), args.seq_len, args.n_windows)
            _, y_slot = windows_like(split_like_corpus(slots), args.seq_len, args.n_windows)
            r[task] = {}
            for key, cond in conds.items():
                lo, hi = BANDS[cond["band"]]
                z = run_one(seed, E, 1, cond, corpus, args, model_cls=BandHomeostaticLM,
                            post=post(k, x_lang, y_slot), r_lo=lo, r_hi=hi)
                z["lesion"].pop("zero_gate1")  # 12.6.138節: ゲート=1 の評価は使わない
                z.pop("threshold", None)
                r[task][key] = z
                sc = z["slot_ce"]
                print(f"seed={seed} {task} {key} gini={z['gini']:.3f} used={z['n_used']:.0f} "
                      f"top{k}={z['topk_share']:.3f} gate={z['gate_mean']:.3f} ce={sc['all']:.4f} "
                      f"b2={sc['target']:.4f} Phub={z['p_hub']:.2f} Prest={z['p_rest']:.2f} "
                      f"lesT={z['hub_lesion_d']['target']:.3f} lesO={z['hub_lesion_d']['other']:.3f}", flush=True)
        runs.append(r)
    out_s = summarize(runs, list(conds))
    verdict = judge(runs, args.eta)
    out = {"環境": env_info(), "設定": vars(args),
           **({"結合元の環境": [part["環境"] for part in parts]} if args.merge else {}),
           "要約": out_s, "判定": verdict, "シード別": runs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({"判定": verdict}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
