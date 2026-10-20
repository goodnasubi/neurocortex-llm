"""ハブ専門家に負荷に見合う容量を与えると、統合のボトルネックが解けるか（12.6.205節の設計）の主実験。

12.6.201〜202節の分かれた土台（`moe_modular.ModularRoutedLM`、η_split = 0.1）の上で、補助損失 0.05・一点の
目標・目標域（均等な幅、12.6.202節の構成）に、目標域で専門家の幅を負荷に比例して配り直す構成（bandcap）と、
同じ幅の並びを負荷と無関係に割り当てる対照（bandrand）を加え（`moe_hub_capacity.HubCapacityLM`）、
統合課題と対照課題の b2 の CE などを測る。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_hub_capacity --seeds 100 \
        --out results/moe_hub_capacity/dev_rev0_seed100.json
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_hub_capacity \
        --merge results/moe_hub_capacity/dev_rev0_seed10*.json --out results/moe_hub_capacity/dev_rev0.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from ..data import CharCorpus
from ..moe_hub_capacity import HubCapacityLM, allocate_widths
from ..moe_integration import integration_corpus, split_like_corpus, windows_like
from ..moe_modular import ModularRoutedLM, region_table
from ..run_env import env_info
from .run_homeostatic_moe import ms
from .run_moe_homeo_band import BANDS
from .run_moe_integration import paired_t
from .run_moe_modular import SLOTS, TASKS, post
from .run_moe_scale import run_one

CAP = {"bandcap": "load", "bandrand": "random"}
FIXED = {"hubcap": "load", "hubrand": "random", "hubeq": "equal"}  # 修正 1（12.6.205節）
NAMES = {"realloc": ("bandcap", "bandrand"), "fixed": ("hubcap", "hubrand")}


def conditions(eta: float, design: str = "realloc") -> dict[str, dict]:
    c = {"aux0.05": {"kind": "plain", "coef": 0.05, "band": "point"}}
    for name in ("point", "band"):
        c[f"{name}{eta:g}"] = {"kind": "select", "eta": eta, "band": name}
    if design == "realloc":
        for name, a in CAP.items():
            c[f"{name}{eta:g}"] = {"kind": "select", "eta": eta, "band": "band", "alloc": a}
    else:
        for name, a in FIXED.items():
            c[f"{name}{eta:g}"] = {"kind": "select", "eta": eta, "band": "point", "fixed": a}
    return c


def hub_design(n_experts: int, n_hubs: int, hub_share: float, d_ff: int, w_min: int, w_max: int,
               seed: int) -> dict[str, tuple[list[int], list[float]]]:
    """修正 1: 専門家 0..n_hubs−1 をハブとする恒常性の目標の比と、各条件の幅。

    目標の比: ハブ r_h = E·hub_share / n_hubs、残り r_n = E·(1 − hub_share) / (E − n_hubs)（和は E）。
    hubcap = 目標の比に比例した幅、hubrand = 同じ幅の並びを乱数の置換で割り当てる、hubeq = 均等な幅。
    """
    E = n_experts
    sp = [E * hub_share / n_hubs] * n_hubs + [E * (1 - hub_share) / (E - n_hubs)] * (E - n_hubs)
    cap = allocate_widths(sp, E * d_ff, w_min, w_max)
    prof = sorted(cap, reverse=True)
    perm = np.random.default_rng(seed).permutation(E)
    rand = [0] * E
    for rank, e in enumerate(perm):
        rand[int(e)] = prof[rank]
    return {"load": (cap, sp), "random": (rand, sp), "equal": ([d_ff] * E, sp)}


def post_cap(k: int, x_lang: torch.Tensor, y_slot: torch.Tensor):
    base = post(k, x_lang, y_slot)

    def f(model, x, y, out):
        base(model, x, y, out)
        if isinstance(model, HubCapacityLM):
            w = model.widths.tolist()
            load = np.asarray(out["load"][0])
            hubs = sorted(range(len(load)), key=lambda e: (-load[e], e))[:k]
            out["widths"] = w
            out["realloc_load"] = model.realloc_load
            out["hub_widths"] = [w[e] for e in hubs]
            out["hubs_widened"] = int(sum(w[e] > model.d_ff for e in hubs))
            out["active_expert_params"] = model.active_expert_params()
    return f


def summarize(runs: list[dict], keys: list[str]) -> dict:
    s: dict = {}
    for task in TASKS:
        s[task] = {}
        for key in keys:
            rs = [r[task][key] for r in runs if key in r[task]]
            d = {m: ms([z[m] for z in rs]) for m in ("gini", "n_used", "gate_mean", "topk_share", "p_hub", "p_rest",
                                                   "hub_target_share", "hub_all_share")}
            d["slot_ce"] = {m: ms([z["slot_ce"][m] for z in rs]) for m in rs[0]["slot_ce"]}
            for name in ("hub_lesion_d", "nonhub_lesion_d", "cut_all_d", "hub_cut_d", "nonhub_cut_d"):
                d[name] = {m: ms([z[name][m] for z in rs]) for m in SLOTS}
            d["all_used_seeds"] = sum(z["all_used"] for z in rs)
            if "widths" in rs[0]:
                d["hubs_widened"] = [z["hubs_widened"] for z in rs]
                d["hub_widths"] = [z["hub_widths"] for z in rs]
                d["max_width"] = ms([max(z["widths"]) for z in rs])
            s[task][key] = d
    return s


def judge(runs: list[dict], eta: float, design: str = "realloc") -> dict:
    """12.6.205節の合格条件（評価用シードを見る前に固定）。主指標 = b2 の CE（target）。
    修正 1 では C・R を構造で決めたハブの条件（hubcap・hubrand）に替える。基準は同じ。"""
    cn, rn = NAMES[design]
    C, R, B, P, A = (f"{cn}{eta:g}", f"{rn}{eta:g}", f"band{eta:g}", f"point{eta:g}", "aux0.05")
    tce = lambda r, t, k: r[t][k]["slot_ce"]["target"]  # noqa: E731
    m = statistics.mean
    n = len(runs)
    need = (7 * n + 9) // 10  # 10 シードで 7
    need_m = (8 * n + 9) // 10  # 10 シードで 8

    def inter(a, b):
        return [(tce(r, "cross", a) - tce(r, "cross", b)) - (tce(r, "within", a) - tce(r, "within", b)) for r in runs]

    def neg(v):
        return m(v) <= -0.01 and sum(x < 0 for x in v) >= need and paired_t(v) <= -2.26

    i_p, i_b, i_r, i_a = inter(C, P), inter(C, B), inter(C, R), inter(C, A)
    widened = [r["cross"][C]["hubs_widened"] for r in runs]
    widened_r = [r["cross"][R]["hubs_widened"] for r in runs]
    bridge = [r["cross"][C]["cut_all_d"]["target"] - r["within"][C]["cut_all_d"]["target"] for r in runs]
    use = [r["cross"][C]["cut_all_d"]["target"] - r["cross"][B]["cut_all_d"]["target"] for r in runs]
    cross_ce = {k: [tce(r, "cross", k) for r in runs] for k in (C, R, B, P, A)}
    return {"eta": eta, "need": need, "design": design, "C": C, "R": R,
            "M_操作（統合課題の C で学習後のハブ2個がともに広げた専門家）": sum(w == 2 for w in widened) >= need_m,
            "V_橋渡し（C で、領域をまたぐ読み取りの切断が統合課題の b2 を対照より 0.1 以上悪くする）":
                m(bridge) >= 0.1 and sum(b > 0 for b in bridge) >= need,
            "H1_交互作用（C−一点）": neg(i_p), "H2_ボトルネックの解消（C−均等な目標域）": neg(i_b),
            "H3_負荷に特有（C−R）": neg(i_r), "H4_交互作用（C−補助損失0.05）": neg(i_a),
            "H5_領域をまたぐ読み取りの利用が増える（統合課題の切断の損害 C−均等な目標域）":
                m(use) >= 0.05 and sum(x > 0 for x in use) >= need,
            "値": {"interaction_point": i_p, "interaction_point_mean": m(i_p), "interaction_point_t": paired_t(i_p),
                  "interaction_band": i_b, "interaction_band_mean": m(i_b), "interaction_band_t": paired_t(i_b),
                  "interaction_rand": i_r, "interaction_rand_mean": m(i_r), "interaction_rand_t": paired_t(i_r),
                  "interaction_aux": i_a, "interaction_aux_mean": m(i_a), "interaction_aux_t": paired_t(i_a),
                  "hubs_widened_cap": widened, "hubs_widened_rand": widened_r, "bridge_cap": bridge,
                  "cut_use_cap_minus_band": use, "cross_b2": cross_ce,
                  "cross_b2_mean": {k: m(v) for k, v in cross_ce.items()}}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    p.add_argument("--conds", type=str, nargs="+", default=None, help="一部の条件だけ学習する（既定は全条件）")
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--eta", type=float, default=0.1, help="分かれた土台の η（12.6.201節の修正 1）")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-lines", type=int, default=8000)
    p.add_argument("--n-concepts", type=int, default=8)
    p.add_argument("--d", type=int, default=48, help="使わない（run_one との互換）")
    p.add_argument("--d-mod", type=int, default=24)
    p.add_argument("--realloc-step", type=int, default=200)
    p.add_argument("--w-min", type=int, default=16)
    p.add_argument("--w-max", type=int, default=192)
    p.add_argument("--design", choices=list(NAMES), default="realloc",
                   help="realloc = 途中で負荷に応じて配り直す（rev0）、fixed = 構造で決めたハブ（修正 1）")
    p.add_argument("--n-hubs", type=int, default=2)
    p.add_argument("--hub-share", type=float, default=0.57,
                   help="修正 1: ハブの目標の負荷の割合（12.6.202節の目標域の上位2個の割合）")
    p.add_argument("--train-steps", type=int, default=600)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--n-windows", type=int, default=256)
    p.add_argument("--k-num", type=int, default=1)
    p.add_argument("--k-den", type=int, default=4)
    p.add_argument("--n-random", type=int, default=30)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", type=str, default="results/moe_hub_capacity/dev.json")
    p.add_argument("--merge", type=str, nargs="+", default=None)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    conds = conditions(args.eta, args.design)
    keys = list(conds) if args.conds is None else args.conds
    E = args.experts
    k = max(1, E * args.k_num // args.k_den)
    runs, parts = [], []
    if args.merge:
        parts = [json.loads(Path(f).read_text()) for f in args.merge]
        by_seed: dict[int, dict] = {}  # 同じシードの別ファイル（条件を分けた実行・修正後の再実行）は条件を合わせる
        for r in (r for part in parts for r in part["シード別"]):
            d = by_seed.setdefault(r["seed"], {"seed": r["seed"], **{t: {} for t in TASKS}})
            for t in TASKS:
                d[t].update(r[t])
        runs = [by_seed[s] for s in sorted(by_seed)]
        args.seeds = [r["seed"] for r in runs]
        keys = [c for c in conds if c in runs[0]["cross"]]
    for seed in ([] if args.merge else args.seeds):
        r = {"seed": seed}
        for task in TASKS:
            text, slots, langs = integration_corpus(task, n_lines=args.n_lines, n_concepts=args.n_concepts, seed=seed)
            corpus = CharCorpus(text)
            x_lang, _ = windows_like(split_like_corpus(langs), args.seq_len, args.n_windows)
            _, y_slot = windows_like(split_like_corpus(slots), args.seq_len, args.n_windows)
            r[task] = {}
            for key in keys:
                cond = conds[key]
                lo, hi = BANDS[cond["band"]]
                kw = {"model_cls": ModularRoutedLM, "region_of_token": region_table(corpus.stoi), "d_mod": args.d_mod}
                if "alloc" in cond:
                    kw.update(model_cls=HubCapacityLM, alloc=cond["alloc"], alloc_seed=seed,
                              realloc_step=args.realloc_step, w_min=args.w_min, w_max=args.w_max)
                if "fixed" in cond:
                    w, sp = hub_design(E, args.n_hubs, args.hub_share, 64, args.w_min, args.w_max, seed)[cond["fixed"]]
                    kw.update(model_cls=HubCapacityLM, widths=w, set_points=sp, w_min=args.w_min, w_max=args.w_max)
                z = run_one(seed, E, 1, cond, corpus, args, post=post_cap(k, x_lang, y_slot), r_lo=lo, r_hi=hi, **kw)
                z["lesion"].pop("zero_gate1")  # 12.6.138節: ゲート=1 の評価は使わない
                z.pop("threshold", None)
                r[task][key] = z
                sc = z["slot_ce"]
                wd = f" widths={z['widths']} hubw={z['hub_widths']}" if "widths" in z else ""
                print(f"seed={seed} {task} {key} gini={z['gini']:.3f} used={z['n_used']:.0f} "
                      f"top{k}={z['topk_share']:.3f} gate={z['gate_mean']:.3f} ce={sc['all']:.4f} "
                      f"b2={sc['target']:.4f} Phub={z['p_hub']:.2f} Prest={z['p_rest']:.2f} "
                      f"hubb2={z['hub_target_share']:.2f}/{z['hub_all_share']:.2f} "
                      f"lesT={z['hub_lesion_d']['target']:.3f} lesO={z['hub_lesion_d']['other']:.3f} "
                      f"cutT={z['cut_all_d']['target']:.3f}{wd}", flush=True)
        runs.append(r)
    out_s = summarize(runs, keys)
    verdict = judge(runs, args.eta, args.design) if set(conds) <= set(keys) else {}
    out = {"環境": env_info(), "設定": vars(args),
           **({"結合元の環境": [part["環境"] for part in parts]} if args.merge else {}),
           "要約": out_s, "判定": verdict, "シード別": runs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({"判定": verdict}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
