"""領域ごとに分かれた土台で、専門家だけが領域をまたげる構成でのハブの利点（12.6.201節の設計）の主実験。

12.6.199節の統合課題・対照課題を、共有の土台（12.6.199節の `BandHomeostaticLM`）と分かれた土台
（`moe_modular.ModularRoutedLM`。言語ごとの埋め込み・GRU・出力層、専門家だけが全モジュールを読む）の上で、
補助損失 0.05・一点の目標・ハブを許す目標域の3条件で学習し、続き（b2）の CE・参加係数・ハブへの経路の偏り・
ハブ除去と領域をまたぐ読み取りの切断の損害を測る。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_modular --seeds 100 \
        --out results/moe_modular/dev_rev0_seed100.json
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_modular \
        --merge results/moe_modular/dev_rev0_seed10*.json --out results/moe_modular/dev_rev0.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from ..data import CharCorpus
from ..moe_homeo_band import BandHomeostaticLM, topk_share
from ..moe_integration import (LANGS, hub_participation, hub_target_share, integration_corpus, participation,
                               slot_ce, split_like_corpus, windows_like)
from ..moe_modular import ModularRoutedLM, region_table
from ..moe_scale import hub_masks
from ..run_env import env_info
from .run_homeostatic_moe import ms
from .run_moe_homeo_band import BANDS
from .run_moe_integration import paired_t
from .run_moe_scale import run_one

TASKS = ("cross", "within")
SUBSTRATES = ("shared", "split")
SLOTS = ("target", "other", "all")


def conditions(args: argparse.Namespace) -> dict[str, dict]:
    c = {f"aux{v:g}": {"kind": "plain", "coef": v, "band": "point"} for v in args.coefs}
    for name in args.bands:
        c[f"{name}{args.eta:g}"] = {"kind": "select", "eta": args.eta, "band": name}
    return c


def _d(after: dict, before: dict) -> dict:
    return {s: after[s] - before[s] for s in SLOTS}


def post(k: int, x_lang: torch.Tensor, y_slot: torch.Tensor):
    def f(model, x, y, out):
        load = np.asarray(out["load"][0])
        choice = model.run(x)["choice"][0].numpy()
        out["topk_share"] = topk_share(load, k)
        base = out["slot_ce"] = slot_ce(model, x, y, y_slot)
        p = participation(choice, x_lang.numpy(), model.n_experts, len(LANGS))
        out["participation"] = [None if np.isnan(v) else float(v) for v in p]
        out["p_hub"], out["p_rest"] = hub_participation(load, p, k)
        out["hub_target_share"], out["hub_all_share"] = hub_target_share(choice, y_slot.numpy(), load, k)
        hub, nonhub = hub_masks(load[None], k, 10 ** 6, np.random.default_rng(0))  # 非ハブ k 個の全組合せ
        out["hub_lesion_d"] = _d(slot_ce(model, x, y, y_slot, disabled=hub, mode="zero"), base)
        les = [_d(slot_ce(model, x, y, y_slot, disabled=m, mode="zero"), base) for m in nonhub]
        out["nonhub_lesion_d"] = {s: float(np.mean([z[s] for z in les])) for s in SLOTS}
        if isinstance(model, ModularRoutedLM):
            out["cut_all_d"] = _d(slot_ce(model, x, y, y_slot, cut_remote=True), base)
            out["hub_cut_d"] = _d(slot_ce(model, x, y, y_slot, cut_remote=hub[0]), base)
            cuts = [_d(slot_ce(model, x, y, y_slot, cut_remote=m[0]), base) for m in nonhub]
            out["nonhub_cut_d"] = {s: float(np.mean([z[s] for z in cuts])) for s in SLOTS}
    return f


def summarize(runs: list[dict], keys: list[str]) -> dict:
    s: dict = {}
    for task in TASKS:
        s[task] = {}
        for key in keys:
            rs = [r[task][key] for r in runs]
            d = {m: ms([z[m] for z in rs]) for m in ("gini", "n_used", "gate_mean", "topk_share", "p_hub", "p_rest",
                                                   "hub_target_share", "hub_all_share")}
            d["slot_ce"] = {m: ms([z["slot_ce"][m] for z in rs]) for m in rs[0]["slot_ce"]}
            for name in ("hub_lesion_d", "nonhub_lesion_d", "cut_all_d", "hub_cut_d", "nonhub_cut_d"):
                if name in rs[0]:
                    d[name] = {m: ms([z[name][m] for z in rs]) for m in SLOTS}
            d["all_used_seeds"] = sum(z["all_used"] for z in rs)
            s[task][key] = d
    return s


def judge(runs: list[dict], eta: float) -> dict:
    """12.6.201節の合格条件（評価用シードを見る前に固定）。主指標 = b2 の CE（target）。"""
    B, P, A = f"split/band{eta:g}", f"split/point{eta:g}", "split/aux0.05"
    Bs, Ps = f"shared/band{eta:g}", f"shared/point{eta:g}"
    tce = lambda r, t, k: r[t][k]["slot_ce"]["target"]  # noqa: E731

    def inter(b, p):
        return [(tce(r, "cross", b) - tce(r, "cross", p)) - (tce(r, "within", b) - tce(r, "within", p)) for r in runs]

    n = len(runs)
    need = (7 * n + 9) // 10  # 10 シードで 7
    i_split, i_aux, i_shared = inter(B, P), inter(B, A), inter(Bs, Ps)
    d3 = [a - b for a, b in zip(i_split, i_shared)]
    split_keys = [k for k in runs[0]["cross"] if k.startswith("split/")]
    gap = {k: statistics.mean([tce(r, "cross", k) - tce(r, "within", k) for r in runs]) for k in runs[0]["cross"]}
    bridge = [r["cross"][B]["cut_all_d"]["target"] - r["within"][B]["cut_all_d"]["target"] for r in runs]
    dp = [r["cross"][B]["p_hub"] - r["cross"][B]["p_rest"] for r in runs]
    rel = lambda z: z["hub_lesion_d"]["target"] / max(z["hub_lesion_d"]["other"], 1e-4)  # noqa: E731
    r_cross, r_within = [rel(r["cross"][B]) for r in runs], [rel(r["within"][B]) for r in runs]
    path = [r["cross"][B]["hub_target_share"] - r["cross"][B]["hub_all_share"] for r in runs]
    m = statistics.mean

    def neg(v):
        return m(v) <= -0.01 and sum(x < 0 for x in v) >= need

    v1 = m(bridge) >= 0.1 and sum(b > 0 for b in bridge) >= need
    h1 = neg(i_split) and paired_t(i_split) <= -2.26
    h2 = neg(i_aux) and paired_t(i_aux) <= -2.26
    h3 = neg(d3)
    h4 = m(dp) >= 0.1 and sum(x > 0 for x in dp) >= need
    h5 = sum(a > b for a, b in zip(r_cross, r_within)) >= need and m(r_cross) > m(r_within)
    h6 = m(path) >= 0.05 and sum(x > 0 for x in path) >= need
    hub_cut = [r["cross"][B]["hub_cut_d"]["target"] for r in runs]
    nonhub_cut = [r["cross"][B]["nonhub_cut_d"]["target"] for r in runs]
    return {"eta": eta, "need": need,
            "V_橋渡し（分かれた土台・目標域で、領域をまたぐ読み取りの切断が統合課題の b2 を対照より 0.1 以上悪くする）": v1,
            "H1_交互作用（分かれた土台、目標域−一点）": h1, "H2_交互作用（分かれた土台、目標域−補助損失0.05）": h2,
            "H3_交互作用が共有の土台より強い": h3, "H4_統合課題でハブの参加係数が高い": h4,
            "H5_ハブ除去の損害が統合課題で b2 に偏る": h5, "H6_b2 の位置がハブに偏る": h6,
            "参考_統合課題の難しさ（全条件で統合課題の b2 の CE が対照より 0.03 以上高い）":
                all(g > 0.03 for k, g in gap.items() if k in split_keys),
            "値": {"interaction_split": i_split, "interaction_split_mean": m(i_split),
                  "interaction_split_t": paired_t(i_split), "interaction_aux": i_aux,
                  "interaction_aux_mean": m(i_aux), "interaction_aux_t": paired_t(i_aux),
                  "interaction_shared": i_shared, "interaction_shared_mean": m(i_shared),
                  "interaction_shared_t": paired_t(i_shared), "split_minus_shared": d3,
                  "split_minus_shared_mean": m(d3), "split_minus_shared_t": paired_t(d3),
                  "gap_cross_minus_within": gap, "bridge": bridge, "p_hub_minus_rest_cross": dp,
                  "lesion_target_over_other_cross": r_cross, "lesion_target_over_other_within": r_within,
                  "hub_path_bias_cross": path, "hub_cut_target_cross": hub_cut,
                  "nonhub_cut_target_cross": nonhub_cut}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    p.add_argument("--substrates", type=str, nargs="+", default=list(SUBSTRATES))
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--coefs", type=float, nargs="+", default=[0.05])
    p.add_argument("--bands", type=str, nargs="+", default=["point", "band"])
    p.add_argument("--eta", type=float, default=0.01, help="12.6.195節の開発用シードで選んだ η*")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-lines", type=int, default=8000)
    p.add_argument("--n-concepts", type=int, default=8)
    p.add_argument("--d", type=int, default=48, help="共有の土台の幅")
    p.add_argument("--d-mod", type=int, default=24, help="分かれた土台のモジュールの幅")
    p.add_argument("--train-steps", type=int, default=600)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--n-windows", type=int, default=256)
    p.add_argument("--k-num", type=int, default=1)
    p.add_argument("--k-den", type=int, default=4)
    p.add_argument("--n-random", type=int, default=30)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", type=str, default="results/moe_modular/dev.json")
    p.add_argument("--merge", type=str, nargs="+", default=None)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    conds = conditions(args)
    keys = [f"{sub}/{c}" for sub in args.substrates for c in conds]
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
            for key in keys:
                sub, cname = key.split("/")
                cond = conds[cname]
                lo, hi = BANDS[cond["band"]]
                kw = {"model_cls": BandHomeostaticLM} if sub == "shared" else \
                    {"model_cls": ModularRoutedLM, "region_of_token": region_table(corpus.stoi), "d_mod": args.d_mod}
                z = run_one(seed, E, 1, cond, corpus, args, post=post(k, x_lang, y_slot), r_lo=lo, r_hi=hi, **kw)
                z["lesion"].pop("zero_gate1")  # 12.6.138節: ゲート=1 の評価は使わない
                z.pop("threshold", None)
                r[task][key] = z
                sc = z["slot_ce"]
                extra = (f" cutT={z['cut_all_d']['target']:.3f} hubcutT={z['hub_cut_d']['target']:.3f} "
                         f"nhcutT={z['nonhub_cut_d']['target']:.3f}") if "cut_all_d" in z else ""
                print(f"seed={seed} {task} {key} gini={z['gini']:.3f} used={z['n_used']:.0f} "
                      f"top{k}={z['topk_share']:.3f} gate={z['gate_mean']:.3f} ce={sc['all']:.4f} "
                      f"b2={sc['target']:.4f} Phub={z['p_hub']:.2f} Prest={z['p_rest']:.2f} "
                      f"hubb2={z['hub_target_share']:.2f}/{z['hub_all_share']:.2f} "
                      f"lesT={z['hub_lesion_d']['target']:.3f} lesO={z['hub_lesion_d']['other']:.3f}{extra}",
                      flush=True)
        runs.append(r)
    out_s = summarize(runs, keys)
    verdict = judge(runs, args.eta) if set(args.substrates) == set(SUBSTRATES) else {}
    out = {"環境": env_info(), "設定": vars(args),
           **({"結合元の環境": [part["環境"] for part in parts]} if args.merge else {}),
           "要約": out_s, "判定": verdict, "シード別": runs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({"判定": verdict}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
