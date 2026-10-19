"""MoE 専門家の予測的プリフェッチの実効速度: 予測器の並列化・複数の読み込み経路・大きな専門家
（12.6.189節の設計）。

12.6.120節の構成（8専門家・1層のルーティング列、小脳の順モデルによる予測）を使い、
{小さな専門家, 大きな専門家} × {予測器を直列, 並列} × {読み込み経路 1本, 2本} の 8 構成で
壁時計時間を測る。時間の揺らぎに対して、各条件を `--reps` 回（条件の順を毎回並べ替えて）測り、
シードごとに中央値を取る。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_expert_prefetch_parallel --seeds 100 101 102 103 104 \
        --out results/expert_prefetch_parallel/dev.json
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..data import CharCorpus, synthetic_corpus
from ..expert_prefetch import (MultiPathExpertStore, RoutedCharLM, hit_rates, make_predictor,
                               routing_trace, simulate_prefetch_v2, train_routed_lm)
from ..run_env import env_info

CONDITIONS = ("none", "last", "freq", "cerebellum", "cerebellum_gated", "oracle")
METRICS = ("wall_s", "stalls", "stall_s", "demand_loads", "prefetch_loads", "wasted_loads", "wasted_bytes",
           "predictor_s")


def config_list(args) -> list[dict]:
    out = []
    for size in ("small", "large"):
        for par in (False, True):
            for paths in (1, 2):
                out.append({"key": f"{size}_{'par' if par else 'ser'}_p{paths}", "size": size,
                            "parallel": par, "paths": paths})
    return out


def _warm(predictor, choices: np.ndarray, hidden: np.ndarray) -> None:
    """速度を測る区間の前の列で予測器を学習させておく（順位は使わず observe だけ）。"""
    for t in range(len(choices) - 1):
        predictor.observe(int(choices[t]), hidden[t], int(choices[t + 1]))


def run_seed(seed: int, args) -> dict:
    torch.manual_seed(seed)
    corpus = CharCorpus(synthetic_corpus(n_sentences=args.n_sentences, seed=seed))
    model = RoutedCharLM(corpus.vocab_size, d=args.d, n_experts=args.n_experts)
    g = torch.Generator().manual_seed(seed)
    ce = train_routed_lm(model, corpus.train, args.train_steps, 32, 64, 3e-3, args.balance_coef, g)
    choices, hidden = routing_trace(model, corpus.val[: args.trace_len])
    w0, w1 = args.warm_len, args.warm_len + args.speed_len
    wc, wh = choices[: w0 + 1], hidden[: w0 + 1]  # 区間の最初の遷移 (w0 → w0+1) は測る側で学ぶ
    sc, sh = choices[w0:w1], hidden[w0:w1]

    out: dict = {"seed": seed, "train_ce": ce,
                 "same_as_prev_rate": float((sc[1:] == sc[:-1]).mean()), "hit_speed_segment": {}, "speed": {}}
    for name in ("last", "freq", "cerebellum"):
        p = make_predictor(name, args.n_experts, args.d, args.cb_lr)
        _warm(p, wc, wh)
        out["hit_speed_segment"][name] = hit_rates(sc, sh, p)

    rng = np.random.default_rng(seed)
    small = [np.concatenate([model.w1[e].detach().numpy().ravel(), model.w2[e].detach().numpy().ravel()])
             for e in range(args.n_experts)]
    # 大きな専門家: (2, d_big, d_ff_big) の float32（既定 4 MB）。ルーティング列は小さな LM のもの
    large = [rng.standard_normal((2, args.big_d, args.big_dff)).astype(np.float32) * 0.02
             for _ in range(args.n_experts)]
    x_big = rng.standard_normal((args.big_batch, args.big_d)).astype(np.float32)
    order_rng = random.Random(seed)
    cfgs = config_list(args)
    raw: dict = {c["key"]: {cond: [] for cond in CONDITIONS} for c in cfgs}
    for rep in range(args.reps):
        jobs = [(c, cond) for c in cfgs for cond in CONDITIONS]
        order_rng.shuffle(jobs)  # 時間の揺らぎ（ドリフト）が条件に偏らないよう、毎回順を並べ替える
        for c, cond in jobs:
            big = c["size"] == "large"
            tmp = Path(tempfile.mkdtemp(prefix="expert_prefetch_par_"))
            try:
                store = MultiPathExpertStore(large if big else small, tmp, args.capacity,
                                             args.big_io_ms if big else args.io_ms, n_paths=c["paths"])
                base = cond.removesuffix("_gated")
                pred = None
                if base in ("last", "freq", "cerebellum"):
                    pred = make_predictor(base, args.n_experts, args.d, args.cb_lr)
                    _warm(pred, wc, wh)
                res = simulate_prefetch_v2(sc, sh, store, pred, args.k_pf,
                                           args.big_compute_ms if big else args.compute_ms,
                                           oracle=(cond == "oracle"),
                                           gate=args.gate if cond.endswith("_gated") else None,
                                           parallel_predictor=c["parallel"], x_batch=x_big if big else None)
                raw[c["key"]][cond].append(asdict(res))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    for c in cfgs:
        out["speed"][c["key"]] = {
            cond: {m: statistics.median(r[m] for r in raw[c["key"]][cond]) for m in METRICS}
            | {"wall_s_reps": [r["wall_s"] for r in raw[c["key"]][cond]]}
            for cond in CONDITIONS}
    return out


def _mean_sd(vals):
    return {"mean": statistics.mean(vals), "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0}


def summarize(runs: list[dict], cfgs: list[dict]) -> dict:
    s: dict = {"hit_speed_segment": {n: {k: _mean_sd([r["hit_speed_segment"][n][k] for r in runs])
                                         for k in ("top1", "top2")} for n in ("last", "freq", "cerebellum")},
               "speed": {}}
    for c in cfgs:
        key = c["key"]
        s["speed"][key] = {cond: {m: _mean_sd([r["speed"][key][cond][m] for r in runs]) for m in METRICS}
                           for cond in CONDITIONS}
        for cond in CONDITIONS:
            red = [1 - r["speed"][key][cond]["wall_s"] / r["speed"][key]["none"]["wall_s"] for r in runs]
            s["speed"][key][cond]["reduction_vs_none"] = _mean_sd(red)
            s["speed"][key][cond]["reduction_per_seed"] = red
    return s


def _speed_pass(runs, s, key: str, cb: str) -> dict:
    """12.6.189節の合格条件（H1・H2 で共通）。"""
    sp = s["speed"][key]
    red = sp[cb]["reduction_vs_none"]["mean"]
    wall_cb = sp[cb]["wall_s"]["mean"]
    simple = min(sp["last"]["wall_s"]["mean"], sp["freq"]["wall_s"]["mean"])
    n_pos = sum(x > 0 for x in sp[cb]["reduction_per_seed"])
    wasted_ok = sp[cb]["wasted_loads"]["mean"] <= sp["freq"]["wasted_loads"]["mean"]
    oracle_red = sp["oracle"]["reduction_vs_none"]["mean"]
    return {"cond": cb, "reduction": red, "wall_cb": wall_cb, "wall_simple_min": simple,
            "seeds_faster_than_none": n_pos, "wasted_ok": wasted_ok,
            "oracle_reduction": oracle_red, "share_of_oracle_gain": red / oracle_red if oracle_red > 0 else None,
            "pass": red >= 0.10 and wall_cb < simple and n_pos >= 4 and wasted_ok}


def judge(runs, s, cb_small: str, cb_large: str) -> dict:
    h1 = _speed_pass(runs, s, "small_par_p2", cb_small)
    h2 = _speed_pass(runs, s, "large_par_p2", cb_large)
    sp = s["speed"]
    # H3a 並列化の寄与（小・1経路）: 直列→並列の短縮が、直列での予測器の時間の半分以上
    ser, par = sp["small_ser_p1"][cb_small], sp["small_par_p1"][cb_small]
    h3a_gain = ser["wall_s"]["mean"] - par["wall_s"]["mean"]
    h3a = {"gain_s": h3a_gain, "predictor_s_serial": ser["predictor_s"]["mean"],
           "pass": h3a_gain >= 0.5 * ser["predictor_s"]["mean"]}
    # H3b 経路の寄与（小・並列）: none 比の短縮率が 1本→2本で 5 ポイント以上増える
    r1 = sp["small_par_p1"][cb_small]["reduction_vs_none"]["mean"]
    r2 = sp["small_par_p2"][cb_small]["reduction_vs_none"]["mean"]
    h3b = {"red_p1": r1, "red_p2": r2, "pass": r2 - r1 >= 0.05}
    # H3c 大きさ（並列・2経路）: 大きな専門家の短縮率が小さな専門家以上
    h3c = {"red_small": h1["reduction"], "red_large": h2["reduction"], "pass": h2["reduction"] >= h1["reduction"]}
    # 整合の確認: none は予測器の直列・並列で変わらないはず（同じ処理）
    sanity = {k: sp[k]["none"]["wall_s"]["mean"] for k in sp}
    return {"H1_small": h1, "H2_large": h2, "H3a_parallel": h3a, "H3b_paths": h3b, "H3c_size": h3c,
            "none_wall_by_config": sanity}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    p.add_argument("--n-sentences", type=int, default=3000)
    p.add_argument("--d", type=int, default=48)
    p.add_argument("--n-experts", type=int, default=8)
    p.add_argument("--train-steps", type=int, default=600)
    p.add_argument("--balance-coef", type=float, default=0.05)
    p.add_argument("--trace-len", type=int, default=8000)
    p.add_argument("--warm-len", type=int, default=2000)
    p.add_argument("--speed-len", type=int, default=600)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--cb-lr", type=float, default=0.2)
    p.add_argument("--capacity", type=int, default=2)
    p.add_argument("--k-pf", type=int, default=1)
    p.add_argument("--gate", type=float, default=0.5)
    p.add_argument("--io-ms", type=float, default=2.0)
    p.add_argument("--compute-ms", type=float, default=1.0)
    p.add_argument("--big-d", type=int, default=512)
    p.add_argument("--big-dff", type=int, default=1024)
    p.add_argument("--big-batch", type=int, default=32)
    p.add_argument("--big-io-ms", type=float, default=4.0)
    p.add_argument("--big-compute-ms", type=float, default=2.0)
    p.add_argument("--cb-small", type=str, default="cerebellum_gated",
                   help="H1 の判定に使う小脳の条件（開発用シードで確定）")
    p.add_argument("--cb-large", type=str, default="cerebellum_gated",
                   help="H2 の判定に使う小脳の条件（開発用シードで確定）")
    p.add_argument("--out", type=str, default="results/expert_prefetch_parallel/dev.json")
    args = p.parse_args()
    torch.set_num_threads(1)  # 学習の非決定性を抑え、計測中のスレッドの取り合いを減らす

    cfgs = config_list(args)
    runs = []
    for seed in args.seeds:
        r = run_seed(seed, args)
        runs.append(r)
        print(f"seed {seed}: ce={r['train_ce']:.3f} same={r['same_as_prev_rate']:.3f} "
              + " ".join(f"{k}={v['top1']:.3f}" for k, v in r["hit_speed_segment"].items()), flush=True)
        for key, conds in r["speed"].items():
            print(f"  {key}: " + " ".join(f"{c}={v['wall_s']:.2f}s/st{v['stalls']:.0f}/w{v['wasted_loads']:.0f}"
                                          for c, v in conds.items()), flush=True)
    s = summarize(runs, cfgs)
    verdict = judge(runs, s, args.cb_small, args.cb_large)
    print(json.dumps(verdict, ensure_ascii=False, indent=1))
    out = {"環境": env_info(), "設定": vars(args), "要約": s, "判定": verdict, "シード別": runs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
