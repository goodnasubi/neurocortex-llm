"""MoE 専門家の予測的プリフェッチと小脳順モデルの結合（12.6.116節の設計）の主実験。

(a) 予測的中率（Top-1/Top-2）と (b) ディスク遅延の模擬下での実効速度（壁時計時間・
待ち回数・無駄 I/O）を別々に測る。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_expert_prefetch --seeds 100 101 102 103 104 \
        --out results/expert_prefetch/dev.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..data import CharCorpus, synthetic_corpus
from ..expert_prefetch import (ExpertDiskStore, RoutedCharLM, hit_rates, make_predictor,
                               routing_trace, simulate_prefetch, train_routed_lm)
from ..run_env import env_info

PREDICTORS = ("last", "freq", "markov", "cerebellum", "cerebellum_onehot")
GATED = ("markov_gated", "cerebellum_gated")  # 修正3: 確信度ゲート付き（prob >= gate のときだけ先読み）
SPEED_CONDITIONS = ("none",) + PREDICTORS + GATED + ("oracle",)


def config_key(capacity: int, k_pf: int, io_ms: float, main_io_ms: float) -> str:
    key = f"cap{capacity}_k{k_pf}"
    return key if io_ms == main_io_ms else f"{key}_io{io_ms:g}"


def run_seed(seed: int, args: argparse.Namespace) -> dict:
    torch.manual_seed(seed)
    corpus = CharCorpus(synthetic_corpus(n_sentences=args.n_sentences, seed=seed))
    model = RoutedCharLM(corpus.vocab_size, d=args.d, n_experts=args.n_experts)
    g = torch.Generator().manual_seed(seed)
    ce = train_routed_lm(model, corpus.train, args.train_steps, 32, 64, 3e-3, args.balance_coef, g)
    data = corpus.val[: args.trace_len]
    choices, hidden = routing_trace(model, data)
    usage = np.bincount(choices, minlength=args.n_experts) / len(choices)

    out: dict = {"seed": seed, "train_ce": ce, "expert_usage": usage.tolist(),
                 "same_as_prev_rate": float((choices[1:] == choices[:-1]).mean()),
                 "hit": {}, "speed": {}}
    for name in PREDICTORS:
        out["hit"][name] = hit_rates(choices, hidden, make_predictor(name, args.n_experts, args.d, args.cb_lr))

    weights = [np.concatenate([model.w1[e].detach().numpy().ravel(), model.w2[e].detach().numpy().ravel()])
               for e in range(args.n_experts)]
    sc, sh = choices[: args.speed_len], hidden[: args.speed_len]
    for capacity, k_pf, io_ms in args.configs:
        key = config_key(capacity, k_pf, io_ms, args.io_ms)
        out["speed"][key] = {}
        for cond in SPEED_CONDITIONS:
            tmp = Path(tempfile.mkdtemp(prefix="expert_prefetch_"))
            try:
                store = ExpertDiskStore(weights, tmp, capacity, io_ms)
                base = cond.removesuffix("_gated")
                pred = make_predictor(base, args.n_experts, args.d, args.cb_lr) if base in PREDICTORS else None
                res = simulate_prefetch(sc, sh, store, pred, k_pf, args.compute_ms, oracle=(cond == "oracle"),
                                        gate=args.gate if cond in GATED else None)
                out["speed"][key][cond] = asdict(res)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    return out


def summarize(runs: list[dict], keys) -> dict:
    def ms(vals):
        return {"mean": statistics.mean(vals), "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0}

    s: dict = {"hit": {}, "speed": {}}
    for name in PREDICTORS:
        s["hit"][name] = {k: ms([r["hit"][name][k] for r in runs]) for k in ("top1", "top2")}
    for key in keys:
        s["speed"][key] = {}
        for cond in SPEED_CONDITIONS:
            s["speed"][key][cond] = {m: ms([r["speed"][key][cond][m] for r in runs])
                                     for m in ("wall_s", "stalls", "stall_s", "demand_loads", "prefetch_loads", "wasted_loads", "wasted_bytes", "predictor_s")}
    return s


def judge(runs: list[dict], s: dict, main_key: str) -> dict:
    """12.6.116節の合格条件。(a) と (b) は独立に判定する。

    (b) の小脳条件は修正3で確信度ゲート付き（cerebellum_gated）に定めた。ゲートなしも参考に記録する。
    """
    cb = [r["hit"]["cerebellum"]["top1"] for r in runs]
    simple = [max(r["hit"]["last"]["top1"], r["hit"]["freq"]["top1"]) for r in runs]
    margin = statistics.mean(cb) - max(s["hit"]["last"]["top1"]["mean"], s["hit"]["freq"]["top1"]["mean"])
    a = margin >= 0.05 and all(c > b for c, b in zip(cb, simple))
    sp = s["speed"][main_key]
    wall = {c: sp[c]["wall_s"]["mean"] for c in SPEED_CONDITIONS}
    reduction = 1 - wall["cerebellum_gated"] / wall["none"]
    b = (reduction >= 0.10 and wall["cerebellum_gated"] < min(wall["last"], wall["freq"])
         and sp["cerebellum_gated"]["wasted_loads"]["mean"] <= sp["freq"]["wasted_loads"]["mean"])
    # 参考: 予測器の計算時間を除いた壁時計時間（予測器が専門家の計算と並列に動けた場合の目安）
    wall_wo_pred = {c: wall[c] - sp[c]["predictor_s"]["mean"] for c in SPEED_CONDITIONS}
    return {"a_pass": a, "a_margin_top1": margin, "b_pass": b, "b_reduction_vs_none": reduction,
            "b_wall_mean": wall, "参考_予測器時間を除く壁時計": wall_wo_pred}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    p.add_argument("--n-sentences", type=int, default=3000)
    p.add_argument("--d", type=int, default=48)
    p.add_argument("--n-experts", type=int, default=8)
    p.add_argument("--train-steps", type=int, default=600)
    p.add_argument("--balance-coef", type=float, default=0.05)
    p.add_argument("--trace-len", type=int, default=8000)
    p.add_argument("--speed-len", type=int, default=1500)
    p.add_argument("--cb-lr", type=float, default=0.2)
    p.add_argument("--io-ms", type=float, default=2.0)
    p.add_argument("--compute-ms", type=float, default=1.0)
    p.add_argument("--gate", type=float, default=0.5,
                   help="確信度ゲート。的中の利得と外れの損失がほぼ等しいとした損益分岐点 0.5")
    p.add_argument("--io-sweep", type=float, nargs="*", default=[0.5, 1.0],
                   help="副次分析: 主構成 (2,1) で io_ms を変えた計測（判定には使わない）")
    p.add_argument("--out", type=str, default="results/expert_prefetch/dev.json")
    args = p.parse_args()
    args.configs = [(2, 1, args.io_ms), (3, 2, args.io_ms)] + [(2, 1, io) for io in args.io_sweep]

    runs = []
    for seed in args.seeds:
        r = run_seed(seed, args)
        runs.append(r)
        print(f"seed {seed}: ce={r['train_ce']:.3f} same={r['same_as_prev_rate']:.3f} "
              + " ".join(f"{k}={v['top1']:.3f}/{v['top2']:.3f}" for k, v in r["hit"].items()), flush=True)
        for key, conds in r["speed"].items():
            print(f"  {key}: " + " ".join(f"{c}={v['wall_s']:.2f}s/st{v['stalls']}/w{v['wasted_loads']}"
                                          for c, v in conds.items()), flush=True)
    s = summarize(runs, [config_key(c, k, io, args.io_ms) for c, k, io in args.configs])
    verdict = judge(runs, s, "cap2_k1")
    print(json.dumps(verdict, ensure_ascii=False, indent=1))
    cfg = {k: v for k, v in vars(args).items()}
    out = {"環境": env_info(), "設定": cfg, "要約": s, "判定": verdict, "シード別": runs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
