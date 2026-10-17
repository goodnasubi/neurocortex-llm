"""恒常性可塑性による MoE の負荷分散（12.6.127節の設計）の主実験。

補助損失なし／補助損失（係数 0.01・0.05）／恒常性の閾値（強さ η 数段階、補助損失なし）で
`RoutedCharLM` を学習し、12.6.124節と同じ指標（負荷の偏り、ハブ2個と非ハブ2個の除去）を比べる。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_homeostatic_moe --seeds 100 101 102 103 104 \
        --out results/homeostatic_moe/dev.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from ..data import CharCorpus, synthetic_corpus
from ..expert_prefetch import RoutedCharLM, train_routed_lm
from ..homeostatic_moe import HomeostaticRoutedCharLM, train_homeostatic_lm
from ..hub_load import eval_ce, expert_choices, gini, lesion_study, make_windows, max_over_mean, normalized_entropy
from ..run_env import env_info

MODES = ("zero", "reroute")


def load_stats(model, x: torch.Tensor, n_experts: int) -> tuple[np.ndarray, dict]:
    load = np.bincount(expert_choices(model, x).numpy().ravel(), minlength=n_experts).astype(np.float64)
    return load, {"gini": gini(load), "max_over_mean": max_over_mean(load), "entropy": normalized_entropy(load),
                  "n_used": int((load > 0).sum())}


def run_one(seed: int, cond: dict, corpus: CharCorpus, args: argparse.Namespace) -> dict:
    torch.manual_seed(seed)  # 条件が違っても初期値・データ順は同じ
    g = torch.Generator().manual_seed(seed)
    if cond["kind"] == "aux":
        model = RoutedCharLM(corpus.vocab_size, d=args.d, n_experts=args.n_experts)
        train_ce = train_routed_lm(model, corpus.train, args.train_steps, 32, args.seq_len, 3e-3, cond["coef"], g)
    else:
        model = HomeostaticRoutedCharLM(corpus.vocab_size, d=args.d, n_experts=args.n_experts,
                                        eta=cond["eta"], alpha=args.alpha)
        train_ce = train_homeostatic_lm(model, corpus.train, args.train_steps, 32, args.seq_len, 3e-3, g)
    model.eval()
    x, y = make_windows(corpus.val, args.seq_len, args.n_windows)
    load, out = load_stats(model, x, args.n_experts)
    out.update({"train_ce": train_ce, "load": load.tolist(), "lesion": {}})
    for mode in MODES:
        r = lesion_study(model, x, y, load, args.k, mode)
        r["ratio"] = r["d_target"] / max(r["d_random_mean"], args.eps)
        out["lesion"][mode] = r
    if cond["kind"] == "homeo":
        th = model.router.threshold
        out["threshold"] = th.tolist()
        out["threshold_range"] = float(th.max() - th.min())
        out["usage_ema"] = model.usage_ema.tolist()
        with model.without_threshold():
            _, off = load_stats(model, x, args.n_experts)
            off["base_ce"] = eval_ce(model, x, y)
        out["threshold_off"] = off
    return out


def conditions(args: argparse.Namespace) -> dict[str, dict]:
    c = {f"aux{v:g}": {"kind": "aux", "coef": v} for v in args.coefs}
    c.update({f"homeo{v:g}": {"kind": "homeo", "eta": v} for v in args.etas})
    return c


def ms(vals) -> dict:
    v = [float(a) for a in vals if not np.isnan(a)]
    if not v:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0}
    return {"mean": statistics.mean(v), "sd": statistics.stdev(v) if len(v) > 1 else 0.0, "n": len(v)}


def summarize(runs: list[dict], keys: list[str], eps: float) -> dict:
    s: dict = {}
    for key in keys:
        rs = [r[key] for r in runs]
        d = {m: ms([r[m] for r in rs]) for m in ("gini", "max_over_mean", "entropy", "n_used", "train_ce")}
        d["all_used_seeds"] = sum(r["n_used"] == len(r["load"]) for r in rs)
        for mode in MODES:
            ls = [r["lesion"][mode] for r in rs]
            dt = [l["d_target"] for l in ls]
            dr = [l["d_random_mean"] for l in ls]
            d[mode] = {"base_ce": ms([l["base_ce"] for l in ls]), "d_target": ms(dt), "d_random": ms(dr),
                       "ratio_of_means": statistics.mean(dt) / max(statistics.mean(dr), eps),
                       "ratio_per_seed": [l["ratio"] for l in ls],
                       "target_gt_random": sum(a > b for a, b in zip(dt, dr)),
                       "spearman_load_single": ms([l["spearman_load_single"] for l in ls]),
                       "hub_load_frac": ms([l["hub_load_frac"] for l in ls]),
                       "d_target_per_load": ms([l["d_target_per_load"] for l in ls]),
                       "d_random_per_load": ms([l["d_random_per_load"] for l in ls])}
        if "threshold_off" in rs[0]:
            d["threshold_range"] = ms([r["threshold_range"] for r in rs])
            d["threshold_off"] = {m: ms([r["threshold_off"][m] for r in rs])
                                  for m in ("gini", "n_used", "base_ce")}
        s[key] = d
    return s


def select_eta(s: dict, etas: list[float]) -> float:
    """12.6.127節の選び方: 比 ≤ 1.5 かつ CE ≤ 補助損失0.05 + 0.01 の最小の η。なければ比が最小の η。"""
    ref = s["aux0.05"]["zero"]["base_ce"]["mean"]
    ok = [e for e in sorted(etas) if s[f"homeo{e:g}"]["zero"]["ratio_of_means"] <= 1.5
          and s[f"homeo{e:g}"]["zero"]["base_ce"]["mean"] <= ref + 0.01]
    if ok:
        return ok[0]
    return min(etas, key=lambda e: s[f"homeo{e:g}"]["zero"]["ratio_of_means"])


def judge(runs: list[dict], s: dict, eta: float) -> dict:
    """12.6.127節の合格条件（主除去 = 出力の除去、η = η*）。"""
    key = f"homeo{eta:g}"
    g0 = [r["aux0"]["gini"] for r in runs]
    gh = [r[key]["gini"] for r in runs]
    h1 = (statistics.mean(g0) - statistics.mean(gh) >= 0.5 and all(b < a for a, b in zip(g0, gh))
          and s[key]["all_used_seeds"] == len(runs))
    h2 = s[key]["zero"]["ratio_of_means"] <= 1.5
    h3 = s[key]["zero"]["base_ce"]["mean"] <= s["aux0.05"]["zero"]["base_ce"]["mean"] + 0.01
    return {"eta": eta, "H1_崩壊を防ぐ": h1, "H2_ハブの非対称性を消す": h2, "H3_性能を損なわない": h3}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    p.add_argument("--coefs", type=float, nargs="+", default=[0.0, 0.01, 0.05])
    p.add_argument("--etas", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3])
    p.add_argument("--eta-star", type=float, default=None, help="評価用シードでは開発用で選んだ η* を渡す")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-sentences", type=int, default=3000)
    p.add_argument("--d", type=int, default=48)
    p.add_argument("--n-experts", type=int, default=8)
    p.add_argument("--train-steps", type=int, default=600)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--n-windows", type=int, default=64)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=str, default="results/homeostatic_moe/dev.json")
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    conds = conditions(args)
    runs = []
    for seed in args.seeds:
        corpus = CharCorpus(synthetic_corpus(n_sentences=args.n_sentences, seed=seed))
        r = {"seed": seed}
        for key, cond in conds.items():
            z = r[key] = run_one(seed, cond, corpus, args)
            zl, rl = z["lesion"]["zero"], z["lesion"]["reroute"]
            extra = (f" th_range={z['threshold_range']:.2f} off_gini={z['threshold_off']['gini']:.3f}"
                     f" off_ce={z['threshold_off']['base_ce']:.3f}") if "threshold" in z else ""
            print(f"seed={seed} {key} gini={z['gini']:.3f} used={z['n_used']} base={zl['base_ce']:.3f} "
                  f"dT={zl['d_target']:.3f} dR={zl['d_random_mean']:.3f} ratio={zl['ratio']:.2f} "
                  f"perload T={zl['d_target_per_load']:.3f} R={zl['d_random_per_load']:.3f} "
                  f"| reroute dT={rl['d_target']:.3f} dR={rl['d_random_mean']:.3f}{extra}", flush=True)
        runs.append(r)
    s = summarize(runs, list(conds), args.eps)
    verdict: dict = {}
    if {0.0, 0.05} <= set(args.coefs):
        eta = args.eta_star if args.eta_star is not None else select_eta(s, args.etas)
        verdict = judge(runs, s, eta)
        verdict["eta_selected_on"] = "引数 --eta-star（開発用シードで選択済み）" if args.eta_star is not None \
            else "このシードでの選択（12.6.127節の選び方）"
    out = {"環境": env_info(), "設定": vars(args), "要約": s, "判定": verdict, "シード別": runs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({"要約": s, "判定": verdict}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
