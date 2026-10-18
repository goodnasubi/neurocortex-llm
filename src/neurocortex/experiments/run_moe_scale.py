"""恒常性 MoE の規模の変化（12.6.137節の設計）の主実験。

専門家の数 E ∈ {8, 16, 32} × MoE 層の数 L ∈ {1, 2} の各規模で、補助損失なし／補助損失 0.05／
ゲート込みの恒常性（12.6.127節、η=0.1）／選択専用の恒常性（12.6.133節、η 数段階）を学習し、
12.6.124・128・134節の指標と、ゲートの大きさを除いた除去の悪化（ゲートの質量あたり・ゲートを 1 に固定）を測る。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_scale --seeds 100 \
        --out results/moe_scale/dev_seed100.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from ..data import CharCorpus, synthetic_corpus
from ..hub_load import make_windows
from ..moe_scale import ScaledRoutedCharLM, eval_ce, lesion_study, load_stats, train_scaled
from ..run_env import env_info
from .run_homeostatic_moe import MODES, ms

PER_LESION = ("d_target_per_load", "d_random_per_load", "d_target_per_gate", "d_random_per_gate")


def scales(args: argparse.Namespace) -> list[tuple[int, int]]:
    return [(e, l) for l in args.layers for e in args.experts]


def conditions(args: argparse.Namespace) -> dict[str, dict]:
    c = {f"aux{v:g}": {"kind": "plain", "coef": v} for v in args.coefs}
    c[f"gate{args.gate_eta:g}"] = {"kind": "gate", "eta": args.gate_eta}
    c.update({f"sel{v:g}": {"kind": "select", "eta": v} for v in args.sel_etas})
    return c


def run_one(seed: int, n_experts: int, n_layers: int, cond: dict, corpus: CharCorpus,
            args: argparse.Namespace, model_cls: type = ScaledRoutedCharLM, steps: int | None = None,
            **model_kw) -> dict:
    """model_cls・steps・model_kw は 12.6.141節（`run_moe_absorb.py`）の拡張。既定では 12.6.137節のまま。"""
    steps = args.train_steps if steps is None else steps
    torch.manual_seed(seed)  # 条件が違っても初期値・データ順は同じ
    g = torch.Generator().manual_seed(seed)
    model = model_cls(corpus.vocab_size, d=args.d, n_experts=n_experts, n_layers=n_layers,
                      kind=cond["kind"], eta=cond.get("eta", 0.0), alpha=args.alpha, **model_kw)
    train_ce = train_scaled(model, corpus.train, steps, 32, args.seq_len, 3e-3, g, cond.get("coef", 0.0))
    model.eval()
    x, y = make_windows(corpus.val, args.seq_len, args.n_windows)
    k = max(1, n_experts * args.k_num // args.k_den)
    load, out = load_stats(model, x)
    out.update({"train_ce": train_ce, "load": load.tolist(), "k": k, "lesion": {}})
    for mode in MODES:
        r = lesion_study(model, x, y, load, k, mode, args.n_random, seed)
        r["ratio"] = r["d_target"] / max(r["d_random_mean"], args.eps)
        out["lesion"][mode] = r
    r = lesion_study(model, x, y, load, k, "zero", args.n_random, seed, gate_one=True)
    r["ratio"] = r["d_target"] / max(r["d_random_mean"], args.eps)
    out["lesion"]["zero_gate1"] = r
    if cond["kind"] != "plain":
        th = model.threshold
        out["threshold"] = th.tolist()
        out["threshold_range"] = float((th.max(1).values - th.min(1).values).mean())  # 層ごとの幅の平均
        _, off = load_stats(model, x, use_threshold=False)
        off["base_ce"] = eval_ce(model, x, y, use_threshold=False)
        off["d_ce"] = off["base_ce"] - out["lesion"]["zero"]["base_ce"]
        out["threshold_off"] = off
        if hasattr(model, "threshold_state"):  # 12.6.141節 (c): 効きを掛ける前の閾値の状態 θ
            st = model.threshold_state
            out["threshold_state_range"] = float((st.max(1).values - st.min(1).values).mean())
    return out


def summarize(runs: list[dict], keys: list[str], eps: float) -> dict:
    s: dict = {}
    for key in keys:
        rs = [r[key] for r in runs]
        d = {m: ms([r[m] for r in rs]) for m in ("gini", "max_over_mean", "entropy", "n_used", "train_ce",
                                                 "gate_mean", "shifted_frac")}
        d["all_used_seeds"] = sum(r["all_used"] for r in rs)
        for mode in (*MODES, "zero_gate1"):
            ls = [r["lesion"][mode] for r in rs]
            dt = [l["d_target"] for l in ls]
            dr = [l["d_random_mean"] for l in ls]
            d[mode] = {"base_ce": ms([l["base_ce"] for l in ls]), "d_target": ms(dt), "d_random": ms(dr),
                       "ratio_of_means": statistics.mean(dt) / max(statistics.mean(dr), eps),
                       "ratio_per_seed": [l["ratio"] for l in ls],
                       "target_gt_random": sum(a > b for a, b in zip(dt, dr)),
                       **{m: ms([l[m] for l in ls]) for m in PER_LESION}}
        if "threshold_off" in rs[0]:
            d["threshold_range"] = ms([r["threshold_range"] for r in rs])
            d["threshold_range_per_seed"] = [r["threshold_range"] for r in rs]
            d["threshold_off"] = {m: ms([r["threshold_off"][m] for r in rs]) for m in ("gini", "n_used", "base_ce", "d_ce")}
        s[key] = d
    return s


def select_eta(s: dict, etas: list[float]) -> float:
    """12.6.127・133節と同じ選び方: 比 ≤ 1.5 かつ CE ≤ 補助損失0.05 + 0.01 の最小の η。なければ比が最小の η。"""
    ref = s["aux0.05"]["zero"]["base_ce"]["mean"]
    ok = [e for e in sorted(etas) if s[f"sel{e:g}"]["zero"]["ratio_of_means"] <= 1.5
          and s[f"sel{e:g}"]["zero"]["base_ce"]["mean"] <= ref + 0.01]
    if ok:
        return ok[0]
    return min(etas, key=lambda e: s[f"sel{e:g}"]["zero"]["ratio_of_means"])


def judge(runs: list[dict], s: dict, eta: float, gate_key: str) -> dict:
    """12.6.137節の合格条件（12.6.133節の H1〜H5 を各規模に当てはめ、H6・H7 を足す）。主除去 = 出力の除去。"""
    key = f"sel{eta:g}"
    g0 = [r["aux0"]["gini"] for r in runs]
    gs = [r[key]["gini"] for r in runs]
    h1 = (statistics.mean(g0) - statistics.mean(gs) >= 0.5 and all(b < a for a, b in zip(g0, gs))
          and s[key]["all_used_seeds"] == len(runs))
    h2 = s[key]["zero"]["ratio_of_means"] <= 1.5
    h3 = s[key]["zero"]["base_ce"]["mean"] <= s["aux0.05"]["zero"]["base_ce"]["mean"] + 0.01
    rs_, rg = s[key]["threshold_range_per_seed"], s[gate_key]["threshold_range_per_seed"]
    h4 = s[key]["threshold_range"]["mean"] <= s[gate_key]["threshold_range"]["mean"] / 5 \
        and all(a < b for a, b in zip(rs_, rg))
    h5 = s[key]["threshold_off"]["d_ce"]["mean"] <= s[gate_key]["threshold_off"]["d_ce"]["mean"] / 2
    sel_g, aux_g = s[key]["zero"]["d_random_per_gate"]["mean"], s["aux0.05"]["zero"]["d_random_per_gate"]["mean"]
    h6 = sel_g <= 0.8 * aux_g
    off = s[key]["threshold_off"]
    h7 = off["gini"]["mean"] <= 0.3 and off["d_ce"]["mean"] <= 0.02
    return {"eta": eta, "比較のゲート込み条件": gate_key, "H1_崩壊を防ぐ": h1, "H2_ハブの非対称性を消す": h2,
            "H3_性能を損なわない": h3, "H4_閾値の幅が小さい": h4, "H5_閾値を外したときの悪化が小さい": h5,
            "H6_ゲートの質量あたりでも頑健": h6, "H6_比_選択専用÷補助損失0.05": sel_g / aux_g,
            "H7_重みが均等な使い方を身につける": h7}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    p.add_argument("--experts", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--layers", type=int, nargs="+", default=[1, 2])
    p.add_argument("--coefs", type=float, nargs="+", default=[0.0, 0.05])
    p.add_argument("--gate-eta", type=float, default=0.1, help="12.6.127節の恒常性の η*")
    p.add_argument("--sel-etas", type=float, nargs="+", default=[0.01, 0.03, 0.1])
    p.add_argument("--eta-star", type=str, nargs="*", default=None,
                   help="評価用シードでは開発用で選んだ η_s* を 'E,L=η' の形で渡す（例: 8,1=0.01）")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-sentences", type=int, default=3000)
    p.add_argument("--d", type=int, default=48)
    p.add_argument("--train-steps", type=int, default=600)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--n-windows", type=int, default=64)
    p.add_argument("--k-num", type=int, default=1, help="ハブの数 k = E × k_num / k_den（既定 E/4、E=8 で 12.6.124節の k=2）")
    p.add_argument("--k-den", type=int, default=4)
    p.add_argument("--n-random", type=int, default=30, help="ランダム除去の組合せの上限（超えたら無作為抽出）")
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", type=str, default="results/moe_scale/dev.json")
    p.add_argument("--merge", type=str, nargs="+", default=None,
                   help="シードごとに別プロセスで出した結果ファイルを結合して要約・判定だけ行う")
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    conds = conditions(args)
    sc = scales(args)
    runs, parts = [], []
    if args.merge:
        parts = [json.loads(Path(f).read_text()) for f in args.merge]
        runs = [r for part in parts for r in part["シード別"]]
        args.seeds = [r["seed"] for r in runs]
    for seed in ([] if args.merge else args.seeds):
        corpus = CharCorpus(synthetic_corpus(n_sentences=args.n_sentences, seed=seed))
        r = {"seed": seed}
        for e, l in sc:
            for key, cond in conds.items():
                z = r[f"E{e}_L{l}/{key}"] = run_one(seed, e, l, cond, corpus, args)
                zl, zg = z["lesion"]["zero"], z["lesion"]["zero_gate1"]
                extra = (f" th_range={z['threshold_range']:.2f} off_gini={z['threshold_off']['gini']:.3f}"
                         f" off_dce={z['threshold_off']['d_ce']:.3f}") if "threshold" in z else ""
                print(f"seed={seed} E{e}_L{l}/{key} gini={z['gini']:.3f} used={z['n_used']:.1f} "
                      f"base={zl['base_ce']:.4f} dT={zl['d_target']:.3f} dR={zl['d_random_mean']:.3f} "
                      f"ratio={zl['ratio']:.2f} perload T={zl['d_target_per_load']:.3f} R={zl['d_random_per_load']:.3f} "
                      f"pergate R={zl['d_random_per_gate']:.3f} gate={z['gate_mean']:.3f} "
                      f"g1 base={zg['base_ce']:.3f} perload R={zg['d_random_per_load']:.3f}{extra}", flush=True)
        runs.append(r)
    stars = dict(kv.split("=") for kv in args.eta_star) if args.eta_star else {}
    out_s, verdict = {}, {}
    for e, l in sc:
        name = f"E{e}_L{l}"
        sub = [{k.split("/")[1]: v for k, v in r.items() if k.startswith(name + "/")} for r in runs]
        s = out_s[name] = summarize(sub, list(conds), args.eps)
        if {0.0, 0.05} <= set(args.coefs):
            star = stars.get(f"{e},{l}")
            eta = float(star) if star is not None else select_eta(s, args.sel_etas)
            verdict[name] = judge(sub, s, eta, f"gate{args.gate_eta:g}")
            verdict[name]["eta_selected_on"] = "引数 --eta-star（開発用シードで選択済み）" if star is not None \
                else "このシードでの選択（12.6.133節の選び方）"
    out = {"環境": env_info(), "設定": vars(args),
           **({"結合元の環境": [part["環境"] for part in parts]} if args.merge else {}),
           "要約": out_s, "判定": verdict, "シード別": runs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({"判定": verdict}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
