"""大きな規模の恒常性 MoE で重みへの取り込みを促す（12.6.141節の設計）の主実験。

規模ごとに、(a) 学習量を 600 × E/8 ステップに増やす、(b) η を取り込み（閾値を外したときのジニ・ΔCE_off）も含めた
基準で選ぶ、(c) 学習の後半に閾値の効きを 0 へ下げる（臨界期の終わり）、を比べる。測定・要約・判定は
`run_moe_scale.py`（12.6.137節）の関数をそのまま使う。

使い方（シード×規模ごとに別プロセスで出して結合する）:
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_absorb --seeds 100 --scales 16,1 \
        --out results/moe_absorb/parts/dev_s100_E16L1.json
    PYTHONPATH=src python -m neurocortex.experiments.run_moe_absorb --merge results/moe_absorb/parts/dev_*.json \
        --out results/moe_absorb/dev.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..data import CharCorpus, synthetic_corpus
from ..moe_absorb import AnnealedScaledRoutedCharLM
from ..run_env import env_info
from .run_moe_scale import judge, run_one, summarize

BASE = ("aux0", "aux0.05", "gate0.1")


def conditions(args: argparse.Namespace, n_experts: int) -> dict[str, dict]:
    """キー → 条件。"long/" は (a) の長い学習、"ann" は (c) の閾値の効きを下げる条件。"""
    c = {"aux0": {"kind": "plain", "coef": 0.0}, "aux0.05": {"kind": "plain", "coef": 0.05},
         "gate0.1": {"kind": "gate", "eta": 0.1}}
    c.update({f"sel{v:g}": {"kind": "select", "eta": v} for v in args.sel_etas})
    c.update({f"ann{v:g}": {"kind": "select", "eta": v, "anneal": True} for v in args.ann_etas})
    steps = args.train_steps * n_experts // 8
    if steps != args.train_steps:  # E=8 は 600 ステップのままなので (a) は基準と同じ（測らない）
        for k in (*BASE, f"sel{args.long_eta:g}"):
            c[f"long/{k}"] = {**c.get(k, {"kind": "select", "eta": args.long_eta}), "steps": steps}
    return c


def run_cond(seed: int, e: int, l: int, cond: dict, corpus: CharCorpus, args: argparse.Namespace) -> dict:
    steps = cond.get("steps", args.train_steps)
    if cond.get("anneal"):
        return run_one(seed, e, l, cond, corpus, args, AnnealedScaledRoutedCharLM, steps,
                       total_steps=steps, anneal_start=args.anneal_start)
    return run_one(seed, e, l, cond, corpus, args, steps=steps)


def select_eta_b(s: dict, etas: list[float]) -> tuple[float, str]:
    """12.6.141節 (b) の選び方。比 ≤ 1.5・CE ≤ 補助損失0.05 + 0.01・外したときのジニ ≤ 0.3・ΔCE_off ≤ 0.02 の最小の η。"""
    ref = s["aux0.05"]["zero"]["base_ce"]["mean"]
    etas = [e for e in etas if f"sel{e:g}" in s]  # 修正1: 規模によって候補の η が違う

    def ok2(e):
        z = s[f"sel{e:g}"]["zero"]
        return z["ratio_of_means"] <= 1.5 and z["base_ce"]["mean"] <= ref + 0.01

    def off(e):
        return s[f"sel{e:g}"]["threshold_off"]

    full = [e for e in sorted(etas) if ok2(e) and off(e)["gini"]["mean"] <= 0.3 and off(e)["d_ce"]["mean"] <= 0.02]
    if full:
        return full[0], "4条件を満たす最小の η"
    two = [e for e in etas if ok2(e)]
    if two:
        return min(two, key=lambda e: off(e)["gini"]["mean"]), "取り込みの条件を満たす η がなく、比と CE を満たす中で外したときのジニが最小"
    return min(etas, key=lambda e: s[f"sel{e:g}"]["zero"]["ratio_of_means"]), "比と CE を満たす η がなく、比が最小"


def analyze(sub: list[dict], args: argparse.Namespace, eta_b: float | None) -> dict:
    """1規模ぶん（シードごとの {キー: 結果} の列）の要約と、手当て (a)(b)(c) の判定。"""
    keys = list(sub[0])
    base_keys = [k for k in keys if not k.startswith("long/")]
    s = summarize(sub, base_keys, args.eps)
    add_state_range(s, sub)
    verdict = {}
    if eta_b is None:
        eta_b, why = select_eta_b(s, args.sel_etas)
    else:
        why = "引数 --eta-b（開発用シードで選択済み）"
    verdict["参考_12.6.138節と同じ_sel0.01"] = judge(sub, s, 0.01, "gate0.1")
    verdict["b_η選択"] = {**judge(sub, s, eta_b, "gate0.1"), "選び方": why}
    for v in args.ann_etas:  # (c): sel0.01 の位置に ann を置いて同じ判定を使う
        csub = [{**{k: r[k] for k in BASE}, f"sel{v:g}": r[f"ann{v:g}"]} for r in sub]
        cs = summarize(csub, list(csub[0]), args.eps)
        verdict[f"c_閾値を弱める_η{v:g}"] = judge(csub, cs, v, "gate0.1")
        verdict[f"c_閾値を弱める_η{v:g}"]["閾値の状態θの幅"] = s[f"ann{v:g}"].get("threshold_state_range")
    long_keys = [k for k in keys if k.startswith("long/")]
    if long_keys:
        lsub = [{k[5:]: r[k] for k in long_keys} for r in sub]
        ls = summarize(lsub, list(lsub[0]), args.eps)
        s["long"] = ls
        verdict["a_学習量"] = judge(lsub, ls, args.long_eta, "gate0.1")
    else:
        verdict["a_学習量"] = {**verdict["参考_12.6.138節と同じ_sel0.01"], "注": "E=8 は 600 ステップのまま（基準と同じ）"}
    for k in verdict:
        v = verdict[k]
        if "H1_崩壊を防ぐ" in v:
            v["取り込みが起きた_H7かつH1〜H3"] = bool(v["H7_重みが均等な使い方を身につける"] and v["H1_崩壊を防ぐ"]
                                              and v["H2_ハブの非対称性を消す"] and v["H3_性能を損なわない"])
    return {"要約": s, "判定": verdict}


def add_state_range(s: dict, sub: list[dict]) -> None:
    for k in s:
        if k.startswith("ann") and "threshold_state_range" in sub[0][k]:
            vals = [r[k]["threshold_state_range"] for r in sub]
            s[k]["threshold_state_range"] = sum(vals) / len(vals)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    p.add_argument("--scales", type=str, nargs="+", default=["8,1", "16,1", "32,1", "32,2"], help="E,L の列")
    p.add_argument("--sel-etas", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3])
    p.add_argument("--ann-etas", type=float, nargs="+", default=[0.01, 0.1], help="(c) の η（主は 0.01）")
    p.add_argument("--anneal-start", type=float, default=0.5)
    p.add_argument("--long-eta", type=float, default=0.01, help="(a) の選択専用の η")
    p.add_argument("--eta-b", type=str, nargs="*", default=None,
                   help="評価用シードでは開発用で選んだ η_b を 'E,L=η' の形で渡す（例: 16,1=0.1）")
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
    p.add_argument("--out", type=str, default="results/moe_absorb/dev.json")
    p.add_argument("--merge", type=str, nargs="+", default=None)
    p.add_argument("--only", type=str, nargs="*", default=None,
                   help="この条件のキーだけ測る（修正1で η を足すとき。--merge で同じシード・規模の部分を合わせる）")
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
                for key, cond in conditions(args, e).items():
                    if args.only and key not in args.only:
                        continue
                    z = runs[key] = run_cond(seed, e, l, cond, corpus, args)
                    zl = z["lesion"]["zero"]
                    extra = (f" th_range={z['threshold_range']:.2f} off_gini={z['threshold_off']['gini']:.3f}"
                             f" off_dce={z['threshold_off']['d_ce']:.3f}") if "threshold_off" in z else ""
                    print(f"seed={seed} E{e}_L{l}/{key} gini={z['gini']:.3f} used={z['n_used']:.1f} "
                          f"base={zl['base_ce']:.4f} ratio={zl['ratio']:.2f} pergate R={zl['d_random_per_gate']:.3f} "
                          f"gate={z['gate_mean']:.3f}{extra}", flush=True)
                parts.append({"環境": env_info(), "seed": seed, "scale": f"E{e}_L{l}", "runs": runs})
    if args.only and not args.merge:  # 足した条件だけの部分は、結合してから要約・判定する
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"環境": env_info(), "設定": vars(args), "部分": parts}, ensure_ascii=False))
        return
    envs = [pt["環境"] for pt in parts]
    merged: dict = {}  # 同じシード・規模の部分（--only で足した条件）を1つにまとめる
    for pt in parts:
        m = merged.setdefault((pt["seed"], pt["scale"]), {**pt, "runs": {}})
        m["runs"].update(pt["runs"])
    parts = list(merged.values())
    stars = dict(kv.split("=") for kv in args.eta_b) if args.eta_b else {}
    scales_seen = list(dict.fromkeys(pt["scale"] for pt in parts))
    out_s, verdict, seeds = {}, {}, {}
    for name in scales_seen:
        ps = sorted((pt for pt in parts if pt["scale"] == name), key=lambda pt: pt["seed"])
        sub = [pt["runs"] for pt in ps]
        e, l = name[1:].split("_L")
        star = stars.get(f"{e},{l}")
        r = analyze(sub, args, float(star) if star is not None else None)
        out_s[name], verdict[name] = r["要約"], r["判定"]
        seeds[name] = {pt["seed"]: pt["runs"] for pt in ps}
    out = {"環境": env_info(), "設定": vars(args), "結合元の環境": envs,
           "要約": out_s, "判定": verdict, **({"シード別": seeds} if args.merge else {"部分": parts})}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False))
    brief = {n: {k: {kk: vv for kk, vv in v.items() if kk.startswith(("H", "取り込み", "eta"))}
                 for k, v in vd.items()} for n, vd in verdict.items()}
    print(json.dumps(brief, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
