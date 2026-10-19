"""共有の表現への文脈のゲイン変調による文字 LM の文脈推定（12.6.175〜176節）。

12.6.173〜174節の課題・規則・判定（run_context_lm.analyze）をそのまま使い、モデルだけを文脈ごとの独立したヘッド
（ContextCharLM）から、共有の表現に文脈信号で乗算的なゲインをかける GainCharLM に替える。文脈なしの対照も同じ形
（文脈1個）にする。規則は初期設定の6規則と診断の active_replay_own（12.6.173節の修正1の事後分布による想起は測らない）。
--fix1 では 12.6.175節の修正1（事後分布で重みづけたゲイン active_soft、予測の混合 active_mix）も測る。

    --phase dev（開発用シード 100〜109）/ eval（評価用シード 0〜9 で一度だけ）
    結果は results/context_lm/context_gain_{phase}.json、集計と判定は --analyze <結果ファイル>

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_gain --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_context_gain --analyze results/context_lm/context_gain_dev.json
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import torch

from ..context_lm import RULES, SOFT_RULES, TrainConfig, evaluate, make_languages, train_models
from ..run_env import env_info
from .run_context_lm import BUFFER, CUE_NS, DEV_SEEDS, EPISODES, EVAL_SEEDS, FAMILIES, POOL, REGIMES, analyze

GAIN_RULES = RULES + ("active_replay_own",)
FIX1_GAIN_RULES = GAIN_RULES + SOFT_RULES  # 修正1（12.6.175節）: 事後分布で重みづけたゲイン・予測の混合


def h0(runs: list[dict], rule: str, n: str = "4") -> dict:
    """H0（12.6.173・175節）を規則 rule で判定する: 手がかり n 個の CE が全組（シードの平均）で文脈なしより低く、
    1回ごとにも 90% 以上で低い。"""
    groups: dict[str, list[dict]] = {}
    for r in runs:
        groups.setdefault(f"{r['family']}-{r['regime']}", []).append(r)
    diff = {nm: sum(r["ce_nocontext"] - r["cues"][n]["ce"][rule] for r in rs) / len(rs)
            for nm, rs in sorted(groups.items())}
    ok_runs = sum(r["cues"][n]["ce"][rule] < r["ce_nocontext"] for r in runs)
    return {f"組ごとの差（文脈なし − {rule}, n={n}）": diff, "下回った回": f"{ok_runs}/{len(runs)}",
            "合格": all(d > 0 for d in diff.values()) and ok_runs >= 0.9 * len(runs)}


def analyze_gain(runs: list[dict]) -> dict:
    """run_context_lm.analyze に、修正1の規則での H0 を加える。"""
    out = analyze(runs)
    for rule in SOFT_RULES:
        if rule in runs[0]["cues"]["4"]["ce"]:
            out["判定"][f"H0_修正1_{rule}"] = h0(runs, rule)
    return out


def run_one(args: tuple[str, str, int], steps: int = 1500, episodes: int = EPISODES, cue_ns=CUE_NS,
            rules: tuple[str, ...] = GAIN_RULES) -> dict:
    fam_name, regime, seed = args
    torch.set_num_threads(1)
    fam, k = FAMILIES[fam_name]
    g = torch.Generator().manual_seed(seed)
    trans, _ = make_languages(fam, g)
    cfg = TrainConfig(regime=regime, steps_per_lang=steps, buffer=BUFFER, k=k, arch="gain")
    trained = train_models(fam, trans, cfg, g)
    ev = evaluate(fam, trans, trained, cfg, tuple(cue_ns), g, episodes=episodes, pool=POOL, rules=rules)
    gains = trained["model"].gains().detach()
    ev.update({"family": fam_name, "regime": regime, "seed": seed, "final_loss": trained["final_loss"],
               # ゲインの散らばり（1 からの平均の絶対偏差、文脈の間の標準偏差の平均）
               "gain_stats": {"mean_abs_dev": float((gains - 1).abs().mean()),
                              "between_ctx_sd": float(gains.std(0).mean())}})
    return ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", type=Path, nargs="+", help="結果ファイルを集計して判定する")
    ap.add_argument("--phase", choices=("dev", "eval"), default="dev")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--fix1", action="store_true", help="修正1の規則（active_soft・active_mix）も測る")
    args = ap.parse_args()
    if args.analyze:
        runs = [r for p in args.analyze for r in json.loads(p.read_text(encoding="utf-8"))["runs"]]
        print(json.dumps(analyze_gain(runs), ensure_ascii=False, indent=1))
        return
    seeds = args.seeds or (DEV_SEEDS if args.phase == "dev" else EVAL_SEEDS)
    jobs = [(f, reg, s) for f in args.families for reg in REGIMES for s in seeds]
    rules = FIX1_GAIN_RULES if args.fix1 else GAIN_RULES
    with Pool(args.workers) as pool:
        runs = pool.starmap(run_one, [(job, 1500, EPISODES, CUE_NS, rules) for job in jobs])
    out = {"環境": env_info(), "設定": {"arch": "gain", "families": {k: [v[0].__dict__, v[1]] for k, v in FAMILIES.items()},
                                     "buffer": BUFFER, "episodes": EPISODES, "pool": POOL, "cue_ns": CUE_NS,
                                     "rules": rules, "seeds": list(seeds)},
           "runs": runs}
    out["集計"] = analyze_gain(runs)
    path = args.out or Path("results/context_lm") / f"context_gain_{'fix1_' if args.fix1 else ''}{args.phase}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["集計"]["判定"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
