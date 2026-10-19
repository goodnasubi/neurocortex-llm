"""文字 LM の文脈推定への、能動的な手がかりの選択と海馬のバッファからの候補の移植（12.6.173〜174節）。

3 族の合成言語（A: 4言語・2次・α = 0.3、B: 7言語・2次・食い違い 15%、C: 4言語・3次・α = 0.05）× 学習の形
（joint・seq）の 6 組で、手がかり 1/2/4/8/16 個の各規則（passive・uniform・active・uniform_replay・active_replay・
active_holdout）の選択精度と次の文字の CE を測る。

    --phase dev（開発用シード 100〜109）/ eval（評価用シード 0〜9 で一度だけ）
    結果は results/context_lm/context_lm_{phase}.json、集計と判定は --analyze <結果ファイル>

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_lm --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_context_lm --analyze results/context_lm/context_lm_dev.json
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import torch

from ..context_lm import RULES, LangFamily, TrainConfig, evaluate, make_languages, train_models
from ..run_env import env_info

DEV_SEEDS = tuple(range(100, 110))
EVAL_SEEDS = tuple(range(10))
CUE_NS = (1, 2, 4, 8, 16)
FAMILIES: dict[str, tuple[LangFamily, int]] = {  # 族: (言語, 窓の長さ k)
    "A": (LangFamily(n_langs=4, vocab=10, order=2, diff_frac=0.3, alpha=0.3), 3),
    "B": (LangFamily(n_langs=7, vocab=10, order=2, diff_frac=0.15, alpha=0.3), 3),
    "C": (LangFamily(n_langs=4, vocab=10, order=3, diff_frac=0.3, alpha=0.05), 4),
}
REGIMES = ("joint", "seq")
BUFFER = 2000
EPISODES = 100
POOL = 4

# 判定条件（12.6.173節で計測前に固定）
GAP_OVER, GAP_NONE = 0.05, 0.02  # 上書きあり: KL_fresh − KL_own_buf ≧ 0.05、上書きなし: ≦ 0.02（nats）


def run_one(args: tuple[str, str, int], steps: int = 1500, episodes: int = EPISODES, cue_ns=CUE_NS) -> dict:
    fam_name, regime, seed = args
    torch.set_num_threads(1)
    fam, k = FAMILIES[fam_name]
    g = torch.Generator().manual_seed(seed)
    trans, _ = make_languages(fam, g)
    cfg = TrainConfig(regime=regime, steps_per_lang=steps, buffer=BUFFER, k=k)
    trained = train_models(fam, trans, cfg, g)
    ev = evaluate(fam, trans, trained, cfg, tuple(cue_ns), g, episodes=episodes, pool=POOL)
    ev.update({"family": fam_name, "regime": regime, "seed": seed, "final_loss": trained["final_loss"]})
    return ev


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def analyze(runs: list[dict]) -> dict:
    """12.6.173節の判定条件を計算する。組 = 族 × 学習の形、1回 = 組 × シード。"""
    groups: dict[str, list[dict]] = {}
    for r in runs:
        groups.setdefault(f"{r['family']}-{r['regime']}", []).append(r)
    out: dict = {"組": {}}
    for name, rs in sorted(groups.items()):
        nat = {key: _mean([r["native"][key] for r in rs]) for key in rs[0]["native"]}
        gap = nat["fresh"] - nat["own_buf"]
        cls = "上書きあり" if gap >= GAP_OVER else ("上書きなし" if gap <= GAP_NONE else "中間")
        row = {"n_seeds": len(rs), "gap": gap, "分類": cls, "native": nat,
               "ce_oracle": _mean([r["ce_oracle"] for r in rs]), "ce_nocontext": _mean([r["ce_nocontext"] for r in rs]),
               "entropy": _mean([r["entropy"] for r in rs]), "acc": {}, "ce": {}, "info": {}}
        for n in rs[0]["cues"]:
            row["acc"][n] = {rule: _mean([r["cues"][n]["acc"][rule] for r in rs]) for rule in rs[0]["cues"][n]["acc"]}
            row["ce"][n] = {rule: _mean([r["cues"][n]["ce"][rule] for r in rs]) for rule in rs[0]["cues"][n]["acc"]}
            row["info"][n] = {rule: _mean([r["cues"][n]["info"][rule] for r in rs]) for rule in rs[0]["cues"][n]["acc"]}
        out["組"][name] = row
    g = out["組"]
    names = list(g)
    verdict: dict = {}

    # H0: 能動的な選択（手がかり4個）の CE が文脈なしより低い（全組の平均で、かつ1回ごとに 90% 以上）
    runs_ok = sum(r["cues"]["4"]["ce"]["active"] < r["ce_nocontext"] for r in runs)
    verdict["H0"] = {"組ごとの差（文脈なし − active, n=4）": {nm: g[nm]["ce_nocontext"] - g[nm]["ce"]["4"]["active"]
                                                       for nm in names},
                     "下回った回": f"{runs_ok}/{len(runs)}"}
    verdict["H0"]["合格"] = all(v > 0 for v in verdict["H0"]["組ごとの差（文脈なし − active, n=4）"].values()) \
        and runs_ok >= 0.9 * len(runs)

    # H1: 能動的な選択 対 passive・uniform（手がかり 2・4 個）
    h1: dict = {}
    ok = True
    for base in ("passive", "uniform"):
        for n in ("2", "4"):
            d_acc = {nm: g[nm]["acc"][n]["active"] - g[nm]["acc"][n][base] for nm in names}
            bad_acc = sum(r["cues"][n]["acc"]["active"] < r["cues"][n]["acc"][base] - 0.02 for r in runs)
            d_ce = {nm: g[nm]["ce"][n][base] - g[nm]["ce"][n]["active"] for nm in names}
            gap_ce = _mean([g[nm]["ce"][n][base] - g[nm]["ce_oracle"] for nm in names])
            closed = _mean(list(d_ce.values())) / gap_ce if gap_ce > 0 else float("nan")
            bad_ce = sum(r["cues"][n]["ce"]["active"] > r["cues"][n]["ce"][base] + 0.005 for r in runs)
            a_ok = _mean(list(d_acc.values())) >= 0.05 and min(d_acc.values()) >= 0.02 and bad_acc <= 0.05 * len(runs)
            c_ok = closed >= 0.3 and min(d_ce.values()) > 0 and bad_ce <= 0.10 * len(runs)
            h1[f"{base}_n{n}"] = {"選択精度の差の平均": _mean(list(d_acc.values())), "最悪の組": min(d_acc.values()),
                                  "−0.02 を下回る回": f"{bad_acc}/{len(runs)}", "CE の差の平均": _mean(list(d_ce.values())),
                                  "閉じた割合": closed, "CE 最悪の組": min(d_ce.values()),
                                  "+0.005 を超えて悪い回": f"{bad_ce}/{len(runs)}", "選択精度の合格": a_ok,
                                  "CE の合格": c_ok}
            ok = ok and a_ok and c_ok
    h1["合格"] = ok
    verdict["H1"] = h1

    verdict["H2"] = _h2(runs, g, names, "active_replay", "active_holdout")
    if "active_replay_cued" in runs[0]["cues"]["8"]["acc"]:
        verdict["H2_修正1"] = _h2(runs, g, names, "active_replay_cued", "active_holdout_cued")
    out["判定"] = verdict
    return out


def _h2(runs: list[dict], g: dict, names: list[str], replay: str, holdout: str) -> dict:
    """H2: 海馬のバッファからの候補（手がかり 8・16 個）。上書きの有無で組を分ける。"""
    over = [nm for nm in names if g[nm]["分類"] == "上書きあり"]
    none = [nm for nm in names if g[nm]["分類"] == "上書きなし"]
    h2: dict = {"上書きあり": over, "上書きなし": none}
    ok_a = bool(over)
    ok_b = bool(none)
    ok_ce = bool(over)
    for n in ("8", "16"):
        gr = {nm: g[nm]["acc"][n][replay] - g[nm]["acc"][n]["active"] for nm in names}
        gh = {nm: g[nm]["acc"][n][holdout] - g[nm]["acc"][n]["active"] for nm in names}
        gce = {nm: g[nm]["ce"][n]["active"] - g[nm]["ce"][n][replay] for nm in names}
        over_runs = [r for r in runs if f"{r['family']}-{r['regime']}" in over]
        bad = sum(r["cues"][n]["acc"][replay] - r["cues"][n]["acc"]["active"] < -0.02 for r in over_runs)
        m_over = _mean([gr[nm] for nm in over])
        m_hold = _mean([gh[nm] for nm in over])
        m_none = _mean([gr[nm] for nm in none])
        h2[f"n{n}"] = {"gain_replay（組ごと）": gr, "gain_holdout（組ごと）": gh, "CE の利得（組ごと）": gce,
                       "上書きありの gain_replay": m_over, "上書きありの gain_holdout": m_hold,
                       "上書きなしの gain_replay": m_none, "上書きありで −0.02 を下回る回": f"{bad}/{len(over_runs)}"}
        if over:
            ok_a = ok_a and m_over >= 0.03 and bad <= 0.10 * len(over_runs) and m_hold <= m_over / 2
            ok_ce = ok_ce and _mean([gce[nm] for nm in over]) > 0
        if none:
            ok_b = ok_b and abs(m_none) <= 0.01
    h2["H2a 合格（上書きありで利得・特異性）"] = ok_a if over else "検証できず（上書きありの組がない）"
    h2["H2b 合格（上書きなしで利得なし）"] = ok_b if none else "検証できず"
    h2["H2-CE 合格（上書きありで CE も改善）"] = ok_ce if over else "検証できず"
    return h2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", type=Path, nargs="+", help="結果ファイルを集計して判定する")
    ap.add_argument("--phase", choices=("dev", "eval"), default="dev")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.analyze:
        runs = [r for p in args.analyze for r in json.loads(p.read_text(encoding="utf-8"))["runs"]]
        print(json.dumps(analyze(runs), ensure_ascii=False, indent=1))
        return
    seeds = args.seeds or (DEV_SEEDS if args.phase == "dev" else EVAL_SEEDS)
    jobs = [(f, reg, s) for f in args.families for reg in REGIMES for s in seeds]
    with Pool(args.workers) as pool:
        runs = pool.map(run_one, jobs)
    out = {"環境": env_info(), "設定": {"families": {k: [v[0].__dict__, v[1]] for k, v in FAMILIES.items()},
                                     "buffer": BUFFER, "episodes": EPISODES, "pool": POOL, "cue_ns": CUE_NS,
                                     "seeds": list(seeds)},
           "runs": runs}
    out["集計"] = analyze(runs)
    path = args.out or Path("results/context_lm") / f"context_lm_{args.phase}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["集計"]["判定"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
