"""時間文脈のタグによる海馬の痕跡の文脈依存の想起と、文字 LM の文脈推定（12.6.179〜180節）。

12.6.175〜176節のゲイン変調のモデル（GainCharLM）と課題・規則をそのまま使い、保存する痕跡に、それを経験したときの
時間文脈のタグ（TCM: Howard & Kahana 2002）をつける。テスト時は、情報源が示したラベルなしの窓（active の候補と
同じ）を聞いて漂流させた今の文脈に近いタグの痕跡を想起して候補にし、期待情報量で選ぶ（active_replay_tcm）。
対照は、タグなしで混ぜたバッファから取る active_replay（12.6.174節と同じ）と、本来の文脈のバッファだけから取る診断
active_replay_own。統制は、リプレイしていない保留サンプルから同じく想起する active_holdout_tcm。

    --phase dev（開発用シード 100〜109）/ eval（評価用シード 0〜9 で一度だけ）
    結果は results/context_lm/context_tcm_{phase}.json、集計と判定は --analyze <結果ファイル>

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_tcm --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_context_tcm --analyze results/context_lm/context_tcm_dev.json
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import torch

from ..context_lm import RULES, TCM_RULES, TrainConfig, evaluate, make_languages, train_models
from ..run_env import env_info
from .run_context_lm import BUFFER, CUE_NS, DEV_SEEDS, EPISODES, EVAL_SEEDS, FAMILIES, POOL, REGIMES, _mean, analyze

TCM_BETA = 0.02  # 文脈の漂流の速さ（時定数 約50文字。12.6.179節で計測前に固定）
TCM_ALL_RULES = RULES + ("active_replay_own",) + TCM_RULES

# 判定条件（12.6.179節で計測前に固定）
NO_HARM = -0.005  # H2-T1: 上書きありの組で gain_tcm の平均がこれ以上（手がかり 2・4・8・16 個のそれぞれ）
BAD_RUN, BAD_FRAC = -0.02, 0.10  # H2-T1: 1回ごとに gain_tcm < −0.02 の回が 10% 以下（手がかり 8・16 個）
CAPTURE = 0.5  # H2-T2: gain_tcm が診断の利得 gain_own の半分以上（手がかり 2・4・8 個の平均）
OWN_MIN = 0.01  # H2-T2: gain_own がこれ未満なら検証できず
NONE_ABS = 0.01  # H2b: 上書きなしの組で |gain_tcm| ≦ 0.01（手がかり 8・16 個）


def h2_tcm(runs: list[dict], g: dict) -> dict:
    """12.6.179節の H2-T1・T2・H2b・H2c・H2-CE と、参考の元の H2a（+0.03）を計算する。"""
    names = list(g)
    over = [nm for nm in names if g[nm]["分類"] == "上書きあり"]
    none = [nm for nm in names if g[nm]["分類"] == "上書きなし"]
    over_runs = [r for r in runs if f"{r['family']}-{r['regime']}" in over]

    def gain(nm: str, n: str, rule: str, base: str = "active") -> float:
        return g[nm]["acc"][n][rule] - g[nm]["acc"][n][base]

    out: dict = {"上書きあり": over, "上書きなし": none, "手がかりごと": {}}
    for n in ("1", "2", "4", "8", "16"):
        row = {}
        for nm in names:
            purity = _mean([r["cues"][n]["tcm_purity"]["replay_tcm"] for r in runs
                            if f"{r['family']}-{r['regime']}" == nm])
            row[nm] = {"gain_tcm": gain(nm, n, "active_replay_tcm"), "gain_replay": gain(nm, n, "active_replay"),
                       "gain_own": gain(nm, n, "active_replay_own"),
                       "gain_holdout_tcm": gain(nm, n, "active_holdout_tcm"),
                       "ce_active": g[nm]["ce"][n]["active"], "ce_tcm": g[nm]["ce"][n]["active_replay_tcm"],
                       "purity_tcm": purity}
        out["手がかりごと"][n] = row
    if not over:
        out["H2-T1"] = out["H2-T2"] = out["H2c"] = out["H2-CE"] = "検証できず（上書きありの組がない）"
    else:
        m = {n: {key: _mean([out["手がかりごと"][n][nm][key] for nm in over])
                 for key in ("gain_tcm", "gain_replay", "gain_own", "gain_holdout_tcm", "ce_active", "ce_tcm")}
             for n in ("2", "4", "8", "16")}
        bad = {n: sum(r["cues"][n]["acc"]["active_replay_tcm"] - r["cues"][n]["acc"]["active"] < BAD_RUN
                      for r in over_runs) for n in ("8", "16")}
        t1 = all(m[n]["gain_tcm"] >= NO_HARM for n in m) and all(b <= BAD_FRAC * len(over_runs) for b in bad.values())
        out["H2-T1"] = {"gain_tcm の平均": {n: m[n]["gain_tcm"] for n in m},
                        "−0.02 を下回る回": {n: f"{b}/{len(over_runs)}" for n, b in bad.items()}, "合格": t1}
        ns = ("2", "4", "8")
        g_tcm, g_own = _mean([m[n]["gain_tcm"] for n in ns]), _mean([m[n]["gain_own"] for n in ns])
        g_hold = _mean([m[n]["gain_holdout_tcm"] for n in ns])
        if g_own < OWN_MIN:
            t2: object = "検証できず（診断の利得が 0.01 未満）"
        else:
            t2 = g_tcm >= CAPTURE * g_own and g_hold <= g_tcm / 2
        out["H2-T2"] = {"gain_tcm（2・4・8 の平均）": g_tcm, "gain_own": g_own, "取り戻した割合": g_tcm / g_own
                        if g_own else float("nan"), "gain_holdout_tcm": g_hold, "合格": t2}
        out["H2c"] = {"gain_tcm − gain_replay": {n: m[n]["gain_tcm"] - m[n]["gain_replay"] for n in ns},
                      "合格": all(m[n]["gain_tcm"] - m[n]["gain_replay"] > 0 for n in ns)}
        out["H2-CE"] = {"CE（active − tcm）": {n: m[n]["ce_active"] - m[n]["ce_tcm"] for n in ("8", "16")},
                        "合格": all(m[n]["ce_active"] - m[n]["ce_tcm"] > 0 for n in ("8", "16"))}
        out["参考_元の H2a（+0.03）"] = all(m[n]["gain_tcm"] >= 0.03 for n in ("8", "16")) \
            and all(b <= BAD_FRAC * len(over_runs) for b in bad.values()) \
            and all(m[n]["gain_holdout_tcm"] <= m[n]["gain_tcm"] / 2 for n in ("8", "16"))
    if none:
        m_none = {n: _mean([out["手がかりごと"][n][nm]["gain_tcm"] for nm in none]) for n in ("8", "16")}
        out["H2b"] = {"上書きなしの gain_tcm": m_none, "合格": all(abs(x) <= NONE_ABS for x in m_none.values())}
    else:
        out["H2b"] = "検証できず"
    t1_ok, t2_ok, b_ok = (out[key] if isinstance(out[key], str) else out[key]["合格"] for key in ("H2-T1", "H2-T2", "H2b"))
    out["失敗が解けた"] = t1_ok is True and t2_ok is True and b_ok is True
    return out


def analyze_tcm(runs: list[dict]) -> dict:
    out = analyze(runs)
    out["判定"]["H2_TCM"] = h2_tcm(runs, out["組"])
    return out


def run_one(args: tuple[str, str, int], steps: int = 1500, episodes: int = EPISODES, cue_ns=CUE_NS,
            beta: float = TCM_BETA) -> dict:
    fam_name, regime, seed = args
    torch.set_num_threads(1)
    fam, k = FAMILIES[fam_name]
    g = torch.Generator().manual_seed(seed)
    trans, _ = make_languages(fam, g)
    cfg = TrainConfig(regime=regime, steps_per_lang=steps, buffer=BUFFER, k=k, arch="gain", tcm_beta=beta)
    trained = train_models(fam, trans, cfg, g)
    ev = evaluate(fam, trans, trained, cfg, tuple(cue_ns), g, episodes=episodes, pool=POOL, rules=TCM_ALL_RULES)
    ev.update({"family": fam_name, "regime": regime, "seed": seed, "final_loss": trained["final_loss"]})
    return ev


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
        print(json.dumps(analyze_tcm(runs)["判定"], ensure_ascii=False, indent=1))
        return
    seeds = args.seeds or (DEV_SEEDS if args.phase == "dev" else EVAL_SEEDS)
    jobs = [(f, reg, s) for f in args.families for reg in REGIMES for s in seeds]
    with Pool(args.workers) as pool:
        runs = pool.map(run_one, jobs)
    out = {"環境": env_info(), "設定": {"arch": "gain", "tcm_beta": TCM_BETA, "item": "2文字組の one-hot",
                                     "families": {k: [v[0].__dict__, v[1]] for k, v in FAMILIES.items()},
                                     "buffer": BUFFER, "episodes": EPISODES, "pool": POOL, "cue_ns": CUE_NS,
                                     "rules": TCM_ALL_RULES, "seeds": list(seeds)},
           "runs": runs}
    out["集計"] = analyze_tcm(runs)
    path = args.out or Path("results/context_lm") / f"context_tcm_{args.phase}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["集計"]["判定"]["H2_TCM"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
