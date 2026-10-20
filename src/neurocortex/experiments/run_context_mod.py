"""文脈のゲイン変調（乗算）と加算の変調の切り分け（12.6.223〜224節）。

12.6.175〜176節では、文脈ごとの独立したヘッド（ContextCharLM、隠れ層1つ）を、共有の表現への乗算的なゲイン変調
（GainCharLM、隠れ層2つ・出力は共有）に替えると oracle の CE が下がった。ただし層の数と出力の共有も同時に変わって
いた。ここでは同じ課題・規則（run_context_lm と同じ族・学習の形・手がかり）で、5つの形を同じデータ・同じシードで
学習して比べる。

    head（12.6.173節）/ head2（ヘッドの2層版）/ gain（12.6.175節）/ bias（同じ位置への加算）/ gainbias（両方）

    --phase dev（開発用シード 100〜104）/ eval（評価用シード 0〜4 で一度だけ）
    結果は results/context_lm/context_mod_{phase}.json、集計と判定は --analyze <結果ファイル>

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_context_mod --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_context_mod --analyze results/context_lm/context_mod_dev.json
"""

from __future__ import annotations

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import torch

from ..context_lm import TrainConfig, evaluate, make_languages, train_models
from ..run_env import env_info
from .run_context_gain import FIX1_GAIN_RULES, GAIN_RULES
from .run_context_lm import BUFFER, CUE_NS, EPISODES, FAMILIES, POOL, REGIMES

ARCHS = ("head", "head2", "gain", "bias", "gainbias")
MOD_ARCHS = ("gain", "bias", "gainbias")  # 共有の表現への変調（active_soft を測れる）
DEV_SEEDS = tuple(range(100, 105))
EVAL_SEEDS = tuple(range(5))

# 判定条件（12.6.223節で計測前に固定）
H1_MIN = 0.010  # oracle の CE の差（bias − gain）の6組の平均の下限（nats）
H1_RUN_FRAC = 0.8  # 1回ごとに gain が低い割合の下限
H2_MIN = 0.010  # active_soft（手がかり4個）の CE の差（bias − gain）の6組の平均の下限
H3_TOTAL_MIN = 0.02  # 12.6.176節の差（head − gain の oracle の CE）が再現したとみなす下限
H3_MAIN, H3_MINOR = 0.5, 0.2  # 乗算に固有の割合がこれ以上なら「主に乗算」、これ以下なら「主に層・共有」
H4_MARGIN = 0.02  # 選択精度（手がかり 2・4 個）で gain が bias を下回ってよい幅
M_SD_MIN = 0.05  # 操作の確認: bias の文脈の間の標準偏差（ユニットの平均）の下限


def rules_for(arch: str) -> tuple[str, ...]:
    return FIX1_GAIN_RULES if arch in MOD_ARCHS else GAIN_RULES + ("active_mix",)


def run_one(args: tuple[str, str, str, int], steps: int = 1500, episodes: int = EPISODES, cue_ns=CUE_NS) -> dict:
    arch, fam_name, regime, seed = args
    torch.set_num_threads(1)
    fam, k = FAMILIES[fam_name]
    g = torch.Generator().manual_seed(seed)
    trans, _ = make_languages(fam, g)
    cfg = TrainConfig(regime=regime, steps_per_lang=steps, buffer=BUFFER, k=k, arch=arch)
    trained = train_models(fam, trans, cfg, g)
    ev = evaluate(fam, trans, trained, cfg, tuple(cue_ns), g, episodes=episodes, pool=POOL, rules=rules_for(arch))
    ev.pop("native_per_lang", None)
    for n in ev["cues"].values():
        n.pop("acc_per_lang", None)
    ev.update({"arch": arch, "family": fam_name, "regime": regime, "seed": seed, "final_loss": trained["final_loss"]})
    m = trained["model"]
    if arch in MOD_ARCHS:  # 変調の散らばり（文脈の間の標準偏差のユニットの平均）
        ev["mod_stats"] = {"gain_between_ctx_sd": float(m.gains().detach().std(0).mean()),
                           "bias_between_ctx_sd": float(m.biases().detach().std(0).mean()),
                           "bias_mean_abs": float(m.biases().detach().abs().mean())}
    return ev


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _t(xs: list[float]) -> float:
    """対応のある差の t 値。"""
    n = len(xs)
    if n < 2:
        return float("nan")
    mu = _mean(xs)
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))
    return mu / (sd / math.sqrt(n)) if sd > 0 else float("inf") * (1 if mu > 0 else -1)


def _paired(runs: list[dict], a: str, b: str, metric) -> dict:
    """a − b の metric の差を、組（族 × 学習の形）× シードで対応させて集計する。"""
    idx = {(r["arch"], r["family"], r["regime"], r["seed"]): r for r in runs}
    groups: dict[str, list[float]] = {}
    for (arch, fam, reg, seed), r in sorted(idx.items()):
        if arch != a or (b, fam, reg, seed) not in idx:
            continue
        groups.setdefault(f"{fam}-{reg}", []).append(metric(r) - metric(idx[(b, fam, reg, seed)]))
    flat = [x for v in groups.values() for x in v]
    return {"組ごと": {k: _mean(v) for k, v in groups.items()}, "6組の平均": _mean(_mean(v) for v in groups.values()),
            "正の組": sum(_mean(v) > 0 for v in groups.values()), "組の数": len(groups),
            "正の回": sum(x > 0 for x in flat), "回の数": len(flat), "t": _t(flat)}


def oracle(r):
    return r["ce_oracle"]


def ce_at(rule: str, n: str = "4"):
    return lambda r: r["cues"][n]["ce"][rule]


def acc_at(rule: str, n: str):
    return lambda r: r["cues"][n]["acc"][rule]


def analyze(runs: list[dict]) -> dict:
    archs = sorted({r["arch"] for r in runs}, key=ARCHS.index)
    table: dict = {}
    for a in archs:
        rs = [r for r in runs if r["arch"] == a]
        row = {"oracle": _mean(map(oracle, rs)), "nocontext": _mean(r["ce_nocontext"] for r in rs),
               "active_n4": _mean(map(ce_at("active"), rs)), "active_mix_n4": _mean(map(ce_at("active_mix"), rs)),
               "acc_active_n2": _mean(map(acc_at("active", "2"), rs)),
               "acc_active_n4": _mean(map(acc_at("active", "4"), rs)),
               "kl_fresh_seq": _mean(r["native"]["fresh"] for r in rs if r["regime"] == "seq"),
               "kl_own_buf_seq": _mean(r["native"]["own_buf"] for r in rs if r["regime"] == "seq")}
        if a in MOD_ARCHS:
            row["active_soft_n4"] = _mean(map(ce_at("active_soft"), rs))
            row["mod_stats"] = {k: _mean(r["mod_stats"][k] for r in rs) for k in rs[0]["mod_stats"]}
        groups: dict[str, list[dict]] = {}
        for r in rs:
            groups.setdefault(f"{r['family']}-{r['regime']}", []).append(r)
        row["組ごとの oracle / 文脈なし / active(n4)"] = {
            k: [_mean(map(oracle, v)), _mean(x["ce_nocontext"] for x in v), _mean(map(ce_at("active"), v))]
            for k, v in sorted(groups.items())}
        table[a] = row
    out: dict = {"形ごと": table, "判定": {}}
    v = out["判定"]
    if not {"gain", "bias"} <= set(archs):
        return out
    # 操作の確認 M: 加算の対照が文脈を使っている（バイアスが文脈の間で分かれ、全組で文脈なしより oracle が低い）
    m_rs = [r for r in runs if r["arch"] == "bias"]
    sd = _mean(r["mod_stats"]["bias_between_ctx_sd"] for r in m_rs)
    beat = table["bias"]["組ごとの oracle / 文脈なし / active(n4)"]
    m_ok = sd >= M_SD_MIN and all(o < nc for o, nc, _ in beat.values())
    v["M"] = {"bias の文脈の間の標準偏差": sd, "全組で oracle < 文脈なし": all(o < nc for o, nc, _ in beat.values()),
              "合格": m_ok}
    h1 = _paired(runs, "bias", "gain", oracle)
    h1["合格"] = (h1["6組の平均"] >= H1_MIN and h1["正の組"] == h1["組の数"]
                 and h1["正の回"] >= H1_RUN_FRAC * h1["回の数"]) if m_ok else "検証できず（M 不合格）"
    v["H1（oracle: bias − gain）"] = h1
    h2 = _paired(runs, "bias", "gain", ce_at("active_soft"))
    h2["合格"] = h2["6組の平均"] >= H2_MIN and h2["正の組"] >= h2["組の数"] - 1
    v["H2（active_soft n=4: bias − gain）"] = h2
    if {"head", "head2"} <= set(archs):
        tot = _paired(runs, "head", "gain", oracle)["6組の平均"]
        mult = h1["6組の平均"]
        depth = _paired(runs, "head", "head2", oracle)["6組の平均"]
        share = mult / tot if tot > 0 else float("nan")
        if tot < H3_TOTAL_MIN:
            say = "12.6.176節の差が再現せず"
        elif share >= H3_MAIN:
            say = "主に乗算のゲイン変調による"
        elif share <= H3_MINOR:
            say = "主に層の数・出力の共有による（乗算に固有ではない）"
        else:
            say = "乗算と層・共有の両方による"
        v["H3（12.6.176節の差の分解）"] = {"head − gain": tot, "乗算に固有（bias − gain）": mult,
                                      "層の数（head − head2）": depth,
                                      "出力の共有と加算の位置（head2 − bias）": _paired(runs, "head2", "bias", oracle)["6組の平均"],
                                      "乗算の割合": share, "言い方": say}
    h4 = {n: _paired(runs, "gain", "bias", acc_at("active", n)) for n in ("2", "4")}
    v["H4（選択精度: gain − bias）"] = {**{f"n{n}": h for n, h in h4.items()},
                                   "合格": all(h["6組の平均"] >= -H4_MARGIN for h in h4.values())}
    ref: dict = {}
    if "gainbias" in archs:
        ref["oracle: gain − gainbias"] = _paired(runs, "gain", "gainbias", oracle)
        ref["active_soft n4: gain − gainbias"] = _paired(runs, "gain", "gainbias", ce_at("active_soft"))
    seq = [r for r in runs if r["regime"] == "seq"]
    ref["seq の KL fresh: bias − gain"] = _paired(seq, "bias", "gain", lambda r: r["native"]["fresh"])
    ref["seq の上書き gap: bias − gain"] = _paired(seq, "bias", "gain",
                                               lambda r: r["native"]["fresh"] - r["native"]["own_buf"])
    ref["active n4 CE: bias − gain"] = _paired(runs, "bias", "gain", ce_at("active"))
    ref["文脈なしを下回る（active_soft n4）"] = {
        a: sum(r["cues"]["4"]["ce"]["active_soft"] < r["ce_nocontext"] for r in runs if r["arch"] == a)
        for a in archs if a in MOD_ARCHS}
    v["参考"] = ref
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", type=Path, nargs="+", help="結果ファイルを集計して判定する")
    ap.add_argument("--phase", choices=("dev", "eval"), default="dev")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--archs", nargs="+", default=list(ARCHS))
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.analyze:
        runs = [r for p in args.analyze for r in json.loads(p.read_text(encoding="utf-8"))["runs"]]
        print(json.dumps(analyze(runs), ensure_ascii=False, indent=1))
        return
    seeds = args.seeds or (DEV_SEEDS if args.phase == "dev" else EVAL_SEEDS)
    jobs = [(a, f, reg, s) for a in args.archs for f in args.families for reg in REGIMES for s in seeds]
    jobs.sort(key=lambda j: j[1] != "B")  # 7言語の族 B（最も重い）を先に
    with Pool(args.workers) as pool:
        runs = pool.map(run_one, jobs)
    out = {"環境": env_info(), "設定": {"archs": args.archs,
                                     "families": {k: [v[0].__dict__, v[1]] for k, v in FAMILIES.items()},
                                     "buffer": BUFFER, "episodes": EPISODES, "pool": POOL, "cue_ns": CUE_NS,
                                     "rules": {a: rules_for(a) for a in args.archs}, "seeds": list(seeds)},
           "runs": runs}
    out["集計"] = analyze(runs)
    path = args.out or Path("results/context_lm") / f"context_mod_{args.phase}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["集計"]["判定"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
