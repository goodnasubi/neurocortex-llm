"""読み出しの出力に効く次元 K を操作する継続学習（12.6.251〜252節）。幅 W = 32・128、古いタスクの出力ビットの数
K = 1・2・4・8、規則 naive・replay・replay_lo。シードごとに入力の組を DRAWS 回引く。

    --phase dev（開発用シード 100〜104）/ eval（評価用シード 0〜4 で一度だけ）
    結果は results/readout_dim/readout_dim_{phase}.json

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_readout_dim --phase dev
    PYTHONPATH=src python -m neurocortex.experiments.run_readout_dim --analyze <結果ファイル> [...]
"""

from __future__ import annotations

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import torch

from ..readout_dim import RULES, draw_spec, run
from ..run_env import env_info
from .run_context_drift_null import spearman

DEV_SEEDS = list(range(100, 105))
EVAL_SEEDS = list(range(0, 5))
WIDTHS = (32, 128)
KS = (1, 2, 4, 8)
DRAWS = 6
N_INPUTS = 10
KEYS = ("d_tot", "f_pot", "e_iso", "f_pot1", "pot_rel", "harm_rep", "harm_ro", "comp", "acc0", "acc1", "pre", "end",
        "old_change", "pre_b0", "change_b0", "new")
JUDGE_RULES = ("replay", "replay_lo")  # 判定に使う規則（naive は古いタスクがほぼ偶然まで壊れるので補助）
# 修正（上限3回）。判定の条件は計測前に決めた。修正は実装の誤りに限る
FIXES: tuple[str, ...] = ()


def jobs(seeds: list[int]) -> list[tuple]:
    return [(s, d, w, k, r) for s in seeds for d in range(DRAWS) for w in WIDTHS for k in KS for r in RULES]


def run_job(job: tuple) -> dict:
    torch.set_num_threads(1)
    s, d, w, k, r = job
    spec = draw_spec(N_INPUTS, max(KS), torch.Generator().manual_seed(s * 100 + d))
    return {"seed": s, "draw": d, "W": w, "K": k, "rule": r, **run(spec, k, w, r, s * 100 + d)}


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [x[key] for x in rows if x.get(key) is not None and math.isfinite(x[key])]
    return sum(vals) / len(vals) if vals else None


def summary(rows: list[dict]) -> dict:
    out: dict = {}
    for r in RULES:
        for w in WIDTHS:
            for k in KS:
                sel = [x for x in rows if x["rule"] == r and x["W"] == w and x["K"] == k]
                if sel:
                    out[f"{r}/W{w}/K{k}"] = {"n": len(sel), **{key: _mean(sel, key) for key in KEYS}}
    return out


def _avg(summ: dict, w: int, k: int, key: str, rules=JUDGE_RULES) -> float | None:
    vals = [summ.get(f"{r}/W{w}/K{k}", {}).get(key) for r in rules]
    return None if any(v is None for v in vals) else sum(vals) / len(vals)


def _in(x: float | None, lo: float, hi: float) -> bool | None:
    return None if x is None else bool(lo <= x <= hi)


def _ratio(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None or b == 0 else a / b


def verdict(summ: dict, rows: list[dict]) -> dict:
    """12.6.251節で計測前に固定した判定条件（値は replay・replay_lo の平均）。"""
    f = {(w, k): _avg(summ, w, k, "f_pot") for w in WIDTHS for k in KS}
    e = {(w, k): _avg(summ, w, k, "e_iso") for w in WIDTHS for k in KS}
    v: dict = {"f_pot": {f"W{w}/K{k}": f[w, k] for w in WIDTHS for k in KS},
               "e_iso": {f"W{w}/K{k}": e[w, k] for w in WIDTHS for k in KS},
               "pot_rel": {f"W{w}/K{k}": _avg(summ, w, k, "pot_rel") for w in WIDTHS for k in KS},
               "old_change": {f"W{w}/K{k}": _avg(summ, w, k, "old_change") for w in WIDTHS for k in KS}}
    # L（主仮説。出力に効く部分空間に入る動きの割合は K とともに増える）: 両方の幅で f_pot(K=8) / f_pot(K=1) ≥ 4
    v["L_ratio"] = {f"W{w}": _ratio(f[w, 8], f[w, 1]) for w in WIDTHS}
    v["L"] = None if None in v["L_ratio"].values() else all(x >= 4 for x in v["L_ratio"].values())
    # P（受け身の薄まり。等方な方向に対する濃縮は K で変わらない）: 両方の幅で e_iso(K=8) / e_iso(K=1) ∈ [0.5, 2]
    v["P_ratio"] = {f"W{w}": _ratio(e[w, 8], e[w, 1]) for w in WIDTHS}
    v["P"] = None if None in v["P_ratio"].values() else all(_in(x, 0.5, 2.0) for x in v["P_ratio"].values())
    # Q（K/W の比が決める）: f_pot(128, 4) / f_pot(32, 1) と f_pot(128, 8) / f_pot(32, 2) が [2/3, 1.5]
    v["Q_ratio"] = {"W128K4/W32K1": _ratio(f[128, 4], f[32, 1]), "W128K8/W32K2": _ratio(f[128, 8], f[32, 2])}
    v["Q"] = None if None in v["Q_ratio"].values() else all(_in(x, 2 / 3, 1.5) for x in v["Q_ratio"].values())
    # W（12.6.248節の幅の効果の再現。K = 1）: f_pot(128, 1) < f_pot(32, 1) かつ pot_rel(128, 1) ≤ pot_rel(32, 1)
    pr = v["pot_rel"]
    v["W"] = None if None in (f[128, 1], f[32, 1], pr["W128/K1"], pr["W32/K1"]) else bool(
        f[128, 1] < f[32, 1] and pr["W128/K1"] <= pr["W32/K1"])
    # R（出力に効く次元が増えると、ビットあたりの保持が下がる）: 幅 32 で old_change(K=8) ≤ old_change(K=1) − 0.01
    oc = v["old_change"]
    v["R"] = None if None in (oc["W32/K8"], oc["W32/K1"]) else bool(oc["W32/K8"] <= oc["W32/K1"] - 0.01)
    # F（測度の妥当性）: 全回（3規則）で harm_rep + harm_ro と old_change の順位相関 ≥ 0.3
    xs = [x["harm_rep"] + x["harm_ro"] for x in rows]
    ys = [x["old_change"] for x in rows]
    v["F_rho"] = spearman(xs, ys)
    v["F"] = None if v["F_rho"] is None else bool(v["F_rho"] >= 0.3)
    return v


def per_seed(rows: list[dict]) -> dict:
    """シードごとの比（判定の規則の平均）。L・P の K=8/K=1 と Q の比が、何シードで基準を満たすか。"""
    seeds = sorted({x["seed"] for x in rows})
    out: dict = {}
    for s in seeds:
        summ = summary([x for x in rows if x["seed"] == s])
        v = verdict(summ, [x for x in rows if x["seed"] == s])
        out[str(s)] = {"L_ratio": v["L_ratio"], "P_ratio": v["P_ratio"], "Q_ratio": v["Q_ratio"],
                       "L": v["L"], "P": v["P"], "Q": v["Q"], "W": v["W"], "R": v["R"]}
    return {"per_seed": out, "n_true": {h: sum(bool(o[h]) for o in out.values()) for h in ("L", "P", "Q", "W", "R")},
            "n": len(out)}


def analyze(paths: list[Path]) -> dict:
    rows: list[dict] = []
    for p in paths:
        rows += json.loads(p.read_text(encoding="utf-8"))["rows"]
    summ = summary(rows)
    return {"seeds": sorted({x["seed"] for x in rows}), "n_rows": len(rows), "summary": summ,
            "per_seed": per_seed(rows), "verdict": verdict(summ, rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", type=Path, nargs="+", help="結果ファイルを集計して判定する")
    ap.add_argument("--phase", choices=("dev", "eval"), default="dev")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.analyze:
        print(json.dumps(analyze(args.analyze), ensure_ascii=False, indent=1))
        return
    seeds = args.seeds or (EVAL_SEEDS if args.phase == "eval" else DEV_SEEDS)
    out_path = args.out or Path(f"results/readout_dim/readout_dim_{args.phase}.json")
    with Pool(args.workers) as pool:
        rows = pool.map(run_job, jobs(seeds), chunksize=1)
    out = {"環境": env_info(), "説明": f"読み出しの出力に効く次元 K の操作（12.6.251〜252節）。phase={args.phase}",
           "seeds": seeds, "widths": WIDTHS, "ks": KS, "rules": RULES, "draws": DRAWS, "n_inputs": N_INPUTS,
           "fixes": FIXES, "rows": [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in x.items()} for x in rows]}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out_path, len(rows), "runs", flush=True)


if __name__ == "__main__":
    main()
