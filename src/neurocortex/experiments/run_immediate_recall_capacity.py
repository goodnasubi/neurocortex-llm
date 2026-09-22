"""ステップ21（海馬モジュールの即時想起率 — 容量圧迫下での劣化, 12.6.50節）の主実験。

ステップ19（`run_immediate_recall.py`）と同じモード×シードループ構造に、書き込み件数
（`n_facts`）を`n_units`に対する比率として振る「圧迫率」の軸を追加する。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_immediate_recall_capacity
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..immediate_recall import run_condition

MODES = ("sdr", "dense", "identity")

# 書き込み件数（n_units=2048に対する圧迫率）。50件はステップ19の再掲・統制用。
DEFAULT_N_FACTS_LIST = (50, 200, 500, 1000, 1600)

DEGRADATION_THRESHOLD = 0.95  # 劣化開始点の判定基準（想起率がこれを明確に下回り始める）


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--value-dim", type=int, default=64)
    ap.add_argument("--n-units", type=int, default=2048)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--n-facts-list", type=int, nargs="+", default=list(DEFAULT_N_FACTS_LIST))
    ap.add_argument("--n-unknown", type=int, default=50)
    ap.add_argument("--gate-slope", type=float, default=20.0)
    ap.add_argument("--gate-bias", type=float, default=0.3)
    ap.add_argument("--out", type=Path,
                    default=Path("results/immediate_recall/capacity_pressure.json"))
    args = ap.parse_args()

    t0 = time.time()

    # all_results[mode][n_facts] = {"recall_rate": [...], "false_positive_rate": [...]}
    all_results: dict[str, dict[int, dict]] = {
        m: {n: {"recall_rate": [], "false_positive_rate": []} for n in args.n_facts_list}
        for m in MODES
    }
    for mode in MODES:
        for n_facts in args.n_facts_list:
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                g = torch.Generator().manual_seed(seed + 60_000)
                r = run_condition(mode, args.d_model, args.value_dim, args.n_units, args.k,
                                  n_facts, args.n_unknown, args.gate_slope, args.gate_bias,
                                  g, seed)
                all_results[mode][n_facts]["recall_rate"].append(r.recall_rate)
                all_results[mode][n_facts]["false_positive_rate"].append(r.false_positive_rate)

    def _stats(values: list[float]) -> dict:
        t = torch.tensor(values)
        return {"mean": float(t.mean()), "std": float(t.std(unbiased=False)), "per_seed": t.tolist()}

    summary: dict[str, dict] = {}
    for mode in MODES:
        summary[mode] = {}
        for n_facts in args.n_facts_list:
            r = all_results[mode][n_facts]
            pressure_ratio = n_facts / args.n_units
            summary[mode][str(n_facts)] = {
                "pressure_ratio": pressure_ratio,
                "recall_rate": _stats(r["recall_rate"]),
                "false_positive_rate": _stats(r["false_positive_rate"]),
            }

    # 主張(a)（統制）: 圧迫率を上げると3モードとも即時想起率がステップ19の満点(1.000)から
    # 明確に低下する圧迫率が存在するか。
    claim_a_per_mode: dict[str, bool] = {}
    for mode in MODES:
        means = [summary[mode][str(n)]["recall_rate"]["mean"] for n in args.n_facts_list]
        claim_a_per_mode[mode] = bool(min(means) < 0.95)
    claim_a_control_ok = bool(all(claim_a_per_mode.values()))

    # 主張(b)（本命）: 同一圧迫率でのsdr優位が過半数の圧迫率で観測されるか。
    n_facts_sorted = sorted(args.n_facts_list)
    sdr_wins = 0
    per_pressure_winner: dict[str, dict] = {}
    for n in n_facts_sorted:
        recalls = {m: summary[m][str(n)]["recall_rate"]["mean"] for m in MODES}
        sdr_better_than_both = recalls["sdr"] > recalls["dense"] and recalls["sdr"] > recalls["identity"]
        if sdr_better_than_both:
            sdr_wins += 1
        per_pressure_winner[str(n)] = {"recall_rate": recalls, "sdr_strictly_best": sdr_better_than_both}
    claim_b_majority_ok = bool(sdr_wins > len(n_facts_sorted) / 2)

    # 主張(b)の代替条件: 劣化開始点（想起率が0.95を明確に下回り始める最小の圧迫率）が
    # sdrの方がdense・identityより高い（＝より高い圧迫率まで耐える）か。
    def _onset_n_facts(mode: str) -> float | None:
        for n in n_facts_sorted:
            if summary[mode][str(n)]["recall_rate"]["mean"] < DEGRADATION_THRESHOLD:
                return n
        return None  # 測定範囲内で劣化開始点に達しなかった

    onset = {m: _onset_n_facts(m) for m in MODES}
    # onsetがNone（劣化しなかった）場合は測定範囲を超えて耐えたとみなし、最大のn_factsより大きい扱いにする。
    onset_comparable = {m: (onset[m] if onset[m] is not None else max(n_facts_sorted) + 1)
                        for m in MODES}
    claim_b_onset_ok = bool(onset_comparable["sdr"] > onset_comparable["dense"]
                            and onset_comparable["sdr"] > onset_comparable["identity"])

    claim_b_ok = bool(claim_b_majority_ok or claim_b_onset_ok)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "ステップ21（海馬モジュールの即時想起率 — 容量圧迫下での劣化, 12.6.50節）: "
                "HippocampalMemoryの3分離層モード（sdr・dense・identity）で、書き込み件数を"
                "n_unitsに対する比率として振り、即時想起率・誤想起率の劣化を実測",
        "条件": vars(args) | {"out": str(args.out)},
        "圧迫率一覧（n_facts: pressure_ratio）": {
            str(n): n / args.n_units for n in n_facts_sorted
        },
        "claim_a_control_ok（圧迫率を上げると3モードとも想起率が1.000から明確に低下するか）":
            claim_a_control_ok,
        "claim_a_per_mode": claim_a_per_mode,
        "claim_b_ok（sdrが優位か: 過半数勝利 or 劣化開始点が遅い）": claim_b_ok,
        "claim_b_majority_ok（過半数の圧迫率でsdrが最良）": claim_b_majority_ok,
        "claim_b_onset_ok（劣化開始点がsdrの方が遅い）": claim_b_onset_ok,
        "sdr_wins_count": sdr_wins,
        "n_pressure_points": len(n_facts_sorted),
        "degradation_onset_n_facts（0.95を明確に下回り始める最小n_facts、Noneは範囲内で未到達）": onset,
        "per_pressure_winner": per_pressure_winner,
        "summary": summary,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n書き出し: {args.out}")
    print("-- 本実験（圧迫率 x モード） --")
    for n in n_facts_sorted:
        print(f"n_facts={n} (pressure={n / args.n_units:.1%})")
        for mode in MODES:
            s = summary[mode][str(n)]
            rr = s["recall_rate"]
            fp = s["false_positive_rate"]
            print(f"  {mode:>10}  recall_rate={rr['mean']:.3f}±{rr['std']:.3f}  "
                  f"false_positive_rate={fp['mean']:.3f}±{fp['std']:.3f}")
    print(f"\n主張(a) claim_a_control_ok: {claim_a_control_ok}  (per_mode={claim_a_per_mode})")
    print(f"主張(b) claim_b_ok: {claim_b_ok}  "
          f"(majority={claim_b_majority_ok} [{sdr_wins}/{len(n_facts_sorted)}], "
          f"onset={claim_b_onset_ok} {onset})")


if __name__ == "__main__":
    main()
