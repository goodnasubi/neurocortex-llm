"""ステップ18（マルチ学習則の学習安定性 — 小脳モジュールを加えた3モジュール構成の
再検証, 12.6.44節）の主実験。

8条件（independent-all / shared-all / shared-hippo-only / shared-rl-only /
shared-cereb-only / shared-hippo-rl / shared-hippo-cereb / shared-rl-cereb）を
比較し、(a) shared-allがindependent-allと比べて明確な劣化を示すか、
(b) ペア共有では平気だが3モジュール目を足すことで新たに劣化するヘッドがあるか
を判定する。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_multi_learning_rule_3way
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..basal_ganglia_core import ParitySpec
from ..multi_learning_rule_3way import CONDITIONS, run_condition


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-inputs", type=int, default=8)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--degrade-threshold", type=float, default=0.1,
                    help="正解率がこの差を超えて劣る、またはMSEがこの比率を超えて悪化すれば劣化とみなす")
    ap.add_argument("--out", type=Path,
                    default=Path("results/multi_learning_rule_3way/gradient_interference_3way.json"))
    args = ap.parse_args()

    t0 = time.time()
    spec = ParitySpec(n_inputs=args.n_inputs, relevant=(0, 1))

    all_results: dict[str, dict] = {
        c: {"hippo_acc": [], "rl_acc": [], "cereb_mse": [], "cosine_mean": {}} for c in CONDITIONS
    }
    for condition in CONDITIONS:
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            g = torch.Generator().manual_seed(seed + 50_000)
            r = run_condition(condition, spec, args.hidden_dim, args.steps, args.batch_size,
                              args.lr, args.n_eval, g)
            if r.hippo_acc is not None:
                all_results[condition]["hippo_acc"].append(r.hippo_acc)
            if r.rl_acc is not None:
                all_results[condition]["rl_acc"].append(r.rl_acc)
            if r.cereb_mse is not None:
                all_results[condition]["cereb_mse"].append(r.cereb_mse)
            for pair, series in r.cosine_similarities.items():
                all_results[condition]["cosine_mean"].setdefault(pair, []).append(
                    sum(series) / len(series))

    def _stats(values: list[float]) -> dict | None:
        if not values:
            return None
        t = torch.tensor(values)
        return {"mean": float(t.mean()), "std": float(t.std(unbiased=False)), "per_seed": t.tolist()}

    summary: dict[str, dict] = {}
    for condition in CONDITIONS:
        r = all_results[condition]
        summary[condition] = {
            "hippo_acc": _stats(r["hippo_acc"]),
            "rl_acc": _stats(r["rl_acc"]),
            "cereb_mse": _stats(r["cereb_mse"]),
            "cosine_similarity_mean_per_seed": {
                pair: _stats(values) for pair, values in r["cosine_mean"].items()
            },
        }

    def _acc_degraded(shared_mean: float, ref_mean: float) -> bool:
        return shared_mean < ref_mean - args.degrade_threshold

    def _mse_degraded(shared_mean: float, ref_mean: float, chance_mse: float) -> bool:
        """MSEは小さいほど良いので、比率で「悪化」を判定する。

        ただしハミング重み回帰は線形ヘッドで容易に解けるため、両条件ともMSEが
        チャンス水準に対してごく小さい（ほぼ0）場合、比率だけで判定すると
        測定ノイズレベルの差を「悪化」と誤検出する。そのためチャンス水準の
        1%を下回る絶対的な増分は無視する。
        """
        if shared_mean - ref_mean < chance_mse * 0.01:
            return False
        return shared_mean > ref_mean * (1.0 + args.degrade_threshold)

    indep_hippo = summary["independent-all"]["hippo_acc"]["mean"]
    indep_rl = summary["independent-all"]["rl_acc"]["mean"]
    indep_cereb = summary["independent-all"]["cereb_mse"]["mean"]

    shared_hippo = summary["shared-all"]["hippo_acc"]["mean"]
    shared_rl = summary["shared-all"]["rl_acc"]["mean"]
    shared_cereb = summary["shared-all"]["cereb_mse"]["mean"]

    # 統制: 単独共有条件がチャンス水準を明確に上回ること（分類0.5、回帰は定数予測のMSEを下回ること）。
    # 定数予測（平均値予測）のMSE = ハミング重みの分散。n_inputsビットの合計は
    # Binomial(n_inputs, 0.5) なので分散は n_inputs/4。
    chance_mse = args.n_inputs / 4.0
    # RLヘッドはステップ14・15でも独立学習時点でrl_acc=0.75〜0.88程度に留まる既知の天井効果
    # （12.6.11節）があるため、分類ヘッドと同じ0.9を閾値にすると健全な学習でも誤って
    # 「崩れている」と判定してしまう。RLについてはチャンス水準(0.5)を明確に上回る0.65を使う。
    control_ok = bool(
        summary["shared-hippo-only"]["hippo_acc"]["mean"] > 0.9 and
        summary["shared-rl-only"]["rl_acc"]["mean"] > 0.65 and
        summary["shared-cereb-only"]["cereb_mse"]["mean"] < chance_mse * 0.5
    )
    if not control_ok:
        print("警告: 統制（単独共有条件の健全性）が崩れています。")

    # 主張(a): shared-allの各ヘッドがindependent-allと比べて明確に劣化していないこと。
    claim_a_ok = bool(
        not _acc_degraded(shared_hippo, indep_hippo) and
        not _acc_degraded(shared_rl, indep_rl) and
        not _mse_degraded(shared_cereb, indep_cereb, chance_mse)
    )

    # 主張(b): ペア共有では劣化しないが、3モジュール目を足すと新たに劣化するヘッドがないこと。
    pair_hippo_rl_hippo = summary["shared-hippo-rl"]["hippo_acc"]["mean"]
    pair_hippo_rl_rl = summary["shared-hippo-rl"]["rl_acc"]["mean"]
    pair_hippo_cereb_hippo = summary["shared-hippo-cereb"]["hippo_acc"]["mean"]
    pair_hippo_cereb_cereb = summary["shared-hippo-cereb"]["cereb_mse"]["mean"]
    pair_rl_cereb_rl = summary["shared-rl-cereb"]["rl_acc"]["mean"]
    pair_rl_cereb_cereb = summary["shared-rl-cereb"]["cereb_mse"]["mean"]

    nonlinear_interference_ok = bool(
        not _acc_degraded(shared_hippo, min(pair_hippo_rl_hippo, pair_hippo_cereb_hippo)) and
        not _acc_degraded(shared_rl, min(pair_hippo_rl_rl, pair_rl_cereb_rl)) and
        not _mse_degraded(shared_cereb, max(pair_hippo_cereb_cereb, pair_rl_cereb_cereb), chance_mse)
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "ステップ18 マルチ学習則の学習安定性（小脳モジュールを加えた3モジュール構成の"
                "再検証）: shared-all（本設計）対independent-all（統制1）対 単独共有・ペア共有の、"
                "同一XORパリティ課題（海馬・基底核）とハミング重み回帰（小脳）での"
                "3ヘッド最終性能とバックボーン勾配コサイン類似度比較（12.6.44節）",
        "条件": vars(args) | {"out": str(args.out)},
        "chance_mse（定数予測の理論MSE）": chance_mse,
        "control_ok": control_ok,
        "claim_a_ok（3モジュール共有がindependent-allと比べ劣化しないか）": claim_a_ok,
        "claim_b_ok（3モジュール目追加による非線形な追加劣化がないか）": nonlinear_interference_ok,
        "summary": summary,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n書き出し: {args.out}")
    print("-- 本実験 --")
    for condition in CONDITIONS:
        s = summary[condition]
        h = s["hippo_acc"]
        r = s["rl_acc"]
        c = s["cereb_mse"]
        h_str = f"{h['mean']:.3f}±{h['std']:.3f}" if h else "N/A"
        r_str = f"{r['mean']:.3f}±{r['std']:.3f}" if r else "N/A"
        c_str = f"{c['mean']:.3f}±{c['std']:.3f}" if c else "N/A"
        print(f"{condition:>20}  hippo_acc={h_str:>14}  rl_acc={r_str:>14}  cereb_mse={c_str}")
    print(f"\n統制 (control_ok): {control_ok}")
    print(f"主張(a) (claim_a_ok): {claim_a_ok}")
    print(f"主張(b) (claim_b_ok): {nonlinear_interference_ok}")


if __name__ == "__main__":
    main()
