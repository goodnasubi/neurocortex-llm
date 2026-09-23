"""ステップ35: キー単位走査への見直し（方式B: 固定k件）の実験スクリプト。

従来の線形走査 vs 候補絞り込み版の性能比較。
"""

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from src.neurocortex.hippocampus import DiskBackedAssociativeStore


def run_experiment():
    """実験実行・結果保存。"""

    key_dim = 128
    value_dim = 64
    key_chunk = 512
    beta = 50.0

    # 実験条件
    N_list = [100000, 300000]
    seeds = [0, 1, 2]
    num_queries_per_run = 32
    num_runs = 5

    results = {}

    for N in N_list:
        results[str(N)] = {}
        print(f"\n{'='*50}")
        print(f"N={N}")
        print(f"{'='*50}")

        for seed in seeds:
            print(f"\n=== seed={seed} ===")
            g = torch.Generator().manual_seed(seed)

            with tempfile.TemporaryDirectory() as tmpdir:
                # ストア初期化
                store = DiskBackedAssociativeStore(
                    key_dim=key_dim,
                    value_dim=value_dim,
                    persist_dir=tmpdir,
                    beta=beta,
                    exact=True,
                    key_chunk=key_chunk,
                )

                # ストアにデータ書き込み
                print(f"  ストアにデータ書き込み中...")
                keys_data = torch.randn(N, key_dim, generator=g)
                values_data = torch.randn(N, value_dim, generator=g)
                store.write(keys_data, values_data)

                # (i) 従来の線形走査 read()
                print(f"  (i) 従来の線形走査", end="")
                read_times_baseline = []
                for run in range(num_runs):
                    query_keys = torch.randn(num_queries_per_run, key_dim, generator=g)
                    start = time.perf_counter()
                    _, _ = store.read(query_keys)
                    elapsed = time.perf_counter() - start
                    read_times_baseline.append(elapsed)
                    print(".", end="", flush=True)
                print()

                baseline_median = np.median(read_times_baseline)
                baseline_p25 = np.percentile(read_times_baseline, 25)
                baseline_p75 = np.percentile(read_times_baseline, 75)
                print(f"      中央値: {baseline_median:.4f}秒")

                # (ii) k = sqrt(N) での候補絞り込み
                k_sqrt_n = max(1, int(N**0.5))
                print(f"  (ii) k={k_sqrt_n} (sqrt(N))", end="")
                read_times_sqrt_n = []
                argmax_match_count_sqrt_n = 0
                candidate_coverage_sqrt_n = 0

                for run in range(num_runs):
                    query_keys = torch.randn(num_queries_per_run, key_dim, generator=g)
                    logits = torch.randn(num_queries_per_run, N, generator=g)
                    attention_weights = torch.softmax(logits, dim=-1)

                    start = time.perf_counter()
                    values_cand, stats_cand = store.read_with_key_candidates(
                        query_keys, attention_weights, k_candidate=k_sqrt_n
                    )
                    elapsed = time.perf_counter() - start
                    read_times_sqrt_n.append(elapsed)

                    # 従来版と比較
                    values_ref, stats_ref = store.read(query_keys)
                    argmax_match_count_sqrt_n += (stats_cand.top1_index == stats_ref.top1_index).sum().item()

                    # 候補内カバレッジ
                    for b in range(num_queries_per_run):
                        candidates = store._get_key_candidates(attention_weights[b], k_sqrt_n)
                        true_best = int(stats_ref.top1_index[b])
                        if true_best >= 0 and true_best in candidates:
                            candidate_coverage_sqrt_n += 1
                    print(".", end="", flush=True)
                print()

                sqrt_n_median = np.median(read_times_sqrt_n)
                sqrt_n_reduction = (1 - sqrt_n_median / baseline_median) * 100
                argmax_match_rate_sqrt_n = (argmax_match_count_sqrt_n / (num_runs * num_queries_per_run)) * 100
                candidate_coverage_rate_sqrt_n = (candidate_coverage_sqrt_n / (num_runs * num_queries_per_run)) * 100

                print(f"      中央値: {sqrt_n_median:.4f}秒 (削減: {sqrt_n_reduction:.1f}%)")
                print(f"      argmax一致率: {argmax_match_rate_sqrt_n:.1f}%")
                print(f"      候補内カバレッジ: {candidate_coverage_rate_sqrt_n:.1f}%")

                # (iii) k = sqrt(N)/2 での候補絞り込み
                k_half_sqrt_n = max(1, int(N**0.5 / 2))
                print(f"  (iii) k={k_half_sqrt_n} (sqrt(N)/2)", end="")
                read_times_half_sqrt_n = []
                argmax_match_count_half_sqrt_n = 0
                candidate_coverage_half_sqrt_n = 0

                for run in range(num_runs):
                    query_keys = torch.randn(num_queries_per_run, key_dim, generator=g)
                    logits = torch.randn(num_queries_per_run, N, generator=g)
                    attention_weights = torch.softmax(logits, dim=-1)

                    start = time.perf_counter()
                    values_cand, stats_cand = store.read_with_key_candidates(
                        query_keys, attention_weights, k_candidate=k_half_sqrt_n
                    )
                    elapsed = time.perf_counter() - start
                    read_times_half_sqrt_n.append(elapsed)

                    # 従来版と比較
                    values_ref, stats_ref = store.read(query_keys)
                    argmax_match_count_half_sqrt_n += (stats_cand.top1_index == stats_ref.top1_index).sum().item()

                    # 候補内カバレッジ
                    for b in range(num_queries_per_run):
                        candidates = store._get_key_candidates(attention_weights[b], k_half_sqrt_n)
                        true_best = int(stats_ref.top1_index[b])
                        if true_best >= 0 and true_best in candidates:
                            candidate_coverage_half_sqrt_n += 1
                    print(".", end="", flush=True)
                print()

                half_sqrt_n_median = np.median(read_times_half_sqrt_n)
                half_sqrt_n_reduction = (1 - half_sqrt_n_median / baseline_median) * 100
                argmax_match_rate_half_sqrt_n = (argmax_match_count_half_sqrt_n / (num_runs * num_queries_per_run)) * 100
                candidate_coverage_rate_half_sqrt_n = (candidate_coverage_half_sqrt_n / (num_runs * num_queries_per_run)) * 100

                print(f"      中央値: {half_sqrt_n_median:.4f}秒 (削減: {half_sqrt_n_reduction:.1f}%)")
                print(f"      argmax一致率: {argmax_match_rate_half_sqrt_n:.1f}%")
                print(f"      候補内カバレッジ: {candidate_coverage_rate_half_sqrt_n:.1f}%")

                # 結果を記録
                seed_key = str(seed)
                results[str(N)][seed_key] = {
                    "baseline": {
                        "median": float(baseline_median),
                        "p25": float(baseline_p25),
                        "p75": float(baseline_p75),
                    },
                    "sqrt_n": {
                        "k": int(k_sqrt_n),
                        "median": float(sqrt_n_median),
                        "reduction_pct": float(sqrt_n_reduction),
                        "argmax_match_rate": float(argmax_match_rate_sqrt_n),
                        "candidate_coverage_rate": float(candidate_coverage_rate_sqrt_n),
                    },
                    "half_sqrt_n": {
                        "k": int(k_half_sqrt_n),
                        "median": float(half_sqrt_n_median),
                        "reduction_pct": float(half_sqrt_n_reduction),
                        "argmax_match_rate": float(argmax_match_rate_half_sqrt_n),
                        "candidate_coverage_rate": float(candidate_coverage_rate_half_sqrt_n),
                    },
                }

    # 結果保存
    results_dir = Path(__file__).parent.parent.parent / "results" / "key_candidate_scan"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "experiment_results.json"

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n結果を保存: {results_file}")
    print("\n=== 実験完了 ===")


if __name__ == "__main__":
    run_experiment()
