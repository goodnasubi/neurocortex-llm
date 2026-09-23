#!/usr/bin/env python3
"""ステップ40：ハイブリッド再ランク深度化による精度向上
フェーズ1-4の実装・実験スクリプト

設計概要：
- 段階1（粗選別）: GPU PQ で k=√N の候補を高速取得
- 段階2（精密再ランク）: 内積・密度・テンプレートの複合スコア
- 段階3（統合判定）: 上位k_refineで最終判定
"""

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from src.neurocortex.hippocampus import DiskBackedAssociativeStore


def measure_stage1_stage3_accuracy(
    store: DiskBackedAssociativeStore,
    test_keys: torch.Tensor,
    true_indices: torch.Tensor,
    k_candidates: int = 707,
    k_refine_values: list = None,
    use_gpu: bool = True,
) -> dict:
    """段階1+段階3の精度測定（再ランク適用）

    Args:
        store: DiskBackedAssociativeStore
        test_keys: テストキー [B, key_dim]
        true_indices: グラウンドトゥルース [B]
        k_candidates: 段階1の候補数（√N推奨）
        k_refine_values: 段階3の上位k値 [100, 150, 200]推奨
        use_gpu: GPU 使用フラグ

    Returns:
        {'stage1': {...}, 'stage3_k100': {...}, ...}
    """
    if k_refine_values is None:
        k_refine_values = [100, 150, 200]

    results = {}

    # 段階1：GPU/CPU PQ 検索
    t0 = time.time()
    if use_gpu:
        out, stats = store.read_with_pq_search_gpu(
            test_keys, k_candidates=k_candidates, M=16
        )
    else:
        out, stats = store.read_with_pq_search(
            test_keys, k_candidates=k_candidates, M=16
        )
    stage1_time = time.time() - t0
    stage1_correct = (stats.arg == true_indices).float().mean().item()

    results["stage1"] = {
        "time_ms": stage1_time * 1000,
        "accuracy": stage1_correct,
        "k_candidates": k_candidates,
    }

    # 段階2+段階3：複合スコア再ランク
    for k_refine in k_refine_values:
        t0 = time.time()
        out_rerank, stats_rerank = store.read_with_hybrid_rerank(
            test_keys,
            k_candidates=k_candidates,
            k_refine=k_refine,
            lambda_inner=0.6,
            lambda_density=0.3,
            lambda_template=0.1,
            use_gpu=use_gpu,
        )
        rerank_time = time.time() - t0
        rerank_correct = (stats_rerank.arg == true_indices).float().mean().item()

        results[f"stage3_k{k_refine}"] = {
            "time_ms": rerank_time * 1000,
            "accuracy": rerank_correct,
            "k_candidates": k_candidates,
            "k_refine": k_refine,
            "overhead_pct": ((rerank_time - stage1_time) / stage1_time * 100)
            if stage1_time > 0 else 0,
        }

    return results


def measure_score_contributions(
    store: DiskBackedAssociativeStore,
    test_keys: torch.Tensor,
    true_indices: torch.Tensor,
    k_candidates: int = 707,
    k_refine: int = 150,
    use_gpu: bool = True,
) -> dict:
    """複合スコア内の各項の寄与度を測定

    主張(c)の検証：内積+密度の寄与が95%以上か確認
    """
    # 各項を個別に測定
    scores_inner = store._compute_inner_scores(test_keys, k_candidates, k_refine, use_gpu)
    scores_density = store._compute_density_scores(test_keys, k_candidates, k_refine)
    scores_template = store._compute_template_scores(test_keys, k_candidates, k_refine)

    # 寄与度分析（スコア値の分散を指標とする）
    var_inner = np.var(scores_inner)
    var_density = np.var(scores_density)
    var_template = np.var(scores_template)

    total_var = var_inner + var_density + var_template

    return {
        "variance": {
            "inner": var_inner,
            "density": var_density,
            "template": var_template,
            "total": total_var,
        },
        "contribution_pct": {
            "inner": (var_inner / total_var * 100) if total_var > 0 else 0,
            "density": (var_density / total_var * 100) if total_var > 0 else 0,
            "template": (var_template / total_var * 100) if total_var > 0 else 0,
        },
    }


def run_benchmark(
    data_dir: Path = Path("results/associative_store_scaling"),
    scales: list = None,
) -> dict:
    """フェーズ2: 大規模ベンチマーク実行"""
    if scales is None:
        scales = [100000, 500000, 1000000]  # 100K, 500K, 1M

    all_results = {
        "timestamp": time.time(),
        "test_configs": [],
    }

    for N in scales:
        print(f"\n{'='*70}")
        print(f"N={N} でのベンチマーク開始")
        print(f"{'='*70}")

        # テストデータの生成
        key_dim = 768
        value_dim = 512
        test_keys = torch.randn(100, key_dim, dtype=torch.float32)
        true_indices = torch.arange(100)  # ダミー：実験時は真値で置き換え

        # ストア初期化
        store = DiskBackedAssociativeStore(
            key_dim=key_dim,
            value_dim=value_dim,
            file_path=data_dir / f"store_n{N}.db",
        )

        # ダミーデータを書き込み（実験時は実データを使用）
        for i in range(min(N, 1000)):  # 実験環境では小規模に
            key = torch.randn(key_dim, dtype=torch.float32)
            value = torch.randn(value_dim, dtype=torch.float64)
            store.write(key, value)

        k_candidates = int(np.sqrt(min(store.write_count, 500000)))

        # 段階1+段階3精度測定
        acc_results = measure_stage1_stage3_accuracy(
            store, test_keys, true_indices, k_candidates=k_candidates
        )

        # スコア寄与度分析
        contrib_results = measure_score_contributions(
            store, test_keys, true_indices, k_candidates=k_candidates
        )

        config_result = {
            "N": N,
            "k_candidates": k_candidates,
            "accuracy": acc_results,
            "contributions": contrib_results,
        }
        all_results["test_configs"].append(config_result)

        print(f"Stage 1 accuracy: {acc_results['stage1']['accuracy']:.4f}")
        for key, val in acc_results.items():
            if key != "stage1" and isinstance(val, dict):
                print(
                    f"  {key}: accuracy={val['accuracy']:.4f}, "
                    f"time={val['time_ms']:.1f}ms, "
                    f"overhead={val.get('overhead_pct', 0):.1f}%"
                )

    return all_results


def main():
    """メイン実行フロー"""
    data_dir = Path("results") / "hybrid_rerank"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("ステップ40: ハイブリッド再ランク深度化")
    print("='*70")

    # フェーズ2: 大規模ベンチマーク
    results = run_benchmark(data_dir, scales=[100000, 500000, 1000000])

    # 結果保存
    output_file = data_dir / "hybrid_rerank_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n結果を {output_file} に保存しました")

    # サマリー出力
    print("\n" + "="*70)
    print("ステップ40 ベンチマーク完了")
    print("="*70)
    for config in results["test_configs"]:
        print(f"\nN={config['N']}:")
        print(f"  Stage 1 accuracy: {config['accuracy']['stage1']['accuracy']:.4f}")
        for key, val in config["accuracy"].items():
            if key != "stage1" and isinstance(val, dict):
                print(
                    f"  {key}: {val['accuracy']:.4f} "
                    f"(overhead: {val.get('overhead_pct', 0):.1f}%)"
                )


if __name__ == "__main__":
    main()
