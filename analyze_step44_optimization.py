#!/usr/bin/env python3
"""
ステップ44.3 最適化検証結果の分析

最適化前後（フル規模 Step 44.4 vs 最適化検証 Step 44.3）の比較
"""

import json
from pathlib import Path
import statistics

def analyze_optimization():
    """最適化効果を分析"""

    # 最適化検証結果を読み込み
    try:
        with open('results/step44_optimized_validation/step44_optimized_validation_summary.json', 'r') as f:
            optimized = json.load(f)
    except FileNotFoundError:
        print("❌ Optimized validation results not found yet")
        return None

    # フル規模結果を読み込み
    with open('results/step44_full_scale_experiment/step44_full_scale_results.json', 'r') as f:
        full_scale = json.load(f)

    print("=" * 80)
    print("ステップ44.3 最適化検証 - 分析結果")
    print("=" * 80)
    print()

    # パラメータスイープから予測値を読み込み
    with open('results/step44_parameter_sweep/loss_weight_sweep_results.json', 'r') as f:
        sweep = json.load(f)
    best = sweep['best_config']

    print("【最適損失重み】")
    print(f"  Hippocampus: {best['hippocampus']:.2f}")
    print(f"  Basal Ganglia: {best['basal_ganglia']:.2f}")
    print(f"  Cerebellum: {best['cerebellum']:.2f}")
    print()

    print("【パラメータスイープ予測値】")
    print(f"  Phase A: {best['phase_a']:.4f}")
    print(f"  Phase B: {best['phase_b']:.4f}")
    print(f"  Phase C: {best['phase_c']:.4f}")
    pred_a_to_b = 100 * (best['phase_b'] - best['phase_a']) / best['phase_a']
    pred_b_to_c = 100 * (best['phase_c'] - best['phase_b']) / best['phase_b']
    print(f"  A→B: {pred_a_to_b:+.2f}%")
    print(f"  B→C: {pred_b_to_c:+.2f}%")
    print()

    print("【小規模検証結果】")
    for phase in ['A', 'B', 'C']:
        phase_summary = optimized['phase_summaries'][phase]
        mean_loss = phase_summary['mean_final_val_loss']
        std_loss = phase_summary['std_final_val_loss']
        print(f"  Phase {phase}: {mean_loss:.4f} ± {std_loss:.4f}")

    print()
    print("【改善分析】")
    summary = optimized['phase_summaries']

    a_loss = summary['A']['mean_final_val_loss']
    b_loss = summary['B']['mean_final_val_loss']
    c_loss = summary['C']['mean_final_val_loss']

    a_to_b_change = b_loss - a_loss
    a_to_b_pct = 100 * a_to_b_change / a_loss

    b_to_c_change = c_loss - b_loss
    b_to_c_pct = 100 * b_to_c_change / b_loss

    print(f"  Phase A→B: {a_to_b_change:+.4f} ({a_to_b_pct:+.2f}%)")
    print(f"  Phase B→C: {b_to_c_change:+.4f} ({b_to_c_pct:+.2f}%)")
    print()

    print("【予測値との比較】")
    print(f"  A→B: 予測 {pred_a_to_b:+.2f}%, 実測 {a_to_b_pct:+.2f}%")
    print(f"  B→C: 予測 {pred_b_to_c:+.2f}%, 実測 {b_to_c_pct:+.2f}%")
    print()

    # 結論
    print("【結論】")
    if a_to_b_pct < 0 and b_to_c_pct < 0:
        print("✓ 最適化損失重みにより両フェーズで改善を確認")
        print("  → フル規模再実験（ステップ44.4 再実行）を推奨")
    elif a_to_b_pct < 0 or b_to_c_pct < 0:
        print("⚠ 片方のフェーズのみ改善")
        print("  → 詳細分析および追加パラメータ調整の検討")
    else:
        print("❌ 改善が確認されない")
        print("  → パラメータ調整の再検討が必要")

if __name__ == "__main__":
    analyze_optimization()
