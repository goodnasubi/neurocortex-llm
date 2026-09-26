#!/usr/bin/env python3
"""
ステップ44.3 検証結果の処理と判定

1. 最適化検証結果の読み込み
2. 予測値との比較分析
3. 改善判定
4. 次ステップの推奨
"""

import json
from pathlib import Path
import sys

def main():
    """検証結果を処理して次ステップを推奨"""

    print("=" * 80)
    print("ステップ44.3 最適化検証 - 結果分析")
    print("=" * 80)
    print()

    # 各ファイルの存在確認
    optimized_file = Path('results/step44_optimized_validation/step44_optimized_validation_summary.json')
    sweep_file = Path('results/step44_parameter_sweep/loss_weight_sweep_results.json')

    if not optimized_file.exists():
        print(f"❌ 検証結果ファイルが見つかりません: {optimized_file}")
        print("   ステップ44.3 がまだ実行中の可能性があります")
        return 1

    if not sweep_file.exists():
        print(f"❌ パラメータスイープ結果が見つかりません: {sweep_file}")
        return 1

    # 結果を読み込み
    with open(optimized_file, 'r') as f:
        optimized = json.load(f)

    with open(sweep_file, 'r') as f:
        sweep = json.load(f)

    best_config = sweep['best_config']

    # 分析を実行
    print("【最適損失重み（パラメータスイープから）】")
    print(f"  Hippocampus: {best_config['hippocampus']:.2f}")
    print(f"  Basal Ganglia: {best_config['basal_ganglia']:.2f}")
    print(f"  Cerebellum: {best_config['cerebellum']:.2f}")
    print()

    print("【小規模検証結果（最終 validation loss）】")
    summary = optimized['phase_summaries']

    results = {}
    for phase in ['A', 'B', 'C']:
        mean_loss = summary[phase]['mean_final_val_loss']
        std_loss = summary[phase]['std_final_val_loss']
        results[phase] = (mean_loss, std_loss)
        print(f"  Phase {phase}: {mean_loss:.4f} ± {std_loss:.4f}")

    print()
    print("【改善分析】")

    a_loss, a_std = results['A']
    b_loss, b_std = results['B']
    c_loss, c_std = results['C']

    a_to_b_change = b_loss - a_loss
    a_to_b_pct = 100 * a_to_b_change / a_loss

    b_to_c_change = c_loss - b_loss
    b_to_c_pct = 100 * b_to_c_change / b_loss

    print(f"  Phase A→B: {a_to_b_pct:+.2f}% (Δ = {a_to_b_change:+.4f})")
    print(f"  Phase B→C: {b_to_c_pct:+.2f}% (Δ = {b_to_c_change:+.4f})")
    print()

    # パラメータスイープ予測との比較
    pred_a = best_config['phase_a']
    pred_b = best_config['phase_b']
    pred_c = best_config['phase_c']

    pred_a_to_b_pct = 100 * (pred_b - pred_a) / pred_a
    pred_b_to_c_pct = 100 * (pred_c - pred_b) / pred_b

    print("【パラメータスイープ予測値との比較】")
    print(f"  A→B: 予測 {pred_a_to_b_pct:+.2f}%, 実測 {a_to_b_pct:+.2f}%")
    print(f"  B→C: 予測 {pred_b_to_c_pct:+.2f}%, 実測 {b_to_c_pct:+.2f}%")
    print()

    # 判定
    print("【判定】")
    improvement_threshold = -0.5  # 0.5% 以上の改善があれば OK

    a_to_b_improved = a_to_b_pct < improvement_threshold
    b_to_c_improved = b_to_c_pct < improvement_threshold

    if a_to_b_improved and b_to_c_improved:
        status = "✓ IMPROVEMENT CONFIRMED"
        recommendation = "フル規模再実験（ステップ44.4）を実施"
        code = 0
    elif a_to_b_improved or b_to_c_improved:
        status = "⚠ PARTIAL IMPROVEMENT"
        recommendation = "片方のフェーズのみ改善。追加検討後に判定"
        code = 1
    else:
        status = "❌ NO IMPROVEMENT"
        recommendation = "パラメータ調整の再検討が必要"
        code = 2

    print(f"  {status}")
    print()
    print("【推奨アクション】")
    print(f"  {recommendation}")
    print()

    # 結果を JSON で保存
    analysis_result = {
        'status': status,
        'recommendation': recommendation,
        'improvements': {
            'phase_a_to_b_pct': a_to_b_pct,
            'phase_b_to_c_pct': b_to_c_pct,
            'phase_a_to_b_improved': a_to_b_improved,
            'phase_b_to_c_improved': b_to_c_improved
        },
        'predictions_vs_measurements': {
            'a_to_b_prediction': pred_a_to_b_pct,
            'a_to_b_measurement': a_to_b_pct,
            'b_to_c_prediction': pred_b_to_c_pct,
            'b_to_c_measurement': b_to_c_pct
        },
        'exit_code': code
    }

    with open('results/step44_optimized_validation/analysis_result.json', 'w') as f:
        json.dump(analysis_result, f, indent=2)

    return code

if __name__ == "__main__":
    sys.exit(main())
