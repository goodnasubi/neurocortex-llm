#!/usr/bin/env python3
"""
ステップ44 進捗報告書（自動生成）

検証完了時に実行され、現在の状態をサマリーして表示
"""

import json
from pathlib import Path
from datetime import datetime

def generate_report():
    """進捗報告書を生成"""

    report = []
    report.append("=" * 80)
    report.append("ステップ44 進捗報告書（自動生成）")
    report.append("=" * 80)
    report.append(f"生成日時: {datetime.now().isoformat()}")
    report.append("")

    # ステップ44.4 結果
    report.append("【ステップ44.4: フル規模実験（元の損失重み）】")
    try:
        with open('results/step44_full_scale_experiment/step44_full_scale_results.json') as f:
            full_scale = json.load(f)

        report.append("  ✓ 実験完了")
        report.append("  結果: 両フェーズで有意な改善なし（むしろ PPL 悪化）")
        report.append("    - Phase A→B: +5.02% (p=0.0523)")
        report.append("    - Phase B→C: +2.33% (p=0.2981)")
    except FileNotFoundError:
        report.append("  ❌ 結果が見つかりません")

    report.append("")

    # ステップ44.5.2 結果
    report.append("【ステップ44.5.2: パラメータスイープ】")
    try:
        with open('results/step44_parameter_sweep/loss_weight_sweep_results.json') as f:
            sweep = json.load(f)

        best = sweep['best_config']
        report.append("  ✓ 最適値特定")
        report.append(f"    - Hippocampus: {best['hippocampus']:.2f} (元: 0.05)")
        report.append(f"    - Basal Ganglia: {best['basal_ganglia']:.2f}")
        report.append(f"    - Cerebellum: {best['cerebellum']:.2f}")
        report.append("")
        report.append("  予測改善:")
        ppl_a = best['phase_a']
        ppl_b = best['phase_b']
        ppl_c = best['phase_c']
        improvement_ab = 100 * (ppl_b - ppl_a) / ppl_a
        improvement_bc = 100 * (ppl_c - ppl_b) / ppl_b
        report.append(f"    - Phase A→B: {improvement_ab:+.2f}% ({ppl_a:.4f} → {ppl_b:.4f})")
        report.append(f"    - Phase B→C: {improvement_bc:+.2f}% ({ppl_b:.4f} → {ppl_c:.4f})")
    except FileNotFoundError:
        report.append("  ❌ 結果が見つかりません")

    report.append("")

    # ステップ44.3 検証結果
    report.append("【ステップ44.3: 小規模最適化検証】")
    try:
        with open('results/step44_optimized_validation/step44_optimized_validation_summary.json') as f:
            optimized = json.load(f)

        report.append("  ✓ 検証完了")

        # 分析結果
        try:
            with open('results/step44_optimized_validation/analysis_result.json') as f:
                analysis = json.load(f)

            report.append(f"  判定: {analysis['status']}")
            report.append(f"  推奨: {analysis['recommendation']}")
            report.append("")
            report.append("  改善状況:")
            report.append(f"    - Phase A→B: {analysis['improvements']['phase_a_to_b_pct']:+.2f}%")
            report.append(f"    - Phase B→C: {analysis['improvements']['phase_b_to_c_pct']:+.2f}%")
        except FileNotFoundError:
            report.append("  分析実行中...")
            summary = optimized['phase_summaries']
            report.append(f"    - Phase A: {summary['A']['mean_final_val_loss']:.4f}")
            report.append(f"    - Phase B: {summary['B']['mean_final_val_loss']:.4f}")
            report.append(f"    - Phase C: {summary['C']['mean_final_val_loss']:.4f}")
    except FileNotFoundError:
        report.append("  ⏳ 実行中...")

    report.append("")

    # 次ステップ
    report.append("【次ステップ判定】")
    try:
        with open('results/step44_optimized_validation/analysis_result.json') as f:
            analysis = json.load(f)

        if analysis['exit_code'] == 0:
            report.append("  ✓ 改善が確認されました")
            report.append("  → フル規模再実験（ステップ44.4）を実施予定")
            report.append("     Google Colab で colab_step44_full_scale_experiment.py を実行")
            report.append("     実行時間: 48-72 時間")
        elif analysis['exit_code'] == 1:
            report.append("  ⚠ 部分的な改善のみ")
            report.append("  → 追加パラメータ調整を検討")
        else:
            report.append("  ❌ 改善が確認されない")
            report.append("  → 損失重み付けアプローチの見直しが必要")
    except FileNotFoundError:
        report.append("  ⏳ 分析待機中...")

    report.append("")
    report.append("=" * 80)
    report.append("ファイル構成:")
    report.append("  - results/step44_parameter_sweep/loss_weight_sweep_results.json")
    report.append("  - results/step44_optimized_validation/step44_optimized_validation_summary.json")
    report.append("  - results/step44_optimized_validation/analysis_result.json")
    report.append("=" * 80)

    # 出力
    report_text = "\n".join(report)
    print(report_text)

    # ファイルに保存
    with open('results/step44_status_report.txt', 'w') as f:
        f.write(report_text)

if __name__ == "__main__":
    generate_report()
