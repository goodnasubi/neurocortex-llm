#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ステップ44.3 結果分析（軽量版 - scipy 不要）

損失値の比較・可視化のみ
"""

import json
import sys
from pathlib import Path
import numpy as np

def load_results(json_file):
    """JSON ファイルから結果を読み込み"""
    with open(json_file, 'r') as f:
        return json.load(f)

def analyze_results(results):
    """結果を分析"""
    print("\n" + "="*80)
    print("ステップ44.3 フェーズ別結果分析")
    print("="*80)

    phases = ['A', 'B', 'C']
    phase_names = {
        'A': '海馬のみ',
        'B': '海馬 + 基底核',
        'C': '全モジュール（海馬 + 基底核 + 小脳）'
    }

    summary = {}

    for phase in phases:
        if phase not in results:
            continue

        phase_data = results[phase][0]  # seed 0
        train_losses = phase_data['train_losses']
        val_losses = phase_data['val_losses']

        summary[phase] = {
            'name': phase_names[phase],
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'avg_train_loss': np.mean(train_losses),
            'avg_val_loss': np.mean(val_losses),
            'train_losses': train_losses,
            'val_losses': val_losses
        }

        print(f"\n【Phase {phase}: {phase_names[phase]}】")
        print(f"  訓練損失:  最終値={train_losses[-1]:.4f}, 平均={np.mean(train_losses):.4f}")
        print(f"  検証損失:  最終値={val_losses[-1]:.4f}, 平均={np.mean(val_losses):.4f}")
        print(f"  学習曲線: {train_losses}")

    # フェーズ間比較
    print("\n" + "="*80)
    print("フェーズ間比較")
    print("="*80)

    if len(summary) >= 3:
        loss_A = summary['A']['final_train_loss']
        loss_B = summary['B']['final_train_loss']
        loss_C = summary['C']['final_train_loss']

        print(f"\nPhase A (基準): {loss_A:.4f}")
        print(f"Phase B (A→B の増分): {loss_B:.4f} ({loss_B - loss_A:+.4f})")
        print(f"Phase C (B→C の増分): {loss_C:.4f} ({loss_C - loss_B:+.4f})")

        if loss_B > loss_A:
            print(f"\n⚠️  警告: 基底核統合後に損失が増加 ({loss_B - loss_A:+.4f})")
        if loss_C > loss_B:
            print(f"⚠️  警告: 小脳統合後に損失が増加 ({loss_C - loss_B:+.4f})")

        # 判定
        print("\n" + "="*80)
        print("ステップ44.4 進行判定")
        print("="*80)

        if loss_B <= loss_A and loss_C <= loss_B:
            print("\n✅ 推奨: ステップ44.4 へ進行可能")
            print("   理由: 段階的統合で損失が安定している")
        else:
            print("\n⚠️  要検討: ステップ44.3 の設計を見直し")
            print("   理由: 段階的統合で損失が増加している可能性")
            print("\n対応案:")
            print("  1. モジュール Loss 重み付けを調整（hippocampus_loss_weight など）")
            print("  2. 基底核・小脳モジュールの初期化を改善")
            print("  3. 学習率・バッチサイズの最適化")

    return summary

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用法: python analyze_step44_results_lite.py <結果ファイル>")
        print("例: python analyze_step44_results_lite.py results/step44_phase_validation/step44_phase_validation_results.json")
        sys.exit(1)

    json_file = sys.argv[1]
    if not Path(json_file).exists():
        print(f"エラー: ファイルが見つかりません: {json_file}")
        sys.exit(1)

    results = load_results(json_file)
    analyze_results(results)

    print("\n" + "="*80)
    print("分析完了")
    print("="*80)
