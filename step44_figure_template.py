#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ステップ44.5 Phase 2: Figure 作成テンプレート

目的:
- ステップ44.4 実験結果の可視化
- 3 phases の perplexity 比較、loss curves、module contribution 分析

実行環境:
- Colab または ローカル GPU
- 依存: matplotlib, numpy, json, pandas

使用方法:
  python step44_figure_template.py --results results/step44_final/results.json
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class Step44FigureGenerator:
    """ステップ44.4 実験結果の Figure 生成"""

    def __init__(self, results_file: str, output_dir: str = "results/step44_figures"):
        """
        Args:
            results_file: analyze_step44_results.py の output JSON
            output_dir: Figure 保存先ディレクトリ
        """
        self.results_file = Path(results_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.results_file, 'r') as f:
            self.results = json.load(f)

        self.phases = ['A', 'B', 'C']
        self.phase_names = {
            'A': 'Backbone + Hippocampus',
            'B': 'Backbone + Hippocampus + Basal Ganglia',
            'C': 'Backbone + Hippocampus + Basal Ganglia + Cerebellum'
        }
        self.colors = {
            'A': '#FF6B6B',  # red
            'B': '#4ECDC4',  # teal
            'C': '#95E1D3'   # mint
        }

    def plot_perplexity_comparison(self):
        """Figure 2: Perplexity comparison (3 phases, 3 seeds, error bar)"""

        fig, ax = plt.subplots(figsize=(10, 6))

        phase_list = []
        means = []
        stds = []

        for phase in self.phases:
            if phase in self.results['results']:
                ppls = self.results['results'][phase]['test_perplexity']
                phase_list.append(phase)
                means.append(np.mean(ppls))
                stds.append(np.std(ppls))

        x_pos = np.arange(len(phase_list))
        colors_list = [self.colors[p] for p in phase_list]

        ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors_list,
               alpha=0.7, edgecolor='black', linewidth=1.5)

        ax.set_xlabel('Phase', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test Perplexity', fontsize=12, fontweight='bold')
        ax.set_title('Perplexity Comparison across Phases', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([self.phase_names[p] for p in phase_list], rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)

        # 数値を bar の上に表示
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.2, f'{mean:.2f}±{std:.2f}',
                   ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure2_perplexity_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir / 'figure2_perplexity_comparison.png'}")
        plt.close()

    def plot_loss_curves(self):
        """Figure 3: Loss curves per phase (epoch-wise training dynamics)"""

        if 'loss_curves' not in self.results:
            print("⚠ Loss curves not in results. Skipping Figure 3.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for idx, phase in enumerate(self.phases):
            ax = axes[idx]

            if phase in self.results['loss_curves']:
                for seed in range(3):
                    train_losses = self.results['loss_curves'][phase][f'seed_{seed}']['train']
                    val_losses = self.results['loss_curves'][phase][f'seed_{seed}']['val']

                    epochs = np.arange(1, len(train_losses) + 1)
                    ax.plot(epochs, train_losses, 'o-', label=f'Seed {seed} (Train)', alpha=0.6)
                    ax.plot(epochs, val_losses, 's--', label=f'Seed {seed} (Val)', alpha=0.6)

            ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
            ax.set_title(f'Phase {phase}: {self.phase_names[phase].split("+")[-1].strip()}',
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure3_loss_curves.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir / 'figure3_loss_curves.png'}")
        plt.close()

    def plot_module_contribution(self):
        """Figure 4: Module contribution breakdown (stacked bar chart)"""

        if 'module_contribution' not in self.results:
            print("⚠ Module contribution not in results. Skipping Figure 4.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Phase B: Hippocampus + Basal Ganglia
        contrib_b = self.results['module_contribution']['phase_B']
        backbone_pct = contrib_b['backbone_loss_ratio']
        hc_pct = contrib_b['hippocampus_loss_ratio']
        bg_pct = contrib_b['basal_ganglia_loss_ratio']

        labels_b = ['Backbone', 'Hippocampus', 'Basal Ganglia']
        sizes_b = [backbone_pct, hc_pct, bg_pct]
        colors_b = ['#FF6B6B', '#4ECDC4', '#95E1D3']

        axes[0].bar(labels_b, sizes_b, color=colors_b, edgecolor='black', linewidth=1.5)
        axes[0].set_ylabel('Loss Contribution (%)', fontsize=11, fontweight='bold')
        axes[0].set_title('Phase B: Module Contribution', fontsize=12, fontweight='bold')
        axes[0].set_ylim([0, 100])
        axes[0].grid(axis='y', alpha=0.3)

        # 数値を bar の上に表示
        for i, v in enumerate(sizes_b):
            axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Phase C: Hippocampus + Basal Ganglia + Cerebellum
        contrib_c = self.results['module_contribution']['phase_C']
        backbone_pct = contrib_c['backbone_loss_ratio']
        hc_pct = contrib_c['hippocampus_loss_ratio']
        bg_pct = contrib_c['basal_ganglia_loss_ratio']
        cb_pct = contrib_c['cerebellum_loss_ratio']

        labels_c = ['Backbone', 'Hippocampus', 'Basal Ganglia', 'Cerebellum']
        sizes_c = [backbone_pct, hc_pct, bg_pct, cb_pct]
        colors_c = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F7DC6F']

        axes[1].bar(labels_c, sizes_c, color=colors_c, edgecolor='black', linewidth=1.5)
        axes[1].set_ylabel('Loss Contribution (%)', fontsize=11, fontweight='bold')
        axes[1].set_title('Phase C: Module Contribution', fontsize=12, fontweight='bold')
        axes[1].set_ylim([0, 100])
        axes[1].grid(axis='y', alpha=0.3)

        # 数値を bar の上に表示
        for i, v in enumerate(sizes_c):
            axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure4_module_contribution.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir / 'figure4_module_contribution.png'}")
        plt.close()

    def plot_statistical_significance(self):
        """Figure 5: Statistical significance (t-test, Cohen's d)"""

        if 'statistical_analysis' not in self.results:
            print("⚠ Statistical analysis not in results. Skipping Figure 5.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        stats_ab = self.results['statistical_analysis']['phase_A_vs_B']
        stats_bc = self.results['statistical_analysis']['phase_B_vs_C']

        comparisons = ['Phase A vs B', 'Phase B vs C']
        cohens_d = [stats_ab['cohens_d'], stats_bc['cohens_d']]
        p_values = [stats_ab['p_value'], stats_bc['p_value']]

        # Cohen's d visualization
        colors_d = ['green' if abs(d) > 0.5 else 'orange' if abs(d) > 0.2 else 'red' for d in cohens_d]
        axes[0].bar(comparisons, cohens_d, color=colors_d, edgecolor='black', linewidth=1.5)
        axes[0].set_ylabel("Cohen's d (Effect Size)", fontsize=11, fontweight='bold')
        axes[0].set_title("Effect Size Comparison", fontsize=12, fontweight='bold')
        axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0].grid(axis='y', alpha=0.3)

        # 数値を bar の上に表示
        for i, d in enumerate(cohens_d):
            axes[0].text(i, d + 0.05 if d > 0 else d - 0.1, f'{d:.3f}', ha='center', va='bottom' if d > 0 else 'top', fontsize=10, fontweight='bold')

        # P-value visualization
        colors_p = ['green' if p < 0.05 else 'red' for p in p_values]
        axes[1].bar(comparisons, [-np.log10(p) for p in p_values], color=colors_p, edgecolor='black', linewidth=1.5)
        axes[1].set_ylabel('-log10(p-value)', fontsize=11, fontweight='bold')
        axes[1].set_title('Statistical Significance (p < 0.05)', fontsize=12, fontweight='bold')
        axes[1].axhline(y=-np.log10(0.05), color='red', linestyle='--', linewidth=2, label='p = 0.05')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)

        # 数値を bar の上に表示
        for i, (p, neg_log_p) in enumerate(zip(p_values, [-np.log10(p) for p in p_values])):
            sig_marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            axes[1].text(i, neg_log_p + 0.1, f'{p:.4f}\n{sig_marker}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure5_statistical_significance.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir / 'figure5_statistical_significance.png'}")
        plt.close()

    def generate_all(self):
        """全 figure を生成"""
        print(f"\n{'='*60}")
        print("ステップ44.5 Figure 生成開始")
        print(f"{'='*60}\n")

        self.plot_perplexity_comparison()
        self.plot_loss_curves()
        self.plot_module_contribution()
        self.plot_statistical_significance()

        print(f"\n{'='*60}")
        print(f"✅ 全 Figure 生成完了: {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='ステップ44.5 Phase 2: Figure 生成')
    parser.add_argument('--results', type=str, required=True, help='analyze_step44_results.py の output JSON')
    parser.add_argument('--output-dir', type=str, default='results/step44_figures', help='Figure 保存先')

    args = parser.parse_args()

    generator = Step44FigureGenerator(args.results, args.output_dir)
    generator.generate_all()


if __name__ == '__main__':
    main()
