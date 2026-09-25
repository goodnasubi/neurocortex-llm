#!/usr/bin/env python3
"""
ステップ44.5: 結果分析・統計検定スクリプト

目的:
- Colab 実験結果（JSON）の統合分析
- Paired t-test による条件A vs B の有意性検定
- 効果量（Cohen's d）の計測
- Loss曲線・PPL 可視化
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='darkgrid')


class Step44Analyzer:
    """ステップ44 結果分析"""

    def __init__(self, result_file: Path):
        self.result_file = Path(result_file)
        with open(self.result_file, 'r') as f:
            self.data = json.load(f)

        self.experiments = self.data.get('experiments', [])
        self.config = self.data.get('config', {})

    def extract_by_condition(self, condition: str) -> List[Dict]:
        """条件ごとに実験結果を抽出"""
        return [exp for exp in self.experiments if exp.get('condition') == condition]

    def compute_final_ppl(self, condition: str) -> List[float]:
        """各 seed の最終 PPL を抽出"""
        experiments = self.extract_by_condition(condition)
        ppls = []
        for exp in experiments:
            val_ppls = exp.get('val_perplexities', [])
            if val_ppls:
                if isinstance(val_ppls[0], list):
                    # (step, ppl) ペア形式
                    final_ppl = val_ppls[-1][1]
                else:
                    # 単純な PPL リスト
                    final_ppl = val_ppls[-1]
                ppls.append(final_ppl)
        return ppls

    def paired_t_test(self) -> Dict:
        """Paired t-test: 条件A vs B"""
        ppls_a = self.compute_final_ppl('A')
        ppls_b = self.compute_final_ppl('B')

        if len(ppls_a) != len(ppls_b):
            print(f"Warning: Condition A has {len(ppls_a)} seeds, B has {len(ppls_b)}")

        # Paired t-test
        min_seeds = min(len(ppls_a), len(ppls_b))
        ppls_a = ppls_a[:min_seeds]
        ppls_b = ppls_b[:min_seeds]

        t_stat, p_value = stats.ttest_rel(ppls_a, ppls_b)

        # Cohen's d (効果量)
        diffs = np.array(ppls_a) - np.array(ppls_b)
        cohens_d = np.mean(diffs) / np.std(diffs) if np.std(diffs) > 0 else 0

        # 改善率
        improvement_pct = (np.mean(ppls_a) - np.mean(ppls_b)) / np.mean(ppls_a) * 100

        return {
            'condition_a_ppl': ppls_a,
            'condition_b_ppl': ppls_b,
            'mean_a': np.mean(ppls_a),
            'mean_b': np.mean(ppls_b),
            'std_a': np.std(ppls_a),
            'std_b': np.std(ppls_b),
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'cohens_d': cohens_d,
            'improvement_pct': improvement_pct
        }

    def plot_ppl_comparison(self, output_path: Path = None):
        """PPL の条件間比較 (box plot)"""
        ppls_a = self.compute_final_ppl('A')
        ppls_b = self.compute_final_ppl('B')

        fig, ax = plt.subplots(figsize=(8, 6))
        data = [ppls_a, ppls_b]
        ax.boxplot(data, labels=['Condition A (Backbone)', 'Condition B (+ Hippocampus)'])
        ax.set_ylabel('Perplexity (final epoch)')
        ax.set_title('ステップ44.2: 条件A vs B の PPL 比較')
        ax.grid(True, alpha=0.3)

        if output_path:
            plt.savefig(output_path / 'ppl_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_loss_curves(self, output_path: Path = None):
        """Loss 曲線の可視化"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 条件A
        for seed, exp in enumerate(self.extract_by_condition('A')):
            losses = exp.get('train_losses', [])
            axes[0].plot(losses, label=f'Seed {seed}', alpha=0.7)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Condition A: Backbone Only')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 条件B
        for seed, exp in enumerate(self.extract_by_condition('B')):
            losses = exp.get('train_losses', [])
            axes[1].plot(losses, label=f'Seed {seed}', alpha=0.7)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Condition B: + Hippocampus Module')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        if output_path:
            plt.savefig(output_path / 'loss_curves.png', dpi=150, bbox_inches='tight')
        plt.close()

    def generate_report(self, output_dir: Path = None) -> str:
        """テキスト形式レポート生成"""
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = self.result_file.parent

        # 統計検定
        stats_result = self.paired_t_test()

        report = f"""
{'='*80}
ステップ44.2 実験結果分析レポート
{'='*80}

【実験設定】
- データセット: {self.config.get('dataset_name', 'N/A')} ({self.config.get('dataset_config', 'N/A')})
- モデル: {self.config.get('backbone_model_id', 'N/A')}
- Epochs: {self.config.get('num_epochs', 'N/A')}
- Seeds: {self.config.get('num_seeds', 'N/A')}
- Batch Size: {self.config.get('batch_size', 'N/A')}

【実験結果】

条件A（Backbone のみ）:
  - 平均 PPL: {stats_result['mean_a']:.2f} ± {stats_result['std_a']:.2f}
  - サンプル: {stats_result['condition_a_ppl']}

条件B（+ 海馬モジュール）:
  - 平均 PPL: {stats_result['mean_b']:.2f} ± {stats_result['std_b']:.2f}
  - サンプル: {stats_result['condition_b_ppl']}

【統計検定】
Paired t-test (条件A vs B):
  - t 統計量: {stats_result['t_statistic']:.4f}
  - p 値: {stats_result['p_value']:.6f}
  - 有意性（α=0.05）: {'★ 有意に改善' if stats_result['is_significant'] and stats_result['improvement_pct'] > 0 else '✗ 有意差なし'}
  - Cohen's d: {stats_result['cohens_d']:.4f}
  - 改善率: {stats_result['improvement_pct']:.2f}%

【判定】
"""
        if stats_result['is_significant']:
            if stats_result['improvement_pct'] > 0:
                report += f"""
✓ 海馬モジュール統合により、ステップ44.2 の目標を達成しました。
  PPL が統計的に有意に改善（p < 0.05）し、
  「脳型LLM の実装実証」の基礎が確立されました。

  次フェーズ: ステップ44.3（基底核統合）→ ステップ44.4（フル規模） へ進行
"""
            else:
                report += f"""
✗ 海馬モジュール統合により PPL が悪化しました（統計的に有意）。
  モジュール設計 or 統合方法を再検討してください。
"""
        else:
            report += f"""
⚠ 有意な改善が検出されませんでした。
  - 改善傾向はあるが、サンプル数（{len(stats_result['condition_a_ppl'])} seeds）での統計力が不足？
  - より多くの seed/epoch での再実験、or 設計修正を検討。
"""

        report += f"""
{'='*80}
生成日時: {Path.cwd()}
"""

        # ファイルに保存
        report_file = output_dir / 'step44_analysis_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        # 図表生成
        self.plot_ppl_comparison(output_dir)
        self.plot_loss_curves(output_dir)

        return report


def main():
    """分析実行"""
    # 結果ファイル指定
    import sys
    if len(sys.argv) > 1:
        result_file = Path(sys.argv[1])
    else:
        # デフォルト: Colab 実験結果ディレクトリ
        result_file = Path('results/step44_hippocampus_validation/step44_full_experiment_results.json')

    if not result_file.exists():
        print(f"Error: Result file not found: {result_file}")
        return

    analyzer = Step44Analyzer(result_file)
    output_dir = result_file.parent
    report = analyzer.generate_report(output_dir)
    print(report)
    print(f"\n✓ Analysis complete. Report saved to: {output_dir}")


if __name__ == "__main__":
    main()
