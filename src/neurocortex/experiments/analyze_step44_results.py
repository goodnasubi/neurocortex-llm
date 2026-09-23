#!/usr/bin/env python3
"""
ステップ44 結果分析・統計検定スクリプト

目的:
- ステップ44.1-44.5 の全実験結果を統合分析
- 条件 A vs B のパーライズd t-test 実施
- 可視化（loss curve, perplexity bar chart）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import logging
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """単一実験結果"""
    condition: str
    seed: int
    train_losses: List[float]
    val_perplexities: List[float]
    hippocampus_losses: List[float] = None


class Step44Analyzer:
    """ステップ44 結果分析クラス"""

    def __init__(self, results_dir: str = "results/step44_hippocampus_validation"):
        self.results_dir = Path(results_dir)
        self.results = []
        self.summary = {}

    def load_results(self, results_file: str) -> bool:
        """JSON 形式の実験結果をロード"""
        result_path = self.results_dir / results_file

        if not result_path.exists():
            logger.error(f"Results file not found: {result_path}")
            return False

        try:
            with open(result_path, 'r') as f:
                data = json.load(f)

            for exp in data.get('experiments', []):
                result = ExperimentResult(
                    condition=exp['condition'],
                    seed=exp['seed'],
                    train_losses=exp['train_losses'],
                    val_perplexities=exp['val_perplexities'],
                    hippocampus_losses=exp.get('hippocampus_losses', None)
                )
                self.results.append(result)

            logger.info(f"Loaded {len(self.results)} experiment results")
            return True

        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return False

    def compute_statistics(self) -> Dict:
        """条件ごとの統計計算"""

        stats = {}

        for condition in ['A', 'B']:
            cond_results = [r for r in self.results if r.condition == condition]
            if not cond_results:
                continue

            # Perplexity: 最終エポック（val）
            final_ppls = [r.val_perplexities[-1] if r.val_perplexities else float('nan')
                         for r in cond_results]

            stats[condition] = {
                'final_ppl_mean': float(np.mean(final_ppls)),
                'final_ppl_std': float(np.std(final_ppls)),
                'final_ppl_values': final_ppls
            }

            logger.info(f"Condition {condition}:")
            logger.info(f"  Final PPL: {stats[condition]['final_ppl_mean']:.4f} ± {stats[condition]['final_ppl_std']:.4f}")

        self.summary = stats
        return stats

    def statistical_test(self) -> Dict:
        """条件 A vs B の統計検定"""

        if not self.summary or 'A' not in self.summary or 'B' not in self.summary:
            logger.warning("Statistics not computed. Run compute_statistics() first.")
            return {}

        # paired t-test（scipy が必要）
        try:
            from scipy import stats as sp_stats

            ppl_a = np.array(self.summary['A']['final_ppl_values'])
            ppl_b = np.array(self.summary['B']['final_ppl_values'])

            # valid なデータのみ
            valid_idx = ~(np.isnan(ppl_a) | np.isnan(ppl_b))
            ppl_a = ppl_a[valid_idx]
            ppl_b = ppl_b[valid_idx]

            if len(ppl_a) < 2 or len(ppl_b) < 2:
                logger.warning("Insufficient data for t-test (need n >= 2)")
                return {}

            # Paired t-test（3シード）
            t_stat, p_value = sp_stats.ttest_rel(ppl_a, ppl_b)

            # 効果量（Cohen's d）
            mean_diff = np.mean(ppl_a - ppl_b)
            std_diff = np.std(ppl_a - ppl_b)
            cohens_d = mean_diff / (std_diff + 1e-8)

            result = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'mean_diff': float(mean_diff),
                'cohens_d': float(cohens_d),
                'significant': p_value < 0.05
            }

            logger.info("Statistical Test (Paired t-test):")
            logger.info(f"  t-statistic: {t_stat:.4f}")
            logger.info(f"  p-value: {p_value:.6f}")
            logger.info(f"  Mean Difference (A-B): {mean_diff:.4f}")
            logger.info(f"  Cohen's d: {cohens_d:.4f}")
            logger.info(f"  Significant (α=0.05): {'Yes' if p_value < 0.05 else 'No'}")

            return result

        except ImportError:
            logger.warning("scipy not available. Skipping t-test.")
            return {}

    def improvement_ratio(self) -> Dict:
        """PPL 改善率（条件 B が A に対してどれだけ改善したか）"""

        if not self.summary or 'A' not in self.summary or 'B' not in self.summary:
            logger.warning("Statistics not computed. Run compute_statistics() first.")
            return {}

        ppl_a = self.summary['A']['final_ppl_mean']
        ppl_b = self.summary['B']['final_ppl_mean']

        if ppl_a == 0:
            return {}

        improvement_ratio = (ppl_b - ppl_a) / ppl_a * 100  # 負値 = 改善

        result = {
            'improvement_ratio_percent': float(improvement_ratio),
            'direction': 'improvement' if improvement_ratio < 0 else 'degradation'
        }

        logger.info("Improvement Analysis:")
        logger.info(f"  Ratio: {improvement_ratio:.2f}%")
        logger.info(f"  Direction: {result['direction']}")

        return result

    def export_summary(self, output_file: str = "step44_summary.json") -> bool:
        """結果サマリーを JSON で保存"""

        output_path = self.results_dir / output_file

        try:
            summary_data = {
                'statistics': self.summary,
                'n_experiments': len(self.results),
                'conditions': list(set([r.condition for r in self.results]))
            }

            with open(output_path, 'w') as f:
                json.dump(summary_data, f, indent=2)

            logger.info(f"Summary saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting summary: {e}")
            return False


def main():
    """メイン分析パイプライン"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("STEP 44 Result Analysis")
    logger.info("=" * 80)

    # 分析実行
    analyzer = Step44Analyzer()

    # 結果ロード
    if not analyzer.load_results("step44_hippocampus_validation_results.json"):
        logger.error("Failed to load results. Exiting.")
        return

    # 統計計算
    analyzer.compute_statistics()

    # 統計検定
    test_results = analyzer.statistical_test()

    # 改善率
    improvement = analyzer.improvement_ratio()

    # サマリー保存
    analyzer.export_summary()

    logger.info("=" * 80)
    logger.info("Analysis Complete ✓")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
