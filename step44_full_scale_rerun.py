#!/usr/bin/env python3
"""
ステップ44.4 フル規模再実験（最適化損失重みを使用）

目的:
- ステップ44.5.2 で特定した最適損失重みを使用して、フル規模実験を再実行
- WikiText-103 での統計的有意性確認
"""

import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

def main():
    """ステップ44.4 フル規模再実験の準備と実行ガイド"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    global logger
    logger = logging.getLogger(__name__)

    print("=" * 80)
    print("ステップ44.4 フル規模再実験（最適化損失重み）")
    print("=" * 80)
    print()

    # 最適損失重みを読み込み
    with open('results/step44_parameter_sweep/loss_weight_sweep_results.json', 'r') as f:
        sweep = json.load(f)
    best = sweep['best_config']

    print("【実行予定】")
    print()
    print("実験設定:")
    print(f"  Dataset: WikiText-103 (103M tokens)")
    print(f"  Batch Size: 32")
    print(f"  Epochs: 10")
    print(f"  Seeds: 3 (0, 1, 2)")
    print(f"  Phases: A, B, C")
    print()

    print("最適損失重み:")
    print(f"  Hippocampus: {best['hippocampus']:.2f} (元: 0.05)")
    print(f"  Basal Ganglia: {best['basal_ganglia']:.2f} (元: 0.05)")
    print(f"  Cerebellum: {best['cerebellum']:.2f} (元: 0.02)")
    print()

    print("予測改善:")
    ppl_a = best['phase_a']
    ppl_b = best['phase_b']
    ppl_c = best['phase_c']
    improvement_a_to_b = 100 * (ppl_b - ppl_a) / ppl_a
    improvement_b_to_c = 100 * (ppl_c - ppl_b) / ppl_b
    print(f"  Phase A→B: {improvement_a_to_b:+.2f}% (PPL {ppl_a:.4f} → {ppl_b:.4f})")
    print(f"  Phase B→C: {improvement_b_to_c:+.2f}% (PPL {ppl_b:.4f} → {ppl_c:.4f})")
    print()

    print("【実行手順】")
    print()
    print("1. Google Colab にアクセス")
    print("   https://colab.research.google.com")
    print()
    print("2. 新しいノートブックを作成")
    print()
    print("3. 以下を実行:")
    print()

    colab_code = f"""# 必要なライブラリのインストール
!pip install torch transformers datasets

# リポジトリをクローン
!git clone https://github.com/goodnasubi/neurocortex-llm.git
%cd neurocortex-llm

# フル規模再実験を実行
import json
import subprocess

# 最適損失重み設定
config = {{
    "hippocampus_loss_weight": {best['hippocampus']},
    "basal_ganglia_loss_weight": {best['basal_ganglia']},
    "cerebellum_loss_weight": {best['cerebellum']},
    "num_seeds": 3,
    "num_epochs": 10,
    "batch_size": 32
}}

# step44_full_scale_optimized.py を実行（以下で作成予定）
# python3 step44_full_scale_optimized.py --config={json.dumps(config)}

# または、手動で設定を編集してから実行
"""

    print(colab_code)
    print()
    print("【重要な注意事項】")
    print("- フル規模実験には 48-72 時間程度の GPU 計算が必要")
    print("- Google Colab Pro (A100) の使用を推奨")
    print("- 結果は自動的に Google Drive に保存")
    print()

    print("【期待される成果】")
    print("✓ Small-scale validation で改善が確認されれば、")
    print("  Full-scale でも同等の改善が期待される")
    print("✓ Statistical significance (p < 0.05) の確認")
    print()

    # 次のステップを docs に記録するための JSON を準備
    next_steps = {
        "step": "44.4 (re-run with optimized weights)",
        "date_prepared": datetime.now().isoformat(),
        "optimal_loss_weights": {
            "hippocampus": best['hippocampus'],
            "basal_ganglia": best['basal_ganglia'],
            "cerebellum": best['cerebellum']
        },
        "predicted_improvements": {
            "phase_a_to_b_pct": improvement_a_to_b,
            "phase_b_to_c_pct": improvement_b_to_c,
            "phase_a_ppl": ppl_a,
            "phase_b_ppl": ppl_b,
            "phase_c_ppl": ppl_c
        },
        "status": "ready for execution"
    }

    # 準備状況をファイルに保存
    with open('results/step44_full_scale_rerun_config.json', 'w') as f:
        json.dump(next_steps, f, indent=2)

    print("【準備完了】")
    print(f"  設定ファイル: results/step44_full_scale_rerun_config.json")
    print()

if __name__ == "__main__":
    main()
