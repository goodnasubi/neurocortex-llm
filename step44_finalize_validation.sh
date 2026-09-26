#!/bin/bash
# ステップ44.3 完了後のワークフロー

set -e  # エラーで停止

cd /home/user/neurocortex-llm

echo "=========================================="
echo "ステップ44.3 検証結果の処理"
echo "=========================================="
echo

# 1. 検証結果が存在するか確認
if [ ! -f results/step44_optimized_validation/step44_optimized_validation_summary.json ]; then
    echo "❌ 検証結果が見つかりません"
    exit 1
fi

# 2. 分析を実行
echo "分析を実行中..."
python3 step44_process_validation_results.py
ANALYSIS_CODE=$?

echo

if [ $ANALYSIS_CODE -eq 0 ]; then
    echo "✓ 改善が確認されました"
    echo "  → フル規模再実験を推奨"
    echo
    echo "次のステップ:"
    echo "1. Google Colab で colab_step44_full_scale_experiment.py を実行"
    echo "2. 実験結果を待機（48-72 時間）"
    echo "3. 統計分析と論文執筆へ"
    echo
    python3 step44_full_scale_rerun.py
elif [ $ANALYSIS_CODE -eq 1 ]; then
    echo "⚠ 部分的な改善のみ"
    echo "  → 追加検討が必要"
else
    echo "❌ 改善が確認されない"
    echo "  → パラメータ調整を再検討"
fi

echo
echo "=========================================="
echo "処理完了"
echo "=========================================="
