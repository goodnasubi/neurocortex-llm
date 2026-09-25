# ステップ44.4 投入準備チェックリスト

**状態**: ✅ 準備完了  
**実行日**: 2026-09-26  
**対象**: Google Colab フル規模実験（WikiText-103 × Phase A/B/C × 3 seed）

---

## 📋 準備状況確認

### コード実装
- ✅ `colab_step44_full_scale_experiment.py`: 新規作成・完成
  - ✅ WikiText-103 データセット対応
  - ✅ Phase A/B/C 自動実行フロー実装
  - ✅ Gradient accumulation 実装（実効 batch size = 64）
  - ✅ Mixed precision (fp16) 実装
  - ✅ Checkpoint/結果保存機能

### 最適化実施
- ✅ Loss 重み付け最適化完了
  - hippocampus_loss_weight: 0.2 → **0.05**
  - basal_ganglia_loss_weight: 0.2 → **0.05**
  - cerebellum_loss_weight: 0.1 → **0.02**
- ✅ 段階的統合による loss 増加が 75-80% 改善確認

### 実行結果の記録
- ✅ docs/brain-structure-research.md に記録
  - ステップ44.3.1 最適化結果
  - ステップ44.4 詳細設計・期待成果

### ドキュメント整備
- ✅ COLAB_STEP44_INSTRUCTIONS.md: ステップ44.4 用に更新
- ✅ STEP44_4_DEPLOYMENT_CHECKLIST.md: このファイル作成

### Git リポジトリ
- ✅ コミット完了: `c308e23 ステップ44.4: フル規模実験準備完了`
- ✅ GitHub push 完了: `master` ブランチ同期

---

## 🚀 Colab 投入手順

### ステップ 1: Colab セッション開始

1. **Google Colab を開く**
   ```
   https://colab.research.google.com
   ```

2. **新規 Notebook を作成** または 既存ノートブック使用

3. **セル 0: リポジトリをクローン**
   ```python
   !git clone https://github.com/goodnasubi/neurocortex-llm.git
   %cd neurocortex-llm
   !git pull origin master  # 最新版を取得
   ```

4. **セル 1: 依存パッケージをインストール**
   ```python
   !pip install torch transformers datasets scipy tqdm numpy -q
   ```

5. **セル 2: GPU 確認**
   ```python
   import torch
   print(f"GPU: {torch.cuda.get_device_name(0)}")
   print(f"CUDA Available: {torch.cuda.is_available()}")
   print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
   ```

### ステップ 2: フル規模実験投入

6. **セル 3: Google Drive マウント**
   ```python
   from google.colab import drive
   drive.mount('/content/gdrive')
   ```

7. **セル 4: スクリプト実行**
   ```python
   %run colab_step44_full_scale_experiment.py
   ```

   **実行時間の目安**:
   - GPU T4: ~70-80 時間（バッチサイズ削減時）
   - GPU A100: ~30-40 時間
   - **推奨**: 夜間実行 or 複数セッション並行

### ステップ 3: 実行中の監視（オプション）

8. **セル 5: リアルタイム結果確認**
   ```python
   import json
   from pathlib import Path
   
   results_dir = Path('/content/gdrive/MyDrive/results/step44_full_scale_experiment')
   results_files = sorted(results_dir.glob('*.json'))
   
   if results_files:
       with open(results_files[-1]) as f:
           data = json.load(f)
           print(json.dumps(data, indent=2)[:1500])
   ```

9. **セル 6: GPU メモリ監視**
   ```python
   import torch
   
   allocated = torch.cuda.memory_allocated() / 1e9
   total = torch.cuda.get_device_properties(0).total_memory / 1e9
   print(f"GPU Memory: {allocated:.2f} GB / {total:.2f} GB")
   print(f"Usage: {allocated/total*100:.1f}%")
   ```

---

## 📊 実行結果の期待値

### Phase 別 Loss の予想

| Phase | 構成 | 予想 Train Loss | 予想 Val Loss | vs Prev |
|---|---|---|---|---|
| A | 海馬のみ | ~10.5-11.0 | ~10.7-11.2 | — |
| B | 海馬 + 基底核 | ~11.0-11.5 | ~11.2-11.7 | +0.5-0.7 |
| C | 全モジュール | ~11.2-11.8 | ~11.4-12.0 | +0.2-0.5 |

**判定基準**:
- ✅ Phase B vs A での loss 増加が ≤ 1.0 → 段階的統合成功
- ✅ Phase C vs B での loss 増加が ≤ 0.5 → 全モジュール統合成功
- ⚠️ 上記を超える → Loss weight 再調整が必要

### 成功の指標

1. **学習の安定性**
   - loss が epoch ごとに単調減少（振動が小さい）
   - gradient norm が NaN/Inf に達さない

2. **モジュール間相互作用**
   - 各モジュール loss が共存できている（dominant なモジュール loss なし）
   - Phase 進行に従い、backbone loss と module loss のバランスが安定

3. **GPU メモリ利用**
   - T4 でも ~7-8 GB 以下（コマンドサイズ安全余裕あり）
   - A100 でも OOM 発生なし

---

## ⚠️ トラブルシューティング

### OOM（メモリ不足）が発生した場合

**症状**: `RuntimeError: CUDA out of memory`

**対応**:
1. `colab_step44_full_scale_experiment.py` の `Step444Config` を編集
   ```python
   batch_size: int = 16  # 32 → 16
   max_seq_length: int = 128  # 256 → 128
   ```

2. Colab の GPU を GPU Premium に変更（A100 に変更）

3. バッチサイズをさらに削減
   ```python
   batch_size: int = 8
   gradient_accumulation_steps: int = 4  # 実効 batch = 32
   ```

### WikiText-103 がダウンロード失敗する場合

**症状**: `ConnectionError` or `HF Hub timeout`

**対応**:
1. ダウンロード再試行（Colab は通常リトライし成功）
   ```python
   !pip install --upgrade huggingface-hub
   ```

2. オフライン時は dummy dataset にフォールバック（自動）

### Colab セッション切断される場合

**症状**: 90 分無操作で切断

**対応**:
1. Chrome DevTools で「no-sleep」コード実行
   ```python
   from IPython.display import HTML
   HTML("""<script>
   setInterval(function() { console.log("staying awake"); }, 5000);
   </script>""")
   ```

2. または別セッションで独立して各 Phase を実行

---

## 📈 結果分析（実験完了後）

実験完了後、ローカル PC で以下を実行：

```bash
# 結果分析スクリプト実行
python analyze_step44_results_lite.py results/step44_full_scale_experiment/step44_full_scale_results.json
```

**出力の解釈**:
- 各 Phase の loss 値を比較
- Phase 間の increase ratio を確認
- ステップ44.5 への進行可否を判定

---

## ✅ 最終確認

- ✅ **コード品質**: Python 構文チェック完了
- ✅ **データ対応**: WikiText-103 ロード機能確認
- ✅ **GPU 対応**: mixed precision / gradient accumulation 実装
- ✅ **ドキュメント**: 投入手順・トラブル対応書き込み完了
- ✅ **GitHub**: 最新版 push 完了

**判定**: **ステップ44.4 投入可能**

---

## 🎯 次ステップ

1. **Colab で ステップ44.4 投入** → 48-72 時間待機
2. **結果分析実行** → `analyze_step44_results_lite.py` で判定
3. **ステップ44.5 へ** → 実験結果に基づき、論文執筆準備 or 追加実験

---

**作成日**: 2026-09-26  
**作成者**: Claude Haiku 4.5  
**ステータス**: ✅ 投入準備完了
