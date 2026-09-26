# Google Colab ステップ44.4 フル規模再実験 実行ガイド

## 概要

ステップ44.3（小規模検証）で最適損失重みの改善が確認された場合、本ガイドに従って Google Colab でフル規模実験を実行してください。

**予定完了日時**: 2026-09-26 05:17 UTC 検証確認後

## 実験設定（最適化損失重み適用）

| パラメータ | 値 | 元の値 |
|-----------|-----|--------|
| Hippocampus loss weight | 0.03 | 0.05 |
| Basal Ganglia loss weight | 0.05 | 0.05 |
| Cerebellum loss weight | 0.02 | 0.02 |

**予測改善**:
- Phase A→B: -3.65%
- Phase B→C: -4.87%

## 実行方法

### Step 1: Google Colab ノートブック準備

1. [Google Colab](https://colab.research.google.com) にアクセス
2. 「ファイル」→「新しいノートブック」で新規作成
3. ノートブックを「neurocortex-step44-experiment」と名前変更

### Step 2: リポジトリのクローン

セル 1:
```python
!git clone https://github.com/goodnasubi/neurocortex-llm.git
%cd neurocortex-llm
```

### Step 3: 必要なパッケージのインストール

セル 2:
```bash
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q datasets transformers numpy scipy scikit-learn tqdm
```

### Step 4: GPU 確認

セル 3:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

**推奨**: A100 以上が望ましい（T4 でも実行可能、ただし実行時間 2倍以上）

### Step 5: 実験実行

セル 4:
```python
%run colab_step44_full_scale_experiment.py
```

**実行時間**:
- A100 with 32GB: 48-72 時間
- V100 with 16GB: 72-96 時間
- T4 with 16GB: 120-144 時間

## 実行完了後

### 結果の確認

1. Colab 右側の「ファイル」パネルから `results/` ディレクトリを確認
2. 以下のファイルが生成されていることを確認:
   - `results/step44_full_scale_rerun_experiment/step44_full_scale_results.json`
   - `results/step44_full_scale_rerun_experiment/phase_summaries.json`

### 結果の github へのプッシュ

セル（最後）:
```bash
!git config user.email "goodnasubi@gmail.com"
!git config user.name "Claude Haiku"
!git add results/
!git commit -m "Step 44.4 RE-RUN: Full-scale experiment with optimized loss weights

- Hippocampus: 0.03 (optimized from 0.05)
- Basal Ganglia: 0.05
- Cerebellum: 0.02
- Dataset: WikiText-103
- Phases: A, B, C (3 seeds × 10 epochs each)"
!git push origin master-m7kte3
```

## 注意事項

### Colab セッション中の作業

- **セッションタイムアウト**: Colab セッションは 12 時間で自動終了
- **対策**: 
  - 実行完了予定時刻前に Colab を確認
  - セッション切断の場合、手動で再度 `%run colab_step44_full_scale_experiment.py` を実行可能
  - スクリプトは途中から再開する機構は有していないため、最初から再実行される

### ディスク容量

- WikiText-103: ~4GB
- 実験ログ・中間ファイル: ~2GB
- **合計**: 6-8GB 必要
- Colab デフォルト: 108GB 利用可能

### メモリ管理

- スクリプトは自動的に GPU メモリを管理
- 予期しないメモリ不足の場合、`torch.cuda.empty_cache()` で解放可能

## トラブルシューティング

### 実行中にエラーが発生した場合

1. **CUDA out of memory**:
   ```python
   torch.cuda.empty_cache()
   # スクリプト再実行
   ```

2. **WikiText-103 ダウンロード失敗**:
   ```python
   # 手動ダウンロード試行
   from datasets import load_dataset
   ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
   ```

3. **git push 失敗**:
   ```bash
   !git remote set-url origin https://<your-token>@github.com/goodnasubi/neurocortex-llm.git
   !git push origin master-m7kte3
   ```

## 完了後の次ステップ

1. ローカル環境で結果の統計分析を実行
   ```bash
   python3 analyze_step44_optimization.py
   ```

2. 図表を生成
   ```bash
   python3 step44_plot_results.py
   ```

3. 論文執筆準備
   - Methods セクション（実験設定の記述）
   - Results セクション（統計結果の記述）
   - Discussion セクション（神経生物学的含意）

4. arXiv preprint へのサブミット準備

---

**更新日**: 2026-09-26
**ステップ**: ステップ44.3 検証完了後に実行
