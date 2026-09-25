# ステップ44.4 Colab フル規模実験ガイド

**目的**: Google Colab Pro で WikiText-103 全データを使用した Phase A/B/C 段階的統合実験（48-72時間）を実行

---

## セットアップ手順

### 1. Google Colab Notebook を開く

```
https://colab.research.google.com
```

### 2. GitHub リポジトリをクローン

```python
!git clone https://github.com/goodnasubi/neurocortex-llm.git
%cd neurocortex-llm
```

### 3. 依存パッケージをインストール

```python
!pip install torch transformers datasets scipy tqdm seaborn matplotlib numpy
```

> **注意**: Colab には PyTorch/GPU ドライバが事前インストール済み

### 4. スクリプトのコピーと環境設定

```python
# ローカルファイルから Colab 対応版を読み込む
%run colab_step44_full_experiment.py
```

---

## 実験の実行

### セッション開始時（必須）

```python
# Google Drive マウント確認
from google.colab import drive
drive.mount('/content/gdrive')

# リポジトリパス確認
import os
os.chdir('/content/neurocortex-llm')
```

### フル実験実行

```python
# 方法1: スクリプト直接実行
!python colab_step44_full_experiment.py

# 方法2: Notebook からの実行（推奨・ログ確認容易）
exec(open('colab_step44_full_experiment.py').read())
```

**想定実行時間**: 48-72 時間（GPU: T4 または A100）

---

## リアルタイム監視

### 実験ログの確認

```python
# Google Drive に保存される結果をリアルタイムで確認
import json
from pathlib import Path

results_dir = Path('/content/gdrive/MyDrive/step44_results/step44_full_experiment')
latest_checkpoint = sorted(results_dir.glob('*.json'))[-1]

with open(latest_checkpoint) as f:
    data = json.load(f)
    print(json.dumps(data, indent=2)[:1000])  # 最初の1000文字表示
```

### GPU メモリ使用状況

```python
import torch
print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### Colab セッションの維持

Colab は 90 分の無操作で切断される。以下で防止：

```python
# セッション中のコード実行状況を表示し続ける
import time
while True:
    time.sleep(60)
    print(f"Session alive at {time.ctime()}")
```

---

## トラブルシューティング

### OOM（メモリ不足）エラー

**原因**: バッチサイズが大きい、またはモデルが大きい

**対策**:
```python
# colab_step44_full_experiment.py の設定を修正
config.batch_size = 8  # デフォルト 16 から 8 に削減
config.gradient_accumulation_steps = 4  # 実効 batch = 32 を維持
```

### タイムアウト・セッション切断

**原因**: 90 分無操作、または長時間実行

**対策**:
- セッション開始から 72 時間以内に完了するよう計画
- Google Drive に中間結果（checkpoint）が自動保存される
- クラッシュ時は最後の checkpoint から再開可能

### GPU がない / T4 では実行時間が長すぎる

**対策**:
- Colab Pro のランタイムを A100 に変更
  - Settings → Compute → GPU を "A100 GPU" に選択
  - (Pro 契約者のみ)

---

## 実験後の処理

### 結果ダウンロード

```python
# Google Drive から ローカルにダウンロード
!cp -r /content/gdrive/MyDrive/step44_results ~/step44_results_backup
```

### 結果分析

実験完了後、ローカルで以下を実行：

```bash
python analyze_step44_results.py results/step44_hippocampus_validation/step44_full_experiment_results.json
```

**生成ファイル**:
- `step44_analysis_report.txt` — テキストレポート
- `ppl_comparison.png` — PPL 比較図
- `loss_curves.png` — Loss 曲線

---

## 並行作業（Colab 実験中に ローカルで実施）

実験が 48-72 時間実行される間に、ステップ44.3 の実装を Jetson/ローカルで進行：

```bash
# Jetson で フェーズ検証実行
python jetson_step44_phase_validation.py
```

---

## チェックリスト

- [ ] Google Colab Pro に契約
- [ ] GitHub リポジトリ アクセス確認
- [ ] 依存パッケージ インストール完了
- [ ] Google Drive マウント 確認
- [ ] スクリプト実行 開始
- [ ] 実験ログ リアルタイム監視開始
- [ ] ローカルで ステップ44.3 実装進行開始

---

## 次ステップ（実験完了後）

1. **結果分析** (Local)
   - `analyze_step44_results.py` で統計検定実行
   - PPL 改善が有意か判定

2. **ステップ44.4 フル規模実験** (Colab, 48-72h)
   - フルモデル（LLaMA-7B など）での実験
   - 基底核・小脳統合フェーズ

3. **ステップ44.5 論文執筆**
   - 結果を整理・分析
   - 神経生物学的妥当性 評価

---

**更新日**: 2026-09-26  
**ステータス**: 実験準備完了。Colab 投入待機中
