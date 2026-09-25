# ステップ44.3 Jetson Nano 軽量インストール

**問題**: 依存パッケージバージョン競合  
**原因**: Jetson OS 標準の古い Python 環境  
**解決策**: 最小限のパッケージのみインストール

---

## インストール手順

### 1. 古いパッケージをクリア（オプション）

```bash
# 競合しているパッケージを削除
pip uninstall -y seaborn datasets huggingface-hub matplotlib pandas

# pip をアップグレード
pip install --upgrade pip setuptools wheel
```

### 2. 最小限の依存パッケージインストール

```bash
# PyTorch for Jetson（ARM64 版）
pip install torch torchvision torchaudio

# 必須パッケージのみ
pip install numpy tqdm scipy

# scipy がエラーの場合
pip install numpy==1.21.0 tqdm scipy --no-deps
```

### 3. リポジトリ確認

```bash
cd ~/neurocortex-llm
ls -la jetson_step44_phase_validation.py
```

### 4. ステップ44.3 実行

```bash
# 全フェーズ (A/B/C) を実行
python jetson_step44_phase_validation.py

# または特定フェーズのみ
python -c "
from jetson_step44_phase_validation import *
config = Step443Config(phase='A')
result = run_phase_validation(config, seed=0)
print(result)
"
```

---

## 実行結果

**成功**: `results/step44_phase_validation/step44_phase_*.json` にファイル生成

**失敗**: torch/numpy エラーの場合
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## ステップ44.3 スクリプトの依存関係

| パッケージ | 用途 | 必須 |
|---|---|---|
| torch | ニューラルネット | ✅ |
| numpy | 数値計算 | ✅ |
| tqdm | プログレスバー | ✅ |
| scipy | 統計計算 | ✅ |
| json | 結果保存 | ✅ |
| logging | ログ出力 | ✅ |
| seaborn | グラフ表示 | ❌ |
| datasets | データロード | ❌ |
| matplotlib | グラフ描画 | ❌ |

**注**: グラフ関連は `analyze_step44_results.py` で使用。  
ステップ44.3 はデータ生成・学習・JSON 保存のみ。

---

## トラブルシューティング

### エラー: `No module named 'torch'`

```bash
pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
```

### エラー: `CUDA out of memory`

```bash
# スクリプトの config を編集
# num_train_samples = 500  # 1000 → 500
# batch_size = 4           # 8 → 4
# seq_length = 64          # 128 → 64
```

### エラー: `numpy version conflict`

```bash
pip install numpy==1.19.5 --force-reinstall
```

---

## 次ステップ

1. ✓ ステップ44.3 実行 → JSON 結果生成
2. → Google Drive に結果をアップロード
3. → ローカル PC で `analyze_step44_results.py` を実行
4. → ステップ44.4 へ進行判定

---

**更新日**: 2026-09-26  
**対象環境**: Jetson Nano 2GB/4GB、JetPack 4.6+
