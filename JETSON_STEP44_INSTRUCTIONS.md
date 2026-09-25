# ステップ44.3 Jetson Nano 実行ガイド

**目的**: Jetson Nano で段階的統合（Phase A/B/C）の検証を実行

---

## 環境セットアップ（初回のみ）

### 1. Jetson OS インストール確認

```bash
# Jetson OS バージョン確認
cat /etc/nv_tegra_release

# 推奨: JetPack 4.6 or later
```

### 2. Python 環境セットアップ

```bash
# Python 3.8+ がデフォルトインストール済み
python3 --version

# pip アップグレード
pip install --upgrade pip setuptools wheel

# 依存パッケージインストール
pip install torch torchvision torchaudio numpy scipy tqdm matplotlib seaborn
```

#### PyTorch for Jetson（重要）

Jetson Nano には ARM64 特化版 PyTorch が必要:

```bash
# 標準 pip では Jetson 対応版が入らない可能性がある
# 代わりに以下を推奨

# オプション1: nvidia-jetson 公式（推奨）
pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html

# オプション2: whl ファイル直接インストール
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
# から ARM64 whl をダウンロードして:
# pip install torch-*.whl
```

### 3. リポジトリをクローン

```bash
cd ~
git clone https://github.com/goodnasubi/neurocortex-llm.git
cd neurocortex-llm
```

---

## 実行方法

### フェーズ順序実行

```bash
# すべてのフェーズ（A/B/C）を順序実行
python jetson_step44_phase_validation.py
```

**想定実行時間**: 
- Phase A（海馬のみ）: 2-3 時間
- Phase B（海馬+基底核）: 2-3 時間
- Phase C（全モジュール）: 2-3 時間
- **合計**: 6-9 時間

### GPU メモリ状況確認

```bash
# Jetson GPU 状態確認
tegrastats

# または
nvidia-smi
```

**メモリ制約**: Jetson Nano 2GB/4GB
- **推奨設定**（現在のデフォルト）:
  - batch_size = 8
  - seq_length = 128
  - num_train_samples = 1000
- **OOM 時**: さらに削減
  - batch_size = 4
  - seq_length = 64

---

## 設定カスタマイズ

### ハイパーパラメータ調整

`jetson_step44_phase_validation.py` の `Step443Config` クラス:

```python
@dataclass
class Step443Config:
    # データセット
    num_train_samples: int = 1000  # 削減時: 500
    num_val_samples: int = 100
    seq_length: int = 128           # 削減時: 64
    
    # 学習設定
    batch_size: int = 8            # 削減時: 4
    learning_rate: float = 5e-5
    num_epochs: int = 3            # クイック検証
    
    # リソース
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = True              # 有効（メモリ削減）
```

### 特定フェーズのみ実行

スクリプトを直接編集:

```python
# main() 関数で phases を変更
phases = ['A']        # Phase A のみ
# phases = ['B']      # Phase B のみ
# phases = ['A', 'B'] # Phase A, B のみ
```

---

## 実行結果

### 出力ファイル

```bash
results/step44_phase_validation/
├── step44_phase_A_results.json
├── step44_phase_B_results.json
├── step44_phase_C_results.json
└── step44_phase_validation_results.json
```

### JSON フォーマット

各フェーズの結果:

```json
{
  "phase": "A",
  "seed": 0,
  "train_losses": [0.123, 0.115, 0.108],
  "val_losses": [0.135, 0.127, 0.120]
}
```

---

## トラブルシューティング

### OOM（メモリ不足）エラー

**症状**: `RuntimeError: CUDA out of memory`

**対応**:

```python
# jetson_step44_phase_validation.py を編集
config = Step443Config()
config.batch_size = 4              # 8 → 4
config.num_train_samples = 500     # 1000 → 500
config.seq_length = 64             # 128 → 64
```

### GPU 認識されない

**確認**:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**False が返る場合**:

```bash
# PyTorch 再インストール（Jetson 特化版）
pip uninstall torch torchvision torchaudio
# 上記「PyTorch for Jetson」セクション を参照
```

### スクリプト実行エラー

**ImportError**:

```bash
# 依存パッケージ再インストール
pip install --upgrade numpy scipy torch tqdm
```

**CUDA Driver の古さ**:

```bash
# JetPack 更新（Jetson ドキュメント参照）
# または CPU モードで実行:
# jetson_step44_phase_validation.py の device を強制変更
# device: str = "cpu"  # CUDA 無視
```

---

## リアルタイム監視

### ログ出力確認

```bash
# stderr / stdout をファイルに保存
python jetson_step44_phase_validation.py > step44_log.txt 2>&1

# リアルタイムで監視
tail -f step44_log.txt
```

### GPU 使用率監視（別ターミナル）

```bash
# ターミナル1: スクリプト実行
python jetson_step44_phase_validation.py

# ターミナル2: GPU 監視
while true; do nvidia-smi; sleep 5; done
```

---

## Colab 実験との並行実行

### スケジュール例（2026-09-26 開始）

**09:00** — Colab で ステップ44.2 フル実験投入開始  
**09:15** — Jetson で ステップ44.3 Phase A 開始  
**12:00** — Phase A 完了 → Phase B 開始  
**15:00** — Phase B 完了 → Phase C 開始  
**18:00** — Phase C 完了、結果保存  
**翌日 09:00** — Colab 実験完了（約 48-72 時間後）  
**09:30** — 結果統合分析開始

---

## 実験結果の GitHub 反映

### 結果をローカルに保存

```bash
# Jetson から ローカル PC へ転送
scp -r jetson_user@jetson_ip:~/neurocortex-llm/results/step44_phase_validation ~/step44_jetson_results
```

### Git に追加・コミット

```bash
# ローカル PC で
cd ~/neurocortex-llm
cp -r ~/step44_jetson_results results/step44_jetson_validation

git add results/step44_jetson_validation/
git commit -m "ステップ44.3: Jetson フェーズ検証結果（Phase A/B/C）"
git push origin master
```

---

## チェックリスト

- [ ] Jetson OS 確認（JetPack 4.6+）
- [ ] Python 3.8+ インストール
- [ ] PyTorch Jetson 版 インストール（GPU 認識確認）
- [ ] リポジトリ クローン
- [ ] jetson_step44_phase_validation.py 実行確認（小規模テスト）
- [ ] Colab 実験投入完了
- [ ] Jetson フェーズ検証開始（スケジュール参照）
- [ ] 結果ファイル確認
- [ ] GitHub へ結果反映

---

## 次ステップ

1. **ステップ44.2 Colab 実験完了待機**（48-72 時間）
2. **ステップ44.3 Jetson 検証完了** → 結果統合
3. **統計分析** (`analyze_step44_results.py`)
4. **ステップ44.4-44.5** へ進行判定

---

**更新日**: 2026-09-26  
**ステータス**: Jetson セットアップ待機中
