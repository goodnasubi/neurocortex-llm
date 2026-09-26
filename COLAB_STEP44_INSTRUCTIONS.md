# ステップ44.4 フル規模実験の実行手順（Colab）

対象スクリプト: `colab_step44_full_scale_experiment.py`（2026-09-26 修正版）

- 評価指標: 全フェーズ共通の backbone `lm_head` の**非重み付き**次トークン CE と PPL（`val_ppls`）。各ヘッドの CE は参考値として `val_head_ce` に記録する。
- 以前の結果ファイル（`step44_full_scale_{A,B,C}_results.json`）は真正性を確認できないため使わない（`docs/brain-structure-research.md` の注記を参照）。

## 規模と所要時間

既定の設定: GPT-2 small 相当（768次元・12層）、WikiText-103 全体、最大256トークン、10エポック × 3フェーズ（A/B/C）× 3シード = 9実行。

所要時間はこの環境では計測できていない。1回目のセッションで、tqdm に表示される 1 ステップあたりの秒数から見積もること（1実行あたり ≒ 秒/step × 1エポックのステップ数 × 10）。1実行でも Colab の1セッション（無料版 最長12時間、Pro 約24時間）に収まらない可能性が高いため、**途中再開を前提に**実行する。

## 手順

### 1. セットアップ（セッションごと）

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/goodnasubi/neurocortex-llm.git
%cd neurocortex-llm
!pip install -q datasets transformers
```

### 2. 実行

結果とチェックポイントは Google Drive に置く（セッションが切れても消えないように）。

```python
OUT = '/content/drive/MyDrive/neurocortex/step44'
!python colab_step44_full_scale_experiment.py --phases A B C \
    --results-dir {OUT} --checkpoint-dir {OUT}/ckpt
```

- 2000 ステップごとと各エポック終了時にチェックポイントを保存する。
- セッションが切れたら、手順1からやり直して**同じコマンドを再実行**すれば、中断した位置（エポック・バッチ）から再開する。中断なしの場合と同一の結果になることを確認済み（`tests/test_colab_step44_full_scale.py`、および偽データでの通し実行）。
- 完了した実行は `{OUT}/phase{A,B,C}_seed{0,1,2}.json` に保存され、再実行時には飛ばされる。
- 複数の Colab アカウントやセッションで分けて実行する場合は、`--phases B --seeds 1` のように指定する。
- WikiText-103 の読み込みに失敗すると停止する（ダミーデータへの切り替えは行わない）。

### 3. 結果の回収

```python
!cat {OUT}/summary.json
```

`summary.json` は完了した実行だけをまとめたもの（各フェーズの最終 val PPL、平均、標準偏差、実行数）。9実行すべての `phase*_seed*.json` を `results/step44_full_scale_rerun/` に置いてコミットすれば、分析を引き継げる。

## 注意

- 基底核ヘッドの critic は、このスクリプトでは学習されない（小規模検証で採用候補とした `td_sg` は未移植）。
- 1ステップ目で GPU メモリ不足になった場合は、`Step444Config.batch_size` を下げ、同じ比率で `gradient_accumulation_steps` を上げる（実効バッチ64を保つ）。
