# ステップ44.4 フル規模実験の実行手順（Kaggle Notebooks）

対象スクリプト: `colab_step44_full_scale_experiment.py`（Kaggle でもそのまま動く）

> Kaggle の画面構成・無料 GPU の枠・実行時間の上限は変わることがある。以下の数値は目安なので、実際の画面と Kaggle のドキュメントで確認すること。この手順は Kaggle 上では検証していない（こちらの環境から Kaggle に接続できないため）。スクリプトの中断・再開は、偽データでの通し実行で確認済み。

## Kaggle の制約と対処

| 制約（目安） | 対処 |
|---|---|
| 1回の実行は最長12時間 | `--max-hours 11` を付ける。11時間でチェックポイントを保存して正常終了する |
| 無料 GPU は週30時間程度 | 9実行は1週間では終わらない見込み。数週間に分けて、同じ手順を繰り返す |
| セッション間でファイルが残らない。残るのは Save Version したときの `/kaggle/working` だけ | 結果とチェックポイントを `/kaggle/working/step44` に置き、次の実行では前回バージョンの出力を入力として読み込んで続ける |
| 出力は20GB程度まで | チェックポイントは1個あたり約3GB。完了した実行のチェックポイントはスクリプトが自動で消す |
| インターネット接続は設定で有効化が必要（電話番号認証が必要） | WikiText-103 と GPT-2 トークナイザーの取得に必要 |

## 初回の準備

1. Kaggle で New Notebook を作る。
2. 右側の Settings で次を設定する。
   - Accelerator: GPU（T4 または P100。T4 ×2 でもスクリプトは1枚だけ使う）
   - Internet: On
3. 下の「実行セル」を貼る。

## 実行セル（毎回同じ）

```python
import os, shutil, subprocess

REPO = '/tmp/neurocortex-llm'   # /kaggle/working に置くと出力が膨らむので /tmp に置く
OUT = '/kaggle/working/step44'
PREV = '/kaggle/input'          # 前回バージョンの出力を入力に追加した場合、この下に現れる

if not os.path.exists(REPO):
    subprocess.run(['git', 'clone', 'https://github.com/goodnasubi/neurocortex-llm.git', REPO], check=True)
subprocess.run(['pip', 'install', '-q', 'datasets', 'transformers'], check=True)

# 前回バージョンの出力（step44 フォルダ）があれば引き継ぐ
for root, dirs, _ in os.walk(PREV):
    if os.path.basename(root) == 'step44':
        print('前回の出力を引き継ぎます:', root)
        shutil.copytree(root, OUT, dirs_exist_ok=True)
        break

os.chdir(REPO)
!python colab_step44_full_scale_experiment.py --phases A B C \
    --results-dir {OUT} --checkpoint-dir {OUT}/ckpt --max-hours 11
!cat {OUT}/summary.json
```

## 繰り返しの流れ

1. **Save Version → Save & Run All (Commit)** でバックグラウンド実行する。ブラウザを閉じても続く。
2. 実行が終わったら、そのバージョンの出力に `step44/`（結果 JSON とチェックポイント）が残っていることを確認する。
3. 次の実行の前に、右側の **Add Input** から、このノートブックの前回バージョンの出力を追加する。自分のノートブックの出力を追加できない場合は、前回の出力から New Dataset を作って、それを追加する。
4. 1 に戻る。スクリプトは完了済みの (phase, seed) を飛ばし、途中のものはチェックポイントから再開する。
5. `summary.json` の A/B/C がそれぞれ `n: 3` になったら完了。

最初の実行のログで、tqdm に表示される 1 ステップあたりの秒数を確認すると、全体の所要時間を見積もれる（1実行あたり ≒ 秒/step × 1エポックのステップ数 × 10エポック）。

## 結果の共有

`step44/phase{A,B,C}_seed{0,1,2}.json` と `summary.json` をダウンロードして共有してもらえれば、分析と docs への記録はこちらで行う（リポジトリでは `results/step44_full_scale_rerun/` に置く）。

## 注意

- 1ステップ目で GPU メモリ不足になった場合は、`Step444Config.batch_size` を下げ、同じ比率で `gradient_accumulation_steps` を上げる（実効バッチ64を保つ）。途中で変えると再開したときに一致しなくなるので、最初の実行の前に決めること。
- 基底核ヘッドの critic は、このスクリプトでは学習されない（`td_sg` は未移植）。
