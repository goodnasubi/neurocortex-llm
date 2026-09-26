# ステップ44.5 Phase 2: 結果分析ガイド

**目的**: ステップ44.4 Colab 実験完了後、結果の統計分析と Figure 生成を実施

**実施者**: ローカル環境（GPU 不要）

**所要時間**: 2-3 時間

---

## Phase 2 フロー

```
1. Colab から結果ダウンロード（results.json）
2. ローカル環境で統計分析実行
3. Figure 生成
4. 論文用 draft results を生成
```

---

## ステップ 1: Colab から結果をダウンロード

### 1.1 Colab セッションから Google Drive へ保存確認

```python
# Colab cell 最後に実行されている確認
import json
results = {
    'A': [...],  # phase A の results
    'B': [...],  # phase B の results
    'C': [...]   # phase C の results
}
with open('/content/gdrive/MyDrive/step44_results/results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### 1.2 Google Drive から ダウンロード

```bash
# ローカル環境から Google Drive にアクセス（手動 or gdrive CLI）
# ファイル: /MyDrive/step44_results/results.json

# または rclone を使用（事前セットアップ必要）
# rclone copy gdrive:/step44_results/ ./results/step44_final/
```

### 1.3 results.json を適切なディレクトリに配置

```bash
mkdir -p results/step44_final
# results.json を配置
ls -la results/step44_final/results.json
```

---

## ステップ 2: analyze_step44_results.py で統計分析

### 2.1 スクリプト実行

```bash
cd /home/user/neurocortex-llm
python analyze_step44_results.py \
  --input results/step44_final/results.json \
  --output results/step44_final/analysis_output.json
```

### 2.2 出力内容確認

```bash
# 統計分析結果を確認
cat results/step44_final/analysis_output.json | jq '.'
```

**出力フォーマット**:
```json
{
  "results": {
    "phase_A": {
      "test_perplexity": [X.XX, X.XX, X.XX],
      "mean_test_ppl": "X.XX",
      "std_test_ppl": "X.XX"
    },
    "phase_B": {...},
    "phase_C": {...}
  },
  "statistical_analysis": {
    "phase_A_vs_B": {
      "t_statistic": "X.XX",
      "p_value": "X.XXXX",
      "cohens_d": "X.XX",
      "significant": true
    },
    "phase_B_vs_C": {...}
  },
  "module_contribution": {
    "phase_B": {
      "backbone_loss_ratio": "X.XX%",
      "hippocampus_loss_ratio": "X.XX%",
      "basal_ganglia_loss_ratio": "X.XX%"
    },
    "phase_C": {...}
  }
}
```

---

## ステップ 3: step44_figure_template.py で Figure 生成

### 3.1 スクリプト実行

```bash
python step44_figure_template.py \
  --results results/step44_final/analysis_output.json \
  --output-dir results/step44_figures
```

### 3.2 出力ファイル確認

```bash
ls -la results/step44_figures/
# 出力:
# - figure2_perplexity_comparison.png
# - figure3_loss_curves.png
# - figure4_module_contribution.png
# - figure5_statistical_significance.png
```

### 3.3 Figure の品質確認

各 figure を画像ビューアで開いて、以下を確認：

- **Figure 2**: 3 phases の perplexity が正しく表示されているか？ error bar は適切か？
- **Figure 3**: Loss curves が epoch 推移で適切に減少しているか？ 3 seeds の差分は大きくないか？
- **Figure 4**: Module contribution が合理的か？（Phase C で全 module が寄与しているか？）
- **Figure 5**: Statistical significance が明確か？（p < 0.05 が視覚的に分かるか？）

---

## ステップ 4: 論文用 Draft Results セクション作成

### 4.1 analyze_step44_results.py の出力を論文フォーマットに変換

```bash
python -c "
import json

with open('results/step44_final/analysis_output.json', 'r') as f:
    data = json.load(f)

# Main results table
print('## 4.4 Full-Scale Experiment Results (WikiText-103)')
print()
print('| Phase | Mean Test PPL | Std Dev | vs Prev | Improvement | Significant |')
print('|---|---|---|---|---|---|')

phases = ['A', 'B', 'C']
prev_ppl = None

for phase in phases:
    if phase in data['results']:
        mean_ppl = float(data['results'][phase]['mean_test_ppl'])
        std_ppl = float(data['results'][phase]['std_test_ppl'])
        
        if prev_ppl is None:
            vs_prev = '-'
            improvement = '-'
            sig = '-'
        else:
            vs_prev = f'{mean_ppl - prev_ppl:+.3f}'
            improvement = f'{100 * (mean_ppl - prev_ppl) / prev_ppl:.1f}%'
            
            if phase == 'B':
                pval = data['statistical_analysis']['phase_A_vs_B']['p_value']
                sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
            else:
                pval = data['statistical_analysis']['phase_B_vs_C']['p_value']
                sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
        
        print(f'| {phase} | {mean_ppl:.2f} | {std_ppl:.2f} | {vs_prev} | {improvement} | {sig} |')
        prev_ppl = mean_ppl
" > results/step44_final/results_table.txt

cat results/step44_final/results_table.txt
```

### 4.2 論文の Results セクションに埋め込み

生成された `results_table.txt` の内容を `step44_paper_outline.md` の Section 4.4 に記入：

```markdown
### 4.4 Full-Scale Experiment Results (WikiText-103)

[上記のテーブルをここに貼り付け]

**Interpretation**:
- Phase A（baseline + hippocampus）: {mean_ppl} ± {std_ppl} perplexity
- Phase B（+ basal ganglia）: {mean_ppl} ± {std_ppl} perplexity
  - vs Phase A: {improvement}%, p={p_value} （t-test, Cohen's d = {cohens_d}）
  - [有意かどうかを判断]

- Phase C（+ cerebellum）: {mean_ppl} ± {std_ppl} perplexity
  - vs Phase B: {improvement}%, p={p_value} （t-test, Cohen's d = {cohens_d}）
  - [有意かどうかを判断]

[Figure 2-5 を引用しながら詳しく解釈]
```

---

## ステップ 5: 判定基準の適用

### 5.1 Hypothesis Verification

```
Phase A vs B: Perplexity 改善が有意（p < 0.05）?
  → YES: BG module が reward prediction 機能を実装 ✓
  → NO: BG module が干渉 → 設計修正（44.5.2）

Phase B vs C: Perplexity 改善が有意？
  → YES: Cerebellum の error correction が機能 ✓
  → NO: Cerebellum module が干渉 → 設計修正（44.5.2）

全て YES?
  → ステップ44.5.1（5B-7B LLM 拡張）推奨
  → 論文を arXiv preprint として投稿準備へ
```

### 5.2 論文執筆の次のステップ

- **結果が良い場合**: Section 5（Discussion）で "Findings are consistent with our hypothesis..." と記述
- **結果が悪い場合**: Section 5 で limitations と future work を詳述、次ステップ（44.5.2）の理由を説明

---

## ファイル構成

```
results/step44_final/
├── results.json                      # Colab から download
├── analysis_output.json              # analyze_step44_results.py の output
└── results_table.txt                 # 論文用 results table

results/step44_figures/
├── figure2_perplexity_comparison.png
├── figure3_loss_curves.png
├── figure4_module_contribution.png
└── figure5_statistical_significance.png

step44_paper_outline.md              # 論文骨子（Section 4.4 - Results 埋め込み）
```

---

## トラブルシューティング

### Q1: results.json が Colab から download できない

**A**: Google Drive をマウント -> file picker から手動ダウンロード

```python
# Colab
from google.colab import files
files.download('/content/gdrive/MyDrive/step44_results/results.json')
```

### Q2: analyze_step44_results.py で scipy エラー

**A**: `pip install scipy` で最新版インストール

```bash
pip install --upgrade scipy
```

### Q3: Figure が生成されない

**A**: matplotlib バックエンド確認

```bash
python -c "import matplotlib; print(matplotlib.get_backend())"
# Agg に設定すること
```

---

## 次のアクション

1. ✅ Phase 2 スクリプト確認（analyze_step44_results.py + figure_template）
2. ✅ ガイド作成完了
3. ⏳ Colab 実験完了待機（48-72h）
4. → Colab 完了後、本ガイドに従い Phase 2 実行
5. → Section 4.4（Results）埋め込み
6. → Section 5（Discussion）執筆
7. → arXiv preprint 投稿

---

**推定実行時間**: 2-3 時間（download + 統計分析 + Figure 生成 + 論文反映）

**必要環境**: Python 3.8+, scipy, matplotlib（GPU 不要）

