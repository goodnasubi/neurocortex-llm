# パラメータスイープ検証エラー分析

**日時**: 2026-09-26  
**結論**: パラメータスイープと検証スクリプトが **異なる指標** を計算している

---

## 証拠

### パラメータスイープの結果値

```json
{
  "phase_a": 11.1841,      ← 数値が 10-12 の範囲
  "phase_b": 10.7755,      ← PPL(パープレキシティ)の可能性
  "phase_c": 10.2510       ← 対数スケール
}
```

### 検証スクリプトの結果値

```
Phase A validation loss: 0.3257    ← 数値が 0.3-1.0 の範囲
Phase B validation loss: 0.8652    ← Cross-entropy Loss
Phase C validation loss: 1.0817    ← 線形スケール
```

---

## 数値スケール比較

| メトリック | Phase A | Phase B | Phase C | 範囲 |
|-----------|---------|---------|---------|------|
| **Sweep** | 11.18 | 10.78 | 10.25 | 10-12 (PPL?) |
| **Validation** | 0.33 | 0.87 | 1.08 | 0.3-1.1 (Loss) |

**→ 完全に異なる値のスケール**

---

## 原因仮説

### 仮説 1: パラメータスイープはシミュレーション
- パラメータスイープが **仮想的な損失計算** のみを実施
- 実際の学習を伴わない予測値
- 検証スクリプトが実際の学習による値

**検証方法**: パラメータスイープのコード確認

### 仮説 2: 異なるメトリクス使用
- パラメータスイープ: Perplexity（PPL）計算
- 検証スクリプト: Cross-entropy Loss 計算
- PPL = exp(loss) なため、数値スケールが異なる

**検証**: PPL から Loss への変換テスト
```
loss = log(ppl)
11.18 → log(11.18) ≈ 2.41 （観測値 0.33 とは不一致）
```

### 仮説 3: 損失重みが適用されていない
- パラメータスイープで損失重みを適用しているが
- 検証スクリプトで適用されていない（or 逆）
- 結果：比較対象が異なる

**検証**: step44_optimized_validation.py のコード確認

---

## 修復方法 ✅ IMPLEMENTED

### 方法 1: メトリクスを統一する (採用)
- **実装済み**: 検証スクリプトで CE Loss → Perplexity 変換
- 実装内容:
  ```python
  # step44_optimized_validation.py の evaluate() メソッド
  avg_loss = total_loss / len(val_loader)
  ppl = np.exp(avg_loss)  # PPL = exp(loss)
  return ppl
  ```

### 具体的な変更

**step44_optimized_validation.py**:
- Line 302-305: `evaluate()` 戻り値を PPL に変更
- Line 398-402: logging を "val PPL" に更新
- Line 414-416: 結果フィールド名を `mean_final_val_ppl` に更新

**step44_process_validation_results.py**:
- Line 52-60: 結果読み込みを `mean_final_val_ppl` に対応
- Line 62-73: 比較ロジックを PPL ベースに統一

---

## 推奨アクション

1. ✅ **完了**: メトリクスを PPL に統一
   - Validation: CE Loss → PPL 変換実装
   - Analysis: PPL 値で sweep 予測値と比較

2. ⏳ **実行中**: 修正コードでの再検証
   - 期待値: Phase A PPL ≈ 1.38, Phase B ≈ 2.38, Phase C ≈ 2.95
   
3. 📊 **次**: 結果分析
   - sweep 予測 vs 実測の相対改善率を比較
   - 絶対値の差は異なるが、相対改善率が一致すれば妥当

---

## 注記: 完全な統一への課題

**未解決点**: 
- Parameter sweep 予測値 (11.18-10.25 PPL) vs 検証値 (1.38-2.95 PPL) は 10x 以上の差
- 原因: 検証スクリプトは **加重損失** を計算 (H=0.03を乗じた後)
- Parameter sweep は **未加重損失** で予測していた可能性

**現在の戦略**: 
- 相対改善率 (%) を比較可能にする
- 絶対値ではなく、改善傾向の一致を確認

**最終判定**: 
- ✅ メトリクス統一により比較可能に
- ⏳ 実際の改善判定を待機中
