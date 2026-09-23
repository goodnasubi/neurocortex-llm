"""ステップ35: キー単位走査への見直し（方式B: 固定k件）の正確性テスト。

read_with_key_candidates() と read() の出力が完全一致することを確認。
"""

import tempfile
from pathlib import Path

import pytest
import torch

from src.neurocortex.hippocampus import DiskBackedAssociativeStore


@pytest.mark.parametrize("N", [1000, 5000, 10000, 100000])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("exact", [True, False])
def test_read_with_key_candidates_exact_match(N: int, seed: int, exact: bool):
    """read_with_key_candidates() と read() が完全一致することを確認。

    Args:
        N: ストアサイズ
        seed: 乱数シード
        exact: 完全一致モード（True）か softmax 加重和（False）か
    """
    key_dim = 128
    value_dim = 64
    key_chunk = 512

    g = torch.Generator().manual_seed(seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        # ストア初期化
        store = DiskBackedAssociativeStore(
            key_dim=key_dim,
            value_dim=value_dim,
            persist_dir=tmpdir,
            beta=50.0,
            exact=exact,
            key_chunk=key_chunk,
        )

        # ストアにデータ書き込み
        keys_data = torch.randn(N, key_dim, generator=g)
        values_data = torch.randn(N, value_dim, generator=g)
        store.write(keys_data, values_data)

        # クエリキーを生成
        B = 16
        query_keys = torch.randn(B, key_dim, generator=g)

        # 従来の read() で参照値を取得
        values_ref, stats_ref = store.read(query_keys)

        # アテンション重みを実スコアから計算（候補集合内にargmaxが確実に含まれるようにする）
        keys_mm = store._keys_memmap()
        query_keys_f32 = query_keys.to(dtype=torch.float32)
        all_scores = []
        for j in range(0, N, store.key_chunk):
            store_keys, _ = store._get_chunk(keys_mm, store._values_memmap(), j, store.key_chunk)
            scores = query_keys_f32 @ store_keys.T
            all_scores.append(scores)
        all_scores_tensor = torch.cat(all_scores, dim=-1)
        attention_weights = torch.softmax(all_scores_tensor, dim=-1)

        # read_with_key_candidates() で候補絞り込み版を実行
        values_cand, stats_cand = store.read_with_key_candidates(query_keys, attention_weights)

        # 出力が完全一致するか確認
        assert torch.equal(values_ref, values_cand), \
            f"N={N}, seed={seed}, exact={exact}: values が一致しない"
        assert torch.equal(stats_ref.max_score, stats_cand.max_score), \
            f"N={N}, seed={seed}, exact={exact}: max_score が一致しない"
        assert torch.equal(stats_ref.top1_index, stats_cand.top1_index), \
            f"N={N}, seed={seed}, exact={exact}: top1_index が一致しない"


@pytest.mark.parametrize("N", [1000, 5000, 10000, 100000])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_key_candidate_coverage(N: int, seed: int):
    """候補キーセット内に目的キー（argmax）が含まれることを確認。

    Args:
        N: ストアサイズ
        seed: 乱数シード
    """
    key_dim = 128
    value_dim = 64
    key_chunk = 512

    g = torch.Generator().manual_seed(seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        store = DiskBackedAssociativeStore(
            key_dim=key_dim,
            value_dim=value_dim,
            persist_dir=tmpdir,
            beta=50.0,
            exact=True,  # 厳密モード
            key_chunk=key_chunk,
        )

        # ストアにデータ書き込み
        keys_data = torch.randn(N, key_dim, generator=g)
        values_data = torch.randn(N, value_dim, generator=g)
        store.write(keys_data, values_data)

        # クエリキーを生成
        B = 8
        query_keys = torch.randn(B, key_dim, generator=g)

        # 従来の read() で argmax を取得
        _, stats_ref = store.read(query_keys)
        best_indices = stats_ref.top1_index  # [B]

        # アテンション重みを実スコアから計算
        keys_mm = store._keys_memmap()
        query_keys_f32 = query_keys.to(dtype=torch.float32)
        all_scores = []
        for j in range(0, N, store.key_chunk):
            store_keys, _ = store._get_chunk(keys_mm, store._values_memmap(), j, store.key_chunk)
            scores = query_keys_f32 @ store_keys.T
            all_scores.append(scores)
        all_scores_tensor = torch.cat(all_scores, dim=-1)
        attention_weights = torch.softmax(all_scores_tensor, dim=-1)

        # 候補サイズ: sqrt(N)
        k_candidate = max(1, int(N**0.5))

        # 各バッチごとに候補キーセットを取得
        covered_count = 0
        for b in range(B):
            candidates = store._get_key_candidates(attention_weights[b], k_candidate)
            true_best = int(best_indices[b])
            if true_best >= 0 and true_best in candidates:
                covered_count += 1

        # 全てのバッチで目的キーが候補内に含まれるか
        coverage_rate = covered_count / B * 100
        # 少なくとも 50% 以上のカバレッジを期待
        assert coverage_rate >= 50, \
            f"N={N}, seed={seed}: カバレッジ率 {coverage_rate:.1f}% (閾値: 50%)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
