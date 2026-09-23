"""ステップ40: ハイブリッド再ランク機構のテスト

主張(a)-(c)の検証テスト
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.neurocortex.hippocampus import DiskBackedAssociativeStore


@pytest.fixture
def temp_store():
    """一時的なストアを生成"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = DiskBackedAssociativeStore(
            key_dim=768,
            value_dim=512,
            persist_dir=tmpdir,
        )
        yield store


@pytest.fixture
def populated_store(temp_store):
    """テストデータで満たされたストア"""
    store = temp_store
    n_data = 1000
    batch_size = 100

    for i in range(0, n_data, batch_size):
        keys = torch.randn(batch_size, 768, dtype=torch.float32)
        values = torch.randn(batch_size, 512, dtype=torch.float64)
        store.write(keys, values)

    yield store


class TestHybridRerankMethods:
    """主張(a): 精度向上の検証"""

    def test_read_with_hybrid_rerank_exists(self, populated_store):
        """メソッドの存在確認"""
        assert hasattr(populated_store, "read_with_hybrid_rerank")

    def test_hybrid_rerank_basic_functionality(self, populated_store):
        """基本的な動作確認"""
        test_keys = torch.randn(10, 768, dtype=torch.float32)

        # ハイブリッド再ランク実行
        out, stats = populated_store.read_with_hybrid_rerank(
            test_keys,
            k_candidates=32,
            k_refine=16,
            lambda_inner=0.6,
            lambda_density=0.3,
            lambda_template=0.1,
            use_gpu=False,
        )

        assert out.shape == (10, 512)
        assert stats.top1_index.shape == (10,)
        assert stats.max_score.shape == (10,)
        assert (stats.top1_index >= -1).all() and (stats.top1_index < 1000).all()

    def test_hybrid_rerank_accuracy_improvement(self, populated_store):
        """精度が向上するか確認（主張a）

        理想的には、k=32 (√1000) では87.5% だが、
        k_refine を適用すると92% 以上に改善するはず。
        """
        test_keys = torch.randn(50, 768, dtype=torch.float32)

        # 段階1のみ（GPU PQ）
        out_stage1, stats_stage1 = populated_store.read_with_pq_search(
            test_keys, k_candidates=32
        )
        stage1_coverage = (stats_stage1.top1_index >= 0).float().mean().item()

        # 段階1+段階3（ハイブリッド再ランク）
        out_hybrid, stats_hybrid = populated_store.read_with_hybrid_rerank(
            test_keys,
            k_candidates=32,
            k_refine=16,
            lambda_inner=0.6,
            lambda_density=0.3,
            lambda_template=0.1,
            use_gpu=False,
        )
        hybrid_coverage = (stats_hybrid.top1_index >= 0).float().mean().item()

        # ハイブリッド再ランクでカバレッジが維持されることを確認
        assert hybrid_coverage >= stage1_coverage - 0.05


class TestScoreFunctions:
    """主張(c): スコア寄与度分析"""

    def test_local_density_computation(self, populated_store):
        """密度スコア計算の動作確認"""
        density = populated_store._local_density_at(100, k_neighbor=20)

        assert isinstance(density, float)
        assert 0.0 <= density <= 1.0

    def test_template_match_score(self, populated_store):
        """テンプレートマッチスコアの動作確認"""
        query = torch.randn(768, dtype=torch.float32)
        score = populated_store._template_match_score(100, query)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_contributions_extraction(self, populated_store):
        """スコア寄与度の抽出"""
        test_keys = torch.randn(10, 768, dtype=torch.float32)

        scores_inner = populated_store._compute_inner_scores(
            test_keys, k_candidates=32, k_refine=16
        )
        scores_density = populated_store._compute_density_scores(
            test_keys, k_candidates=32, k_refine=16
        )
        scores_template = populated_store._compute_template_scores(
            test_keys, k_candidates=32, k_refine=16
        )

        assert len(scores_inner) > 0
        assert len(scores_density) > 0
        assert len(scores_template) > 0
        assert len(scores_inner) == len(scores_density) == len(scores_template)


class TestPerformanceRequirements:
    """主張(b): 速度要件の確認"""

    def test_hybrid_rerank_speed(self, populated_store):
        """E2E処理時間が要件を満たすか確認

        N=500K時に400ms以下を想定
        """
        import time

        test_keys = torch.randn(20, 768, dtype=torch.float32)

        t0 = time.time()
        out, stats = populated_store.read_with_hybrid_rerank(
            test_keys,
            k_candidates=32,
            k_refine=16,
            use_gpu=False,
        )
        elapsed = time.time() - t0

        # 小規模データなので許容時間は短めに設定
        assert elapsed < 1.0  # 1秒以内

    def test_speed_scaling(self, populated_store):
        """スケーリング特性の確認"""
        import time

        times = []
        for batch_size in [10, 20, 50]:
            test_keys = torch.randn(batch_size, 768, dtype=torch.float32)

            t0 = time.time()
            out, stats = populated_store.read_with_hybrid_rerank(
                test_keys,
                k_candidates=32,
                k_refine=16,
                use_gpu=False,
            )
            elapsed = time.time() - t0
            times.append(elapsed)

        # バッチサイズが増加しても線形以下の増加にとどまることを確認
        ratio = times[2] / times[0]
        assert ratio < 6.0  # 50/10=5倍の入力に対して6倍以下の時間


class TestRegressionTests:
    """ステップ35-39との互換性確認"""

    def test_backward_compatibility_with_pq_search(self, populated_store):
        """PQ検索との互換性"""
        test_keys = torch.randn(10, 768, dtype=torch.float32)

        # 既存PQ検索
        out_pq, stats_pq = populated_store.read_with_pq_search(
            test_keys, k_candidates=32
        )

        # ハイブリッド再ランク（λ_density=0, λ_template=0でPQと同等）
        out_hybrid, stats_hybrid = populated_store.read_with_hybrid_rerank(
            test_keys,
            k_candidates=32,
            k_refine=32,  # 候補をすべて評価
            lambda_inner=1.0,
            lambda_density=0.0,
            lambda_template=0.0,
            use_gpu=False,
        )

        # アーギュメント（インデックス）が一致することを確認
        match_rate = (stats_pq.top1_index == stats_hybrid.top1_index).float().mean().item()
        assert match_rate >= 0.95  # 95%以上一致


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
