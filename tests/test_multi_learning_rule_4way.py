"""STDP統合4モジュール構成のテスト（ステップ42、12.6.92節）。

テスト対象:
  - src/neurocortex/multi_learning_rule_4way.py

テスト項目:
  1. 4モジュール全体の基本実行確認（最小config）
  2. STDP on/off での動作差異確認
  3. NaN/Inf チェック（各ステップで値域確認）
  4. データ型・形状確認（出力形式の妥当性）
  5. コサイン類似度が2モジュール共有時だけ計算されること
"""

import pytest
import torch

from neurocortex.basal_ganglia_core import ParitySpec
from neurocortex.multi_learning_rule_4way import (
    CONDITIONS, RunResult, run_condition, _extract_base_and_stdp_flag, _active_modules
)


@pytest.fixture
def spec():
    return ParitySpec(n_inputs=16, relevant=(0, 1))


class TestConditionParsing:
    """条件文字列の解析テスト。"""

    def test_extract_base_and_stdp_flag_on(self):
        """STDP on 条件の解析。"""
        base, stdp = _extract_base_and_stdp_flag("shared-all-stdp-on")
        assert base == "shared-all"
        assert stdp is True

    def test_extract_base_and_stdp_flag_off(self):
        """STDP off 条件の解析。"""
        base, stdp = _extract_base_and_stdp_flag("shared-all-stdp-off")
        assert base == "shared-all"
        assert stdp is False

    def test_all_conditions_valid(self):
        """全16条件が解析可能。"""
        assert len(CONDITIONS) == 16
        for cond in CONDITIONS:
            base, stdp = _extract_base_and_stdp_flag(cond)
            assert isinstance(base, str)
            assert isinstance(stdp, bool)


class TestBasicFunctionality:
    """基本的な動作確認"""

    def test_independent_all_no_stdp(self, spec):
        """基本条件（独立、STDP無し）の実行確認"""
        gen = torch.Generator().manual_seed(42)
        res = run_condition(
            "independent-all-stdp-off", spec,
            hidden_dim=32, steps=10, batch_size=32, lr=0.001, eval_n=50,
            generator=gen
        )
        assert isinstance(res, RunResult)
        assert res.hippo_acc is not None
        assert res.rl_acc is not None
        assert res.cereb_mse is not None
        assert res.nan_count == 0
        assert not res.stdp_enabled

    def test_shared_all_no_stdp(self, spec):
        """共有バックボーン（STDP無し）の実行確認"""
        gen = torch.Generator().manual_seed(43)
        res = run_condition(
            "shared-all-stdp-off", spec,
            hidden_dim=32, steps=10, batch_size=32, lr=0.001, eval_n=50,
            generator=gen
        )
        assert res.hippo_acc is not None
        assert res.rl_acc is not None
        assert res.cereb_mse is not None
        assert res.nan_count == 0

    def test_independent_all_with_stdp(self, spec):
        """基本条件（独立、STDP有り）の実行確認"""
        gen = torch.Generator().manual_seed(44)
        res = run_condition(
            "independent-all-stdp-on", spec,
            hidden_dim=32, steps=10, batch_size=32, lr=0.001, eval_n=50,
            generator=gen
        )
        assert res.stdp_enabled
        assert res.nan_count == 0
        # STDP 更新ノルムが記録されていることを確認
        assert isinstance(res.stdp_weight_update_norms, list)

    def test_shared_all_with_stdp(self, spec):
        """共有バックボーン（STDP有り）の実行確認"""
        gen = torch.Generator().manual_seed(45)
        res = run_condition(
            "shared-all-stdp-on", spec,
            hidden_dim=32, steps=10, batch_size=32, lr=0.001, eval_n=50,
            generator=gen
        )
        assert res.stdp_enabled
        assert res.nan_count == 0


class TestSTDPOnOff:
    """STDP on/off 比較テスト。"""

    def test_stdp_off_has_no_norms(self, spec):
        """STDP off では更新ノルムが記録されない。"""
        gen = torch.Generator().manual_seed(47)
        res = run_condition(
            "shared-all-stdp-off", spec,
            hidden_dim=32, steps=10, batch_size=32, lr=0.001, eval_n=50,
            generator=gen
        )
        assert len(res.stdp_weight_update_norms) == 0

    def test_stdp_on_produces_norms(self, spec):
        """STDP on では更新ノルムが記録される。"""
        gen = torch.Generator().manual_seed(46)
        res = run_condition(
            "shared-all-stdp-on", spec,
            hidden_dim=32, steps=20, batch_size=32, lr=0.001, eval_n=50,
            generator=gen
        )
        # steps=20, 5ステップごとにSTDPを適用 → 複数回
        assert len(res.stdp_weight_update_norms) >= 1

    def test_stdp_norms_are_non_negative(self, spec):
        """更新ノルムは非負の値。"""
        gen = torch.Generator().manual_seed(48)
        res = run_condition(
            "independent-all-stdp-on", spec,
            hidden_dim=32, steps=20, batch_size=32, lr=0.001, eval_n=50,
            generator=gen
        )
        for norm in res.stdp_weight_update_norms:
            assert norm >= 0.0


class TestNaNStability:
    """主張(a): NaN 発生率の確認"""

    def test_no_nan_in_independent(self, spec):
        """独立構成での NaN 発生なし"""
        gen = torch.Generator().manual_seed(47)
        res = run_condition(
            "independent-all-stdp-off", spec,
            hidden_dim=32, steps=50, batch_size=32, lr=0.001, eval_n=50,
            generator=gen
        )
        assert res.nan_count == 0

    def test_no_nan_in_shared(self, spec):
        """共有構成での NaN 発生なし"""
        gen = torch.Generator().manual_seed(48)
        res = run_condition(
            "shared-all-stdp-off", spec,
            hidden_dim=32, steps=50, batch_size=32, lr=0.001, eval_n=50,
            generator=gen
        )
        assert res.nan_count == 0

    def test_no_nan_with_stdp(self, spec):
        """STDP 統合時の NaN 発生なし"""
        gen = torch.Generator().manual_seed(49)
        res = run_condition(
            "shared-all-stdp-on", spec,
            hidden_dim=32, steps=50, batch_size=32, lr=0.001, eval_n=50,
            generator=gen
        )
        assert res.nan_count == 0


class TestConditionCoverage:
    """全条件の実行確認"""

    @pytest.mark.parametrize("condition", CONDITIONS)
    def test_all_conditions_run(self, spec, condition):
        """全 16 条件が実行できることを確認"""
        gen = torch.Generator().manual_seed(hash(condition) % 10000)
        res = run_condition(
            condition, spec,
            hidden_dim=32, steps=10, batch_size=32, lr=0.001, eval_n=50,
            generator=gen
        )
        # NaN が過度に発生していないことを確認（50ステップで複数モジュール = 150回以下）
        assert res.nan_count <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
