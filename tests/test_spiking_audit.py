"""共有監査ユーティリティ（`spiking_audit.py`、12.6.40節・ステップ16）自体の単体テスト。

既知の「合格するはずの」ケース（`models.py`のスパイキング版RecallModel、
12.6.1節Test Aで合格確認済み）と「不合格になるはずの」ケース（12.9節の
階段関数方式そのもの、`activation="step"`）の両方で、ユーティリティが
正しく判定できることを先に確認する。
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.hippocampus import HippocampalMemory, fit_stdp  # noqa: E402
from neurocortex.models import RecallModel  # noqa: E402
from neurocortex.neurons import NeuronConfig  # noqa: E402
from neurocortex.tasks import RecallSpec, make_recall_batch  # noqa: E402

from spiking_audit import (  # noqa: E402
    assert_membrane_ablation_degrades,
    assert_not_permutation_invariant,
    warn_if_degenerate,
)

SPEC = RecallSpec(n_cue=8, n_filler=8, max_delay=16)


def _tokens(n: int, delay: int, generator: torch.Generator) -> torch.Tensor:
    tokens, _ = make_recall_batch(SPEC, n, generator, delay=delay)
    return tokens


def test_assert_not_permutation_invariant_passes_for_spiking_model() -> None:
    torch.manual_seed(0)
    model = RecallModel(vocab_size=SPEC.pad_id + 1, n_classes=SPEC.n_cue, activation="spiking")
    model.eval()
    g = torch.Generator().manual_seed(0)
    tokens = _tokens(8, delay=15, generator=g)
    assert_not_permutation_invariant(lambda t: model.forward_all(t), tokens)


def test_assert_not_permutation_invariant_fails_for_step_model() -> None:
    """既知の不合格ケース: 12.9節の階段関数方式は順列不変なので、監査は必ず検出できること。"""
    torch.manual_seed(0)
    model = RecallModel(vocab_size=SPEC.pad_id + 1, n_classes=SPEC.n_cue, activation="step")
    model.eval()
    g = torch.Generator().manual_seed(0)
    tokens = _tokens(8, delay=15, generator=g)
    with pytest.raises(AssertionError):
        assert_not_permutation_invariant(lambda t: model.forward_all(t), tokens)


def test_assert_membrane_ablation_degrades_detects_real_drop() -> None:
    torch.manual_seed(0)
    model = RecallModel(vocab_size=SPEC.pad_id + 1, n_classes=SPEC.n_cue, activation="spiking")
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    for _ in range(300):
        tokens, labels = make_recall_batch(SPEC, 32, g, delay=15)
        loss = torch.nn.functional.cross_entropy(model(tokens), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        tokens, labels = make_recall_batch(SPEC, 500, torch.Generator().manual_seed(999), delay=15)
        pred = model(tokens).argmax(-1)
        return float((pred == labels).float().mean())

    baseline, ablated = assert_membrane_ablation_degrades(evaluate, model.set_carry_membrane,
                                                           min_drop=0.2)
    assert baseline > ablated


def test_assert_membrane_ablation_raises_when_no_real_drop() -> None:
    """既知の不合格ケース: アブレーションしても性能が落ちない（＝時間ダイナミクスを
    使っていない）架空のモデルに対しては、監査が必ず検出できること。
    """
    def evaluate_constant() -> float:
        return 0.9  # 常に同じ値 = アブレーションの影響を一切受けないふり

    def noop_set_carry_membrane(enabled: bool) -> None:
        pass

    with pytest.raises(AssertionError):
        assert_membrane_ablation_degrades(evaluate_constant, noop_set_carry_membrane, min_drop=0.2)


# --- ステップ29（12.6.66節）: 主張(b) warn_if_degenerate ---------------------
#
# まず合成テンソル（コサイン類似度を正確に制御できる）で境界条件を保証し、
# 続いて実際の`fit_stdp`（ステップ26〜28が使う学習則そのもの）で小規模に
# 再学習した重みでも、倍率が大きいほど縮退が進む（＝しきい値を跨ぐ）という
# 実験の定性的な傾向が再現されることを確認する。


def test_warn_if_degenerate_synthetic_below_threshold_no_warning() -> None:
    """互いにほぼ直交する行ベクトル（縮退なし）では警告が出ないこと。"""
    torch.manual_seed(0)
    weight = torch.eye(16) + 0.01 * torch.randn(16, 16)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 警告が出たら例外にして即座に検出する
        max_cos = warn_if_degenerate(weight, threshold=0.99)
    assert max_cos < 0.99


def test_warn_if_degenerate_synthetic_above_threshold_warns() -> None:
    """ほぼ同一方向を向く行ベクトル（縮退）では確実に`UserWarning`が出ること。"""
    torch.manual_seed(0)
    base = torch.randn(1, 16)
    weight = base.expand(16, -1) + 1e-6 * torch.randn(16, 16)
    with pytest.warns(UserWarning):
        max_cos = warn_if_degenerate(weight, threshold=0.99)
    assert max_cos >= 0.99


def _small_scale_stdp_weight(multiplier: float, seed: int = 0) -> torch.Tensor:
    """ステップ26〜28（`run_stdp_stability.py`）と同じ`fit_stdp`を、実行時間を
    抑えた小規模設定（d_model・n_units・n_trainを縮小）で走らせ、実際のSTDP
    学習則から得られる重みを返す。ステップ27・28が確定させた「倍率が大きいほど
    縮退が進む」という定性的傾向が、この小規模設定でも再現されることを前提に、
    (b)の検証を実際の学習コード経路に対して行う（合成テンソルのみに頼らない）。
    """
    d_model, n_units, k, n_train, t_len, epochs = 32, 64, 4, 1500, 8, 20
    sep = HippocampalMemory(d_model, value_dim=d_model, n_units=n_units, k=k,
                             mode="sdr", seed=seed).separator
    g_data = torch.Generator().manual_seed(seed + 500)
    spikes = (torch.rand(n_train, t_len, d_model, generator=g_data) < 0.1).float()
    g_fit = torch.Generator().manual_seed(seed + 7_000)
    fit_stdp(sep, spikes, epochs=epochs, a_plus=0.01 * multiplier, a_minus=0.008 * multiplier,
             tau=0.9, generator=g_fit)
    return sep.weight.detach()


def test_warn_if_degenerate_real_stdp_default_point_no_warning() -> None:
    """既定点相当（倍率1.0）の実際のSTDP学習では警告が出ないこと。"""
    weight = _small_scale_stdp_weight(multiplier=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        max_cos = warn_if_degenerate(weight, threshold=0.99)
    assert max_cos < 0.99


def test_warn_if_degenerate_real_stdp_high_multiplier_warns() -> None:
    """高倍率（縮退が進む方向）の実際のSTDP学習では確実に警告が出ること。

    小規模設定（d_model=32・n_units=64）ではステップ27・28本体の倍率グリッド
    （3倍・10倍）そのままでは縮退が閾値まで進まないため、同じ「倍率を上げると
    縮退が進む」という定性的傾向を保ったまま、この設定で実際に閾値を跨ぐ
    50倍点を用いる（本テストの目的は`warn_if_degenerate`が実際の`fit_stdp`
    経路の出力に対して正しく反応することの確認であり、規模依存の絶対倍率の
    一致は要求しない）。
    """
    weight = _small_scale_stdp_weight(multiplier=50.0)
    with pytest.warns(UserWarning):
        max_cos = warn_if_degenerate(weight, threshold=0.99)
    assert max_cos >= 0.99


# --- ステップ29（12.6.66節）: 主張(a) CI/pre-commitが使う実行コマンドの
#     失敗検知能力（意図的に壊したダミーテストへのpytest実行） -----------------

_BROKEN_TEST_FILE = textwrap.dedent(
    """
    # 意図的に壊した偽のspiking自己検証: 順列不変性チェックを外し、
    # 常に「合格」するふりをするダミー実装。
    def test_fake_permutation_invariance_check_that_always_passes() -> None:
        assert True  # 本来行うべき順列不変性の検証を一切行わない

    def test_intentionally_failing_smoke_check() -> None:
        assert False, "CI/pre-commitのpytest実行が失敗を検知できるかの意図的な失敗"
    """
)


def test_pytest_detects_intentionally_broken_dummy_test(tmp_path: Path) -> None:
    """CI（`.github/workflows/tests.yml`）・pre-commit（`.pre-commit-config.yaml`）
    がいずれも`pytest`をそのまま呼び出す構成であるため、`pytest`自体が意図的に
    壊れたテストに対して非0終了コードを返すことを確認すれば、CI/pre-commit
    双方の失敗検知能力を代理的に検証したことになる（実際のpush・GitHub Actions
    実行結果の確認は範囲外。設計文書12.6.66節の合格条件を参照）。
    """
    broken_file = tmp_path / "test_broken_dummy.py"
    broken_file.write_text(_BROKEN_TEST_FILE, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(broken_file)],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0, (
        f"意図的に壊したダミーテストに対してpytestが非0終了コードを返さなかった。\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
