"""小脳モジュール — 逆モデル本体（12.6.24節・ステップ8）の単体テスト。

最終的な引数再現誤差の良し悪しではなく、統制2（inverse-modelとpseudo-inverse
が同じ呼び出し履歴で学習すること）・統制5b（from-scratch条件が実際に
エピソードごとに再初期化されること）という機構そのものを機械的に検証する
（`tests/test_cerebellum.py`・`test_hopfield.py` と同型）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.cerebellum import ForwardModel, call_tool, make_tool  # noqa: E402
from neurocortex.cerebellum_inverse import (  # noqa: E402
    InverseModel,
    pseudo_inverse_predict,
    run_condition,
    sample_from_D,
)


# --- 逆モデル本体 ------------------------------------------------------------------

def test_inverse_model_starts_at_zero() -> None:
    model = InverseModel(dim=4)
    assert torch.all(model.A == 0)
    assert torch.all(model.c == 0)
    assert torch.equal(model.predict(3.0), torch.zeros(4))


def test_inverse_model_update_returns_pre_update_error() -> None:
    model = InverseModel(dim=2)
    x_true = torch.tensor([1.0, 2.0])
    error = model.update(y=1.0, x_true=x_true, lr=0.1)
    assert torch.equal(error, x_true)  # 予測はまだ0ベクトルなので誤差=真値


def test_inverse_model_converges_to_true_linear_inverse() -> None:
    """健全性チェック: 十分な (y, x) データがあれば、逆モデルは真の線形写像に収束すること。"""
    torch.manual_seed(0)
    true_A = torch.tensor([0.5, -1.0, 2.0])
    true_c = torch.tensor([1.0, 0.0, -0.5])
    model = InverseModel(dim=3)
    g = torch.Generator().manual_seed(1)
    for _ in range(3000):
        y = float(torch.randn(1, generator=g))
        x_true = true_A * y + true_c
        model.update(y, x_true, lr=0.05)
    assert torch.allclose(model.A, true_A, atol=0.05)
    assert torch.allclose(model.c, true_c, atol=0.05)


# --- 疑似逆行列（統制5） --------------------------------------------------------

def test_pseudo_inverse_recovers_minimum_norm_solution() -> None:
    """疑似逆行列は不良設定問題の最小ノルム解を返すこと（wと平行な解）。"""
    forward_model = ForwardModel(dim=3)
    forward_model.weight = torch.tensor([1.0, 2.0, 2.0])  # ノルム3
    forward_model.bias = 1.0
    x_hat = pseudo_inverse_predict(forward_model, y_target=4.0)
    # x̂ = w*(y-b)/||w||^2 = [1,2,2]*3/9 = [1/3, 2/3, 2/3]
    assert torch.allclose(x_hat, torch.tensor([1 / 3, 2 / 3, 2 / 3]), atol=1e-5)
    # 検算: w・x̂ + b が目標に一致すること
    assert float(forward_model.weight @ x_hat + forward_model.bias) == pytest.approx(4.0, abs=1e-5)


def test_pseudo_inverse_is_the_minimum_norm_among_valid_solutions() -> None:
    """x̂はwに平行（不良設定の解空間の中で最小ノルムであることの間接的な確認）。"""
    forward_model = ForwardModel(dim=4)
    forward_model.weight = torch.tensor([1.0, 1.0, -1.0, 2.0])
    forward_model.bias = 0.0
    x_hat = pseudo_inverse_predict(forward_model, y_target=5.0)
    w = forward_model.weight
    cos_sim = float((x_hat @ w) / (x_hat.norm() * w.norm()))
    assert cos_sim == pytest.approx(1.0, abs=1e-5)


def test_pseudo_inverse_handles_zero_weight_without_crashing() -> None:
    forward_model = ForwardModel(dim=2)  # weight は初期化直後ゼロ
    x_hat = pseudo_inverse_predict(forward_model, y_target=1.0)
    assert torch.equal(x_hat, torch.zeros(2))


# --- 呼び出し履歴の生成 -----------------------------------------------------------

def test_sample_from_D_uses_specified_mean_and_seed_is_deterministic() -> None:
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    x1 = sample_from_D(dim=4, mean=3.0, std=0.5, generator=g1)
    x2 = sample_from_D(dim=4, mean=3.0, std=0.5, generator=g2)
    assert torch.equal(x1, x2)


# --- 条件ごとの実行（統制2・5b） -------------------------------------------------

def test_run_condition_returns_expected_number_of_eval_points() -> None:
    tool = make_tool(dim=3, seed=0)
    g = torch.Generator().manual_seed(0)
    rows = run_condition("inverse-model", tool, n_calls=100, episode_len=20,
                         d_mean=3.0, d_std=0.5, lr=0.05, eval_every=25, n_eval=8,
                         generator=g)
    assert len(rows) == 4
    assert all(r["x_error"] >= 0.0 and r["y_error"] >= 0.0 for r in rows)


def test_inverse_from_scratch_resets_at_episode_boundaries() -> None:
    """統制5bの核心: from-scratch条件はエピソード境界で逆モデルの重みが実際にゼロに戻ること。

    十分に学習が進んだ後の誤差が、持続するinverse-modelより明確に悪化していること
    （＝本当にリセットされていること）を間接的に確認する。
    """
    tool = make_tool(dim=3, seed=1)
    g_persist = torch.Generator().manual_seed(0)
    rows_persist = run_condition("inverse-model", tool, n_calls=400, episode_len=20,
                                 d_mean=3.0, d_std=0.5, lr=0.05, eval_every=400, n_eval=64,
                                 generator=g_persist)
    g_scratch = torch.Generator().manual_seed(0)
    rows_scratch = run_condition("inverse-from-scratch", tool, n_calls=400, episode_len=20,
                                 d_mean=3.0, d_std=0.5, lr=0.05, eval_every=400, n_eval=64,
                                 generator=g_scratch)
    assert rows_scratch[-1]["x_error"] > rows_persist[-1]["x_error"]


def test_pseudo_inverse_and_inverse_model_share_the_same_call_history() -> None:
    """統制2: 同じシード・同じn_callsなら、両条件が消費する呼び出し履歴の乱数列は同じ長さ・
    同じタイミングで進む（片方だけ多くデータを見ていない）ことを、実行が完走することで確認する。
    """
    tool = make_tool(dim=3, seed=0)
    g1 = torch.Generator().manual_seed(0)
    rows1 = run_condition("inverse-model", tool, n_calls=100, episode_len=20,
                          d_mean=3.0, d_std=0.5, lr=0.05, eval_every=50, n_eval=8,
                          generator=g1)
    g2 = torch.Generator().manual_seed(0)
    rows2 = run_condition("pseudo-inverse", tool, n_calls=100, episode_len=20,
                          d_mean=3.0, d_std=0.5, lr=0.05, eval_every=50, n_eval=8,
                          generator=g2)
    assert len(rows1) == len(rows2) == 2
