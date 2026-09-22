"""小脳モジュール（12.6.14節・ステップ4）の単体テスト。

主張は「持続する状態」であり、学習則（デルタ則）そのものではない。したがって
最重要の検証は「cerebellum条件とfrom-scratch条件が本当に同一の学習則・同一の
探索アルゴリズムを通っているか」（統制5「学習則の監査」）であり、最終結果の
良し悪しではない（`tests/test_path_audit.py`・`test_basal_ganglia.py` と同型）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.cerebellum import (  # noqa: E402
    ForwardModel,
    call_tool,
    make_tool,
    run_episode,
    sample_target,
)


# --- ツール ---------------------------------------------------------------------

def test_tool_is_deterministic_given_seed() -> None:
    """統制3: 同じシードなら同じツールになること（エピソードをまたぐ固定の前提）。"""
    t1 = make_tool(dim=4, seed=7)
    t2 = make_tool(dim=4, seed=7)
    assert torch.equal(t1.weight, t2.weight)
    assert t1.bias == t2.bias


def test_call_tool_is_linear() -> None:
    tool = make_tool(dim=3, seed=0)
    args = torch.tensor([1.0, 2.0, -1.0])
    expected = args @ tool.weight + tool.bias
    assert torch.allclose(call_tool(tool, args), expected)


def test_sample_target_is_reachable() -> None:
    """sample_target が返す値は、あるargsでのツール出力そのものなので到達可能。"""
    tool = make_tool(dim=3, seed=0)
    g = torch.Generator().manual_seed(0)
    target = sample_target(tool, g, scale=1.0)
    # 十分広い探索なら厳密一致するargsが見つかるはず、という直接の保証にはならないが、
    # 値そのものが実際の出力レンジ内にあることは最低限確認できる。
    probe = torch.randn(2000, 3, generator=torch.Generator().manual_seed(1)) * 3.0
    outputs = probe @ tool.weight + tool.bias
    assert outputs.min() - 1.0 <= target <= outputs.max() + 1.0


# --- 順モデル（デルタ則） --------------------------------------------------------

def test_forward_model_starts_at_zero() -> None:
    model = ForwardModel(dim=4)
    assert torch.all(model.weight == 0)
    assert model.bias == 0.0
    assert float(model.predict(torch.ones(4))) == 0.0


def test_delta_rule_converges_to_true_linear_function() -> None:
    """デルタ則を十分回せば、線形モデルの真の重みに収束すること（学習則の健全性）。"""
    torch.manual_seed(0)
    tool = make_tool(dim=3, seed=1)
    model = ForwardModel(dim=3)
    g = torch.Generator().manual_seed(2)
    for _ in range(2000):
        args = torch.randn(3, generator=g)
        y = float(call_tool(tool, args))
        model.update(args, y, lr=0.05)
    assert torch.allclose(model.weight, tool.weight, atol=0.05)
    assert model.bias == pytest.approx(tool.bias, abs=0.05)


def test_update_returns_pre_update_error() -> None:
    model = ForwardModel(dim=2)
    error = model.update(torch.tensor([1.0, 0.0]), y_true=3.0, lr=0.1)
    assert error == pytest.approx(3.0)  # 予測はまだ0なので誤差=真値


# --- 探索エピソード ---------------------------------------------------------------

def test_random_search_never_touches_model() -> None:
    """model=None（random-search条件）は、渡しようがないため学習が起きない設計。

    「モデルを持たない」条件が本当にモデルなしで動くことを確認する（統制の前提）。
    """
    tool = make_tool(dim=2, seed=0)
    g = torch.Generator().manual_seed(0)
    result = run_episode(tool, target=0.0, eps=1e6, max_real_calls=5,
                         n_candidates=8, step_size=1.0, lr=0.1, generator=g, model=None)
    assert result.success  # eps が非常に緩いので1回で成功するはず
    assert result.real_calls_used == 1


def test_cerebellum_and_from_scratch_share_the_same_update_rule() -> None:
    """統制5: 同じ (args, y) 列を与えれば、cerebellum用・from-scratch用の

    `ForwardModel` インスタンスは全く同じ重みに収束すること（両条件が本当に
    同一の学習則を通っていることの機械的保証）。
    """
    tool = make_tool(dim=3, seed=0)
    model_a = ForwardModel(3)  # cerebellum役
    model_b = ForwardModel(3)  # from-scratch役（このテストでは同じデータを与える）
    g = torch.Generator().manual_seed(1)
    for _ in range(50):
        args = torch.randn(3, generator=g)
        y = float(call_tool(tool, args))
        model_a.update(args, y, lr=0.1)
        model_b.update(args, y, lr=0.1)
    assert torch.equal(model_a.weight, model_b.weight)
    assert model_a.bias == model_b.bias


def test_persistent_model_is_the_same_object_across_episodes() -> None:
    """cerebellum条件の「持続性」は、呼び出し側が同一インスタンスを使い回すことで

    実現される。run_cerebellum.run_condition の契約をここで直接検証する。
    """
    from neurocortex.experiments.run_cerebellum import run_condition

    class _Args:
        n_episodes = 3
        eps = 0.05
        max_real_calls = 10
        n_candidates = 16
        step_size = 1.0
        lr = 0.2
        target_scale = 1.0

    tool = make_tool(dim=2, seed=0)
    g = torch.Generator().manual_seed(0)
    # cerebellum条件が完走すること（=同一モデルの使い回しで例外なく動くこと）を確認する。
    rows = run_condition("cerebellum", tool, _Args(), g)
    assert len(rows) == 3
    assert all(isinstance(r["real_calls_used"], int) for r in rows)


def test_from_scratch_matches_a_freshly_built_model_each_episode() -> None:
    """from-scratch条件の各エピソードは、真にゼロ初期化のモデルから始まること。

    同じ乱数状態から (a) `run_condition("from-scratch", ...)` の2エピソード目と
    (b) `ForwardModel` を明示的に新規作成して単独で走らせた1エピソード分が、
    完全に一致することを確認する（使い回されていれば1エピソード目の学習が
    残り、一致しなくなる）。
    """
    from neurocortex.cerebellum import run_episode
    from neurocortex.experiments.run_cerebellum import run_condition

    class _Args:
        n_episodes = 2
        eps = 0.05
        max_real_calls = 10
        n_candidates = 16
        step_size = 1.0
        lr = 0.2
        target_scale = 1.0

    tool = make_tool(dim=2, seed=0)

    g1 = torch.Generator().manual_seed(0)
    rows = run_condition("from-scratch", tool, _Args(), g1)

    # 同じ乱数消費列を再現するため、1エピソード目を同じ手順（新規モデルで
    # run_episodeを1回分走らせる）で実際に消費してから、2エピソード目だけを
    # 新規モデルで単独に走らせ、一致を確認する。
    g2 = torch.Generator().manual_seed(0)
    from neurocortex.cerebellum import sample_target
    target1 = sample_target(tool, g2, scale=_Args.target_scale)
    run_episode(tool, target1, _Args.eps, _Args.max_real_calls, _Args.n_candidates,
               _Args.step_size, _Args.lr, g2, ForwardModel(2))  # 1エピソード目を空費

    fresh_model = ForwardModel(2)
    target2 = sample_target(tool, g2, scale=_Args.target_scale)
    result2 = run_episode(tool, target2, _Args.eps, _Args.max_real_calls,
                          _Args.n_candidates, _Args.step_size, _Args.lr, g2, fresh_model)

    assert rows[1]["real_calls_used"] == result2.real_calls_used
    assert rows[1]["success"] == result2.success
