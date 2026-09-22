"""統制5「経路の監査」（12.6.1節）。

テストAの構成に、位置を跨ぐ演算（アテンション・畳み込み・cumsum 等）が存在しない
ことを機械的に保証する。12.9節と同種の失敗（別経路で解けていた）の予防。

保証は3段構えで行う:
  (a) 構文木（AST）上に禁止演算の呼び出しが現れないことの静的検査
  (b) 使われているモジュール型がホワイトリストに収まることの検査
  (c) ヤコビアンによる動的検査（最も強い。実際の依存関係そのものを測る）
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex import models, neurons  # noqa: E402
from neurocortex.models import EmbeddingFreeRecallModel, RecallModel  # noqa: E402
from neurocortex.neurons import NeuronConfig  # noqa: E402

# 位置（トークン）方向に情報を混ぜうる演算。テストAの経路上に存在してはならない。
# 文字列一致ではなく構文木上の識別子として検査するので、docstring中の言及は誤検知しない。
FORBIDDEN_NAMES = frozenset({
    "cumsum", "cumprod", "logcumsumexp",
    "conv1d", "conv2d", "conv3d", "Conv1d", "Conv2d",
    "softmax", "log_softmax", "einsum", "roll", "flip", "fliplr",
    "scatter", "scatter_add", "gather", "unfold", "matmul", "bmm",
    "scaled_dot_product_attention", "MultiheadAttention", "GRU", "LSTM", "RNN",
})

ALLOWED_MODULE_TYPES = (
    nn.Embedding, nn.Linear, nn.ModuleList,
    neurons.ALIFNeuron, neurons.StepActivation,
)


def _forbidden_identifiers(module) -> set[str]:
    """モジュールの構文木から、禁止演算にあたる識別子・演算子の使用を抽出する。"""
    tree = ast.parse(inspect.getsource(module))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            found.add(node.attr)
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            found.add(node.id)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            found.add("@ (行列積演算子)")
    return found


def test_source_has_no_cross_position_ops() -> None:
    """(a) テストAの経路を構成するソースに、位置を跨ぐ演算が存在しないこと。"""
    for module in (models, neurons):
        found = _forbidden_identifiers(module)
        assert not found, f"{module.__name__} に位置を跨ぐ演算がある: {sorted(found)}"


def test_audit_detects_a_planted_violation() -> None:
    """(a') 監査そのものが機能していることの確認（線形アテンションを検出できるか）。"""
    from neurocortex import attention

    assert _forbidden_identifiers(attention), "監査が明らかな違反を見逃している"


@pytest.mark.parametrize("activation", ["spiking", "step"])
def test_only_whitelisted_module_types(activation: str) -> None:
    """(b) モデルが位置独立なモジュールとニューロンだけで構成されていること。"""
    model = RecallModel(18, 8, activation=activation)
    for name, m in model.named_modules():
        if name == "":
            continue
        assert isinstance(m, ALLOWED_MODULE_TYPES), f"{name}: 想定外の型 {type(m)}"


def _cross_position_jacobian(model: EmbeddingFreeRecallModel, t: int = 8, d: int = 8):
    """出力[位置i] の 入力[位置j] に対する勾配ノルム行列 [T, T] を返す。"""
    jac = torch.zeros(t, t)
    for i in range(t):
        x = torch.randn(1, t, d, requires_grad=True)
        torch.manual_seed(i)
        out = model.forward_from_dense(x)[0, i].sum()
        (g,) = torch.autograd.grad(out, x, allow_unused=True)
        jac[i] = g[0].abs().sum(dim=-1)
    return jac


def _make(activation: str, carry: bool = True) -> EmbeddingFreeRecallModel:
    torch.manual_seed(0)
    m = EmbeddingFreeRecallModel(
        18, 8, d_model=8, n_layers=2, cfg=NeuronConfig(beta=0.95, alpha=5.0),
        activation=activation,
    )
    m.set_carry_membrane(carry)
    return m


def test_step_model_has_no_cross_position_dependence() -> None:
    """(c-1) 対照群は出力[i]が入力[i]にしか依存しない＝位置間経路が存在しない。"""
    jac = _cross_position_jacobian(_make("step"))
    off_diagonal = jac - torch.diag(torch.diagonal(jac))
    assert float(off_diagonal.abs().max()) == 0.0


def test_spiking_model_depends_on_past_only() -> None:
    """(c-2) スパイキング版は過去に依存し、未来には依存しない（因果性）。"""
    jac = _cross_position_jacobian(_make("spiking"))
    future = torch.triu(jac, diagonal=1)
    assert float(future.abs().max()) == 0.0, "未来の入力に依存している（因果性の破れ）"
    past = torch.tril(jac, diagonal=-1)
    assert float(past.abs().max()) > 0.0, "過去への経路が存在しない"


def test_membrane_is_the_only_cross_position_path() -> None:
    """(c-3) 膜電位の持ち越しを切ると位置間依存が完全に消える。

    これが「位置間を結ぶ経路は膜電位だけ」ということの機械的な証明になる。
    """
    jac = _cross_position_jacobian(_make("spiking", carry=False))
    off_diagonal = jac - torch.diag(torch.diagonal(jac))
    assert float(off_diagonal.abs().max()) == 0.0


# --- ステップ1（海馬モジュール, 12.6.4節）統制5 ------------------------------
#
# 「書き込みに勾配を要しない」という主張は、コード上で保証しない限り主張にならない。
# 皮質バックボーンが凍結されたままであることを機械的に確かめる。

def test_hippocampus_leaves_the_cortex_frozen() -> None:
    from neurocortex.experiments.run_hippocampus import build_backbone, run_condition
    from neurocortex.tasks import FactSpec, make_fact_batch

    spec = FactSpec(n_prefix=2, n_suffix=32, n_object=8)
    model = build_backbone(spec, d_model=16, n_layers=1, n_heads=2,
                           seed=0, train_steps=2, lr=1e-3, batch_size=8)
    assert not any(p.requires_grad for p in model.parameters()), "皮質が凍結されていない"

    before = [p.detach().clone() for p in model.parameters()]
    prompts, objects = make_fact_batch(spec, 16, torch.Generator().manual_seed(0))

    class _Args:
        n_units, sep_seed, cue_noise, tap = 128, 0, 0.0, "ln_f"

    row = run_condition(model, spec, prompts, objects, "sdr", _Args(),
                        beta=50.0, gain=16.0, k=8, sep_seed=0)
    assert row["n_stored"] == 16
    # 書き込み・読み出しを経ても皮質の重みが1ビットも変わらないこと。
    for p, q in zip(model.parameters(), before):
        assert torch.equal(p, q), "海馬の書き込みが皮質の重みを変えている"
