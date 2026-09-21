"""スパイキングニューロン自体の性質を検証するテスト。

12.7節「スパイキングであることの自己検証の必要性」に対応する常設テスト。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.neurons import ALIFNeuron, NeuronConfig, StepActivation  # noqa: E402
from neurocortex.surrogate import spike  # noqa: E402


@pytest.fixture
def cfg() -> NeuronConfig:
    return NeuronConfig()


def test_spike_output_is_strictly_binary(cfg: NeuronConfig) -> None:
    """ALIFの出力は厳密に {0, 1} のみ。"""
    x = torch.randn(4, 32, 16) * 3.0
    out = ALIFNeuron(cfg)(x)
    assert torch.isin(out, torch.tensor([0.0, 1.0])).all()


def test_forward_is_exact_heaviside() -> None:
    """順伝播はサロゲートではなく厳密な階段関数であること。"""
    x = torch.tensor([-1.0, -1e-9, 0.0, 1e-9, 1.0])
    assert torch.equal(spike(x), torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]))


def test_surrogate_gradient_flows_and_is_not_ste() -> None:
    """勾配が流れ、かつ straight-through（恒等）ではないこと。"""
    x = torch.linspace(-2, 2, 64, requires_grad=True)
    spike(x, alpha=2.0).sum().backward()
    assert x.grad is not None
    assert (x.grad > 0).all(), "arctanサロゲートは全域で正の勾配を持つ"
    # STE なら勾配は一定（1 または 0/1 の矩形）。arctan は 0 付近で最大の単峰形。
    g = x.grad
    assert g.argmax().item() in (31, 32), "勾配は x=0 付近で最大になるはず"
    assert g[0] < g[32] * 0.1, "裾では勾配が十分小さいはず（STEなら一定）"


def test_alif_has_genuine_temporal_dynamics(cfg: NeuronConfig) -> None:
    """同一の定常入力に対して出力が位置ごとに変化する（＝時間ダイナミクスがある）。

    12.9節で棄却された方式は定常入力に対し完全に定常な出力を返す。ここが両者の分岐点。
    """
    x = torch.full((1, 12, 1), 0.6)
    out = ALIFNeuron(cfg)(x).flatten()
    assert out.unique().numel() > 1, "定常入力に定常応答なら時間ダイナミクスが無い"


def test_alif_is_not_permutation_invariant(cfg: NeuronConfig) -> None:
    """位置を入れ替えると出力が変わる（位置間に依存がある）。"""
    x = torch.randn(1, 16, 4)
    perm = torch.randperm(16)
    neuron = ALIFNeuron(cfg)
    assert not torch.equal(neuron(x)[:, perm], neuron(x[:, perm]))


def test_step_activation_is_permutation_invariant(cfg: NeuronConfig) -> None:
    """対照群（階段関数）は位置間の依存を一切持たない。"""
    x = torch.randn(1, 16, 4)
    perm = torch.randperm(16)
    step = StepActivation(cfg)
    assert torch.equal(step(x)[:, perm], step(x[:, perm]))


def test_step_activation_equals_a_staircase_function(cfg: NeuronConfig) -> None:
    """12.9節の再現: 階段関数方式は単調な n_steps+1 値の階段に厳密に退化する。"""
    step = StepActivation(cfg)
    grid = torch.linspace(-3.0, 5.0, 100_001).view(1, -1, 1)
    r = step(grid).flatten()
    assert sorted(set(r.tolist())) == [i / cfg.n_steps for i in range(cfg.n_steps + 1)]
    assert (r[1:] - r[:-1] < 0).sum().item() == 0, "単調非減少であること"
    # 折れ目だけから組み立てた階段関数と、ランダム入力で厳密一致すること
    edges = grid.flatten()[1:][(r[1:] - r[:-1]) > 0]
    rnd = torch.empty(50_000).uniform_(-3.0, 5.0)
    rebuilt = torch.stack([(rnd >= e).float() for e in edges]).sum(0) / cfg.n_steps
    assert torch.equal(step(rnd.view(1, -1, 1)).flatten(), rebuilt)


def test_plain_lif_reduces_to_quantized_relu() -> None:
    """漏れも適応も無い場合、量子化ReLUと厳密に一致する（12.9節の核心）。"""
    cfg = NeuronConfig(beta=1.0, kappa=0.0, theta=1.0, n_steps=4)
    x = torch.empty(1, 100_000, 1).uniform_(-2.0, 3.0)
    got = StepActivation(cfg)(x)
    want = torch.clamp(torch.floor(4.0 * x) / 4.0, 0.0, 1.0)
    assert torch.equal(got, want)


def test_soft_reset_subtracts_threshold(cfg: NeuronConfig) -> None:
    """ソフトリセット（減算型）であり、ハードリセット（ゼロ代入）ではないこと。

    大きな入力を1回だけ与えた次の位置で、残差 (I - theta) * beta が効いて
    追加入力なしでも発火しうることで確認する。
    """
    cfg = NeuronConfig(beta=1.0, kappa=0.0, theta=1.0)
    x = torch.zeros(1, 2, 1)
    x[0, 0, 0] = 2.5  # 発火後、ソフトリセットなら 1.5 が残る
    out = ALIFNeuron(cfg)(x).flatten()
    assert out.tolist() == [1.0, 1.0], "ハードリセットなら2位置目は発火しない"


def test_membrane_carry_can_be_disabled(cfg: NeuronConfig) -> None:
    """carry_membrane=False で位置間の依存が消えること（統制2の介入が正しく効く）。"""
    x = torch.randn(1, 16, 4)
    neuron = ALIFNeuron(cfg, carry_membrane=False)
    perm = torch.randperm(16)
    assert torch.equal(neuron(x)[:, perm], neuron(x[:, perm]))
