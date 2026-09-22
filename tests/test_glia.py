"""グリア層のBCM型ロードバランシング（12.6.17節・ステップ5）の単体テスト。

最終的な均等化性能ではなく、統制2（局所性）・統制3（凍結による推論時のみの
適応不能性）という機構そのものを機械的に検証する
（`tests/test_path_audit.py`・`test_basal_ganglia.py` と同型）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.glia import (  # noqa: E402
    AuxLossCorrection,
    LoadBalanceSpec,
    bcm_update,
    flip_bias,
    imbalance,
    make_bias,
    proportional_update,
    route_batch,
)


# --- 偏り・ルーティング --------------------------------------------------------

def test_flip_bias_negates() -> None:
    bias = make_bias(6, seed=0)
    assert torch.equal(flip_bias(bias), -bias)


def test_make_bias_is_deterministic() -> None:
    assert torch.equal(make_bias(6, seed=3), make_bias(6, seed=3))


def test_route_batch_usage_sums_to_one() -> None:
    spec = LoadBalanceSpec(n_experts=5, batch_size=32)
    bias = make_bias(5, seed=0)
    correction = torch.zeros(5)
    g = torch.Generator().manual_seed(0)
    usage, soft_probs = route_batch(spec, bias, correction, g)
    assert usage.sum() == pytest.approx(1.0)
    assert torch.allclose(soft_probs.sum(dim=-1), torch.ones(32), atol=1e-5)


def test_large_bias_concentrates_usage_without_correction() -> None:
    """補正なしでは、偏った地のバイアスが利用率の不均衡を作ること（課題設計の前提）。"""
    spec = LoadBalanceSpec(n_experts=6, batch_size=256, noise_scale=0.5)
    bias = make_bias(6, seed=1, scale=3.0)
    correction = torch.zeros(6)
    g = torch.Generator().manual_seed(0)
    usage, _ = route_batch(spec, bias, correction, g)
    assert imbalance(usage, spec.target_rate) > 0.02


def test_imbalance_is_zero_for_uniform_usage() -> None:
    spec = LoadBalanceSpec(n_experts=4)
    uniform = torch.full((4,), 0.25)
    assert imbalance(uniform, spec.target_rate) == pytest.approx(0.0)


# --- 局所性の監査（統制2） -------------------------------------------------------

def test_bcm_update_is_local_to_each_expert() -> None:
    """専門家iの補正は専門家iの利用率のみに依存し、他の専門家の値を変えても動かない。"""
    usage_a = torch.tensor([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
    usage_b = torch.tensor([0.5, 0.9, 0.0, 0.0, 0.0, 0.0])  # 専門家1以降だけ変える
    corr_a, corr_b = torch.zeros(6), torch.zeros(6)
    msq_a, msq_b = torch.zeros(6), torch.zeros(6)
    bcm_update(corr_a, usage_a, msq_a, alpha=1.0, beta=0.1, lr=0.5)
    bcm_update(corr_b, usage_b, msq_b, alpha=1.0, beta=0.1, lr=0.5)
    assert corr_a[0] == pytest.approx(corr_b[0]), "専門家0の補正が他の専門家の利用率に影響されている"
    assert msq_a[0] == pytest.approx(msq_b[0])


def test_proportional_update_is_local_to_each_expert() -> None:
    usage_a = torch.tensor([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
    usage_b = torch.tensor([0.5, 0.9, 0.0, 0.0, 0.0, 0.0])
    corr_a, corr_b = torch.zeros(6), torch.zeros(6)
    proportional_update(corr_a, usage_a, target_rate=1 / 6, lr=0.5)
    proportional_update(corr_b, usage_b, target_rate=1 / 6, lr=0.5)
    assert corr_a[0] == pytest.approx(corr_b[0])


# --- 補正則の方向の健全性 --------------------------------------------------------

def test_bcm_suppresses_overactive_expert() -> None:
    """慢性的に過活動な専門家は補正が下がる方向（抑制）に動くこと（8.2節のLTD相当）。"""
    correction = torch.zeros(1)
    msq = torch.zeros(1)
    usage = torch.tensor([0.9])  # 目標(例えば1/6)よりずっと高い、慢性的な過活動を模す
    for _ in range(20):
        bcm_update(correction, usage, msq, alpha=1.0, beta=0.2, lr=0.1)
    assert correction.item() < 0


def test_proportional_pushes_toward_target_rate() -> None:
    correction = torch.zeros(1)
    usage = torch.tensor([0.9])
    proportional_update(correction, usage, target_rate=1 / 6, lr=0.5)
    assert correction.item() < 0  # 利用率が目標より高いので抑制方向


# --- 凍結（統制3・主張a） --------------------------------------------------------

def test_frozen_aux_loss_does_not_update() -> None:
    """統制3・主張(a)の核心: freeze()後は勾配ステップが一切起きないこと。"""
    torch.manual_seed(0)
    aux = AuxLossCorrection(n_experts=4, lr=0.5)
    before = aux.correction.detach().clone()
    spec = LoadBalanceSpec(n_experts=4, batch_size=32)
    bias = make_bias(4, seed=0, scale=3.0)
    g = torch.Generator().manual_seed(0)

    aux.freeze()
    for _ in range(10):
        usage, soft_probs = route_batch(spec, bias, aux.correction, g)
        loss = aux.step(soft_probs, spec.target_rate)
        assert loss == 0.0
    assert torch.equal(aux.correction.detach(), before), "凍結後もcorrectionが動いている"


def test_unfrozen_aux_loss_reduces_loss() -> None:
    """健全性チェック: 凍結しなければ通常通り損失が下がっていくこと。"""
    torch.manual_seed(0)
    aux = AuxLossCorrection(n_experts=4, lr=0.5)
    spec = LoadBalanceSpec(n_experts=4, batch_size=64)
    bias = make_bias(4, seed=0, scale=3.0)
    g = torch.Generator().manual_seed(1)
    losses = []
    for _ in range(100):
        usage, soft_probs = route_batch(spec, bias, aux.correction, g)
        losses.append(aux.step(soft_probs, spec.target_rate))
    assert losses[-1] < losses[0]


def test_bcm_and_proportional_keep_adapting_when_aux_loss_would_be_frozen() -> None:
    """主張(a)のエンドツーエンド確認: 「推論時のみ」を模した後も、局所則は

    利用率を目標に近づけ続けられること（勾配・凍結という概念自体が存在しない）。
    """
    torch.manual_seed(0)
    spec = LoadBalanceSpec(n_experts=4, batch_size=64, noise_scale=0.5)
    bias = make_bias(4, seed=2, scale=3.0)
    correction = torch.zeros(4)
    msq = torch.zeros(4)
    g = torch.Generator().manual_seed(3)
    imbalances = []
    for _ in range(200):
        usage, _ = route_batch(spec, bias, correction, g)
        bcm_update(correction, usage, msq, alpha=1.0, beta=0.1, lr=0.5)
        imbalances.append(imbalance(usage, spec.target_rate))
    assert imbalances[-1] < imbalances[0]
