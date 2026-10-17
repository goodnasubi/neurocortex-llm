"""ハブ専門家への負荷集中（12.6.123節）のテスト。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neurocortex.expert_prefetch import RoutedCharLM
from neurocortex.hub_load import (eval_ce, functional_degree, gini, lesion_study, lesioned_forward,
                                  make_windows, max_over_mean, normalized_entropy, spearman,
                                  transition_matrix)


def _model(seed: int = 0):
    torch.manual_seed(seed)
    model = RoutedCharLM(vocab_size=12, d=8, n_experts=4, d_ff=8)
    x, y = make_windows(torch.randint(0, 12, (400,)), seq_len=16, n_windows=10)
    return model, x, y


def test_load_metrics():
    assert gini([5, 5, 5, 5]) == pytest.approx(0.0)
    assert gini([0, 0, 0, 8]) == pytest.approx(0.75)  # (E-1)/E
    assert normalized_entropy([3, 3, 3]) == pytest.approx(1.0)
    assert max_over_mean([0, 0, 0, 8]) == pytest.approx(4.0)
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_unlesioned_forward_matches_model():
    model, x, _ = _model()
    with torch.no_grad():
        ref_logits, _, ref_choice, _ = model(x)
        for mode in ("zero", "reroute"):
            logits, choice = lesioned_forward(model, x, None, mode)
            assert torch.allclose(logits, ref_logits, atol=1e-6)
            assert torch.equal(choice, ref_choice)
            off = torch.zeros(4, dtype=torch.bool)
            logits, _ = lesioned_forward(model, x, off, mode)
            assert torch.allclose(logits, ref_logits, atol=1e-6)


def test_zero_lesion_removes_expert_output_only():
    model, x, _ = _model()
    with torch.no_grad():
        _, _, choice, h = model(x)
        e = int(choice.flatten().mode().values)
        dis = torch.zeros(4, dtype=torch.bool)
        dis[e] = True
        logits, ch = lesioned_forward(model, x, dis, "zero")
        assert torch.equal(ch, choice)  # ルーティングは変えない
        m = choice == e
        assert torch.allclose(logits[m], model.head(h)[m], atol=1e-6)  # 残差だけが通る
        ref, _, _, _ = model(x)
        assert torch.allclose(logits[~m], ref[~m], atol=1e-6)


def test_reroute_never_picks_disabled():
    model, x, _ = _model()
    dis = torch.tensor([True, True, False, False])
    with torch.no_grad():
        _, ch = lesioned_forward(model, x, dis, "reroute")
    assert set(ch.unique().tolist()) <= {2, 3}


def test_transition_graph():
    choices = torch.tensor([[0, 1, 0, 1], [2, 2, 2, 2]])
    m = transition_matrix(choices, 3)
    assert m[0, 1] == 2 and m[1, 0] == 1 and m[2, 2] == 3 and m.sum() == 6
    degree, strength = functional_degree(m)
    assert degree.tolist() == [1, 1, 0]  # 自己遷移は結合に数えない
    assert strength[2] == 0.0


def test_lesion_study_hubs_and_unused_experts():
    model, x, y = _model()
    load = np.array([10.0, 0.0, 30.0, 0.0])
    r = lesion_study(model, x, y, load, k=2, mode="zero")
    assert r["hubs"] == [2, 0]
    assert r["base_ce"] == pytest.approx(eval_ce(model, x, y))
    assert len(r["d_single"]) == 4
    assert r["hub_load_frac"] == pytest.approx(1.0)
    assert r["n_random_with_load"] == 0 and np.isnan(r["d_random_per_load"])
