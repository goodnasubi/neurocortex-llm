"""ハブ専門家に負荷に見合う容量を与える MoE（12.6.205節）のテスト。"""

from __future__ import annotations

import argparse

import numpy as np
import pytest
import torch

from neurocortex.data import CharCorpus
from neurocortex.experiments.run_moe_hub_capacity import conditions, judge, post_cap
from neurocortex.experiments.run_moe_scale import run_one
from neurocortex.moe_hub_capacity import HubCapacityLM, allocate_widths
from neurocortex.moe_integration import integration_corpus, split_like_corpus, windows_like
from neurocortex.moe_modular import ModularRoutedLM, region_table


def _corpus(task="cross", n=60, seed=1):
    text, slots, langs = integration_corpus(task, n_lines=n, seed=seed)
    return CharCorpus(text), slots, langs


def _pair(c, **kw):
    torch.manual_seed(3)
    base = ModularRoutedLM(c.vocab_size, region_of_token=region_table(c.stoi), d_mod=8, d_ff=16, n_experts=4)
    torch.manual_seed(3)
    cap = HubCapacityLM(c.vocab_size, region_of_token=region_table(c.stoi), d_mod=8, d_ff=16, n_experts=4,
                        w_min=4, w_max=40, **kw)
    return base, cap


def test_allocate_widths_sum_bounds_and_order():
    w = allocate_widths([0.5, 0.2, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0], 512, 16, 192)
    assert sum(w) == 512 and min(w) >= 16 and max(w) <= 192
    assert w[0] >= w[1] >= w[3] >= w[4] == 16
    assert allocate_widths([1, 1, 1, 1], 64, 4, 40) == [16] * 4  # 均等な負荷なら均等な幅
    with pytest.raises(ValueError):
        allocate_widths([1, 1], 100, 4, 40)


def test_equal_width_matches_modular_model():
    """再配分の前は、`ModularRoutedLM` と同じ初期値・同じ出力（予備の単位は別の乱数で使われない）。"""
    c, _, _ = _corpus()
    base, cap = _pair(c)
    assert torch.equal(cap.w1[0][..., :16], base.w1[0]) and torch.equal(cap.w2[0][:, :16], base.w2[0])
    assert torch.equal(cap.head_w, base.head_w)
    x = c.train[None, :40]
    assert torch.allclose(cap.run(x)["logits"], base.run(x)["logits"], atol=1e-5)
    assert torch.allclose(cap.run(x, cut_remote=True)["logits"], base.run(x, cut_remote=True)["logits"], atol=1e-5)
    assert cap.active_expert_params() == base.w1[0].numel() + base.w2[0].numel()


def test_reallocate_keeps_budget_and_zeroes_new_units():
    c, _, _ = _corpus()
    _, cap = _pair(c)
    x = c.train[None, :40]
    before = cap.active_expert_params()
    w = cap.reallocate([0.6, 0.2, 0.1, 0.1])
    assert w[0] == max(w) and sum(w) == 64
    assert cap.active_expert_params() == before  # 有効なパラメータ数は揃う
    # 広げた単位は出力側（w2 の行）を 0 にして加える
    assert (cap.w2[0][0, 16:w[0]] == 0).all()


def test_random_allocation_is_permutation_of_load_profile():
    c, _, _ = _corpus()
    _, a = _pair(c, alloc="load")
    _, r = _pair(c, alloc="random", alloc_seed=5)
    load = [0.6, 0.2, 0.1, 0.1]
    wa, wr = a.reallocate(load), r.reallocate(load)
    assert sorted(wa) == sorted(wr)
    perm = np.random.default_rng(5).permutation(4)
    assert wr[int(perm[0])] == max(wa)  # 負荷と無関係な置換で割り当てる


def test_realloc_happens_once_during_training():
    c, _, _ = _corpus()
    _, cap = _pair(c, kind="select", eta=0.1, r_lo=0.5, r_hi=2.5, realloc_step=3)
    for _ in range(5):
        choice = torch.zeros(1, 2, 10, dtype=torch.long)  # 専門家 0 に全部送る
        cap.homeostatic_step(choice[0][None])
        if int(cap.steps_seen) < 3:
            assert cap.widths.tolist() == [16] * 4
    assert cap.realloc_load is not None and int(cap.widths[0]) == max(cap.widths.tolist())
    assert int(cap.widths.sum()) == 64


def test_gradients_only_reach_active_units():
    c, _, _ = _corpus()
    _, cap = _pair(c)
    cap.reallocate([0.4, 0.3, 0.2, 0.1])
    x, y = c.train[None, :40], c.train[None, 1:41]
    logits = cap.run(x)["logits"]
    torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).backward()
    for e, w in enumerate(cap.widths.tolist()):
        assert (cap.w1[0].grad[e, :, w:] == 0).all() and (cap.w2[0].grad[e, w:] == 0).all()


def test_run_one_records_widths_and_judge_shape():
    c, slots, langs = _corpus(n=80)
    args = argparse.Namespace(d=8, alpha=0.05, train_steps=4, seq_len=16, n_windows=4, k_num=1, k_den=4,
                              n_random=2, eps=1e-4)
    x_lang, _ = windows_like(split_like_corpus(langs), 16, 4)
    _, y_slot = windows_like(split_like_corpus(slots), 16, 4)
    conds = conditions(0.1)
    assert set(conds) == {"aux0.05", "point0.1", "band0.1", "bandcap0.1", "bandrand0.1"}
    z = run_one(0, 4, 1, conds["bandcap0.1"], c, args, model_cls=HubCapacityLM, post=post_cap(1, x_lang, y_slot),
                r_lo=0.5, r_hi=2.5, region_of_token=region_table(c.stoi), d_mod=8, alloc="load", alloc_seed=0,
                realloc_step=2, w_min=4, w_max=160)
    assert sum(z["widths"]) == 4 * 64 and len(z["hub_widths"]) == 1 and "cut_all_d" in z

    def fake(ce_c):
        r = {"seed": 0}
        for t in ("cross", "within"):
            r[t] = {k: {"slot_ce": {"target": 0.5}, "cut_all_d": {"target": 0.0}, "hubs_widened": 2} for k in conds}
        r["cross"]["bandcap0.1"]["slot_ce"]["target"] = ce_c
        return r
    v = judge([fake(0.3 + 0.01 * i) for i in range(10)], 0.1)
    assert v["H1_交互作用（C−一点）"] and v["H2_ボトルネックの解消（C−均等な目標域）"]
    assert v["M_操作（統合課題の C で学習後のハブ2個がともに広げた専門家）"]


def test_fixed_hubs_widths_and_set_points():
    """修正 1: 構造で決めたハブ。幅は目標の比に比例し、和は均等な幅と同じ。恒常性は専門家ごとの目標に寄せる。"""
    from neurocortex.experiments.run_moe_hub_capacity import hub_design
    d = hub_design(8, 2, 0.57, 64, 16, 192, seed=0)
    (wc, sp), (wr, sp_r), (we, _) = d["load"], d["random"], d["equal"]
    assert sum(wc) == sum(wr) == sum(we) == 512 and sorted(wc) == sorted(wr)
    assert wc[0] == wc[1] == max(wc) and abs(sum(sp) - 8) < 1e-9 and sp == sp_r
    assert set(conditions(0.1, "fixed")) == {"aux0.05", "point0.1", "band0.1", "hubcap0.1", "hubrand0.1", "hubeq0.1"}
    c, _, _ = _corpus()
    torch.manual_seed(3)
    m = HubCapacityLM(c.vocab_size, region_of_token=region_table(c.stoi), d_mod=8, d_ff=16, n_experts=4, w_min=4,
                      w_max=40, widths=[28, 12, 12, 12], set_points=[2.0, 2 / 3, 2 / 3, 2 / 3], kind="select", eta=0.1)
    assert m.widths.tolist() == [28, 12, 12, 12] and m.active_expert_params() == 64 * (32 + 8)
    m.usage_ema.copy_(torch.tensor([[0.5, 1 / 6, 1 / 6, 1 / 6]]))
    assert torch.allclose(m.drive_error(), torch.zeros(1, 4), atol=1e-5)  # 目標どおりなら閾値は動かない
    m.usage_ema.copy_(torch.full((1, 4), 0.25))
    e = m.drive_error()[0]
    assert e[0] < 0 and (e[1:] > 0).all()  # ハブは使われなさすぎ、残りは使われすぎ
    for _ in range(3):
        m.homeostatic_step(torch.zeros(1, 2, 10, dtype=torch.long))
    assert m.widths.tolist() == [28, 12, 12, 12]  # 再配分しない
    with pytest.raises(ValueError):
        HubCapacityLM(c.vocab_size, region_of_token=region_table(c.stoi), d_mod=8, d_ff=16, n_experts=4,
                      w_min=4, w_max=40, widths=[30, 12, 12, 12])
