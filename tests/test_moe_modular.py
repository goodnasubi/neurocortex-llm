"""領域ごとに分かれた土台の MoE（12.6.201節）のテスト。"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from neurocortex.data import CharCorpus
from neurocortex.experiments.run_moe_modular import post
from neurocortex.experiments.run_moe_scale import run_one
from neurocortex.moe_integration import LANGS, integration_corpus, split_like_corpus, windows_like
from neurocortex.moe_modular import ModularRoutedLM, region_table, sticky_regions


def _corpus(task="cross", n=60, seed=1):
    text, slots, langs = integration_corpus(task, n_lines=n, seed=seed)
    return CharCorpus(text), slots, langs


def _model(c, **kw):
    torch.manual_seed(0)
    return ModularRoutedLM(c.vocab_size, region_of_token=region_table(c.stoi), d_mod=8, d_ff=16, **kw)


def test_region_table_marks_consonants_only():
    c, _, _ = _corpus()
    tab = region_table(c.stoi)
    for ch, i in c.stoi.items():
        want = next((m for m, (cons, _) in enumerate(LANGS) if ch in cons), -1)
        assert tab[i] == want
    assert max(tab) == len(LANGS) - 1


def test_sticky_regions_match_corpus_annotation():
    c, _, langs = _corpus(n=40)
    tab = torch.as_tensor(region_table(c.stoi))
    x = c.train[None, :]
    reg = sticky_regions(x, tab)[0].numpy()
    ann = langs[:len(c.train)]
    assert (reg == ann).all()  # 区切り・母音は直前の子音の領域（コーパスの注釈と同じ）
    assert sticky_regions(torch.tensor([[c.stoi["|"], c.stoi["\n"]]]), tab).tolist() == [[0, 0]]


def test_no_direct_path_between_modules():
    """専門家を全部除くと、ある領域の位置の予測は他の領域の文字に依らない（直接の結合がない）。"""
    c, _, _ = _corpus()
    m = _model(c, n_experts=4)
    line_a = "pata|baba dada\n"
    line_b = "kete|baba dada\n"  # a1 だけを同じ言語の別の語に替える（領域の並びは同じ）
    xa = torch.tensor([[c.stoi[ch] for ch in line_a]])
    xb = torch.tensor([[c.stoi[ch] for ch in line_b]])
    assert (sticky_regions(xa, m.region_tab) == sticky_regions(xb, m.region_tab)).all()
    off = torch.ones(1, 4, dtype=torch.bool)
    j = slice(line_a.index("|") + 1, None)
    la, lb = m.run(xa, disabled=off)["logits"], m.run(xb, disabled=off)["logits"]
    assert torch.allclose(la[0, j], lb[0, j])
    la, lb = m.run(xa)["logits"], m.run(xb)["logits"]
    assert not torch.allclose(la[0, j], lb[0, j])  # 専門家があれば領域をまたいで届く


def test_cut_remote_isolates_single_expert_and_mask_matches_all():
    c, _, _ = _corpus()
    m1 = _model(c, n_experts=1)  # 経路選択の情報も使えない
    line_a, line_b = "pata|baba dada\n", "kete|baba dada\n"
    xa = torch.tensor([[c.stoi[ch] for ch in line_a]])
    xb = torch.tensor([[c.stoi[ch] for ch in line_b]])
    j = slice(line_a.index("|") + 1, None)
    la, lb = m1.run(xa, cut_remote=True)["logits"], m1.run(xb, cut_remote=True)["logits"]
    assert torch.allclose(la[0, j], lb[0, j])
    m = _model(c, n_experts=4)
    x = c.train[:64][None]
    full = m.run(x, cut_remote=True)["logits"]
    mask = m.run(x, cut_remote=torch.ones(4, dtype=torch.bool))["logits"]
    none = m.run(x, cut_remote=torch.zeros(4, dtype=torch.bool))["logits"]
    assert torch.allclose(full, mask) and torch.allclose(none, m.run(x)["logits"])


def test_run_outputs_and_homeostasis():
    c, _, _ = _corpus()
    m = _model(c, n_experts=4, kind="select", eta=0.1, r_lo=0.5, r_hi=2.5)
    x = c.train[:64].view(2, 32)
    r = m.run(x)
    assert r["logits"].shape == (2, 32, c.vocab_size)
    assert r["choice"].shape == (1, 2, 32) and r["gate"].shape == (1, 2, 32)
    assert r["router"].shape == (1, 2, 32, 4)
    m.homeostatic_step(r["choice"])
    assert torch.allclose(m.threshold.mean(), torch.zeros(()), atol=1e-6)


def test_run_one_with_post_records_cut_and_lesions():
    c, slots, langs = _corpus(n=300)
    args = argparse.Namespace(d=48, alpha=0.05, train_steps=3, seq_len=32, n_windows=8, k_num=1, k_den=4,
                              n_random=5, eps=1e-4)
    x_lang, _ = windows_like(split_like_corpus(langs), 32, 8)
    _, y_slot = windows_like(split_like_corpus(slots), 32, 8)
    z = run_one(0, 8, 1, {"kind": "select", "eta": 0.01}, c, args, model_cls=ModularRoutedLM,
                post=post(2, x_lang, y_slot), region_of_token=region_table(c.stoi), d_mod=8, r_lo=0.5, r_hi=2.5)
    for name in ("cut_all_d", "hub_cut_d", "nonhub_cut_d", "hub_lesion_d", "nonhub_lesion_d"):
        assert set(z[name]) == {"target", "other", "all"} and all(np.isfinite(list(z[name].values())))
    assert 0 <= z["hub_target_share"] <= 1


def test_conditions_use_split_eta_only_for_split_substrate():
    from neurocortex.experiments.run_moe_modular import conditions, split_eta
    args = argparse.Namespace(coefs=[0.05], bands=["point", "band"], eta=0.01, split_eta=None)
    assert split_eta(args) == 0.01
    args.split_eta = 0.1
    assert set(conditions(args, split_eta(args))) == {"aux0.05", "point0.1", "band0.1"}
    assert conditions(args, args.eta)["band0.01"] == {"kind": "select", "eta": 0.01, "band": "band"}
