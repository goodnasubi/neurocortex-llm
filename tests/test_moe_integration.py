"""領域をまたぐ統合が要る課題（12.6.199節）のテスト。"""

from __future__ import annotations

import random

import numpy as np
import torch

from neurocortex.data import CharCorpus
from neurocortex.hub_load import make_windows
from neurocortex.moe_homeo_band import BandHomeostaticLM
from neurocortex.moe_integration import (LANGS, SLOT_A1, SLOT_B1, SLOT_B2, SLOT_SEP, hub_participation,
                                         hub_target_share, integration_corpus, make_lexicons, participation,
                                         slot_ce, split_like_corpus, windows_like)


def _lines(task, seed=3, n=200):
    text, slots, langs = integration_corpus(task, n_lines=n, seed=seed)
    return text, slots, langs


def _lang_of(word):
    return next(i for i, (cons, _) in enumerate(LANGS) if word[0] in cons)


def test_lexicons_are_cognates_with_disjoint_consonants():
    lex = make_lexicons(8, random.Random(0))
    assert len(lex) == len(LANGS) and all(len(set(w)) == 8 for w in lex)
    for c in range(8):
        shapes = {tuple((LANGS[l][0] + LANGS[l][1]).index(ch) for ch in lex[l][c]) for l in range(len(LANGS))}
        assert len(shapes) == 1  # 同じ概念は全言語で子音・母音の番号が同じ
    for a in range(len(LANGS)):
        for b in range(a + 1, len(LANGS)):
            assert not set(LANGS[a][0]) & set(LANGS[b][0])


def test_cross_task_switches_language_and_within_keeps_it():
    for task in ("cross", "within"):
        text, _, _ = _lines(task)
        for line in text.strip().split("\n"):
            a1, rest = line.split("|")
            b1, b2 = rest.split(" ")
            assert _lang_of(b1) == _lang_of(b2)
            assert (_lang_of(a1) != _lang_of(b1)) == (task == "cross")


def test_b2_concept_is_a_fixed_permutation_of_a1_shared_across_tasks():
    seed = 5
    lex = make_lexicons(8, random.Random(seed))
    maps = {}
    for task in ("cross", "within"):
        text, _, _ = _lines(task, seed=seed)
        lines = text.strip().split("\n")
        for line in lines:
            a1, rest = line.split("|")
            b2 = rest.split(" ")[1]
            c1 = lex[_lang_of(a1)].index(a1)
            c2 = lex[_lang_of(b2)].index(b2)
            assert maps.setdefault(c1, c2) == c2  # 同じシードなら課題によらず同じ置換
        maps_a1 = [lex[_lang_of(l.split("|")[0])].index(l.split("|")[0]) for l in lines]
        maps.setdefault("a1_" + task, maps_a1)
    assert maps["a1_cross"] == maps["a1_within"]  # 行の乱数（a1 の概念）は課題で同じ
    assert sorted(v for k, v in maps.items() if isinstance(k, int)) == sorted(set(
        v for k, v in maps.items() if isinstance(k, int)))


def test_annotations_align_with_text():
    text, slots, langs = _lines("cross", n=50)
    assert len(text) == len(slots) == len(langs)
    for ch, s in zip(text, slots):
        assert (s == SLOT_SEP) == (ch in " |\n")
    line = text.split("\n")[0]
    n_a1 = line.index("|")
    assert set(slots[:n_a1]) == {SLOT_A1}
    assert slots[n_a1 + 1] == SLOT_B1 and slots[len(line) - 1] == SLOT_B2
    assert langs[0] != langs[len(line) - 1]


def test_windows_like_matches_make_windows():
    text, slots, _ = _lines("within", n=300)
    corpus = CharCorpus(text)
    x, y = make_windows(corpus.val, 16, 10)
    sv = split_like_corpus(slots)
    xs, ys = windows_like(sv, 16, 10)
    assert xs.shape == x.shape == ys.shape
    sep = [corpus.stoi[c] for c in " |\n"]
    assert torch.equal(torch.isin(y, torch.tensor(sep)), ys == SLOT_SEP)
    assert torch.equal(torch.isin(x, torch.tensor(sep)), xs == SLOT_SEP)


def test_slot_ce_partitions_all_tokens():
    text, slots, _ = _lines("cross", n=300)
    corpus = CharCorpus(text)
    x, y = make_windows(corpus.val, 16, 8)
    _, ys = windows_like(split_like_corpus(slots), 16, 8)
    torch.manual_seed(0)
    m = BandHomeostaticLM(corpus.vocab_size, d=8, n_experts=4, d_ff=8)
    m.eval()
    ce = slot_ce(m, x, y, ys)
    n_t = int((ys == SLOT_B2).sum())
    n = ys.numel()
    assert abs(ce["all"] - (ce["target"] * n_t + ce["other"] * (n - n_t)) / n) < 1e-5
    dis = torch.zeros(1, 4, dtype=torch.bool)
    dis[0, :] = True
    assert slot_ce(m, x, y, ys, disabled=dis)["all"] != ce["all"]


def test_participation_extremes():
    choice = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    label = np.array([0, 1, 2, 3, 2, 2, 2, 2])
    p = participation(choice, label, 3, 4)
    assert abs(p[0] - 1.0) < 1e-12 and abs(p[1]) < 1e-12 and np.isnan(p[2])
    load = np.array([4.0, 4.0, 0.0])
    hub, rest = hub_participation(load, p, 1)
    assert hub == 1.0 and rest == 0.0


def test_hub_target_share():
    choice = np.array([0, 0, 1, 2])
    y_slot = np.array([SLOT_B2, SLOT_B1, SLOT_B2, SLOT_SEP])
    t, a = hub_target_share(choice, y_slot, np.array([2.0, 1.0, 1.0]), 1)
    assert t == 0.5 and a == 0.5
