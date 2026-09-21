"""課題生成と言語モデル部分の健全性テスト。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.attention import linear_attention_causal  # noqa: E402
from neurocortex.data import CharCorpus, synthetic_corpus  # noqa: E402
from neurocortex.lm import DenseBaseline, SpikingLM, count_params  # noqa: E402
from neurocortex.tasks import (  # noqa: E402
    OrderSpec,
    RecallSpec,
    make_order_batch,
    make_recall_batch,
)


@pytest.fixture
def gen() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(0)
    return g


def test_recall_cue_and_filler_alphabets_are_disjoint(gen: torch.Generator) -> None:
    """cue記号とfiller記号は互いに素な集合から引かれること（12.6.1節）。"""
    spec = RecallSpec(max_delay=16)
    x, y = make_recall_batch(spec, 256, gen, delay=16)
    cue_pos = x.shape[1] - 1 - 16
    assert torch.equal(x[:, cue_pos], y), "cue位置のトークンが正解ラベルと一致すること"
    fillers = x[:, cue_pos + 1 : -1]
    assert (fillers >= spec.n_cue).all(), "fillerがcue記号集合と重なっている"
    assert (fillers < spec.n_cue + spec.n_filler).all()


def test_recall_zero_delay_puts_cue_at_the_answer_position(gen: torch.Generator) -> None:
    """N=0 は解答位置そのものが cue（統制1が成立する構成であること）。"""
    spec = RecallSpec(max_delay=16)
    x, y = make_recall_batch(spec, 64, gen, delay=0)
    assert torch.equal(x[:, -1], y)


def test_recall_delay_is_randomised_across_the_batch(gen: torch.Generator) -> None:
    """delay を指定しなければバッチ内で遅延が変わる（固定タイミングの学習を防ぐ）。"""
    spec = RecallSpec(max_delay=16)
    x, _ = make_recall_batch(spec, 256, gen)
    cue_positions = (x < spec.n_cue).float().argmax(dim=1)
    assert cue_positions.unique().numel() > 5


def test_recall_label_is_uninferable_from_fillers(gen: torch.Generator) -> None:
    """ラベルとfillerが独立であること（抜け道の排除）。"""
    spec = RecallSpec(max_delay=16)
    x, y = make_recall_batch(spec, 4096, gen, delay=16)
    fillers = x[:, -16:-1]
    # ラベルごとのfiller分布がほぼ一様なら、fillerからラベルは推定できない
    for label in range(spec.n_cue):
        subset = fillers[y == label]
        if subset.numel() == 0:
            continue
        counts = torch.bincount(subset.flatten() - spec.n_cue, minlength=spec.n_filler).float()
        assert counts.std() / counts.mean() < 0.15


def test_order_task_labels_match_the_symbol_order(gen: torch.Generator) -> None:
    spec = OrderSpec(seq_len=16)
    x, y = make_order_batch(spec, 128, gen)
    for b in range(x.shape[0]):
        pa = int((x[b] == spec.a_id).nonzero())
        pb = int((x[b] == spec.b_id).nonzero())
        assert int(y[b]) == int(pa > pb)


def test_linear_attention_is_causal() -> None:
    """線形アテンションが未来を参照しないこと。"""
    torch.manual_seed(0)
    q, k, v = (torch.rand(1, 2, 8, 4) + 0.1 for _ in range(3))
    base = linear_attention_causal(q, k, v)
    v2 = v.clone()
    v2[:, :, 5:, :] += 10.0  # 位置5以降だけを変える
    got = linear_attention_causal(q, k, v2)
    assert torch.allclose(base[:, :, :5], got[:, :, :5], atol=1e-6)
    assert not torch.allclose(base[:, :, 5:], got[:, :, 5:])


def test_single_layer_linear_attention_state_is_order_invariant() -> None:
    """テストBの根拠: 単層の線形アテンション状態は前文脈の順序に不変であること。"""
    torch.manual_seed(0)
    q, k, v = (torch.rand(1, 1, 8, 4) + 0.1 for _ in range(3))
    perm = torch.randperm(7)  # 最終位置以外を並べ替える
    kp = torch.cat([k[:, :, :7][:, :, perm], k[:, :, 7:]], dim=2)
    vp = torch.cat([v[:, :, :7][:, :, perm], v[:, :, 7:]], dim=2)
    assert torch.allclose(
        linear_attention_causal(q, k, v)[:, :, -1],
        linear_attention_causal(q, kp, vp)[:, :, -1],
        atol=1e-5,
    )


@pytest.mark.parametrize("cls", [SpikingLM, DenseBaseline])
def test_lm_is_causal(cls) -> None:
    """言語モデルが未来のトークンを参照しないこと。"""
    torch.manual_seed(0)
    model = cls(vocab_size=20, d_model=32, n_layers=2, n_heads=2, max_len=12).eval()
    x = torch.randint(0, 20, (1, 12))
    x2 = x.clone()
    x2[0, 7:] = (x2[0, 7:] + 1) % 20
    with torch.no_grad():
        a, b = model(x), model(x2)
    assert torch.allclose(a[:, :7], b[:, :7], atol=1e-5)


def test_lm_parameter_counts_match_between_groups() -> None:
    """スパイキング版と密なベースラインのパラメータ数が一致すること（公平な比較）。"""
    kw = dict(vocab_size=28, d_model=192, n_layers=3, n_heads=4, max_len=64)
    assert count_params(SpikingLM(**kw)) == count_params(DenseBaseline(**kw))


def test_lm_size_is_within_the_step0_budget() -> None:
    """12.11節が定めるステップ0の規模（1〜3Mパラメータ）に収まること。"""
    n = count_params(SpikingLM(vocab_size=28, d_model=192, n_layers=3, n_heads=4, max_len=64))
    assert 1_000_000 <= n <= 3_000_000, n


def test_synthetic_corpus_is_reproducible() -> None:
    """合成コーパスが外部ダウンロードなしで再現可能であること。"""
    assert synthetic_corpus(100, seed=3) == synthetic_corpus(100, seed=3)
    assert synthetic_corpus(100, seed=3) != synthetic_corpus(100, seed=4)
    c = CharCorpus(synthetic_corpus(200, seed=0))
    assert c.vocab_size > 10 and c.train.numel() > 0 and c.val.numel() > 0
