import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from step44_optimized_validation import BrainInspiredLLMPhase, OptimizedValidationConfig


def _model(phase, **kw):
    torch.manual_seed(0)
    cfg = OptimizedValidationConfig(phase=phase, vocab_size=20, **kw)
    return BrainInspiredLLMPhase(cfg).eval()


def test_backbone_is_causal():
    m = _model("A")
    x = torch.randint(0, 20, (1, 10))
    x2 = x.clone()
    x2[0, 7:] = (x2[0, 7:] + 1) % 20
    with torch.no_grad():
        h1 = m.backbone[1](m.backbone[0](x), src_mask=torch.nn.Transformer.generate_square_subsequent_mask(10), is_causal=True)
        h2 = m.backbone[1](m.backbone[0](x2), src_mask=torch.nn.Transformer.generate_square_subsequent_mask(10), is_causal=True)
    assert torch.allclose(h1[0, :7], h2[0, :7], atol=1e-5)


def test_hippocampus_ce_independent_of_weights_and_phase():
    x = torch.randint(0, 20, (2, 12))
    ces = []
    for phase, w in [("A", 0.03), ("C", 1.0)]:
        m = _model(phase, hippocampus_loss_weight=w)
        with torch.no_grad():
            ces.append(m(x, x)["hippocampus_ce"].item())
    assert abs(ces[0] - ces[1]) < 1e-5


def test_td_critic_loss_is_bounded_below():
    from step44_optimized_validation import BasalGangliaHead
    x = torch.randint(0, 20, (2, 12))
    for mode in ("td", "td_rpe", "td_sg", "td_rpe_sg", "none"):
        torch.manual_seed(0)
        head = BasalGangliaHead(16, 20, mode=mode)
        with torch.no_grad():
            head.critic[-1].bias.fill_(-1e3)
        out = head(torch.randn(2, 12, 16), x)
        assert out["loss"].item() >= 0


def test_pos_encoding_off_matches_legacy_backbone():
    m = _model("C")
    assert not hasattr(m, "pos_embedding")
    x = torch.randint(0, 20, (2, 10))
    with torch.no_grad():
        legacy = m.backbone[1](m.backbone[0](x), src_mask=torch.nn.Transformer.generate_square_subsequent_mask(10), is_causal=True)
        assert torch.allclose(m.encode(x), legacy)
    # 位置埋め込みありでも、既存モジュールの初期値は変わらない
    m_pe = _model("C", pos_encoding=True)
    for k, v in m.state_dict().items():
        assert torch.equal(v, m_pe.state_dict()[k]), k


def test_pos_encoding_is_causal_and_position_aware():
    m = _model("A", pos_encoding=True)
    x = torch.randint(0, 20, (1, 10))
    x2 = x.clone()
    x2[0, 7:] = (x2[0, 7:] + 1) % 20
    same = torch.full((1, 10), 3)
    with torch.no_grad():
        h1, h2, hs = m.encode(x), m.encode(x2), m.encode(same)
    assert torch.allclose(h1[0, :7], h2[0, :7], atol=1e-5)
    # 同じトークン列でも位置ごとに表現が異なる
    assert not torch.allclose(hs[0, 5], hs[0, 6], atol=1e-3)
