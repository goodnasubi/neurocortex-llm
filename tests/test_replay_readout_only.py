"""リプレイの勾配の行き先を絞る対照（12.6.255〜256節）のテスト。コーパスは取得せず、小さな合成の文字列で動かす。"""

import pytest
import torch

from neurocortex.experiments import run_replay_readout_only as ro
from neurocortex.realtext_forgetting import split_replay_logp, train_forgetting
from test_realtext_forgetting import _cfg, _data
from test_realtext_head_probe import _fake, _model


def test_split_logp_matches_seq_logp():
    """detach の有無で値は seq_logp と同じ。"""
    m = _model()
    x = torch.randint(0, 8, (5, 8))
    c = torch.tensor([0, 1, 2, 1, 0])
    ref = m.seq_logp(x, c)
    for dt in (False, True):
        for dr in (False, True):
            assert torch.allclose(split_replay_logp(m, x, c, dt, dr), ref, atol=1e-6)


@pytest.mark.parametrize("rule,trunk_grad,readout_grad", [("replay_ro", False, True), ("replay_body", True, False)])
def test_gradient_routing(rule, trunk_grad, readout_grad):
    """replay_ro は胴体に、replay_body は読み出しに勾配を流さない。"""
    from neurocortex.realtext_forgetting import REPLAY_SPLIT
    m = _model()
    x = torch.randint(0, 8, (4, 9))
    c = torch.tensor([1, 1, 2, 2])
    lp = split_replay_logp(m, x[:, :-1], c, *REPLAY_SPLIT[rule])
    torch.nn.functional.nll_loss(lp.reshape(-1, lp.shape[-1]), x[:, 1:].reshape(-1)).backward()
    tg = m.mid.weight.grad
    assert (tg is not None and tg.abs().sum() > 0) == trunk_grad
    hg = m.head_w.grad
    assert (hg is not None and hg[1:].abs().sum() > 0) == readout_grad
    assert hg is None or hg[0].abs().sum() == 0  # リプレイにない文脈の読み出しは変わらない


def test_split_rules_differ_from_replay_and_naive():
    """replay_ro・replay_body は学習を終え、replay・naive とも違う R 行列になる。replay は従来どおり。"""
    runs = {r: train_forgetting(_data(), 8, _cfg(r), [2, 0, 1], torch.Generator().manual_seed(5),
                                torch.Generator().manual_seed(6))["R_ce"] for r in ("naive", "replay", "replay_ro", "replay_body")}
    assert runs["replay_ro"][0] == runs["naive"][0] == runs["replay"][0]  # 最初の言語ではリプレイなし
    for a in ("replay_ro", "replay_body"):
        for b in ("naive", "replay"):
            assert runs[a][-1] != runs[b][-1]


def test_analyze_judgments():
    runs = []
    for s in ("R", "G"):
        for k in range(5):
            for cond, f, m, i in (("naive", 1.5, 1.0 + 0.01 * k, 0.2 + 0.01 * k), ("replay", 0.2, 0.05, 0.8),
                                  ("replay_ro", 0.5, 0.1 + 0.001 * k, 0.21), ("replay_body", 0.4, 0.3, 0.6 + 0.01 * k)):
                r = _fake(cond, s, k, f, m, i)
                r["metrics_nce"] = {"ACC": -1.0, "LA": -1.0 - 0.01 * k}
                runs.append(r)
    j = ro.analyze(runs)["判定"]
    assert j["H1"]["合格"] and j["H1"]["ro_M/naive_M"] < 0.2
    assert j["H2"]["合格"] and j["H3"]["合格"] and j["H4"]["合格"]
    assert abs(j["参考"]["replay_ro: F の減りの replay に対する割合"] - 1.0 / 1.3) < 1e-9


def test_jobs():
    assert len(ro.jobs_for("dev", ("R", "G"))) == 2 * 4 * 5
