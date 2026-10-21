"""境界の推定が難しい設定（戻ってくる言語・1断片ずつの割り当て、12.6.249〜250節）のテスト。コーパスは取得せず、
小さな合成の文字列で動かす。"""

import torch

from neurocortex.context_realtext import TransformerContextLM
from neurocortex.experiments import run_realtext_hard_boundary as hb
from neurocortex.realtext_boundary_free import batch_nll
from neurocortex.realtext_hard_boundary import (HardConfig, assignment_stats, frag_nll, stream_blocks,
                                                stream_metrics, train_hard)
from test_realtext_forgetting import _data


def test_stream_blocks_returning_order():
    for seed in range(20):
        b = stream_blocks(4, [2, 0, 3, 1], 3, torch.Generator().manual_seed(seed))
        assert b[:4] == [2, 0, 3, 1] and len(b) == 12
        assert all(sorted(b[4 * k:4 * k + 4]) == [0, 1, 2, 3] for k in range(3))
        assert all(x != y for x, y in zip(b, b[1:]))  # 同じ言語が続かない


def test_frag_nll_mean_matches_batch_nll():
    torch.manual_seed(0)
    m = TransformerContextLM(8, 4, 8, 16, 1, 2, hidden=16).eval()
    seg = torch.randint(1, 8, (5, 9), generator=torch.Generator().manual_seed(1))
    assert torch.allclose(frag_nll(m, seg, 3).mean(0), batch_nll(m, seg, 3), atol=1e-6)


def test_assignment_stats():
    # 言語 0,1 が区間 [0, 1, 0]、1区間 2 ステップ × バッチ 2 断片
    blocks = [0, 1, 0]
    lang = [0] * 4 + [1] * 4 + [0] * 4
    block = [0] * 4 + [1] * 4 + [2] * 4
    st = assignment_stats([0] * 4 + [1] * 4 + [0] * 4, lang, block, blocks, 2, 2)
    assert st["ARI"] == 1.0 and st["react"] == 1.0 and st["split"] == 0.0 and st["merged"] == 0
    assert st["delays"] == [0, 0]
    # 戻ってきた言語に新しい文脈を作った（分けすぎ）。最初のステップは古い文脈のまま（遅れ 1）
    st = assignment_stats([0] * 4 + [1] * 4 + [1, 1, 2, 2], lang, block, blocks, 2, 2)
    assert st["react"] == 0.0 and st["split"] == 0.5 and st["n_ctx"] == 3 and st["extra"] == 1
    # 全部を1つにまとめた（統合しすぎ）
    st = assignment_stats([0] * 12, lang, block, blocks, 2, 2)
    assert st["merged"] == 1 and st["ARI"] == 0.0


def test_stream_metrics():
    # 言語 p=0,1 の区間 [0, 1, 0]。rows は CE（小さいほど良い）
    rows = [[1.0, 3.0], [1.5, 1.2], [1.1, 1.4]]
    m = stream_metrics(rows, [0, 1, 0])
    assert abs(m["ACC"] + 1.25) < 1e-9
    assert abs(m["FGT"] - ((-1.5 + 1.0) + (-1.4 + 1.2)) / 2) < 1e-9  # 区間 1 の p=0、区間 2 の p=1
    assert abs(m["LA1"] + (1.0 + 1.2) / 2) < 1e-9
    assert abs(m["AVG"] + (1.0 + 1.35 + 1.25) / 3) < 1e-9


def _cfg(rule, asg, delta=0.3):
    return HardConfig(rule=rule, assign=asg, steps_per_lang=12, visits=3, batch=6, replay_batch=4, seq_len=8,
                      buffer=20, d=16, layers=1, heads=2, hidden=16, eval_segments=10, max_ctx=10, delta=delta)


def _run(rule, asg, delta=0.3):
    return train_hard(_data(), 8, _cfg(rule, asg, delta), [2, 0, 1], torch.Generator().manual_seed(5),
                      torch.Generator().manual_seed(6))


def test_given_assignments():
    out = _run("replay", "id")
    assert out["seg"]["n_ctx"] == 3 and out["seg"]["ARI"] == 1.0 and out["seg"]["react"] == 1.0
    assert len(out["R_ce"]) == 9 and set(out["metrics"]) == {"ACC", "FGT", "AVG", "LA1"}
    blk = _run("replay", "blk")
    assert blk["seg"]["n_ctx"] == 9 and blk["seg"]["react"] == 0.0 and blk["seg"]["split"] == 1.0
    assert blk["blocks"] == out["blocks"]  # 流れは割り当てに依らない


def test_inferred_assignments_extremes():
    """δ が大きいと全部が1つの文脈にまとまり、小さいと上限まで文脈を作る。frag と batch のどちらも動く。"""
    for asg in ("frag", "batch"):
        big = _run("naive", asg, delta=1e9)
        assert big["seg"]["n_ctx"] == 1 and big["seg"]["merged"] == 2
        small = _run("replay", asg, delta=-1e9)
        assert small["seg"]["n_ctx"] == 10
        assert len(small["R_ce"]) == 9 and all(len(r) == 3 for r in small["R_ce"])
    frag = _run("naive", "frag", delta=1e9)
    assert len(frag["surprise"]["switch"]) == 8  # 区間の最初の断片ごと（最初の区間を除く）


def _fake(cond, s, seed, acc, fgt, ari_=1.0, react=1.0):
    return {"cond": cond, "set": s, "seed": seed, "delta": 0.3,
            "metrics": {"ACC": acc, "FGT": fgt, "AVG": acc, "LA1": acc}, "R_id": [[1.0, 0.9]],
            "seg": {"n_ctx": 3, "ARI": ari_, "merged": 0, "react": react, "split": 0.0, "delays": [0, 1]},
            "surprise": {"within": [0.1, 0.2, 0.5], "switch": [1.2, 1.5]}}


def test_analyze_judgments():
    runs = []
    for s in ("R", "G"):
        for seed in range(5):
            e = 0.01 * seed
            runs += [_fake("naive_id", s, seed, -2.0, -0.5), _fake("naive_frag", s, seed, -2.1 + e, -0.6),
                     _fake("replay_id", s, seed, -1.5, -0.05), _fake("replay_frag", s, seed, -1.52 + e, -0.06),
                     _fake("replay_batch", s, seed, -1.5, -0.05), _fake("replay_blk", s, seed, -1.6 - e, -0.1)]
    j = hb.analyze(runs)["判定"]
    assert j["V"]["合格"] and j["A1"]["合格"] and j["A2"]["合格"] and j["H1"]["合格"] and j["H2"]["合格"]
    assert j["H3"]["合格"] and abs(j["参考"]["replay_frag: 驚き"]["言語の中で δ を超える割合"] - 1 / 3) < 1e-9
    for r in runs:
        if r["cond"] == "replay_frag":
            r["seg"]["ARI"], r["seg"]["react"] = 0.5, 0.4
            r["metrics"]["ACC"] -= 0.2
    j = hb.analyze(runs)["判定"]
    assert not j["A1"]["合格"] and not j["A2"]["合格"] and not j["H2"]["合格"]


def test_jobs():
    assert len(hb.jobs_for("dev", ("R", "G"))) == 2 * len(hb.CONDS) * 5 and len(hb.CONDS) == 8
    assert {j[2] for j in hb.jobs_for("eval", ("R",))} == set(range(5))
