"""言語の境界を与えない継続学習（12.6.245〜246節）のテスト。コーパスは取得せず、小さな合成の文字列で動かす。"""

import torch

from neurocortex.context_realtext import TransformerContextLM
from neurocortex.experiments import run_realtext_boundary_free as bf
from neurocortex.realtext_boundary_free import BoundaryConfig, ari, assign, batch_nll, segmentation, train_boundary_free
from test_realtext_forgetting import _data


def test_assign_rules():
    """直前の文脈の驚きが δ 以下なら留まり、超えたら驚きが δ 以下の別の文脈に戻り、どれも超えれば新しい文脈を作る。"""
    assert assign(torch.zeros(0), None, 0, 0.3, 5) == 0
    assert assign(torch.tensor([0.1, -1.0]), 0, 2, 0.3, 5) == 0  # 持続（より驚きの小さい文脈があっても）
    assert assign(torch.tensor([0.9, 0.2]), 0, 2, 0.3, 5) == 1  # 再活性化
    assert assign(torch.tensor([0.9, 0.5]), 0, 2, 0.3, 5) == 2  # 新しい文脈
    assert assign(torch.tensor([0.9, 0.5]), 0, 2, 0.3, 2) == 1  # 上限に達したら驚きが最小の文脈


def test_ari_and_segmentation():
    assert ari([0, 0, 1, 1], [5, 5, 7, 7]) == 1.0  # 名前の付け替えに依らない
    assert abs(ari([0, 1, 0, 1], [0, 0, 1, 1]) + 0.5) < 1e-9
    seg = segmentation([0] * 10 + [0] * 3 + [1] * 7 + [1] * 10, 10, 3)
    assert seg["delays"] == [3, 10] and seg["missed"] == 1 and seg["n_ctx"] == 2
    assert seg["runs"] == [[0, 0, 13], [1, 13, 17]] and seg["major"] == [0, 2]


def test_batch_nll_matches_seq_logp():
    torch.manual_seed(0)
    m = TransformerContextLM(8, 4, 8, 16, 1, 2, hidden=16).eval()
    seg = torch.randint(1, 8, (5, 9), generator=torch.Generator().manual_seed(1))
    nll = batch_nll(m, seg, 3)
    for c in range(3):
        lp = m.seq_logp(seg[:, :-1], torch.full((5,), c))
        ref = -lp.gather(-1, seg[:, 1:, None])[..., 0].mean()
        assert abs(float(nll[c]) - float(ref.detach())) < 1e-5


def _cfg(rule, boundary, delta=0.3):
    return BoundaryConfig(rule=rule, boundary=boundary, steps_per_lang=30, batch=8, replay_batch=4, seq_len=8,
                          buffer=20, d=16, layers=1, heads=2, hidden=16, eval_segments=10, max_ctx=6, delta=delta)


def _run(rule, boundary, delta=0.3):
    return train_boundary_free(_data(), 8, _cfg(rule, boundary, delta), [2, 0, 1], torch.Generator().manual_seed(5),
                               torch.Generator().manual_seed(6))


def test_given_boundary_creates_one_context_per_language():
    out = _run("replay", "given")
    assert out["seg"]["n_ctx"] == 3 and out["seg"]["ARI"] == 1.0 and out["seg"]["delays"] == [0, 0]
    assert len(out["R_ce"]) == 3 and set(out["metrics_nce"]) == {"ACC", "BWT", "F", "LA"}
    assert out["R_id"][0][0] == 1.0  # 文脈が1つなら同定は常に正しい


def test_infer_equals_given_when_assignment_matches():
    """推定の割り当てが境界を与えた割り当てと一致すれば、学習も評価もまったく同じになる（乱数の流れが同じ）。
    δ が大きいと境界を見逃して1つの文脈にまとまる。"""
    given = _run("naive", "given")
    big = _run("naive", "infer", delta=1e9)
    assert big["seg"]["n_ctx"] == 1 and big["seg"]["missed"] == 2
    small = _run("naive", "infer", delta=-1e9)  # 毎ステップ新しい文脈（上限まで）
    assert small["seg"]["n_ctx"] == 6
    for d in (0.05, 0.3, 1.0):
        inf = _run("naive", "infer", delta=d)
        if inf["seg"]["runs"] == given["seg"]["runs"]:
            assert inf["R_ce"] == given["R_ce"]


def _fake(cond, s, seed, acc, bwt, la, ari_=1.0, delays=(2, 3)):
    return {"cond": cond, "set": s, "seed": seed, "order": [0, 1, 2],
            "metrics_nce": {"ACC": acc, "BWT": bwt, "F": -bwt, "LA": la}, "R_id": [[1.0, 0.9]],
            "seg": {"n_ctx": 3, "ARI": ari_, "delays": list(delays), "missed": 0, "runs": [[0, 0, 1]]},
            "trace": [[None, 0], [0.0, 0], [1.0, 1], [0.1, 1], [2.0, 2], [0.05, 2]]}


def test_analyze_judgments():
    runs, shared = [], {"replay_sh": {}, "naive_sh": {}}
    for s in ("R", "G"):
        for seed in range(5):
            e = 0.01 * seed
            runs += [_fake("naive_tf", s, seed, -2.4 + e, -1.4 + e, -1.36), _fake("replay_tf", s, seed, -1.5 + e, -0.1, -1.38),
                     _fake("naive_bd", s, seed, -2.4 + e, -1.4 + e, -1.36), _fake("replay_bd", s, seed, -1.48 + e, -0.1, -1.38)]
            shared["replay_sh"][(s, seed)] = {"ACC": -1.6 - e, "BWT": -0.3}
            shared["naive_sh"][(s, seed)] = {"ACC": -2.5, "BWT": -1.5 - e}
    j = bf.analyze(runs, shared)["判定"]
    assert j["S1"]["合格"] and j["S2"]["合格"] and j["V"]["合格"] and j["H1"]["合格"] and j["H2"]["合格"]
    assert j["H3"]["合格"] and abs(j["H3"]["差"] + 0.02) < 1e-9 and j["H4"]["合格"]
    runs[1]["seg"]["delays"] = [500, 1500]
    runs[1]["seg"]["ARI"] = -0.5
    j = bf.analyze(runs, shared)["判定"]
    assert not j["S1"]["合格"] and j["S2"]["replay_tf"]["検出の割合"] == 0.9


def test_jobs():
    assert len(bf.jobs_for("dev", ("R", "G"))) == 2 * len(bf.CONDS) * 5
    assert {j[2] for j in bf.jobs_for("eval", ("R",))} == set(range(5))
    assert {j[2] for j in bf.jobs_for("pre", ("R",), seeds=[999])} == {999}
