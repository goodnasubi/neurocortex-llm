"""12.6.251〜252節: 読み出しの出力に効く次元 K の操作。"""

import torch

from neurocortex.experiments.run_readout_dim import KS, WIDTHS, summary, verdict
from neurocortex.readout_dim import RULES, draw_spec, drift_stats, run, xor_labels


def test_draw_spec_pairs_disjoint_and_nested():
    spec = draw_spec(10, 8, torch.Generator().manual_seed(0))
    pairs = [*spec.old_pairs, spec.new_pair]
    assert len(set(pairs)) == 9 and all(a < b for a, b in pairs)


def test_xor_labels():
    bits = torch.tensor([[1.0, 1.0, -1.0], [1.0, -1.0, -1.0]])
    y = xor_labels(bits, ((0, 1), (1, 2)))
    assert y.tolist() == [[0.0, 1.0], [1.0, 0.0]]


def test_drift_stats_isotropic_fraction_and_potent_only():
    g = torch.Generator().manual_seed(0)
    w, k, n = 64, 4, 4000
    h0 = torch.randn(n, w, generator=g)
    w0 = torch.randn(k, w, generator=g)
    head = (w0, torch.zeros(k))
    y = (h0 @ w0.T > 0).float()
    st = drift_stats(h0, h0 + 0.1 * torch.randn(n, w, generator=g), head, head, y)
    assert abs(st["f_pot"] - k / w) < 0.01 and abs(st["e_iso"] - 1) < 0.15 and st["comp"] == 0
    # 出力に効く部分空間の中だけの動きは f_pot = 1、零空間だけなら 0
    q, _ = torch.linalg.qr(w0.T)
    dp = torch.randn(n, k, generator=g) @ q.T
    assert abs(drift_stats(h0, h0 + dp, head, head, y)["f_pot"] - 1) < 1e-4
    dn = torch.randn(n, w, generator=g)
    dn = dn - dn @ q @ q.T
    st = drift_stats(h0, h0 + dn, head, head, y)
    assert st["f_pot"] < 1e-6 and abs(st["pot_rel"]) < 1e-3


def test_drift_stats_readout_compensation():
    g = torch.Generator().manual_seed(1)
    h0 = torch.randn(500, 16, generator=g)
    w0 = torch.randn(2, 16, generator=g)
    dh = torch.randn(500, 16, generator=g)
    y = (h0 @ w0.T > 0).float()
    st = drift_stats(h0, h0 + dh, (w0, torch.zeros(2)), (w0, torch.zeros(2)), y)
    assert st["harm_ro"] == 0 and st["acc0"] == 1


def test_run_deterministic_and_k_shares_inputs():
    spec = draw_spec(10, 8, torch.Generator().manual_seed(3))
    a = run(spec, 2, 32, "replay", 7, pretrain_steps=50, steps=20)
    b = run(spec, 2, 32, "replay", 7, pretrain_steps=50, steps=20)
    assert a == b
    for r in RULES:
        x = run(spec, 4, 32, r, 7, pretrain_steps=50, steps=20)
        assert 0 <= x["f_pot"] <= 1 and 0 <= x["pre"] <= 1
    # naive では古い読み出しは変わらないので読み出しの寄与は 0
    assert run(spec, 1, 32, "naive", 7, pretrain_steps=50, steps=20)["harm_ro"] == 0


def test_verdict_on_synthetic_summary():
    rows = []
    for r in RULES:
        for w in WIDTHS:
            for k in KS:
                f = k / w
                rows.append({"rule": r, "W": w, "K": k, "seed": 0, "draw": 0, "d_tot": 1.0, "f_pot": f,
                             "e_iso": w * f / k, "f_pot1": f, "pot_rel": 0.5 - 0.001 * w, "harm_rep": -0.1 * k,
                             "harm_ro": 0.0, "comp": 0.0, "acc0": 1.0, "acc1": 1.0, "pre": 1.0, "end": 1.0,
                             "old_change": -0.01 * k, "pre_b0": 1.0, "change_b0": 0.0, "new": 1.0})
    v = verdict(summary(rows), rows)
    assert v["L"] and v["P"] and v["Q"] and v["W"] and v["R"] and v["F"]
