import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import step44_continual_forgetting as cf
import step44_continual_replay as cr
import step44_replay_factorial_analysis as fa
from test_step44_replay_ablation import _args, _data


def test_factorial_cells_share_storage_and_differ_in_replay():
    d = _data()
    a = cf.run_continual("A", 3, d, _args(), log=lambda s: None)
    lo = cr.run_continual_replay(3, d, _args(memory_per_task=3, replay_size=1, label="Mlo_r1"), log=lambda s: None)
    hi = cr.run_continual_replay(3, d, _args(memory_per_task=3, replay_size=4, label="Mlo_r4"), log=lambda s: None)
    big = cr.run_continual_replay(3, d, _args(memory_per_task=6, replay_size=1, label="Mhi_r1"), log=lambda s: None)
    # 同じ M なら保存断片は同一（再生数だけが違う）。M を増やすと保存断片は小さい M の保存を先頭に含む。
    assert lo["stored_chunk_idx"] == hi["stored_chunk_idx"]
    assert all(b[:3] == s for b, s in zip(big["stored_chunk_idx"], lo["stored_chunk_idx"]))
    assert lo["n_replayed_per_task"][1:] != hi["n_replayed_per_task"][1:]
    assert lo["n_replayed_per_task"] == big["n_replayed_per_task"]
    for r in (lo, hi, big):
        assert r["ce"][0] == a["ce"][0]  # 最初のタスクは A と同一
    assert lo["ce"][1] != hi["ce"][1] and lo["ce"][1] != big["ce"][1]


def _write(dirp, p, s, f, la, ex_last, cfg):
    m = {k: 1.0 for k in fa.ca.METRICS}
    m.update(F_ce=f, LA_ce=la)
    (dirp / f"phase{p}_seed{s}.json").write_text(json.dumps({
        "order": ["a", "b"], "n_params": 1, "train_seconds": 1.0, "metrics": m, "ce": [[5.0, 9.0], [ex_last, 5.0]],
        "config": cfg, "env": {"torch": "x"}}))


def _setup(tmp_path, m8r16_ex=5.6):
    dirs = [tmp_path / x for x in ("n", "s", "r", "b")]
    for x in dirs:
        x.mkdir()
    nd, sd, rd, bd = dirs
    c = lambda M, r: {"lr": 5e-4, "memory_per_task": M, "replay_size": r}
    for s in range(3):
        j = 0.01 * s
        _write(bd, "V", s, 3.8 + j, 5.0, 9.0, c(64, 16))
        _write(bd, "A", s, 3.7 + j, 5.0, 9.0, c(64, 16))
        _write(sd, "Rsmall", s, 0.5 + j, 5.0, 5.7 + j, c(8, 2))
        _write(nd, "M8r16", s, 0.3 - j, 5.0, m8r16_ex + j, c(8, 16))
        _write(nd, "M64r2", s, 0.2 + 2 * j, 5.0, 5.2 - j, c(64, 2))
        _write(rd, "R", s, 0.08 + j / 2, 5.0, 5.2 + j, c(64, 16))
    return dirs


def test_analysis_verdicts(tmp_path):
    dirs = _setup(tmp_path)
    out = fa.analyze(*dirs, [0, 1, 2], uni={"a": 5.5, "b": 5.0})
    assert out["reuse_issues"] == [] and set(out["first_row_ce_maxdiff_vs_A"].values()) == {0.0}
    assert out["H1_verdict"] == "支持" and out["H2_verdict"] == "支持"
    cr_ = out["context_retention"]
    assert cr_["R"]["retains_context"] and cr_["M64r2"]["retains_context"]
    assert not cr_["Rsmall"]["retains_context"] and not cr_["M8r16"]["retains_context"]
    assert out["H3_verdict"] == "支持"
    assert abs(out["interaction_F_ce"]["mean"] - ((0.08 - 0.3) - (0.2 - 0.5) + 0.005)) < 1e-9  # seed 項の平均 0.5×0.01
    assert "H3" in fa.to_markdown(out)
    # M8r16 が文脈を保持すると H3 不支持
    (tmp_path / "x").mkdir()
    dirs2 = _setup(tmp_path / "x", m8r16_ex=5.2)
    assert fa.analyze(*dirs2, [0, 1, 2], uni={"a": 5.5, "b": 5.0})["H3_verdict"] == "不支持"


def test_main_effect_and_reuse_check(tmp_path):
    v = lambda p, u: {"verdict_paired": p, "verdict_unpaired": u}
    assert fa.main_effect([v("改善", "改善"), v("改善", "差なし")]) == "部分支持"
    assert fa.main_effect([v("差なし", "差なし"), v("悪化", "悪化")]) == "不支持"
    nd, sd, rd, bd = _setup(tmp_path)
    _write(nd, "M8r16", 1, 0.3, 5.0, 5.6, {"lr": 1e-3, "memory_per_task": 8, "replay_size": 2})
    iss = fa.analyze(nd, sd, rd, bd, [0, 1, 2])["reuse_issues"]
    assert any("config.lr" in i for i in iss) and any("(M, r)" in i for i in iss)
