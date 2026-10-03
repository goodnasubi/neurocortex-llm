import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from step44_full_scale_analysis import analyze, main


def _write(root, phase, seed, final):
    # エポック1〜3は適当な値、最終エポックが評価対象
    (root / f"phase{phase}_seed{seed}.json").write_text(json.dumps(
        {"phase": phase, "seed": seed, "val_ppls": [final + 30, final + 10, final + 3, final],
         "val_losses": [], "train_losses": [], "val_head_ce": []}))


def test_clear_difference_is_detected(tmp_path):
    for s, (a, b) in enumerate([(34.4, 35.4), (34.8, 35.9), (34.1, 35.0)]):
        _write(tmp_path, "A", s, a)
        _write(tmp_path, "B", s, b)
    c = analyze(tmp_path)["comparisons"]["A->B_val"]
    assert c["verdict_paired"] == "悪化" and c["verdict_unpaired"] == "悪化"
    assert c["n_paired"] == 3 and c["mean_diff"] > 0


def test_no_difference(tmp_path):
    for s, (a, b) in enumerate([(34.4, 34.6), (34.8, 34.5), (34.1, 34.2)]):
        _write(tmp_path, "A", s, a)
        _write(tmp_path, "B", s, b)
    c = analyze(tmp_path)["comparisons"]["A->B_val"]
    assert c["verdict_paired"] == "差なし" and c["verdict_unpaired"] == "差なし"


def test_missing_seed_uses_common_seeds_for_paired(tmp_path):
    for s in range(3):
        _write(tmp_path, "A", s, 34.0 + 0.3 * s)
    for s in (0, 2):
        _write(tmp_path, "B", s, 33.0 + 0.3 * s)
    out = analyze(tmp_path)
    c = out["comparisons"]["A->B_val"]
    assert c["paired_seeds"] == [0, 2] and c["n_paired"] == 2
    assert c["n_a"] == 3 and c["n_b"] == 2
    assert all(abs(d + 1.0) < 1e-9 for d in c["per_seed"])
    assert set(out["missing"]) == {"B1", "C0", "C1", "C2"}
    # フェーズ C がない比較も落ちずに判定不能になる
    assert out["comparisons"]["B->C_val"]["verdict_paired"] == "判定不能"


def test_single_seed_is_undecidable(tmp_path, capsys):
    _write(tmp_path, "A", 0, 34.40)
    _write(tmp_path, "B", 0, 34.47)
    out = main([str(tmp_path), str(tmp_path / "no_such.json")])
    c = out["comparisons"]["A->B_val"]
    assert c["verdict_paired"] == "判定不能" and c["verdict_unpaired"] == "判定不能"
    assert (tmp_path / "analysis.json").exists()
    assert "判定不能" in capsys.readouterr().out
