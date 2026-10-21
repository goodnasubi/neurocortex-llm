import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from step44_vanilla_analysis import analyze, verdict


def _w(d, p, s, v):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"phase{p}_seed{s}.json").write_text(json.dumps({"val_ppls": [v + 20, v], "test_ppl": v - 10}))


def test_verdict():
    assert verdict(1.0, 0.4) == "悪化" and verdict(-1.0, 0.4) == "改善" and verdict(0.7, 0.4) == "差なし"


def test_analyze(tmp_path):
    for s, base in enumerate([100.0, 100.4, 99.7]):
        _w(tmp_path / "v", "V", s, base)
        for p, o in zip("ABC", [0.0, 0.3, 5.0]):
            _w(tmp_path / "abc", p, s, base + o)
    out = analyze(tmp_path / "abc", tmp_path / "v")
    assert out["comparisons"]["V->A_val"]["verdict_paired"] == "差なし"
    c = out["comparisons"]["V->C_val"]
    assert abs(c["mean_diff"] - 5.0) < 1e-9 and c["verdict_unpaired"] == "悪化"
    assert out["phases"]["V"]["test"]["mean"] > 0
