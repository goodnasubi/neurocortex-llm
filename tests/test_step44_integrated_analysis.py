import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from step44_integrated_analysis import NO_DATA, integrate, main


def _write(root, phase, seed, final, with_test=False):
    heads = {"A": {"hippocampus_loss": 4.0}, "B": {"hippocampus_loss": 4.0, "basal_ganglia_loss": 3.9}}.get(
        phase, {"hippocampus_loss": 4.0, "basal_ganglia_loss": 3.9, "cerebellum_loss": 3.9})
    d = {"phase": phase, "seed": seed, "val_ppls": [final + 30, final + 5, final], "val_losses": [], "train_losses": [],
         "val_head_ce": [heads] * 3, "config": {"hidden_size": 8, "hippocampus_loss_weight": 0.05}}
    if with_test:
        d["test_ppl"] = final + 1
    (root / f"phase{phase}_seed{seed}.json").write_text(json.dumps(d))


def _fill(root, offsets, with_test=False):
    root.mkdir(parents=True, exist_ok=True)
    for s, base in enumerate([100.0, 100.4, 99.7]):
        for p, o in zip("ABC", offsets):
            _write(root, p, s, base + o, with_test)


def test_integrate_agreement_and_no_data(tmp_path):
    _fill(tmp_path / "r", [0, 0.2, 0.3], with_test=True)
    _fill(tmp_path / "f", [0, 0.1, 0.1])
    out = integrate(tmp_path / "r", tmp_path / "f")
    g = out["direction_agreement"]["A->B_val"]
    assert g["sign_agree"] is True and not g["improved_anywhere"]
    full = out["scales"]["full"]
    assert full["test_ppl"]["A"] == NO_DATA          # test_ppl が無いものは捏造しない
    assert full["cost"]["学習時間（秒）"] == NO_DATA
    assert "basal_ganglia_loss" not in full["aux"]["A"]["head_ce_final"]
    assert abs(full["aux"]["C"]["head_ce_final"]["cerebellum_loss"]["mean"] - 3.9) < 1e-9
    assert out["scales"]["reduced"]["test_ppl"]["A"]["n"] == 3


def test_main_writes_json(tmp_path, capsys):
    _fill(tmp_path / "r", [0, 0, 0])
    _fill(tmp_path / "f", [0, 0, 0])
    main([str(tmp_path / "r"), str(tmp_path / "f"), str(tmp_path / "o")])
    assert (tmp_path / "o" / "analysis.json").exists()
    assert "規模間の方向一致" in capsys.readouterr().out
