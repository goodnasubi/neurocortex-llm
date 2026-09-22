"""ステップ27（12.6.62節の設計）の単体テスト。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.hippocampus import HippocampalMemory, fit_stdp  # noqa: E402
from neurocortex.experiments.run_stdp_degeneracy_impact import (  # noqa: E402
    recompute_n_unique_from_stability,
)


def test_fit_stdp_raw_update_callback_fires_once_per_epoch() -> None:
    """新規追加した`raw_update_callback`が、各epoch末にちょうど1回ずつ呼ばれること。"""
    mem = HippocampalMemory(8, value_dim=8, n_units=32, k=4, mode="sdr")
    spikes = (torch.rand(20, 3, 8) > 0.5).float()
    epochs_seen: list[int] = []
    norms: list[float] = []
    fit_stdp(mem.separator, spikes, epochs=4, batch_size=8,
              generator=torch.Generator().manual_seed(0),
              raw_update_callback=lambda ep, norm: (epochs_seen.append(ep), norms.append(norm)))
    assert epochs_seen == [0, 1, 2, 3]
    assert all(n >= 0.0 for n in norms)


def test_fit_stdp_raw_update_callback_does_not_change_weights() -> None:
    """`raw_update_callback`を渡しても、STDPアルゴリズム自体（学習後の重み）は変わらないこと。"""
    spikes = (torch.rand(20, 3, 8) > 0.5).float()

    mem1 = HippocampalMemory(8, value_dim=8, n_units=32, k=4, mode="sdr")
    fit_stdp(mem1.separator, spikes, epochs=2, batch_size=8,
              generator=torch.Generator().manual_seed(0))

    mem2 = HippocampalMemory(8, value_dim=8, n_units=32, k=4, mode="sdr")
    fit_stdp(mem2.separator, spikes, epochs=2, batch_size=8,
              generator=torch.Generator().manual_seed(0),
              raw_update_callback=lambda ep, norm: None)

    assert torch.allclose(mem1.separator.weight, mem2.separator.weight)


def test_fit_stdp_without_callbacks_is_unchanged() -> None:
    """両コールバックとも`None`（既定）のときは従来どおり動作すること（回帰確認）。"""
    mem = HippocampalMemory(8, value_dim=8, n_units=32, k=4, mode="sdr")
    spikes = (torch.rand(20, 3, 8) > 0.5).float()
    info = fit_stdp(mem.separator, spikes, epochs=2, batch_size=8,
                     generator=torch.Generator().manual_seed(0))
    assert info["rule"] == "stdp"
    assert info["epochs"] == 2


def test_recompute_n_unique_from_stability_missing_file(tmp_path: Path) -> None:
    """存在しないJSONを指定した場合は`available: False`を返すこと。"""
    result = recompute_n_unique_from_stability(tmp_path / "no_such.json")
    assert result == {"available": False}


def test_recompute_n_unique_from_stability_aggregates_epoch1_and_last(tmp_path: Path) -> None:
    """既定点（1倍）のepoch1・epoch末の`n_unique_rows`を正しく再集計すること。"""
    stability = {
        "rows": [
            {
                "multiplier": 1.0, "diverged": False,
                "epoch_records": [
                    {"epoch": 0, "n_unique_rows": 100},
                    {"epoch": 1, "n_unique_rows": 80},
                ],
            },
            {
                "multiplier": 1.0, "diverged": False,
                "epoch_records": [
                    {"epoch": 0, "n_unique_rows": 90},
                    {"epoch": 1, "n_unique_rows": 70},
                ],
            },
            # 別倍率・発散した行は集計対象外
            {
                "multiplier": 3.0, "diverged": False,
                "epoch_records": [{"epoch": 0, "n_unique_rows": 10},
                                   {"epoch": 1, "n_unique_rows": 10}],
            },
            {
                "multiplier": 1.0, "diverged": True,
                "epoch_records": [{"epoch": 0, "n_unique_rows": 5},
                                   {"epoch": 1, "n_unique_rows": 1}],
            },
        ],
    }
    path = tmp_path / "stability.json"
    path.write_text(json.dumps(stability), encoding="utf-8")

    result = recompute_n_unique_from_stability(path, multiplier=1.0)
    assert result["available"] is True
    assert result["n_seeds"] == 2
    assert result["epoch1_n_unique_mean"] == 95.0  # (100+90)/2
    assert result["epoch20_n_unique_mean"] == 75.0  # (80+70)/2
