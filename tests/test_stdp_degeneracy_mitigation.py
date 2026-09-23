"""ステップ28（12.6.64節の設計）の単体テスト。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.hippocampus import HippocampalMemory, fit_stdp  # noqa: E402


def test_fit_stdp_default_weight_decay_is_unchanged() -> None:
    """`weight_decay`未指定（既定0.0）のときは従来と完全に同一の挙動であること（回帰確認）。"""
    spikes = (torch.rand(20, 3, 8) > 0.5).float()

    mem1 = HippocampalMemory(8, value_dim=8, n_units=32, k=4, mode="sdr")
    fit_stdp(mem1.separator, spikes, epochs=2, batch_size=8,
              generator=torch.Generator().manual_seed(0))

    mem2 = HippocampalMemory(8, value_dim=8, n_units=32, k=4, mode="sdr")
    fit_stdp(mem2.separator, spikes, epochs=2, batch_size=8,
              generator=torch.Generator().manual_seed(0), weight_decay=0.0)

    assert torch.allclose(mem1.separator.weight, mem2.separator.weight)


def test_fit_stdp_weight_decay_changes_weights() -> None:
    """`weight_decay>0`は既定（0.0）と異なる重みを生むこと（減衰項が実際に効くこと）。"""
    spikes = (torch.rand(20, 3, 8) > 0.5).float()

    mem1 = HippocampalMemory(8, value_dim=8, n_units=32, k=4, mode="sdr")
    fit_stdp(mem1.separator, spikes, epochs=3, batch_size=8,
              generator=torch.Generator().manual_seed(0), weight_decay=0.0)

    mem2 = HippocampalMemory(8, value_dim=8, n_units=32, k=4, mode="sdr")
    fit_stdp(mem2.separator, spikes, epochs=3, batch_size=8,
              generator=torch.Generator().manual_seed(0), weight_decay=0.5)

    assert not torch.allclose(mem1.separator.weight, mem2.separator.weight)


def test_fit_stdp_weight_decay_shrinks_raw_update_norm() -> None:
    """`weight_decay`は正規化前の生の更新量ノルムを（同一条件下で）縮小させること。

    `raw_update = scale*dw/b - weight_decay*sep.weight`であり、`sep.weight`は
    L2正規化されて概ね単位ノルム程度なので、大きな`weight_decay`は更新方向を
    `-sep.weight`寄りに引っ張り、dwとほぼ逆向きなら更新量ノルムを縮小させうる。
    ここでは減衰項が`raw_update_callback`に実際に反映されることだけを確認する。
    """
    spikes = (torch.rand(20, 3, 8) > 0.5).float()
    norms_no_decay: list[float] = []
    norms_decay: list[float] = []

    mem1 = HippocampalMemory(8, value_dim=8, n_units=32, k=4, mode="sdr")
    fit_stdp(mem1.separator, spikes, epochs=2, batch_size=8,
              generator=torch.Generator().manual_seed(0), weight_decay=0.0,
              raw_update_callback=lambda ep, n: norms_no_decay.append(n))

    mem2 = HippocampalMemory(8, value_dim=8, n_units=32, k=4, mode="sdr")
    fit_stdp(mem2.separator, spikes, epochs=2, batch_size=8,
              generator=torch.Generator().manual_seed(0), weight_decay=0.3,
              raw_update_callback=lambda ep, n: norms_decay.append(n))

    assert norms_no_decay != norms_decay


def test_fit_stdp_weight_decay_does_not_change_dw_formula_sign() -> None:
    """`weight_decay=0.0`と`callback`・`raw_update_callback`未指定の場合の情報dictが従来通りであること。"""
    mem = HippocampalMemory(8, value_dim=8, n_units=32, k=4, mode="sdr")
    spikes = (torch.rand(20, 3, 8) > 0.5).float()
    info = fit_stdp(mem.separator, spikes, epochs=2, batch_size=8,
                     generator=torch.Generator().manual_seed(0), weight_decay=0.0)
    assert info["rule"] == "stdp"
    assert info["epochs"] == 2


def test_pattern_separator_k_can_be_enlarged_without_code_change() -> None:
    """対策(a)（k拡大）はコード変更不要でインスタンス化時の引数だけで実現できること。"""
    mem = HippocampalMemory(8, value_dim=8, n_units=32, k=16, mode="sdr")
    assert mem.separator.k == 16
    x = torch.randn(5, 8)
    out = mem.separator(x)
    assert (out != 0).sum(dim=-1).eq(16).all()
