"""臨界期の間と閉じた後でルータの学習則を切り替える（12.6.185節）。

12.6.182節で、ゲートを通る LM の勾配は閉じる前の取り込み（A）を助け、閉じた後には漂流を作った。反実仮想の教師の
三要素則だけの学習（とくにソフトな教師 tf_srand3）は、閉じた後の漂流を抑えた。ここでは臨界期の間（開いている期間）は
ゲートの勾配で学び、閉じた後は三要素則（教師）だけで学ぶ。`moe_tf.py` ほかの既存の模型の挙動は変えない。

- `SwitchLM`: `ThreeFactorLM` の継承。
  - `switch="closing"`: 閉じ始め（s < 1）から、ゲートの LM の勾配をルータに送らない（`ConsolLM` の gate_detach と同じ）。
  - `switch="closed"`: 閉じ終わり（s = 0）から切る（閉じる期間の間はゲートの勾配で学ぶ）。
  - `soft_after=True`: 閉じ始めまではハードな教師、閉じ始めからはソフトな教師（評価する専門家と乱数の消費は同じ）。
  既定（switch=None・soft_after=False）では `ThreeFactorLM` と同じ学習経過（`tests/test_moe_sw.py`）。
"""

from __future__ import annotations

import torch

from .moe_tf import ThreeFactorLM


class SwitchLM(ThreeFactorLM):
    """開いている期間はゲートの勾配で、閉じた後は教師の三要素則だけでルータを学習する。"""

    def __init__(self, *args, switch: str | None = None, soft_after: bool = False, **kw) -> None:
        super().__init__(*args, **kw)
        if switch not in (None, "closing", "closed"):
            raise ValueError(f"未知の switch: {switch}")
        if switch and (self.kind != "select" or self.n_layers != 1):
            raise ValueError("switch は1層の選択専用（select）の模型だけ")
        if soft_after and (self.soft or self.kind_t not in ("top", "rand") or self.teacher_homeo or self.react_probe):
            raise ValueError("soft_after はハードな部分評価の教師（top・rand）だけ")
        self.switch, self.soft_after = switch, soft_after

    def detach_gate(self) -> bool:
        sw = getattr(self, "switch", None)
        if sw == "closing" and self.closing:
            return True
        if sw == "closed" and self.closing and self.strength <= 0.0:
            return True
        return super().detach_gate()

    def teacher_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.soft_after:
            self.soft = self.closing
        return super().teacher_loss(x, y)
