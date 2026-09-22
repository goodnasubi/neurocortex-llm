"""スパイキング自己検証テストの常設化（12.6.40節・ステップ16）— 共有監査ユーティリティ。

12.9節・`test_neurons.py`が確立した「ALIFNeuronクラス単体」の識別テストを、
実際にそれを使うモジュール（`lm.py`・`order_model.py`）の**使われ方**に対して
適用するための、再利用可能な監査関数を提供する。`test_`プレフィックスを持たない
ため、pytestに単体テストファイルとして収集されない。

  assert_not_permutation_invariant(...)         … 量子化への退化を検出する代理指標。
                                                    純粋な位置ごとの量子化（12.9節が
                                                    棄却した方式）は時間位置を入れ替えても
                                                    出力の集合が変わらない（順列不変）。
                                                    `test_neurons.py`の
                                                    `test_step_activation_is_permutation_invariant`
                                                    / `test_alif_is_not_permutation_invariant`
                                                    と同じ操作的定義を、モジュール全体の
                                                    実際の出力に対して適用する。
  assert_membrane_ablation_degrades(...)         … `set_carry_membrane(False)`で膜電位の
                                                    持ち越しを無効化すると、実タスクでの
                                                    性能が明確に落ちることを確認する
                                                    （`run_identification.py`・`run_order.py`
                                                    が個別に実装していたパターンの一般化）。
"""

from __future__ import annotations

from typing import Callable

import torch


def assert_not_permutation_invariant(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    tokens: torch.Tensor,
    seq_dim: int = 1,
    atol: float = 1e-5,
    output_has_seq_dim: bool = True,
) -> None:
    """`forward_fn(tokens)`の出力が、系列次元の並べ替えに対して不変でないことを確認する。

    量子化への退化（12.9節）は位置ごとに独立な写像になるため、入力の並べ替えは
    出力の対応する並べ替えと完全に一致する（順列不変）。時間ダイナミクスが
    本当に機能していれば、並べ替えは出力を単純な並べ替え以上に変化させる。

    `output_has_seq_dim=True`（既定、例: `SpikingLM`の各位置の出力）の場合は、
    `torch.equal`ではなく`atol`付きの許容誤差で「並べ替えた入力の出力」と
    「元の出力を並べ替えたもの」を比較する（等変性のチェック。埋め込み・線形層を
    経た実モデルでは、行の順序が変わるとバッチ化行列積の内部実装が異なる浮動小数点
    丸め経路を取ることがあり実測で最大1e-8オーダーの差が出るため、厳密な
    `torch.equal`ではなく許容誤差を使う）。

    `output_has_seq_dim=False`（例: `OrderModel`のように最終位置だけを出力するモデル）
    の場合は出力に系列次元が無いため等変性は定義できず、単純に「並べ替え前後で
    出力が変わること」（不変性の否定）だけを確認する。
    """
    out = forward_fn(tokens)
    perm = torch.randperm(tokens.shape[seq_dim])
    tokens_perm = tokens.index_select(seq_dim, perm)
    out_perm_input = forward_fn(tokens_perm)
    if output_has_seq_dim:
        out_perm_output = out.index_select(seq_dim, perm)
        assert not torch.allclose(out_perm_input, out_perm_output, atol=atol), (
            "出力が入力の並べ替えに対して単純に追従している（順列不変 = 量子化への退化の疑い）"
        )
    else:
        assert not torch.allclose(out_perm_input, out, atol=atol), (
            "出力が入力の並べ替えに対して変化しない（順列不変 = 量子化への退化の疑い）"
        )


def assert_membrane_ablation_degrades(
    evaluate_fn: Callable[[], float],
    set_carry_membrane: Callable[[bool], None],
    min_drop: float = 0.05,
) -> tuple[float, float]:
    """`set_carry_membrane(False)`で膜電位の持ち越しを無効化すると、`evaluate_fn()`が
    明確に低下する（`min_drop`以上）ことを確認する。戻り値: (baseline, ablated)。
    """
    set_carry_membrane(True)
    baseline = evaluate_fn()
    set_carry_membrane(False)
    ablated = evaluate_fn()
    set_carry_membrane(True)
    assert baseline - ablated >= min_drop, (
        f"膜電位ゼロ化による性能低下が小さすぎる（baseline={baseline:.4f}, "
        f"ablated={ablated:.4f}, 必要な低下={min_drop}）。"
        "時間ダイナミクスが実際にタスク性能へ寄与していない可能性がある"
    )
    return baseline, ablated
