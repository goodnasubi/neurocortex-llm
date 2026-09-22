"""小脳モジュール — 逆モデル本体（12.6.24節・ステップ8）。

ステップ4（`cerebellum.py`）で検証したのは持続する順モデルのみだった。12.3.4節の
もう一方の要素——目標値から呼び出し引数を直接予測する**逆モデル**——を対象にする。

線形ツール `y = w・x + b`（`x`が2次元以上）は不良設定（1つの`y`に対応する`x`が
無数にある）である。素朴には、逆モデルを別に学習する価値は、既に持続している
順モデルの**疑似逆行列**（最小ノルム解 `x̂ = w(y-b)/‖w‖²`）で代替できてしまい
かねない（12.6.6節・12.6.20節と同型の罠）。したがって主張は`y`の再現度ではなく、
**引数`x`そのものの再現誤差**（真の生成分布`D`がどれだけ最小ノルム解から離れて
いても正しく捉えられるか）に置く。

  inverse-model（本設計）        … `(y, x)` 履歴から直接学習する逆写像。
                                   エピソードをまたいで持続する
  pseudo-inverse（対照群、統制5）… 同じ履歴で学習する持続的な順モデルの
                                   疑似逆行列。追加パラメータを一切持たない
  inverse-from-scratch（統制5b） … inverse-modelと同一構成だが、持続しない
                                   （エピソードごとに再初期化）
"""

from __future__ import annotations

import torch

from .cerebellum import ForwardModel, Tool, call_tool


class InverseModel:
    """逆モデル `ĝ(y) = A・y + c`（スカラー→ベクトル）。デルタ則でのみ更新する。"""

    def __init__(self, dim: int) -> None:
        self.A = torch.zeros(dim)
        self.c = torch.zeros(dim)

    def predict(self, y: float) -> torch.Tensor:
        return self.A * y + self.c

    def update(self, y: float, x_true: torch.Tensor, lr: float) -> torch.Tensor:
        """1件の (y, x_true) でデルタ則更新する。戻り値は更新前の予測誤差ベクトル。"""
        error = x_true - self.predict(y)
        self.A += lr * error * y
        self.c += lr * error
        return error


def pseudo_inverse_predict(forward_model: ForwardModel, y_target: float) -> torch.Tensor:
    """持続する順モデル `ŷ = ŵ・x + b̂` の疑似逆行列（最小ノルム解）で `x` を予測する。

    不良設定の線形系 `ŵ・x = y_target - b̂` の最小ノルム解は
    `x̂ = ŵ (y_target - b̂) / ‖ŵ‖²`。
    """
    w = forward_model.weight
    denom = float(w @ w)
    if denom < 1e-12:
        return torch.zeros_like(w)
    return w * (y_target - forward_model.bias) / denom


def sample_from_D(dim: int, mean: float, std: float, generator: torch.Generator) -> torch.Tensor:
    """呼び出し引数の真の生成分布 `D`（原点から離れた固定平均のガウス分布）から1件引く。"""
    return mean + std * torch.randn(dim, generator=generator)


def run_condition(condition: str, tool: Tool, n_calls: int, episode_len: int,
                  d_mean: float, d_std: float, lr: float, eval_every: int, n_eval: int,
                  generator: torch.Generator, lr_forward: float | None = None) -> list[dict]:
    """1条件・1シードぶんの実行。`eval_every`呼び出しごとの評価結果のリストを返す。

    各要素は {"x_error": 引数の再現誤差の平均, "y_error": yの再現誤差の平均}。

    `lr`は逆モデル（入力がスカラーの`y`、デルタ則の安定域は`y`のスケールに
    依存）に、`lr_forward`（省略時は`lr`と同値）は順モデル（入力がベクトルの
    `x`、安定域は`x`のスケールに依存）に使う。`D`の平均を原点から離す設計上、
    `y`と`x`のスケールが大きく異なりうるため、デルタ則の安定な学習率を別々に
    持たせる必要がある。
    """
    if lr_forward is None:
        lr_forward = lr
    dim = tool.weight.shape[0]
    inverse_model = InverseModel(dim)
    forward_model = ForwardModel(dim)

    rows: list[dict] = []
    for call_idx in range(1, n_calls + 1):
        if condition == "inverse-from-scratch" and (call_idx - 1) % episode_len == 0:
            inverse_model = InverseModel(dim)

        x = sample_from_D(dim, d_mean, d_std, generator)
        y = float(call_tool(tool, x))

        if condition in ("inverse-model", "inverse-from-scratch"):
            inverse_model.update(y, x, lr)
        elif condition == "pseudo-inverse":
            forward_model.update(x, y, lr_forward)
        else:
            raise ValueError(condition)

        if call_idx % eval_every == 0:
            x_errors, y_errors = [], []
            for _ in range(n_eval):
                x_star = sample_from_D(dim, d_mean, d_std, generator)
                y_star = float(call_tool(tool, x_star))
                if condition == "pseudo-inverse":
                    x_hat = pseudo_inverse_predict(forward_model, y_star)
                else:
                    x_hat = inverse_model.predict(y_star)
                y_hat = float(call_tool(tool, x_hat))
                x_errors.append(float((x_hat - x_star).norm()))
                y_errors.append(abs(y_hat - y_star))
            rows.append({
                "call": call_idx,
                "x_error": sum(x_errors) / n_eval,
                "y_error": sum(y_errors) / n_eval,
            })
    return rows
