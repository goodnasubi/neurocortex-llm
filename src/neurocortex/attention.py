"""因果的線形アテンション（12.3.1節）。

状態 S_t = Σ_{i<=t} φ(k_i) v_i^T を逐次更新する。softmaxの二乗オーダーを避け、
スパイキングニューロンと同じく「位置方向に持ち越される状態」として実装する。
12.3.1節の方針に従い、独立した関数境界として保つ（将来のカーネル化のため）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_attention_causal(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """因果的線形アテンション。

    Args:
        q, k: [B, H, T, D]（非負であること。呼び出し側で elu+1 等をかける）
        v:    [B, H, T, Dv]
    Returns:
        [B, H, T, Dv]
    """
    b, h, t, d = q.shape
    dv = v.shape[-1]
    s = q.new_zeros(b, h, d, dv)  # キー・バリュー外積の累積状態
    z = q.new_zeros(b, h, d)      # 正規化項
    outs = []
    for i in range(t):
        s = s + k[:, :, i, :].unsqueeze(-1) * v[:, :, i, :].unsqueeze(-2)
        z = z + k[:, :, i, :]
        num = torch.einsum("bhd,bhde->bhe", q[:, :, i, :], s)
        den = torch.einsum("bhd,bhd->bh", q[:, :, i, :], z).unsqueeze(-1)
        outs.append(num / (den + eps))
    return torch.stack(outs, dim=2)


class CausalLinearAttention(nn.Module):
    """マルチヘッド因果的線形アテンション層。"""

    def __init__(self, d_model: int, n_heads: int = 4) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model は n_heads で割り切れる必要がある")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split(z: torch.Tensor) -> torch.Tensor:
            return z.view(b, t, self.n_heads, self.d_head).transpose(1, 2)

        # φ(x) = elu(x) + 1 で非負化（Katharopoulos et al. 2020）
        q = F.elu(split(q)) + 1.0
        k = F.elu(split(k)) + 1.0
        v = split(v)
        y = linear_attention_causal(q, k, v)
        y = y.transpose(1, 2).reshape(b, t, d)
        return self.out(y)
