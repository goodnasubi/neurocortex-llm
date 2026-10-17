"""入力分布からのラベルなし文脈推定（12.6.111節）。

12.6.110節で、入力分布がタスク間で同じ課題では、少しずつ変わる文脈を（ラベルを使っても）サンプル単位で識別できない
ことが分かった。入力分布がタスクで異なれば、ラベルなしで文脈を推定でき、混ざったバッチもサンプル単位で分けられる
はずである。

  プロトタイプ … 各文脈は入力ビットごとの Bernoulli パラメータ μ_c を持つ。割り当てたサンプルの累積カウントによる
                 Beta 事後平均（指数移動平均だと、選ばれた偏ったサンプルに引きずられて歪む。12.6.111節）
  新しい文脈   … バッチの対数尤度が、既存の最もよい文脈より「新しい原因」（一様分布、事前 log α）のほうが高いとき作る
  割り当て     … 既存の文脈の中で、入力の尤度と粘着性のある事前（最近の割り当て比率）から、バッチ内の混合比を EM で
                 推定してサンプルごとに選ぶ

推定にラベルは使わない。ラベルはモデルの学習だけに使い、テスト時の手がかりもラベルなしである。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .context_inference import make_batch
from .hippocampus_replay import make_model


def run_unsupervised_sequence(specs: list, hidden_dim: int, generator: torch.Generator,
                              eval_generator: torch.Generator, n_slots: int | None = None, log_alpha: float = -5.0,
                              prior_ema: float = 0.9, em_iters: int = 20, kappa: float = 8.0,
                              pretrain_steps: int = 500, pretrain_weight_decay: float = 0.01, steps: int = 500,
                              batch: int = 64, lr: float = 0.05, buffer_size: int = 50, cue_n: int = 16,
                              eval_n: int = 2000, transition_steps: int = 0) -> dict:
    """戻り値: {"R": 正確度行列, "assign": (タスク, そのバッチで最も多く割り当てた文脈) の列, "n_contexts",
    "sample_purity": 学習中のサンプル単位の割り当ての純度（真のタスクごとの最頻文脈の割合の平均）}"""
    T = len(specs)
    C = n_slots or T + 2
    model = make_model(specs[0].n_inputs + C, hidden_dim)

    def with_ctx(bits: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        oh = torch.zeros(bits.shape[0], C)
        oh[torch.arange(bits.shape[0]), ctx] = 1.0
        return torch.cat([bits, oh], dim=1)

    mu: list[torch.Tensor] = []
    pos_count: list[torch.Tensor] = []
    n_count: list[float] = []
    prior: list[float] = []

    def loglik(bits: torch.Tensor) -> torch.Tensor:
        """[n, 文脈数] のサンプルごとの対数尤度"""
        pos = (bits > 0).float()
        m = torch.stack(mu).clamp(0.05, 0.95)
        return pos @ m.log().T + (1 - pos) @ (1 - m).log().T

    def infer(bits: torch.Tensor) -> torch.Tensor:
        n, d = bits.shape
        if not mu or (loglik(bits).max(dim=1).values.sum() < n * d * math.log(0.5) + log_alpha and len(mu) < C):
            # 新しい原因: バッチの尤度が一様分布に負ける
            pos_count.append((bits > 0).float().sum(dim=0))
            n_count.append(float(bits.shape[0]))
            mu.append((pos_count[-1] + 1) / (n_count[-1] + 2))
            # バッチ単位で新しい原因があると判断したので、新しい文脈に相応の事前を与える。0 から始めると EM が
            # 新しい文脈に割り当てられず、前の文脈のプロトタイプが新タスク側にずれてしまう
            for c in range(len(prior)):
                prior[c] *= 0.5
            prior.append(1.0 if len(mu) == 1 else 0.5)
        ll = loglik(bits)
        pri = torch.tensor(prior) + 1e-3
        pri = pri / pri.sum()
        pi = pri.clone()
        for _ in range(em_iters):
            r = torch.softmax(ll + pi.log(), dim=1)
            pi = (r.sum(dim=0) + kappa * pri) / (n + kappa)
        ctx = (ll + pi.log()).argmax(dim=1)
        frac = torch.bincount(ctx, minlength=len(mu)).float() / n
        for c in range(len(mu)):
            prior[c] = prior_ema * prior[c] + (1 - prior_ema) * float(frac[c])
            m = ctx == c
            if m.any():
                pos_count[c] += (bits[m] > 0).float().sum(dim=0)
                n_count[c] += float(m.sum())
                mu[c] = (pos_count[c] + 1) / (n_count[c] + 2)
        return ctx

    @torch.no_grad()
    def accuracy(j: int) -> float:
        cb, _ = make_batch(specs[j], cue_n, eval_generator)  # ラベルは使わない
        c = int(loglik(cb).sum(dim=0).argmax())
        bits, labels = make_batch(specs[j], eval_n, eval_generator)
        return float((model(with_ctx(bits, torch.full((eval_n,), c))).argmax(-1) == labels).float().mean())

    buf: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    buf_count: dict[int, int] = {}
    assign: list[tuple[int, int]] = []
    sample_counts = torch.zeros(T, C)
    R: list[list[float]] = []
    for k, spec in enumerate(specs):
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=pretrain_weight_decay if k == 0 else 0.0)
        for step in range(pretrain_steps if k == 0 else steps):
            bits, labels = make_batch(spec, batch, generator)
            src = torch.full((batch,), k)
            if k > 0 and step < transition_steps:
                old_bits, old_labels = make_batch(specs[k - 1], batch, generator)
                take_new = torch.rand(batch, generator=generator) < (step + 1) / (transition_steps + 1)
                bits = torch.where(take_new[:, None], bits, old_bits)
                labels = torch.where(take_new, labels, old_labels)
                src = torch.where(take_new, k, k - 1)
            ctx = infer(bits)
            for t, c in zip(src.tolist(), ctx.tolist()):
                sample_counts[t, c] += 1
            assign.append((k, int(torch.bincount(ctx, minlength=C).argmax())))
            x, y = with_ctx(bits, ctx), labels
            if buf and k > 0:
                bb = torch.cat([b for v in buf.values() for b, _ in v])
                bl = torch.cat([lab for v in buf.values() for _, lab in v])
                bc = torch.cat([torch.full((b.shape[0],), c) for c, v in buf.items() for b, _ in v])
                idx = torch.randint(bb.shape[0], (batch,), generator=generator)
                x = torch.cat([x, with_ctx(bb[idx], bc[idx])])
                y = torch.cat([y, bl[idx]])
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            for c in ctx.unique().tolist():
                m = ctx == c
                n = min(buffer_size - buf_count.get(c, 0), int(m.sum()))
                if n > 0:
                    buf_count[c] = buf_count.get(c, 0) + n
                    buf.setdefault(c, []).append((bits[m][:n], labels[m][:n]))
        R.append([accuracy(j) for j in range(T)])
    purity = float((sample_counts.max(dim=1).values / sample_counts.sum(dim=1).clamp_min(1)).mean())
    return {"R": R, "assign": assign, "n_contexts": len(mu), "sample_purity": purity}
