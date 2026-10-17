"""サンプルごとの事後推定による文脈推定（12.6.110節）。12.6.109節で、バッチ全体に1つの文脈を硬く割り当てる
推定（context_inference.py の inferred）は、タスクが少しずつ混ざる移行期間に破綻した。本モジュールは潜在原因推論
（Gershman, Blei & Niv 2010）に近い形で、サンプルごとに文脈の事後確率を計算して割り当てる。

  事後 ∝ 事前 × 尤度
  尤度   … その文脈の予測が当たれば 1 − ε、外れれば ε
  事前   … 最近の割り当て比率の指数移動平均（粘着性）。新しい原因は事前 α・尤度 0.5（中華料理店過程の簡略版）
  混合比 … バッチ内の混合比を EM で推定し、事前で正則化する（尤度だけでは区別できないサンプルの配分に必要）

最新の文脈が確立済みで、新しい原因を選んだサンプルの件数（バッチをまたいで減衰付きで累積）が min_new に達したときだけ、新しい文脈を割り当てる。
最新の文脈が確立するまでは、その尤度を 0.5 とし、それ自体が「新しい原因」の受け皿になる（作った直後の文脈は前の
文脈と同じ予測をするため、実際の尤度を使うと次々と新しい文脈が作られてしまう）。どの文脈でも説明できず、まだ新しい
文脈も作らないサンプルは保留し、学習にも保存にも使わない（既存の文脈を汚さないため）。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .context_inference import make_batch
from .hippocampus_replay import make_model


def run_posterior_sequence(specs: list, hidden_dim: int, generator: torch.Generator, eval_generator: torch.Generator,
                           n_slots: int | None = None, alpha: float = 0.1, prior_ema: float = 0.9,
                           min_new: int = 8, eps: float = 0.02, new_decay: float = 0.9, em_iters: int = 20, kappa: float = 8.0, tau: float = 0.4, ema: float = 0.9, pretrain_steps: int = 500,
                           pretrain_weight_decay: float = 0.01, steps: int = 500, batch: int = 64, lr: float = 0.05,
                           buffer_size: int = 50, cue_n: int = 16, eval_n: int = 2000,
                           transition_steps: int = 0) -> dict:
    """戻り値: {"R": 正確度行列, "assign": (タスク, そのバッチで最も多く割り当てた文脈) の列, "n_contexts"}"""
    T = len(specs)
    C = n_slots or T + 2
    model = make_model(specs[0].n_inputs + C, hidden_dim)

    def with_ctx(bits: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        oh = torch.zeros(bits.shape[0], C)
        oh[torch.arange(bits.shape[0]), ctx] = 1.0
        return torch.cat([bits, oh], dim=1)

    @torch.no_grad()
    def label_prob(bits: torch.Tensor, labels: torch.Tensor, c: int) -> torch.Tensor:
        """予測が当たれば 1 − ε、外れれば ε。モデルの確信度そのものを使うと、確信度の低い文脈が食い違うサンプルを
        吸収し、それで学習してさらに確信度が下がるという悪循環になる（12.6.110節）。"""
        pred = model(with_ctx(bits, torch.full((bits.shape[0],), c))).argmax(dim=-1)
        return torch.where(pred == labels, 1.0 - eps, eps)

    @torch.no_grad()
    def mean_loss(bits: torch.Tensor, labels: torch.Tensor, c: int) -> float:
        return float(F.cross_entropy(model(with_ctx(bits, torch.full((bits.shape[0],), c))), labels))

    used = [0]
    prior = {0: 1.0}
    ema_loss: dict[int, float] = {}
    buf: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    buf_count: dict[int, int] = {}
    assign: list[tuple[int, int]] = []
    new_count = [0.0]

    def infer(bits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """バッチ内の混合比を EM で推定し、サンプルごとに事後確率が最大の文脈を選ぶ。

        尤度だけでは、2つのタスクで正解が一致するサンプル（半分）を区別できない。バッチ内の混合比 π を
        正解が食い違うサンプルから推定すれば、一致するサンプルも π に従って配分される。未確立の最新の文脈と
        「新しい原因」の尤度は 0.5（まだ何も予測できない）とする。π は前のバッチまでの事前（粘着性）で正則化する。
        """
        n = bits.shape[0]
        if len(used) == 1 and ema_loss.get(0, math.inf) >= tau:
            return torch.zeros(n, dtype=torch.long)  # 最初の文脈を学習中
        newest_ok = ema_loss.get(used[-1], math.inf) < tau
        cols = [label_prob(bits, labels, c) if (c != used[-1] or newest_ok) else torch.full((n,), 0.5) for c in used]
        can_add = newest_ok and len(used) < C
        if can_add:
            cols.append(torch.full((n,), 0.5))
        lik = torch.stack(cols, dim=1).clamp_min(1e-6)
        pri = torch.tensor([prior[c] for c in used] + ([alpha] if can_add else []))
        pri = (pri + 1e-3) / (pri + 1e-3).sum()
        pi = pri.clone()
        for _ in range(em_iters):
            r = pi * lik
            r = r / r.sum(dim=1, keepdim=True)
            pi = (r.sum(dim=0) + kappa * pri) / (n + kappa)
        choice = (pi * lik).argmax(dim=1)
        is_new = choice == len(used)
        new_count[0] = new_decay * new_count[0] + int(is_new.sum())
        if can_add and new_count[0] >= min_new:
            used.append(len(used))
            prior[used[-1]] = 0.0
            new_count[0] = 0.0
            return torch.tensor(used)[choice]
        # どの文脈でも説明できず、新しい文脈もまだ作らないサンプルは保留する（学習・保存に使わない）
        ctx = torch.tensor(used)[choice.clamp(max=len(used) - 1)]
        ctx[is_new] = -1
        return ctx

    @torch.no_grad()
    def accuracy(j: int) -> float:
        cb, cl = make_batch(specs[j], cue_n, eval_generator)
        c = min(used, key=lambda u: mean_loss(cb, cl, u))
        bits, labels = make_batch(specs[j], eval_n, eval_generator)
        return float((model(with_ctx(bits, torch.full((eval_n,), c))).argmax(-1) == labels).float().mean())

    R: list[list[float]] = []
    for k, spec in enumerate(specs):
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=pretrain_weight_decay if k == 0 else 0.0)
        for step in range(pretrain_steps if k == 0 else steps):
            bits, labels = make_batch(spec, batch, generator)
            if k > 0 and step < transition_steps:
                old_bits, old_labels = make_batch(specs[k - 1], batch, generator)
                take_new = torch.rand(batch, generator=generator) < (step + 1) / (transition_steps + 1)
                bits = torch.where(take_new[:, None], bits, old_bits)
                labels = torch.where(take_new, labels, old_labels)
            ctx = infer(bits, labels)
            keep = ctx >= 0
            counts = torch.bincount(ctx[keep], minlength=C).float()
            assign.append((k, int(counts.argmax())))
            frac = counts / counts.sum().clamp_min(1.0)
            for c in used:
                prior[c] = prior_ema * prior[c] + (1 - prior_ema) * float(frac[c])
            bits, labels, ctx = bits[keep], labels[keep], ctx[keep]
            x, y = with_ctx(bits, ctx), labels
            if buf and k > 0:
                bb = torch.cat([b for v in buf.values() for b, _ in v])
                bl = torch.cat([l for v in buf.values() for _, l in v])
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
                cl = mean_loss(bits[m], labels[m], c)
                ema_loss[c] = cl if c not in ema_loss else ema * ema_loss[c] + (1 - ema) * cl
                n = min(buffer_size - buf_count.get(c, 0), int(m.sum()))
                if n > 0:
                    buf_count[c] = buf_count.get(c, 0) + n
                    buf.setdefault(c, []).append((bits[m][:n], labels[m][:n]))
        R.append([accuracy(j) for j in range(T)])
    return {"R": R, "assign": assign, "n_contexts": len(used)}
