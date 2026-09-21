"""識別テスト（12.6.1節）の学習・評価ルーチン。

学習予算の対称性（統制3）を担保するため、スパイキング版と対照群（階段関数）は
この同一関数・同一引数・同一シードで学習される。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from ..models import RecallModel
from ..neurons import NeuronConfig
from ..tasks import RecallSpec, make_recall_batch


@dataclass
class TrainConfig:
    """学習設定。両群で完全に同一の値を使う（12.6.1節 統制3）。"""

    steps: int = 3000
    batch_size: int = 64
    lr: float = 3e-3
    d_model: int = 64
    n_layers: int = 2
    eval_batches: int = 16
    eval_batch_size: int = 128
    grad_clip: float = 1.0


def _set_seed(seed: int) -> torch.Generator:
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed + 10_000)
    return g


@torch.no_grad()
def evaluate(
    model: RecallModel,
    spec: RecallSpec,
    tcfg: TrainConfig,
    seed: int,
    delay: int | None,
) -> float:
    """指定した遅延での正答率。"""
    model.eval()
    g = torch.Generator()
    g.manual_seed(seed + 999_983)
    correct = 0
    total = 0
    for _ in range(tcfg.eval_batches):
        x, y = make_recall_batch(spec, tcfg.eval_batch_size, g, delay=delay)
        pred = model(x).argmax(dim=-1)
        correct += int((pred == y).sum())
        total += y.numel()
    return correct / total


def train_recall(
    spec: RecallSpec,
    ncfg: NeuronConfig,
    tcfg: TrainConfig,
    activation: str,
    seed: int,
    train_delay: int | None = None,
    verbose: bool = False,
) -> tuple[RecallModel, dict]:
    """遅延想起課題でモデルを1本学習する。

    Args:
        train_delay: None なら遅延を 0..max_delay でランダムに振る。
    Returns:
        (学習済みモデル, 学習ログ)
    """
    g = _set_seed(seed)
    model = RecallModel(
        vocab_size=spec.vocab_size,
        n_classes=spec.n_cue,
        d_model=tcfg.d_model,
        n_layers=tcfg.n_layers,
        cfg=ncfg,
        activation=activation,
    )
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr)
    losses: list[float] = []
    for step in range(tcfg.steps):
        model.train()
        x, y = make_recall_batch(spec, tcfg.batch_size, g, delay=train_delay)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()
        losses.append(loss.detach().item())
        if verbose and (step + 1) % max(1, tcfg.steps // 10) == 0:
            print(f"  step {step + 1}/{tcfg.steps} loss={sum(losses[-50:]) / 50:.4f}", flush=True)
    log = {
        "final_loss": sum(losses[-50:]) / 50,
        "diverged": not all(l == l for l in losses[-50:]),
        "firing_rates": model.firing_rates(),
        "train_config": asdict(tcfg),
        "neuron_config": asdict(ncfg),
    }
    return model, log
