"""`SpikingLM`（`lm.py`）の常設識別テスト（12.6.40節・ステップ16）。

`ALIFNeuron`クラス単体は`test_neurons.py`で検証済みだが、`SpikingLM`自身の
使われ方（3層×3活性化のブロック構成、次トークン予測タスク）が本当に時間
ダイナミクスを活かしているかは、これまで検証されていなかった。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neurocortex.data import load_corpus  # noqa: E402
from neurocortex.lm import SpikingLM  # noqa: E402
from neurocortex.neurons import NeuronConfig  # noqa: E402

from spiking_audit import assert_membrane_ablation_degrades, assert_not_permutation_invariant  # noqa: E402


def _make_model_and_corpus(seed: int = 0):
    corpus = load_corpus(None, seed=seed)
    ncfg = NeuronConfig(beta=0.95)
    model = SpikingLM(vocab_size=corpus.vocab_size, d_model=32, n_layers=2, n_heads=2,
                      max_len=32, cfg=ncfg)
    return model, corpus


def test_not_permutation_invariant_on_untrained_model() -> None:
    """統制（V1）: 学習前でも、埋め込み・時間位置エンコーディング自体が既に位置依存だが、
    ここではモデル全体の出力が並べ替えに対して単純に追従しないことを確認する。
    """
    torch.manual_seed(0)
    model, corpus = _make_model_and_corpus()
    model.eval()
    g = torch.Generator().manual_seed(0)
    tokens, _ = corpus.batch("train", 4, 16, g)
    assert_not_permutation_invariant(lambda t: model(t), tokens)


def test_membrane_ablation_degrades_validation_loss() -> None:
    """V2: `set_carry_membrane(False)`にすると、次トークン予測の検証損失が明確に悪化すること。"""
    torch.manual_seed(0)
    model, corpus = _make_model_and_corpus()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    g = torch.Generator().manual_seed(0)
    for _ in range(300):
        x, y = corpus.batch("train", 16, 32, g)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    @torch.no_grad()
    def evaluate_neg_loss() -> float:
        # 損失は低いほど良いので、`assert_membrane_ablation_degrades`が仮定する
        # 「高いほど良い」指標に合わせて符号を反転する。
        model.eval()
        eval_g = torch.Generator().manual_seed(999)
        x, y = corpus.batch("val", 64, 32, eval_g)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        return -float(loss)

    baseline, ablated = assert_membrane_ablation_degrades(evaluate_neg_loss,
                                                           model.set_carry_membrane, min_drop=0.02)
    assert baseline > ablated
