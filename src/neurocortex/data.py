"""文字レベルのコーパスとデータローダ。

既定は外部ダウンロード不要の合成コーパス（シード固定で完全に再現可能）。
`--data` で任意のテキストファイルも渡せる。
"""

from __future__ import annotations

import random
from pathlib import Path

import torch

_SUBJECTS = ["the cat", "a dog", "the bird", "my friend", "the robot", "a child", "the river"]
_VERBS = ["sees", "chases", "holds", "builds", "hides", "finds", "watches"]
_OBJECTS = ["a stone", "the box", "some bread", "the lamp", "a flower", "the key", "my hat"]
_ADVERBS = ["quickly", "quietly", "again", "at dawn", "in the garden", "near the wall"]
_CONNECTIVES = [", and then ", ", but ", ". later ", ". meanwhile ", ". so "]


def synthetic_corpus(n_sentences: int = 6000, seed: int = 0) -> str:
    """規則的だが自明でない合成英文コーパスを生成する。

    語順に長距離の依存（接続詞で結ばれた節の再帰）を含めることで、文字レベルでも
    単なる頻度表以上の構造を持たせる。
    """
    rng = random.Random(seed)
    out: list[str] = []
    for _ in range(n_sentences):
        parts = []
        for _ in range(rng.randint(1, 3)):
            s = f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)}"
            if rng.random() < 0.5:
                s += f" {rng.choice(_ADVERBS)}"
            parts.append(s)
        text = parts[0]
        for p in parts[1:]:
            text += rng.choice(_CONNECTIVES) + p
        out.append(text + ".\n")
    return "".join(out)


class CharCorpus:
    """文字レベルの語彙化と train/val 分割を行う。"""

    def __init__(self, text: str, val_fraction: float = 0.1) -> None:
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n_val = int(len(data) * val_fraction)
        self.train = data[:-n_val]
        self.val = data[-n_val:]

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def batch(
        self, split: str, batch_size: int, seq_len: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(入力 [B, T], 次文字ラベル [B, T]) を返す。"""
        src = self.train if split == "train" else self.val
        idx = torch.randint(0, len(src) - seq_len - 1, (batch_size,), generator=generator)
        x = torch.stack([src[i : i + seq_len] for i in idx])
        y = torch.stack([src[i + 1 : i + seq_len + 1] for i in idx])
        return x, y


def load_corpus(path: str | None = None, seed: int = 0) -> CharCorpus:
    """path が None なら合成コーパス、あればそのテキストファイルを読む。"""
    if path is None:
        return CharCorpus(synthetic_corpus(seed=seed))
    return CharCorpus(Path(path).read_text(encoding="utf-8"))
