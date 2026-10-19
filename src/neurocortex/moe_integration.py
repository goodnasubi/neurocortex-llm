"""領域をまたぐ統合が要る課題で、ハブ専門家の性能上の利点を調べる部品（12.6.199節）。

皮質のハブ（rich club）は領域をまたぐ情報の統合を担う（van den Heuvel & Sporns, 2011, 2013;
Bullmore & Sporns, 2012; Zamora-López et al., 2010）。12.6.196節では単一の分布の文字 LM でハブの利点が
見えなかったので、複数の言語（2つの族）の混合で、片方の領域（言語 i の前半）の手がかりで、もう片方の
領域（言語 j の後半）の続きが決まる課題を作る。

1行 = "a1|b1 b2\\n"。a1 は言語 i、b1・b2 は言語 j の語。語は概念の番号 c ∈ [0, n_concepts) を
言語ごとの語彙（同源語。`make_lexicons`）で表したもの。b1 は無作為、b2 の概念は π(c(a1))（全言語で共通の
置換 π）。with_a2=True で前半に無作為な語 a2 を挟む（"a1 a2|b1 b2"。12.6.199節の予備調整で学べなかった形）。
- 統合課題（cross）: j ≠ i。b2 を当てるには、前半の領域の概念（a1）と後半の領域の言語（b1）の結合が要る。
- 対照課題（within）: j = i。依存の距離・置換・語彙は同じで、手がかりと続きが同じ領域の中にある。

言語は文字の集合で分かれる（族ごとに母音、言語ごとに子音）ので、領域は入力の文字から分かる。
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F

from .moe_scale import ScaledRoutedCharLM

# 族 0 = 母音 "ae"、族 1 = 母音 "ou"。言語ごとに子音の集合が違う（文字は言語の間で重ならない子音と、族で共有の母音）
LANGS = (("ptk", "ae"), ("bdg", "ae"), ("mnl", "ou"), ("srv", "ou"))
SLOT_A1, SLOT_A2, SLOT_B1, SLOT_B2, SLOT_SEP = 0, 1, 2, 3, 4


def make_lexicons(n_concepts: int, rng: random.Random) -> list[list[str]]:
    """言語ごとに CVCV の語を n_concepts 個。同じ概念の語は全言語で子音・母音の番号が同じ（同源語）。

    概念 c の語 = cons[p] vow[q] cons[r] vow[s]（(p, q, r, s) は概念ごとに共通、文字は言語ごと）。
    言語の間の対応は体系的なので、概念は番号の型として言語によらず表せる。
    """
    n_c, n_v = len(LANGS[0][0]), len(LANGS[0][1])
    shapes = [(p, q, r, s) for p in range(n_c) for q in range(n_v) for r in range(n_c) for s in range(n_v)]
    if n_concepts > len(shapes):
        raise ValueError(n_concepts)
    chosen = rng.sample(shapes, n_concepts)
    return [[cons[p] + vows[q] + cons[r] + vows[s] for p, q, r, s in chosen] for cons, vows in LANGS]


def integration_corpus(task: str, n_lines: int = 8000, n_concepts: int = 8, seed: int = 0,
                       with_a2: bool = False) -> tuple[str, np.ndarray, np.ndarray]:
    """(テキスト, 文字ごとの枠の番号, 文字ごとの領域（言語）の番号)。

    枠: a1=0, a2=1, b1=2, b2=3, 区切り（空白・"|"・改行）=4。領域: 前半の文字と "|" は i、後半と改行は j。
    語彙と置換 π はシードで決まり、task が違っても同じシードなら同じ。
    """
    if task not in ("cross", "within"):
        raise ValueError(task)
    rng = random.Random(seed)
    lex = make_lexicons(n_concepts, rng)
    perm = list(range(n_concepts))
    rng.shuffle(perm)
    rng_l = random.Random(seed + 7919)  # 行の乱数は task で消費が変わらないよう別にする
    k = len(LANGS)
    chars, slots, langs = [], [], []

    def put(s: str, slot: int, lang: int) -> None:
        chars.append(s)
        slots.extend([slot] * len(s))
        langs.extend([lang] * len(s))

    for _ in range(n_lines):
        i = rng_l.randrange(k)
        off = rng_l.randrange(1, k)
        j = (i + off) % k if task == "cross" else i
        c1, c2, c3 = rng_l.randrange(n_concepts), rng_l.randrange(n_concepts), rng_l.randrange(n_concepts)
        put(lex[i][c1], SLOT_A1, i)
        if with_a2:
            put(" ", SLOT_SEP, i)
            put(lex[i][c2], SLOT_A2, i)
        put("|", SLOT_SEP, i)
        put(lex[j][c3], SLOT_B1, j)
        put(" ", SLOT_SEP, j)
        put(lex[j][perm[c1]], SLOT_B2, j)
        put("\n", SLOT_SEP, j)
    return "".join(chars), np.asarray(slots, dtype=np.int64), np.asarray(langs, dtype=np.int64)


def split_like_corpus(arr: np.ndarray, val_fraction: float = 0.1) -> np.ndarray:
    """`CharCorpus` と同じ分割で検証側を返す。"""
    n_val = int(len(arr) * val_fraction)
    return arr[-n_val:]


def windows_like(arr: np.ndarray, seq_len: int, n_windows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`hub_load.make_windows` と同じ窓で、入力側 [N,T] と目標側 [N,T] の注釈を返す。"""
    n = min(n_windows, (len(arr) - 1) // seq_len)
    t = torch.as_tensor(arr)
    xs = torch.stack([t[i * seq_len:(i + 1) * seq_len] for i in range(n)])
    ys = torch.stack([t[i * seq_len + 1:(i + 1) * seq_len + 1] for i in range(n)])
    return xs, ys


@torch.no_grad()
def slot_ce(model: ScaledRoutedCharLM, x: torch.Tensor, y: torch.Tensor, y_slot: torch.Tensor, **kw) -> dict:
    """枠ごとの CE（目標の文字の枠で分ける）。"target" = b2、"other" = b2 以外、"all" = 全体。"""
    logits = model.run(x, **kw)["logits"]
    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none")
    s = y_slot.reshape(-1)
    out = {"all": float(nll.mean()), "target": float(nll[s == SLOT_B2].mean()),
           "other": float(nll[s != SLOT_B2].mean())}
    for name, v in (("a1", SLOT_A1), ("a2", SLOT_A2), ("b1", SLOT_B1), ("sep", SLOT_SEP)):
        out[name] = float(nll[s == v].mean())
    return out


def participation(choice: np.ndarray, label: np.ndarray, n_experts: int, n_labels: int) -> np.ndarray:
    """専門家ごとの参加係数 P_e = (1 − Σ_s (k_es / k_e)^2) / (1 − 1/S)（Guimerà & Amaral, 2005 を [0,1] に正規化）。

    k_es = 専門家 e に送られた領域 s のトークン数。全領域から均等に受けると 1、1領域だけなら 0。
    使われない専門家は nan。
    """
    k = np.zeros((n_experts, n_labels))
    np.add.at(k, (choice.ravel(), label.ravel()), 1)
    tot = k.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = 1 - ((k / tot[:, None]) ** 2).sum(1)
    p = p / (1 - 1 / n_labels)
    p[tot == 0] = np.nan
    return p


def hub_participation(load: np.ndarray, p: np.ndarray, k: int) -> tuple[float, float]:
    """負荷上位 k 個（ハブ）と残り（使われた専門家のみ）の参加係数の平均。"""
    order = sorted(range(len(load)), key=lambda e: (-load[e], e))
    hub, rest = p[order[:k]], p[order[k:]]
    rest = rest[~np.isnan(rest)]
    return float(np.nanmean(hub)), float(rest.mean()) if len(rest) else float("nan")


def hub_target_share(choice: np.ndarray, y_slot: np.ndarray, load: np.ndarray, k: int) -> tuple[float, float]:
    """ハブ（負荷上位 k 個）に送られた割合: b2 を予測する位置での割合と、全位置での割合。"""
    order = sorted(range(len(load)), key=lambda e: (-load[e], e))
    is_hub = np.isin(choice, order[:k])
    tgt = y_slot == SLOT_B2
    return float(is_hub[tgt].mean()), float(is_hub.mean())
