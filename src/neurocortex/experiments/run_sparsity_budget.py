"""イベント駆動で省略可能な計算量の試算（12.3.1節 2026-09-22 の設計訂正の検証）。

「発火しないニューロンは計算コストを消費しない」というイベント駆動性が成立するのは、
**スパイクが行列積の入力になっている場合に限られる**。本スクリプトは SpikingLM の
線形演算（重み行列積）を「スパイクを入力に取るもの」と「連続値を入力に取るもの」に
分類し、実測の発火率を掛けて削減率を求める。

計上の方針:
  - 重み行列積の MAC 数のみを数える（トークンあたり）。LayerNorm・残差加算・
    サロゲート・膜電位更新は要素ごとの演算であり、行列積に対して桁が小さいので除く
  - 線形アテンションの状態更新（k⊗v の外積と q との縮約）は**重みを持たない**演算で
    あり、入力は q, k, v（連続値）なので、イベント駆動では省略できない。別枠で計上する
  - 出力ヘッド（語彙への射影）と埋め込みは、配置に依らず密である。別枠で計上する

削減率の定義: ある行列積の入力がスパイク列で発火率 r のとき、非ゼロ入力だけを処理する
実装では MAC 数が r 倍になる。したがって省略できるのは (1 - r) 倍である。

使い方:
    python -m neurocortex.experiments.run_sparsity_budget \
        --weights results/weights/spiking.npz --data docs/brain-structure-research.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..data import load_corpus
from ..lm import SpikingLM
from ..neurons import NeuronConfig


def block_sites(d: int) -> list[dict]:
    """1ブロックあたりの重み行列積サイト（トークンあたりMAC数と入力の種別）。

    新配置 `x → LN → ALIF → Linear` では qkv・fc1・fc2 がスパイクを入力に取る。
    旧配置 `Linear → ALIF` では fc2 のみであった（両方を計上して比較する）。
    """
    return [
        # name, MAC/token, 新配置でスパイク入力か, 旧配置でスパイク入力か, 発火率の参照先
        {"name": "qkv", "macs": 3 * d * d, "spike_new": True, "spike_old": False, "act": "act_attn"},
        {"name": "attn.out", "macs": d * d, "spike_new": False, "spike_old": False, "act": None},
        {"name": "fc1", "macs": 4 * d * d, "spike_new": True, "spike_old": False, "act": "act_fc1"},
        {"name": "fc2", "macs": 4 * d * d, "spike_new": True, "spike_old": True, "act": "act_fc2"},
    ]


@torch.no_grad()
def measure_firing_rates(model: SpikingLM, corpus, seq_len: int, batch_size: int,
                         n_batches: int, seed: int) -> list[list[float]]:
    """ブロックごと・サイトごとの発火率 [n_layers][3] を実データで測る。"""
    g = torch.Generator()
    g.manual_seed(seed)
    model.eval()
    acc: list[list[float]] = [[0.0, 0.0, 0.0] for _ in model.blocks]
    for _ in range(n_batches):
        x, _ = corpus.batch("val", batch_size, seq_len, g)
        model(x)
        for i, blk in enumerate(model.blocks):
            for j, name in enumerate(("act_attn", "act_fc1", "act_fc2")):
                acc[i][j] += float(getattr(blk, name).last_firing_rate)
    return [[v / n_batches for v in row] for row in acc]


def main() -> None:
    p = argparse.ArgumentParser(description="イベント駆動で省略可能な計算量の試算")
    p.add_argument("--weights", default="results/weights/spiking.npz",
                   help="学習済み重み(.npz)。無ければランダム初期化で測る")
    p.add_argument("--data", default="docs/brain-structure-research.md")
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--max-len", type=int, default=128,
                   help="位置埋め込みの長さ。export_weights の既定(128)に合わせる")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-batches", type=int, default=8)
    p.add_argument("--beta", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--out", default="results/sparsity_budget.json")
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    corpus = load_corpus(args.data, seed=args.seed)
    torch.manual_seed(args.seed)
    model = SpikingLM(corpus.vocab_size, d_model=args.d_model, n_layers=args.n_layers,
                      n_heads=args.n_heads, max_len=args.max_len,
                      cfg=NeuronConfig(beta=args.beta))
    wpath = Path(args.weights)
    loaded = False
    if wpath.exists():
        z = np.load(wpath)
        sd = {k: torch.from_numpy(z[k]) for k in z.files}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        loaded = not missing and not unexpected
        if not loaded:
            print(f"警告: 重みの形が一致しない（missing={list(missing)}, "
                  f"unexpected={list(unexpected)}）。ランダム初期化で続行する", flush=True)
    else:
        print(f"警告: {wpath} が無いのでランダム初期化で測る", flush=True)

    rates = measure_firing_rates(model, corpus, args.seq_len, args.batch_size,
                                 args.n_batches, args.seed)

    d = args.d_model
    sites = block_sites(d)
    rows = []
    tot = new_spike = old_spike = 0.0
    new_saved = old_saved = 0.0
    for li in range(args.n_layers):
        for s in sites:
            r = rates[li][("act_attn", "act_fc1", "act_fc2").index(s["act"])] if s["act"] else None
            m = float(s["macs"])
            tot += m
            if s["spike_new"]:
                new_spike += m
                new_saved += m * (1.0 - r)
            if s["spike_old"]:
                old_spike += m
                old_saved += m * (1.0 - r)
            rows.append({"layer": li, "site": s["name"], "macs_per_token": s["macs"],
                         "firing_rate": r, "spike_input_new": s["spike_new"],
                         "spike_input_old": s["spike_old"]})

    # 別枠: 重みを持たない線形アテンションの状態演算と、出力ヘッド
    d_head = d // args.n_heads
    attn_state = args.n_layers * args.n_heads * (3 * d_head * d_head)  # k⊗v, q·S, q·z
    head = d * corpus.vocab_size

    res = {
        "args": vars(args),
        "weights_loaded": loaded,
        "vocab_size": corpus.vocab_size,
        "firing_rates_per_layer": rates,
        "mean_firing_rate": float(np.mean(rates)),
        "sites": rows,
        "weight_macs_per_token_blocks": tot,
        "aux_macs_per_token": {"linear_attention_state": attn_state, "output_head": head},
        "new_placement": {
            "spike_input_share": new_spike / tot,
            "skippable_share_of_block_macs": new_saved / tot,
            "skippable_share_including_aux": new_saved / (tot + attn_state + head),
        },
        "old_placement": {
            "spike_input_share": old_spike / tot,
            "skippable_share_of_block_macs": old_saved / tot,
            "skippable_share_including_aux": old_saved / (tot + attn_state + head),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    def pct(x: float) -> str:
        return f"{100 * x:.1f}%"

    print(f"平均発火率: {pct(res['mean_firing_rate'])}")
    print(f"ブロック内の重み行列積: {tot:,.0f} MAC/token "
          f"(別枠: アテンション状態 {attn_state:,} / ヘッド {head:,})")
    for tag in ("old_placement", "new_placement"):
        r = res[tag]
        print(f"[{tag}] スパイク入力の占める割合 {pct(r['spike_input_share'])} / "
              f"省略可能 {pct(r['skippable_share_of_block_macs'])}（ブロック内） / "
              f"{pct(r['skippable_share_including_aux'])}（別枠込み）")
    print(f"保存: {args.out}")


if __name__ == "__main__":
    main()
