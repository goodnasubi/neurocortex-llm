"""Rust実装の正当性検証に使う参照出力を PyTorch から書き出す（12.11.2節）。

Jetson の torch 1.12 CPU ビルドは素の nn.Linear が散発的に非有限値を返すため、
参照は **WSL2 の torch 2.14** で取る（12.11.2節の注記）。

出力 `results/rust/reference.npz`:
    tokens          [B, T] int64          … Rust側のベンチでも同じ列を使う
    spiking_logits  [B, T, V] float32
    dense_logits    [B, T, V] float32
    spikes          [B, L*(T*d + T*d + T*4d)] float32
                     … 系列ごとに layer→(act_attn, act_fc1, act_fc2) の順で平坦化。
                       Rust 側の SpikeLog と同じ並びにしてある

使い方:
    PYTHONPATH=src python tools/dump_reference.py --out results/rust/reference.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from neurocortex.lm import DenseBaseline, SpikingLM
from neurocortex.neurons import ALIFNeuron, NeuronConfig


def build(tag: str, meta: dict):
    a, v = meta["args"], meta["vocab_size"]
    if tag == "spiking":
        m = SpikingLM(v, d_model=a["d_model"], n_layers=a["n_layers"], n_heads=a["n_heads"],
                      max_len=a.get("max_len", 128), cfg=NeuronConfig(beta=a["beta"]))
    else:
        m = DenseBaseline(v, d_model=a["d_model"], n_layers=a["n_layers"], n_heads=a["n_heads"],
                          max_len=a.get("max_len", 128))
    z = np.load(f"{meta['_wdir']}/{tag}.npz")
    m.load_state_dict({k: torch.from_numpy(z[k]) for k in z.files})
    return m.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", default="results/weights")
    ap.add_argument("--out", default="results/rust/reference.npz")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--data", default=None,
                    help="実テキストから val バッチを取る（未指定なら randint。"
                         "12.6.3節の発火率9.8%は実テキストでの値）")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    meta = json.loads(Path(args.weights_dir, "meta.json").read_text(encoding="utf-8"))
    meta["_wdir"] = args.weights_dir
    vocab = meta["vocab_size"]

    g = torch.Generator()
    g.manual_seed(args.seed)
    if args.data:
        from neurocortex.data import load_corpus
        corpus = load_corpus(args.data, seed=args.seed)
        if corpus.vocab_size != vocab:
            print(f"注意: コーパスの語彙数 {corpus.vocab_size} が重みの {vocab} と異なる"
                  f"（本文書の更新による。トークンは語彙内にクリップする）")
        tokens, _ = corpus.batch("val", args.batch_size, args.seq_len, g)
        tokens = tokens.clamp_max(vocab - 1)
    else:
        tokens = torch.randint(0, vocab, (args.batch_size, args.seq_len), generator=g)

    payload: dict[str, np.ndarray] = {"tokens": tokens.numpy().astype(np.int64)}
    rates: list[float] = []

    # float32 と float64 の両方で参照を取る。
    # スパイキングモデルは閾値判定を含むため、和の順序が変わるだけで
    # ごく一部のユニットの発火が反転する。どちらも「正しいfloat32実装」でありうるので、
    # Rust 実装はこの2つの参照の**いずれか**と厳密一致すればよい、という基準にする。
    for dtype, suffix in ((torch.float32, ""), (torch.float64, "_f64")):
        spiking = build("spiking", meta).to(dtype)
        dense = build("dense", meta).to(dtype)
        captured: list[torch.Tensor] = []
        handles = []
        for blk in spiking.blocks:
            for name in ("act_attn", "act_fc1", "act_fc2"):
                mod = getattr(blk, name)
                assert isinstance(mod, ALIFNeuron)
                handles.append(mod.register_forward_hook(
                    lambda _m, _i, out: captured.append(out.detach().float().clone())))
        with torch.no_grad():
            sp_logits = spiking(tokens)
            dn_logits = dense(tokens)
        for h in handles:
            h.remove()
        for arr, name in ((sp_logits, "spiking"), (dn_logits, "dense")):
            if not torch.isfinite(arr).all():
                raise SystemExit(f"{name} の logits に非有限値がある（12.11.2節の有限性検査）")
        # 系列ごとに [layer][site] の順で平坦化（Rust の SpikeLog と同じ並び）
        spikes = np.stack([
            np.concatenate([c[b].reshape(-1).numpy() for c in captured])
            for b in range(args.batch_size)
        ])
        uniq = np.unique(spikes)
        if not set(uniq.tolist()) <= {0.0, 1.0}:
            raise SystemExit(f"スパイク列が二値でない: {uniq[:10]}")
        payload[f"spiking_logits{suffix}"] = sp_logits.float().numpy()
        payload[f"dense_logits{suffix}"] = dn_logits.float().numpy()
        payload[f"spikes{suffix}"] = spikes.astype(np.float32)
        if not suffix:
            rates = [round(float(c.mean()), 4) for c in captured]

    n_flip = int((payload["spikes"] != payload["spikes_f64"]).sum())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)
    print(f"保存: {out}  tokens={tuple(tokens.shape)} spikes={payload['spikes'].shape} "
          f"平均発火率={float(payload['spikes'].mean()):.4f}")
    print("層別発火率:", rates)
    print(f"float32 と float64 で発火が反転したユニット: {n_flip} / {payload['spikes'].size}")


if __name__ == "__main__":
    main()
