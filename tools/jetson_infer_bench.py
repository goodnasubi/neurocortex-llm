"""Jetson上で学習済みLMを推論させ、トークンあたりのエネルギーを測る（12.6節 副指標）。

powerlog.py から --  の後ろに置いて実行されることを想定。
標準出力の最後に JSON を1行出す。
"""
from __future__ import annotations
import argparse, json, sys, time
import numpy as np
import torch

sys.path.insert(0, "/home/ikeda/neurocortex/src")
from neurocortex.lm import DenseBaseline, SpikingLM, count_params
from neurocortex.neurons import NeuronConfig


def load(tag, meta, device):
    a = meta["args"]; v = meta["vocab_size"]
    if tag == "spiking":
        m = SpikingLM(v, d_model=a["d_model"], n_layers=a["n_layers"],
                      n_heads=a["n_heads"], cfg=NeuronConfig(beta=a["beta"]))
    else:
        m = DenseBaseline(v, d_model=a["d_model"], n_layers=a["n_layers"], n_heads=a["n_heads"])
    z = np.load(f"/home/ikeda/neurocortex/weights/{tag}.npz")
    sd = {k: torch.from_numpy(z[k]) for k in z.files}
    m.load_state_dict(sd)
    return m.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["spiking", "dense"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    meta = json.load(open("/home/ikeda/neurocortex/weights/meta.json"))
    dev = torch.device(a.device)
    m = load(a.tag, meta, dev)
    g = torch.Generator(); g.manual_seed(0)
    x = torch.randint(0, meta["vocab_size"], (a.batch_size, a.seq_len), generator=g).to(dev)

    with torch.no_grad():
        for _ in range(3):
            m(x)
        if a.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(a.iters):
            m(x)
        if a.device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0

    tokens = a.iters * a.batch_size * a.seq_len
    res = {"tag": a.tag, "device": a.device, "params": count_params(m), "tokens": tokens,
           "sec": round(dt, 3), "tokens_per_sec": round(tokens / dt, 1)}
    if a.tag == "spiking":
        res["firing_rates"] = [round(r, 4) for r in m.firing_rates() if r is not None]
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
