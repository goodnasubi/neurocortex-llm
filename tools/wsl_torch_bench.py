"""WSL2(torch 2.14) 上で PyTorch 版のスループットを測る（Rust実装との比較基準）。

入力トークンは Rust ベンチと同じ results/rust/reference_corpus.npz を使う。
最後に密な行列積の素の性能（GFLOPS）も出す。自作の Rust カーネルが
手抜きでないかを開示するための校正値（12.11.2節 公平性）。

使い方: PYTHONPATH=src python tools/wsl_torch_bench.py [スレッド数]
"""
import json,sys,time,numpy as np,torch
from pathlib import Path
sys.path.insert(0,"src")
from neurocortex.lm import DenseBaseline, SpikingLM
from neurocortex.neurons import NeuronConfig
th=int(sys.argv[1]) if len(sys.argv)>1 else 1
torch.set_num_threads(th)
meta=json.loads(Path("results/weights/meta.json").read_text());a=meta["args"];v=meta["vocab_size"]
z=np.load("results/rust/reference_corpus.npz");tok=torch.from_numpy(z["tokens"]).long()
for tag in ("spiking","dense"):
    m=(SpikingLM(v,d_model=a["d_model"],n_layers=a["n_layers"],n_heads=a["n_heads"],cfg=NeuronConfig(beta=a["beta"]))
       if tag=="spiking" else DenseBaseline(v,d_model=a["d_model"],n_layers=a["n_layers"],n_heads=a["n_heads"]))
    w=np.load(f"results/weights/{tag}.npz");m.load_state_dict({k:torch.from_numpy(w[k]) for k in w.files});m.eval()
    it=10 if tag=="spiking" else 60
    with torch.no_grad():
        for _ in range(3): m(tok)
        t0=time.time()
        for _ in range(it): m(tok)
        dt=time.time()-t0
    n=it*tok.numel()
    print(f"RESULT {json.dumps({'impl':'pytorch','mode':tag,'threads':th,'tokens':n,'sec':round(dt,3),'tokens_per_sec':round(n/dt,1)})}")
# 密な行列積のピーク（比較の校正用）
x=torch.randn(64,192);W=torch.randn(768,192)
t0=time.time();n=0
while time.time()-t0<2.0:
    for _ in range(200): x@W.T
    n+=200
dt=time.time()-t0
print(f"CALIB torch linear 64x192x768: {2*64*192*768*n/dt/1e9:.2f} GFLOPS ({th} threads)")
