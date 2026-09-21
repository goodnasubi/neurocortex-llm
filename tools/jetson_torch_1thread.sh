#!/bin/bash
# PyTorch 側を Rust と同じ「1コアのみ」で測り直す（スレッド数を揃えた比較のため）。
# torch.set_num_threads(1) だけでは oneDNN の OMP プールが 4 コアを使ってしまうので、
# OMP_NUM_THREADS も 1 に固定する（実測: 4761 tok/s → 1827 tok/s と大きく変わる）。
cd ~/neurocortex
run() {
  echo "### torch $1 device=$2 OMP=1 iters=$3"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 tools/powerlog.py --out p_torch_$1_$2_omp1.csv -- \
    python3 tools/infer_bench.py --tag $1 --device $2 --iters $3 --threads 1 2>&1 |
    grep -E "RESULT|powerlog"
}
run spiking cpu "${1:-12}"
run dense cpu "${2:-50}"
