#!/bin/bash
# Jetson上でエネルギー測定を回す補助スクリプト（12.6.2節の再取得用）。
cd ~/neurocortex
run() {
  echo "### $1 $2 iters=$3"
  python3 tools/powerlog.py --out p_$1_$2.csv -- \
    python3 tools/infer_bench.py --tag $1 --device $2 --iters $3 2>&1 | grep -E "RESULT|powerlog"
}
run spiking cpu 30
run spiking cuda 40
run dense cpu 200
run dense cuda 1000
