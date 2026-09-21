#!/bin/bash
# Rust 3モードを語彙1028の重み・1コアで電力込み測定（12.11.3節の条件統一）
cd ~/neurocortex
rm -f p_1core_r2_rust_.csv
B=./rust/target/release/neurocortex-rs
for M in event spiking-dense dense; do
  IT=150
  [ "$M" = event ] && IT=400
  echo "### rust $M iters=$IT vocab=1028"
  OMP_NUM_THREADS=1 python3 tools/powerlog.py --out "p_1core_r2_rust_${M}.csv" -- \
    taskset -c 0 "$B" bench --mode "$M" --tokens results/rust/reference_r2.npz \
      --weights-dir weights_r2_pre --vocab 1028 --iters "$IT"
done
