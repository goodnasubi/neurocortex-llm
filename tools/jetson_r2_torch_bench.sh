#!/bin/bash
# PyTorch側を「実行時間 ≒ 測定窓」になる反復数で測り直す（12.11.3節の条件統一）。
#
# 従来は1回の測定が10〜15秒で、そのうち import torch と重み読み込みに数秒かかるため、
# 測定窓の3〜5割が低電力の起動時間で占められ、平均電力が過小評価されていた。
# Rust側は起動が一瞬なのでこの希釈がなく、**PyTorchに有利な非対称性**になっていた。
# 各条件とも実測区間が約100秒になる反復数にして揃える。
cd ~/neurocortex
run() {
  tag=$1; wdir=$2; label=$3; dev=$4; iters=$5
  echo "### $label / $tag / $dev / iters=$iters"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 tools/powerlog.py --out "p_r2_${label}_${tag}_${dev}.csv" -- \
    python3 tools/infer_bench.py --tag "$tag" --device "$dev" --iters "$iters" --threads 1 \
      --weights-dir "$HOME/neurocortex/$wdir"
}
run spiking weights_r2_pre  pre  cpu  150
run spiking weights_r2_post post cpu  170
run dense   weights_r2_pre  pre  cpu  350
run spiking weights_r2_pre  pre  cuda 130
run spiking weights_r2_post post cuda 170
run dense   weights_r2_pre  pre  cuda 6300
