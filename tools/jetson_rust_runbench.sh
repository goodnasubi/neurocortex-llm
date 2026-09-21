#!/bin/bash
# Jetson 上で Rust 実装3モードのエネルギーを実測する（12.11.1節の手法）。
# Jetson の ~/neurocortex/ に置いて実行する。
#   bash jetson_rust_runbench.sh [iters]
cd ~/neurocortex
BIN=./rust/target/release/neurocortex-rs
TOK=${TOK:-results/rust/reference.npz}   # PyTorch側のベンチと同じ randint(seed=0) のトークン列
run() {
  mode=$1; iters=$2
  echo "### rust $mode iters=$iters tokens=$TOK"
  python3 tools/powerlog.py --out p_rust_${mode}.csv -- \
    $BIN bench --mode $mode --tokens $TOK --weights-dir weights --iters $iters 2>&1 |
    grep -E "RESULT|mW|J|平均|エネルギー"
}
IT_E=${IT_E:-300}
IT_D=${IT_D:-150}
run event "$IT_E"
run spiking-dense "$IT_D"
run dense "$IT_D"
