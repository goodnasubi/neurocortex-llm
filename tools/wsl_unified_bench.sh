#!/bin/bash
# x86(WSL2) 側のスループットを12.11.3節の統一条件で測る。
#   - 語彙1028の同一重み（results/weights_r2_pre）
#   - 同一トークン列（results/rust/reference_r2.npz）
#   - 1スレッド固定（taskset -c 0 + OMP_NUM_THREADS=1）
#   - 測定区間を約100秒に揃える（反復数固定だと実装ごとに窓の長さが変わる）
set -e
cd "$(dirname "$0")/.."
BIN="${BIN:-$HOME/.cache/ncrs-verify/x86_64-unknown-linux-musl/release/neurocortex-rs}"
SECS="${SECS:-100}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Rust: tokens/sec が既知なので、約 $SECS 秒になる反復数を与える
run_rust() {
  mode=$1; iters=$2
  taskset -c 0 "$BIN" bench --mode "$mode" --tokens results/rust/reference_r2.npz \
    --weights-dir results/weights_r2_pre --vocab 1028 --iters "$iters"
}
taskset -c 0 "$BIN" calib
run_rust event         "${EV_IT:-4200}"
run_rust spiking-dense "${DN_IT:-2600}"
run_rust dense         "${DN_IT:-2600}"
taskset -c 0 python tools/wsl_torch_bench.py 1 results/weights_r2_pre \
  results/rust/reference_r2.npz "$SECS"
