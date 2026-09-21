#!/bin/bash
# 12.6.3節のCPUエネルギー測定を「1コア固定」でやり直す（12.11.2節の訂正を受けて）。
#
# 既定の4コア実行では torch 1.12 の CPU カーネルが非有限値を返し、測定値が壊れる。
# OMP_NUM_THREADS=1 に固定すると発火率がWSL2と一致し、数値的に健全になる。
# infer_bench.py は出力の有限性を毎回検査し、壊れていれば終了コード3で落ちる。
#
# 旧配置(post)と新配置(pre)を同一コーパスで学習し直した重みで比較する。
#   bash jetson_cpu_1core_rebench.sh
cd ~/neurocortex
run() {
  tag=$1; wdir=$2; label=$3; iters=$4
  echo "### $label / tag=$tag / OMP=1 / iters=$iters"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 tools/powerlog.py --out p_1core_${label}_${tag}.csv -- \
    python3 tools/infer_bench.py --tag $tag --device cpu --iters $iters --threads 1 \
      --weights-dir "$wdir" 2>&1 | grep -E "RESULT|powerlog|警告"
}
run spiking "$HOME/neurocortex/weights_r2_pre"  pre  "${1:-12}"
run spiking "$HOME/neurocortex/weights_r2_post" post "${1:-12}"
# 密なベースラインは配置に依存しないので1回でよい
run dense   "$HOME/neurocortex/weights_r2_pre"  pre  "${2:-40}"
