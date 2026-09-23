cd /mnt/c/Users/tks_i/Claude/Projects/neurocortex-llm
source ~/venvs/neurocortex/bin/activate
export OMP_NUM_THREADS=4 PYTHONPATH=src
for noise in 0.0 0.02 0.1; do
  python -m neurocortex.experiments.run_hippocampus --tap block0.attn \
     --counts 1 10 100 1000 10000 --seeds 5 --cue-noise "$noise" \
     --out "results/hippocampus/recall_curve_block0attn_noise${noise}.json" \
     > "results/hippocampus/run_block0attn_noise${noise}.log" 2>&1
  echo "=== noise=$noise ==="; tail -8 "results/hippocampus/run_block0attn_noise${noise}.log"
done
