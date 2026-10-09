#!/bin/bash
# Timing benchmark (benchmarks/bench_packing.py) on one whole GPU: every model
# and training-set size, 5 repeats each, with the environment recorded.
#   sbatch --account=def-yourpi --gpus=h100:1 cluster/drac/bench.sbatch
#SBATCH --job-name=brainfold-bench
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/drac/env.sh"
source "$BRAINFOLD_VENV/bin/activate"
out=$BRAINFOLD_RESULTS/bench-$SLURM_JOB_ID
mkdir -p "$out"
nvidia-smi -q > "$out/nvidia-smi.txt"
pip freeze > "$out/pip-freeze.txt"
for model in eegnet atcnet; do
  for n_trials in 288 4608; do
    python "$REPO/benchmarks/bench_packing.py" --model $model --n-trials $n_trials --repeats 5 \
      --out "$out/$model-$n_trials.csv"
  done
done
