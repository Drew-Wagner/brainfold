#!/bin/bash
# Training time and peak memory per pack size K (recipes/moabb/sweep.py), one
# recipe configuration per array task, on one MIG slice. Run prepare.sbatch first.
#   sbatch --account=def-yourpi --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --array=0-11 cluster/drac/sweep.sbatch
# Array index i runs CONFIGS[i] from env.sh. Extra arguments go to sweep.py.
#SBATCH --job-name=brainfold-sweep
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/drac/env.sh"
source "$BRAINFOLD_VENV/bin/activate"
config=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
out=$BRAINFOLD_RESULTS/sweep-$SLURM_ARRAY_JOB_ID
mkdir -p "$out"
python "$REPO/recipes/moabb/sweep.py" "hparams/$config.yaml" --cache-dir "$BRAINFOLD_CACHE" \
  --out "$out/$config.csv" --wandb "$BRAINFOLD_WANDB_PROJECT" "$@"
