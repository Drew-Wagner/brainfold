#!/bin/bash
# Preprocess and cache every configuration's trials, on CPU, before the GPU jobs.
#   sbatch --account=def-yourpi cluster/drac/prepare.sbatch
#SBATCH --job-name=brainfold-prepare
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G  # Lee2019_ERP: 14 GB of trials, stacked into one array
#SBATCH --time=4:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/drac/env.sh"
source "$BRAINFOLD_VENV/bin/activate"
for config in "${CONFIGS[@]}"; do
  python "$REPO/recipes/moabb/train.py" "hparams/$config.yaml" --prepare-only --cache-dir "$BRAINFOLD_CACHE"
done
