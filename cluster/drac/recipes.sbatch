#!/bin/bash
# One recipe configuration per array task, packed and with the braindecode
# baseline, on one whole GPU. Run prepare.sbatch first.
#   sbatch --account=def-yourpi --gpus=h100:1 --array=0-11 cluster/drac/recipes.sbatch
# Array index i runs CONFIGS[i] from env.sh. Extra arguments go to train.py, e.g.
#   sbatch ... cluster/drac/recipes.sbatch --set train.seeds=[0,1,2]
#SBATCH --job-name=brainfold-recipes
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/drac/env.sh"
source "$BRAINFOLD_VENV/bin/activate"
config=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
out=$BRAINFOLD_RESULTS/recipes-$SLURM_ARRAY_JOB_ID
mkdir -p "$out"
nvidia-smi --query-gpu=name,driver_version --format=csv > "$out/$config.gpu.csv"
python "$REPO/recipes/moabb/train.py" "hparams/$config.yaml" --baseline --cache-dir "$BRAINFOLD_CACHE" \
  --out "$out/$config.csv" "$@"
