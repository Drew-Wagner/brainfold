#!/bin/bash
# One recipe configuration and seed per array task, on one MIG slice; seed 0
# also trains the braindecode baseline. Run prepare.sbatch first.
#   sbatch --account=def-yourpi --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --array=0-23,30-35 cluster/drac/recipes.sbatch
#   sbatch --account=def-yourpi --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --array=24-29 cluster/drac/recipes.sbatch  # Lee2019_ERP
# Array index i runs CONFIGS[i / 3] from env.sh with seed i % 3. Extra
# arguments go to train.py and apply to every task; set each configuration's
# train.pack_size in its hparams file.
#SBATCH --job-name=brainfold-recipes
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/drac/env.sh"
source "$BRAINFOLD_VENV/bin/activate"
n_seeds=3
config=${CONFIGS[$((SLURM_ARRAY_TASK_ID / n_seeds))]}
seed=$((SLURM_ARRAY_TASK_ID % n_seeds))
baseline=
if [ "$seed" = 0 ]; then baseline=--baseline; fi  # one seed is enough to compare run by run
out=$BRAINFOLD_RESULTS/recipes-$SLURM_ARRAY_JOB_ID
mkdir -p "$out"
python "$REPO/recipes/moabb/train.py" "hparams/$config.yaml" --set "train.seeds=[$seed]" $baseline \
  --cache-dir "$BRAINFOLD_CACHE" --out "$out/$config-seed$seed.csv" --wandb "$BRAINFOLD_WANDB_PROJECT" "$@"
