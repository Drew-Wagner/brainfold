#!/bin/bash
# One recipe configuration and seed per array task, on one MIG slice. Run
# prepare.sbatch first. Packed models only: add --baseline (passed on to
# train.py) to also train every run alone with braindecode, about 10x longer.
#   sbatch --account=def-yourpi --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --array=0-23 cluster/drac/recipes.sbatch
#   sbatch --account=def-yourpi --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --time=12:00:00 --array=30-35 cluster/drac/recipes.sbatch  # Lee2019_SSVEP
#   sbatch --account=def-yourpi --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --time=12:00:00 --array=24-29 cluster/drac/recipes.sbatch  # Lee2019_ERP
# Array index i runs CONFIGS[i / 3] from env.sh with seed i % 3. Extra
# arguments go to train.py and apply to every task; set each configuration's
# train.pack_size in its hparams file.
# On a 2g.20gb slice (job 63860031), the packed models took at most 10 min
# (Lee2019_MI ATCNet), plus a few minutes of data loading, so 30 min. With
# --baseline, seed 0 took up to 37 min, and Lee2019_MI ATCNet 2 h 26 min: ask
# for 1 h and 4 h. Lee2019_ERP and Lee2019_SSVEP are untimed.
#SBATCH --job-name=brainfold-recipes
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/drac/env.sh"
source "$BRAINFOLD_VENV/bin/activate"
n_seeds=3
config=${CONFIGS[$((SLURM_ARRAY_TASK_ID / n_seeds))]}
seed=$((SLURM_ARRAY_TASK_ID % n_seeds))
out=$BRAINFOLD_RESULTS/recipes-$SLURM_ARRAY_JOB_ID
mkdir -p "$out"
python "$REPO/recipes/moabb/train.py" "hparams/$config.yaml" --set "train.seeds=[$seed]" \
  --cache-dir "$BRAINFOLD_CACHE" --out "$out/$config-seed$seed.csv" --wandb "$BRAINFOLD_WANDB_PROJECT" "$@"
