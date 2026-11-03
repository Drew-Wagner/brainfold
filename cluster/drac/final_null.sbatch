#!/bin/bash
# Label-permutation null of a finished nested search's final models (recipes/moabb/final_null.py):
# 100 permutations of the training labels within runs, per subject, trained as the final
# evaluation (augmented) and tested on the held-out session's true labels.
#   sbatch --account=def-ravanelm_gpu --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 \
#     --dependency=afterok:<search job> --export=ALL,SB_CONFIG=bciciv1_eegnet,SEARCH_DIR=<its heldout dir> \
#     cluster/drac/final_null.sbatch
# Time, BCI Competition IV data set 1: 400 augmented models on 200 training trials. On the RTX 4080
# SUPER a step-2 trial (24 augmented models, 133 training trials, 625 epochs) took 59 s, so about
# 30 min for typical hyperparameters; the 2g.20gb slice took about twice the 4080's time on the cue
# checks' null (job 2042573): about 1 h, and 2 h 30 min leaves room for a larger chosen model.
#SBATCH --job-name=brainfold-final-null
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/drac/env.sh"
source "$BRAINFOLD_VENV/bin/activate"
config=${SB_CONFIG:?set SB_CONFIG, e.g. bciciv1_eegnet}
cd "$REPO/recipes/moabb"
python final_null.py "hparams/sb_moabb/$config.yaml" --search "${SEARCH_DIR:?set SEARCH_DIR}" --permutations 100 "$@"
