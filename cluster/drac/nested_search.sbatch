#!/bin/bash
# The nested search (recipes/moabb/nested_search.py) with their Adam: array index i
# searches for, trains on the other sessions, and tests on session i (0 or 1 for BNCI2014001).
#   sbatch --account=def-yourpi --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --array=0-1 cluster/drac/nested_search.sbatch
# Resumable: resubmit the same array index with the same RESULTS_ID to continue, e.g.
#   sbatch ... --array=1 --export=ALL,RESULTS_ID=<first job id> cluster/drac/nested_search.sbatch
# Extra arguments go to nested_search.py (e.g. --optimizer schedule-free).
# Time: on the RTX 4080 SUPER a typical step-1 trial (625 epochs, 16 filters) took
# 57 s for one held-out session's 54 models, against 25 s for sb_search.py's 18;
# sb_search.py's whole Adam run took 3 h 02 min on a 2g.20gb slice (job 65405367_0).
# One held-out session is about 2.3x its search and half its final evaluation:
# about 7 h. The 12 h request is 1.7x that; the search resumes if it doesn't finish.
#SBATCH --job-name=brainfold-sb-nested
#SBATCH --gpus=1
#SBATCH --cpus-per-task=3  # billed as the max of CPUs, memory and GPU: 3 CPUs stay under a 2g.20gb slice (see CLAUDE.md)
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/drac/env.sh"
source "$BRAINFOLD_VENV/bin/activate"
config=${SB_CONFIG:-bnci2014001_eegnet}
held_out=$SLURM_ARRAY_TASK_ID
out=$BRAINFOLD_RESULTS/sb-nested-${RESULTS_ID:-$SLURM_ARRAY_JOB_ID}/$config/heldout$held_out
mkdir -p "$out"
cd "$REPO/recipes/moabb"
python nested_search.py "hparams/sb_moabb/$config.yaml" --out "$out" --set "nested.held_out=[$held_out]" \
  --wandb "$BRAINFOLD_WANDB_PROJECT" "$@"
