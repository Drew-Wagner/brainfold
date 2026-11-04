#!/bin/bash
# Label-permutation null for the cue checks (recipes/moabb/cue_checks/cue_null.py): 100 permutations of the
# training labels within runs, per subject and held-out session, for each held-out session's chosen
# configuration. Array index 0: EEG with the configuration's own band; 1: the 3 EOG channels above 8 Hz.
#   sbatch --account=def-ravanelm_gpu --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --array=0-1 cluster/drac/cue_null.sbatch
# Time: 1800 models per task. On the RTX 4080 SUPER, 27 of these models trained in 70-80 s, so about
# 80 min per task; the 2g.20gb slice took 0.8-1.5x the 4080's time on the recipes: up to about 2 h.
#SBATCH --job-name=brainfold-cue-null
#SBATCH --gpus=1
#SBATCH --cpus-per-task=3  # billed as the max of CPUs, memory and GPU: 3 CPUs stay under a 2g.20gb slice (see CLAUDE.md)
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/drac/env.sh"
source "$BRAINFOLD_VENV/bin/activate"
variants=(eeg eog_fmin8)
variant=${variants[$SLURM_ARRAY_TASK_ID]}
out=$BRAINFOLD_RESULTS/cue-null-${RESULTS_ID:-$SLURM_ARRAY_JOB_ID}
mkdir -p "$out"
cd "$REPO/recipes/moabb"
python cue_checks/cue_null.py --variant "$variant" --permutations 100 --pack-size 54 \
  --out "$out/cue_null_$variant.csv" "$@"
