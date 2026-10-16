#!/bin/bash
# The improved search protocol (recipes/moabb/PROTOCOL_IDEAS.md) with schedule-free
# AdamW: step 0 (preprocessing), step 1 (architecture and training), step 2 (EEG
# augmentation), 50 TPE trials each, then 10 seeds over the folds. Selection is nested
# by test session: one search per test session, on the folds that test on it, run as
# separate array tasks. Task i searches for session SESSIONS[i % n] in replicate
# search i / n (idea 7: replicate r > 0 uses TPE seed r and trial seeds from r;
# replicate 0 uses the yaml's seed, 1986).
#   sbatch --account=def-yourpi --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --array=0-1 cluster/drac/sb_improved.sbatch
#   sbatch ... --array=2-5 cluster/drac/sb_improved.sbatch   # replicate searches 1 and 2
# SESSIONS lists the dataset's sessions (MOABB's names; BNCI2014001's by default).
# Resumable: resubmit the same array index with the same RESULTS_ID to continue an
# interrupted search (the trial that was running is redone), e.g.
#   sbatch ... --array=1 --export=ALL,RESULTS_ID=<first job id> cluster/drac/sb_improved.sbatch
# Extra arguments go to sb_search.py.
# Time, per task (one test session), from the RTX 4080 SUPER (README, "The improved
# protocol"; the 2g.20gb slice took 0.8-1.5x its time on the recipes) and job 65405367:
# step 0, 50 trials of EEGNet-8,2 on 54 cells, about 70 s training (139 s measured
# for 108 cells) and 13 s preprocessing each: 1.2 h. Step 1: their step 1 took about 1.2 h on 2g.20gb (job
# 65405367, Adam, 18 cells, 250-1000 epochs); here 54 cells (3x) and always 1000
# epochs (1.6x): about 5.5 h. Step 2 trains step 1's choice with augmentation (2.8x
# slower on EEGNet-8,2) to its chosen epoch: about 2-3 h for a model of EEGNet-8,2's
# size, more for larger ones. The final evaluation: about 0.3 h. About 9-10 h per task
# on 2g.20gb; 18 h asks for about 1.8x that. The search resumes if it doesn't finish.
#SBATCH --job-name=brainfold-sb-improved
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/drac/env.sh"
source "$BRAINFOLD_VENV/bin/activate"
config=${SB_CONFIG:-bnci2014001_eegnet}
read -r -a sessions <<< "${SESSIONS:-0train 1test}"
session=${sessions[$((SLURM_ARRAY_TASK_ID % ${#sessions[@]}))]}
replicate=$((SLURM_ARRAY_TASK_ID / ${#sessions[@]}))
seeds=()
if [ "$replicate" -gt 0 ]; then
  seeds=(--tpe-seed "$replicate" --trial-seed "$replicate")
fi
out=$BRAINFOLD_RESULTS/sb-improved-${RESULTS_ID:-$SLURM_ARRAY_JOB_ID}/$config-r$replicate/test-$session
mkdir -p "$out"
cd "$REPO/recipes/moabb"
python sb_search.py "hparams/sb_moabb/improved/$config.yaml" --optimizer schedule-free --out "$out" \
  --test-session "$session" "${seeds[@]}" --wandb "$BRAINFOLD_WANDB_PROJECT" "$@"
