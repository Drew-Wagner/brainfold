#!/bin/bash
# Version 2 of the search protocol (recipes/moabb/PROTOCOL.md) with schedule-free AdamW:
# preprocessing fixed by rule (4-40 Hz, 125 Hz, the window from the paradigm block, every
# EEG channel), step 1 (architecture and training) and step 2 (EEG augmentation, no
# augmentation as its first trial), 50 TPE trials each, then 10 seeds over the folds.
# Selection is nested by test session: one search per test session, run as separate array
# tasks. Task i searches for session SESSIONS[i % n] in replicate search i / n (replicate
# r > 0 uses TPE seed r and trial seeds from r; replicate 0 uses the yaml's seed, 1986).
# Task 0 first runs the confound audit (sb_audit.py), whose report goes next to the
# searches (audit.md, audit.csv); it's skipped if audit.md exists.
#   sbatch --account=def-yourpi --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --array=0-1 cluster/drac/sb_v2.sbatch
#   sbatch ... --array=2-5 cluster/drac/sb_v2.sbatch   # replicate searches 1 and 2
# SESSIONS lists the test sessions (MOABB's names; BNCI2014001's by default; for a dataset
# with one session, <session>-half1 and <session>-half2).
# Resumable: resubmit the same array index with the same RESULTS_ID to continue an
# interrupted search (the trial that was running is redone), e.g.
#   sbatch ... --array=1 --export=ALL,RESULTS_ID=<first job id> cluster/drac/sb_v2.sbatch
# Extra arguments go to sb_search.py.
# Time, per task (one test session: 54 cells per trial on BNCI2014001), measured on the
# RTX 4080 SUPER (README, "Version 2 of the protocol"), then scaled to the 2g.20gb slice:
# - step 1: a model from the middle of the priors (14.7k parameters) takes 170 s per trial
#   (EEGNet-8,2: 60 s); 50 trials, 2.4 h;
# - step 2: the same model with augmentation to epoch 1000, 170 s with every transform off
#   and 283 s with all 7 on; about 230 s for a typical trial (half of them on), 3.2 h; less
#   if step 1 chose an earlier epoch, more if it chose a larger model;
# - the final evaluation (90 cells) about 0.2 h; the audit (task 0 only) 0.08 h;
# about 5.9 h on the RTX 4080 SUPER. On 2g.20gb, version 1's step 0 (job 65435553: 50
# trials of EEGNet-8,2 on 54 cells in about 3 h, 216 s per trial) took about 2.6x the RTX
# 4080 SUPER's time for the same trial (README: 70 s training and 13 s preprocessing), so
# about 15 h per task. 24 h asks for about 1.6x that; the search resumes if it doesn't finish.
#SBATCH --job-name=brainfold-sb-v2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
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
results=$BRAINFOLD_RESULTS/sb-v2-${RESULTS_ID:-$SLURM_ARRAY_JOB_ID}/$config
out=$results-r$replicate/test-$session
mkdir -p "$out"
cd "$REPO/recipes/moabb"
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ] && [ ! -e "$results-audit/audit.md" ]; then
  python sb_audit.py "hparams/sb_moabb/v2/$config.yaml" --optimizer schedule-free --out "$results-audit" \
    --wandb "$BRAINFOLD_WANDB_PROJECT"
fi
python sb_search.py "hparams/sb_moabb/v2/$config.yaml" --optimizer schedule-free --out "$out" \
  --test-session "$session" "${seeds[@]}" --wandb "$BRAINFOLD_WANDB_PROJECT" "$@"
