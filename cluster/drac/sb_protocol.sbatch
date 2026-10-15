#!/bin/bash
# SpeechBrain-MOABB's protocol (recipes/moabb/sb_search.py): two steps of 50 TPE
# trials, then 10 seeds over every fold. Array index 0 runs it with their Adam and
# cyclic learning rate (BatchNorm momentum added to step 1), 1 with schedule-free AdamW.
#   sbatch --account=def-yourpi --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --array=0-1 cluster/drac/sb_protocol.sbatch
# Resumable: resubmit the same array index with the same RESULTS_ID to continue an
# interrupted search (the trial that was running is redone), e.g.
#   sbatch ... --array=1 --export=ALL,RESULTS_ID=<first job id> cluster/drac/sb_protocol.sbatch
# Extra arguments go to sb_search.py.
# Time, from the RTX 4080 SUPER (the 2g.20gb slice took 0.8-1.5x its time on the
# recipes): step-1 trials took 11-98 s plus 12 s of preprocessing, so step 1 takes
# about 1-2 h; step-2 trials train step 1's best model with augmentation, up to
# 616 s for a model as large as their tuned one (10 min x 50 = 8.5 h); the final
# 10 seeds take about as long as 10 such trials (1.7 h). About 12 h on the 4080,
# 17 h on 2g.20gb, a bit more with schedule-free (1000 epochs per trial). The
# 24 h request covers that; the search resumes if it doesn't finish.
#SBATCH --job-name=brainfold-sb-protocol
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/drac/env.sh"
source "$BRAINFOLD_VENV/bin/activate"
config=${SB_CONFIG:-bnci2014001_eegnet}
optimizers=(adam schedule-free)
optimizer=${optimizers[$SLURM_ARRAY_TASK_ID]}
out=$BRAINFOLD_RESULTS/sb-protocol-${RESULTS_ID:-$SLURM_ARRAY_JOB_ID}/$config-$optimizer
mkdir -p "$out"
cd "$REPO/recipes/moabb"
python sb_search.py "hparams/sb_moabb/$config.yaml" --optimizer "$optimizer" --out "$out" \
  --wandb "$BRAINFOLD_WANDB_PROJECT" "$@"
