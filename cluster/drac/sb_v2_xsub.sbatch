#!/bin/bash
# Version 2 of the search protocol, cross-subject (hparams/sb_moabb/v2_xsub/; PROTOCOL.md):
# version 2's steps and rules with the subjects in 3 groups, one nested search per test group
# on the other subjects, and leave-one-training-subject-out validation. Task i searches for
# group TEST_GROUPS[i % n] in replicate search i / n (replicate r > 0 uses TPE seed r and trial
# seeds from r). No audit: it measures the dataset, and ran with sb_v2.sbatch.
#   sbatch --account=def-yourpi --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --array=0-2 cluster/drac/sb_v2_xsub.sbatch
# TEST_GROUPS lists the test groups (BNCI2014001's by default; sb_protocol.subject_groups names them).
# SB_PROTOCOL picks another protocol directory under hparams/sb_moabb/, e.g. the search over subject
# alignment: --export=ALL,SB_PROTOCOL=v2_xsub_align
# Resumable: resubmit the same array index with the same RESULTS_ID to continue an
# interrupted search (the trial that was running is redone), e.g.
#   sbatch ... --array=1 --export=ALL,RESULTS_ID=<first job id> cluster/drac/sb_v2_xsub.sbatch
# Extra arguments go to sb_search.py.
# Time: see the submission in PROTOCOL.md's log (measured locally before the first run).
#SBATCH --job-name=brainfold-sb-v2-xsub
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/drac/env.sh"
source "$BRAINFOLD_VENV/bin/activate"
# fail fast on a node whose GPU torch can't reach (job 65792547_0 ran 18 h on the CPU on fc10715)
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to torch on $(hostname)'"
config=${SB_CONFIG:-bnci2014001_eegnet}
protocol_dir=${SB_PROTOCOL:-v2_xsub}
read -r -a groups <<< "${TEST_GROUPS:-sub01-02-03 sub04-05-06 sub07-08-09}"
group=${groups[$((SLURM_ARRAY_TASK_ID % ${#groups[@]}))]}
replicate=$((SLURM_ARRAY_TASK_ID / ${#groups[@]}))
seeds=()
if [ "$replicate" -gt 0 ]; then
  seeds=(--tpe-seed "$replicate" --trial-seed "$replicate")
fi
out=$BRAINFOLD_RESULTS/sb-${protocol_dir//_/-}-${RESULTS_ID:-$SLURM_ARRAY_JOB_ID}/$config-r$replicate/test-$group
mkdir -p "$out"
cd "$REPO/recipes/moabb"
python sb_search.py "hparams/sb_moabb/$protocol_dir/$config.yaml" --optimizer schedule-free --out "$out" \
  --test-session "$group" "${seeds[@]}" --wandb "$BRAINFOLD_WANDB_PROJECT" "$@"
