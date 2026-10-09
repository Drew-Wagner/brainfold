# Sourced by every script here. Override any of these in your environment.
module load StdEnv/2023 python/3.12  # check `module spider python` for the versions on your cluster

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export BRAINFOLD_VENV=${BRAINFOLD_VENV:-$HOME/venvs/brainfold}
export BRAINFOLD_DATA=${BRAINFOLD_DATA:?set BRAINFOLD_DATA to a directory in your project space, e.g. ~/projects/def-yourpi/$USER/brainfold}  # raw data, cache, results
export MNE_DATA=$BRAINFOLD_DATA/mne_data  # where MOABB downloads the datasets
export BRAINFOLD_CACHE=$BRAINFOLD_DATA/cache  # preprocessed trials (recipes/moabb/train.py --cache-dir)
export BRAINFOLD_RESULTS=$BRAINFOLD_DATA/results
mkdir -p "$MNE_DATA" "$BRAINFOLD_CACHE" "$BRAINFOLD_RESULTS"

# The recipe configurations (recipes/moabb/hparams/<name>.yaml)
CONFIGS=(
  bnci2014001_eegnet bnci2014001_atcnet
  lee2019mi_eegnet lee2019mi_atcnet
  bnci2014009_eegnet bnci2014009_atcnet
  nakanishi2015_eegnet nakanishi2015_atcnet
  lee2019erp_eegnet lee2019erp_atcnet
  lee2019ssvep_eegnet lee2019ssvep_atcnet
)
