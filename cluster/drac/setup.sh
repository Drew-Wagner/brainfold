#!/bin/bash
# Run once on a LOGIN node (compute nodes have no internet): creates the
# virtualenv and downloads the datasets.
#
#   export BRAINFOLD_DATA=~/projects/def-yourpi/$USER/brainfold
#   bash cluster/drac/setup.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

if [ ! -d "$BRAINFOLD_VENV" ]; then
  virtualenv --no-download "$BRAINFOLD_VENV"
fi
source "$BRAINFOLD_VENV/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch  # the cluster's wheel, built for its CUDA
# The rest from PyPI. The preprocessed-trial cache is keyed by these three versions; pinning them to the
# versions a cache was prepared with elsewhere lets it be copied to $BRAINFOLD_DATA/cache and reused.
pip install -e "$REPO[braindecode,recipes]" braindecode==1.8.1 moabb==1.7.2 mne==1.13.2
python -c "import torch, braindecode, brainfold; print('torch', torch.__version__, 'braindecode', braindecode.__version__)"

python "$REPO/cluster/drac/download.py" "${CONFIGS[@]}"
