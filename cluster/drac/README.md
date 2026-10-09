# Running on DRAC (Digital Research Alliance of Canada) clusters

Scripts to run the pack-size sweep and the recipes on a DRAC GPU cluster,
on H100 MIG slices: the GPUs an EEG job realistically gets there. A MIG slice
has its own compute units and memory, so timings aren't disturbed by other
jobs. Next to every result, `-env.json` records the GPU (with its MIG
profile), the library versions, the git commit and the Slurm job.

Submit everything from the repository root. Compute nodes have no internet
access, so the virtualenv and the datasets are set up on a login node first.

```bash
# 1. Once, on a login node: the virtualenv and the raw datasets (Lee2019: about 60 GB for MI, 100 GB for ERP).
export BRAINFOLD_DATA=~/projects/def-yourpi/$USER/brainfold   # put this in your ~/.bashrc
bash cluster/drac/setup.sh
wandb login   # once, to sync the offline W&B runs later

# 2. Preprocess and cache every configuration's trials, on CPU.
sbatch --account=def-yourpi cluster/drac/prepare.sbatch

# 3. Time and peak memory per pack size K, per configuration, on the main slice and the smallest.
sbatch --account=def-yourpi --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --array=0-11 cluster/drac/sweep.sbatch
sbatch --account=def-yourpi --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 --array=0-11 cluster/drac/sweep.sbatch

# 4. Set train.pack_size in each recipes/moabb/hparams/<config>.yaml from the sweep, then run every
#    recipe and seed (array index = 3 x configuration + seed). Lee2019_ERP's data takes about 14 GB,
#    so it needs 3g.40gb.
sbatch --account=def-yourpi --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --array=0-23,30-35 cluster/drac/recipes.sbatch
sbatch --account=def-yourpi --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --array=24-29 cluster/drac/recipes.sbatch

# 5. On a login node: upload the offline W&B runs.
wandb sync $BRAINFOLD_DATA/results/wandb/offline-run-*
```

Results go to `$BRAINFOLD_DATA/results/`, and to offline W&B runs under
`results/wandb/` (project `$BRAINFOLD_WANDB_PROJECT`, default `brainfold`):

- `sweep-<job>/<config>.csv`: per pack size K, seconds per epoch and peak
  memory, and the memory predicted for K before it ran (see `sweep.py`). The
  CSV is rewritten after every K, so a job that fails keeps what it measured.
- `recipes-<job>/<config>-seed<s>.csv`: one row per run and method (packed or
  braindecode, which seed 0 also trains), with its metrics, training and
  evaluation times and the pack's peak memory. Next to it, `-losses.csv` has
  every run's training loss per epoch and `-logits.npz` its test logits. All
  three are rewritten after every pack, so a job that runs out of time keeps
  the packs it finished.

`bench.sbatch` runs `benchmarks/bench_packing.py` (random BNCI2014001-shaped
data, with the torch.func comparison) the same way.

## Check before the first run

These scripts have not been run on a DRAC cluster yet. Check these first,
interactively (`salloc --account=def-yourpi --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 --time=0:30:00`):

- **Modules:** `env.sh` loads `StdEnv/2023 python/3.12`. See `module spider python`.
- **GPU names:** the MIG profiles above are Fir's and Nibi's; list a cluster's
  with `sinfo -o "%G"`. `nvidia-smi -L` in the job should show the MIG device.
- **A short run:** `python recipes/moabb/sweep.py hparams/bnci2014001_eegnet.yaml --max-pack-size 4`.
- **Time limits:** `recipes.sbatch` asks for 12 h. The Lee2019 configurations
  with ATCNet take the longest, because seed 0 also trains 108 braindecode
  models one at a time.
- **Licenses:** some MOABB datasets ask you to accept their terms on first
  download. If `download.py` stops on one, accept it and run `setup.sh` again.
