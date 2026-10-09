# Running on DRAC (Digital Research Alliance of Canada) clusters

Scripts to run the timing benchmark and the recipes on a DRAC GPU cluster.
A dedicated headless GPU gives more trustworthy timings than a workstation
that is also driving a display. Every result directory records the GPU, the
driver and the Python environment it ran with.

Submit everything from the repository root. Compute nodes have no internet
access, so the virtualenv and the datasets are set up on a login node first.

```bash
# 1. Once, on a login node: the virtualenv and the raw datasets (Lee2019: about 60 GB for MI, 100 GB for ERP).
export BRAINFOLD_DATA=~/projects/def-yourpi/$USER/brainfold   # put this in your ~/.bashrc
bash cluster/drac/setup.sh

# 2. Preprocess and cache every configuration's trials, on CPU.
sbatch --account=def-yourpi cluster/drac/prepare.sbatch

# 3. Once that has finished: the timing benchmark, and every recipe (one array task each).
sbatch --account=def-yourpi --gpus=h100:1 cluster/drac/bench.sbatch
sbatch --account=def-yourpi --gpus=h100:1 --array=0-11 cluster/drac/recipes.sbatch
```

Results go to `$BRAINFOLD_DATA/results/`:

- `bench-<job>/`: one CSV per model and training-set size, with every repeat,
  plus `nvidia-smi -q` and `pip freeze`.
- `recipes-<job>/`: one CSV per configuration, with one row per run and method
  (packed or braindecode), its metrics and its training time.

To run with more seeds, pass extra arguments through to `train.py`:
`sbatch ... cluster/drac/recipes.sbatch --set train.seeds=[0,1,2]`. The
configurations, in array order, are listed in `env.sh`.

## Check before the first run

These scripts have not been run on a DRAC cluster yet. Check these first:

- **Modules:** `env.sh` loads `StdEnv/2023 python/3.12`. See `module spider python`.
- **GPU type:** `--gpus=h100:1` asks for a whole H100. The type names differ
  between clusters, and some clusters split GPUs into MIG slices: ask for a
  whole GPU, so the timings are comparable.
- **Time limits:** `recipes.sbatch` asks for 12 h. The Lee2019 configurations
  with ATCNet take the longest, because the braindecode baseline trains 108
  models one at a time.
- **Licenses:** some MOABB datasets ask you to accept their terms on first
  download. If `download.py` stops on one, accept it and run `setup.sh` again.
