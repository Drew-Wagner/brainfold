# brainfold

Train every fold, seed and subject of an EEG decoding benchmark at once.

EEG decoding models are small and EEG datasets are small, so training one
model at a time leaves most of a GPU idle. Results also vary a lot across
seeds, folds and subjects, so one run says little. A careful evaluation
(seeds x folds x subjects) means hundreds of runs.

brainfold packs K such runs into one network and trains them together:

- **Exact.** Each member computes what a separate braindecode model would.
  The tests check forward passes, BatchNorm running statistics and Adam
  training steps against K separate models in float64.
- **Fast.** 4-8x less training time per model on a GPU (see
  [Benchmarks](#benchmarks)), partly from packing and partly from computing
  each model's first block in a cheaper order.
- **braindecode-compatible.** Packed models take braindecode's arguments, and
  each member converts to and from an ordinary braindecode model, for
  evaluation, saving and the Hugging Face Hub.

The goal is to make thorough evaluation cheap: report the spread over seeds
and folds instead of a single number, for about the cost of one run.

## Models

| model | class | braindecode | speedup at K=18 (RTX 4080 SUPER) |
|---|---|---|---|
| EEGNet (Lawhern et al., 2018) | `PackedEEGNet` | >= 1.2 | 4.3-7.1x |
| ATCNet (Altaheri et al., 2022) | `PackedATCNet` | >= 1.4 | 7.6-7.8x |

More models will follow; see [Adding a model](#adding-a-model).

## Install

```bash
pip install "brainfold @ git+https://github.com/Drew-Wagner/brainfold"
```

The package depends only on PyTorch. Extras: `[braindecode]` for
`to_braindecode()` and `from_braindecode()`, `[recipes]` for the recipes.

For development:

```bash
uv sync --group dev --extra braindecode --extra recipes
uv run pytest
```

## Quickstart

```python
import torch
import torch.nn.functional as F
from brainfold import PackedEEGNet

model = PackedEEGNet.from_seeds(seeds, n_chans=22, n_outputs=4, n_times=512).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for x, y in batches:  # x: (batch, K, n_chans, n_times), y: (batch, K); column k is member k's batch
    logits = model(x)  # (batch, K, n_outputs)
    losses = F.cross_entropy(logits.permute(0, 2, 1), y, reduction="none").mean(0)  # (K,)
    optimizer.zero_grad()
    losses.sum().backward()  # SUM, not mean: each member gets exactly its own gradient
    optimizer.step()         # one Adam over the packed parameters == K separate Adams
```

The loop is the same for every model. Each member sees its own data, so the K
columns can hold different folds, subjects or seeds. Not every training trick
keeps the members independent: see
[Keeping members independent](#keeping-members-independent). The
[recipes](#recipes) have a complete loop with per-member shuffling, a cosine
schedule, evaluation and a baseline.

## Common interface

Every packed model has the same interface:

- **Construction.** `PackedX(K, **kwargs)` is K braindecode `X(**kwargs)`: the
  same argument names, defaults and meaning, after the pack size. That
  includes `chs_info`, `sfreq` and `input_window_seconds` for inferring
  `n_chans` and `n_times`.
- **Seeds.** `PackedX.from_seeds(seeds, ...)` initializes member k from
  `seeds[k]` alone, so a member's initial weights don't depend on what it is
  packed with.
- **Shapes.** Input `(batch, K, n_chans, n_times)`, output `(batch, K, n_outputs)`.
- **Per-member state.** `member_state_dict(k)` returns member k's weights and
  BatchNorm statistics in braindecode's format, `load_member_state_dict(k, state)`
  overwrites member k alone, and `to_state_dicts()` / `load_state_dicts(states)`
  do all members.
- **braindecode.** `model.to_braindecode()` returns K ordinary braindecode
  models, and `PackedX.from_braindecode(models)` packs existing ones.
- **Precomputed input moments.** In training, the first BatchNorm's
  statistics come from moments of the input (see [How it works](#how-it-works)).
  `moments = model.input_moments(X)` computes them once per trial, and
  `model(X[idx], tuple(m[idx] for m in moments))` gives the same result as
  `model(X[idx])` without recomputing them every step. The recipes do this.

For example, per-member early stopping:

```python
for k in range(K):
    if val_acc[k] > best_acc[k]:
        best_acc[k], best_state[k] = val_acc[k], model.member_state_dict(k)
...
model.load_state_dicts(best_state)  # every member at its own best epoch
```

## Model notes

### EEGNet

`PackedEEGNet` computes EEGNet's first block (temporal conv -> BatchNorm ->
spatial conv) spatial-first, which does about 11x less work in that block for
22 electrodes (see [How it works](#how-it-works)). Most of its speedup comes
from there.

- With K=1 it is about 2x *slower* than braindecode's `EEGNet`; the gain
  appears from about K=4.
- Unsupported options raise a `ValueError`: cropped decoding (a
  `final_conv_length` shorter than the remaining time) and activations with
  parameters, such as `PReLU`, which would be shared across the pack.
  `batch_norm_momentum` must be a float (not `None`).
- `batch_norm_affine=False` works, but braindecode (as of 1.8.1) fails to
  build such an `EEGNet`, so `to_braindecode()` does too.

### ATCNet

`ATCNet` is a conv block followed by `n_windows` (default 5) branches, each an
attention block and a TCN with its own weights, reading overlapping windows of
the conv block's output. braindecode runs the branches one after another.
`PackedATCNet` packs them the way it packs members: all K x `n_windows`
attention blocks and TCNs run as one grouped op each. The conv block starts
with EEGNet's first block and uses the same spatial-first computation.

- It is already about 2x faster than braindecode's `ATCNet` with K=1, because
  the windows' branches run together.
- It applies the same shrinking of kernels, pools and windows as `ATCNet` for
  short inputs, so `n_times` around 1125 (4.5 s at 250 Hz) is what the
  defaults expect.
- The tests cover the default configuration and variants: other sizes,
  `concat`, one window, short inputs, `conv_max_norm_const`.
- `model.source_optimizer_param_groups()` gives the official code's L2 weight
  decay groups, as braindecode's `ATCNet` does (conv and TCN kernels 0.009,
  final layer 0.5). With Adam or SGD this keeps members independent.
- **braindecode's `concat=True` fails** (as of 1.8.1) when `n_windows > 1`: its
  forward pairs windows with final layers, and with `concat` there is only
  one, so only the first window runs and the final layer gets the wrong input
  size. `PackedATCNet(concat=True)` concatenates all windows, as `ATCNet`
  documents; the tests compare it with braindecode's own submodules run that
  way.
- The attention block's extra dropout is fixed at 0.3 in braindecode, and in
  `PackedATCNet` too (`model.att_block_drop`).
- `tcn_activation`s with parameters, such as `PReLU`, raise a `ValueError`.

## Recipes

[`recipes/moabb`](recipes/moabb) trains and evaluates the packed models on
MOABB datasets, with one hparams file per dataset and model. Data loading and
preprocessing use braindecode (`MOABBDataset`, band-pass, resampling,
exponential moving standardization, `create_windows_from_events`), and the
preprocessed trials are cached in `~/.cache/brainfold`.

| task | hparams | dataset | protocol | runs per seed | metrics |
|---|---|---|---|---|---|
| motor imagery | `bnci2014001_{eegnet,atcnet}` | BNCI2014001 (BCI Competition IV 2a): 9 subjects, 22 channels, 4 classes | cross-session | 18 | accuracy, kappa |
| motor imagery | `lee2019mi_{eegnet,atcnet}` | Lee2019_MI (OpenBMI): 54 subjects, 62 channels, 2 classes | cross-session | 108 | accuracy, kappa |
| P300 | `bnci2014009_{eegnet,atcnet}` | BNCI2014_009: 10 subjects, 16 channels, 1 target in 6 | cross-session | 30 | ROC AUC, balanced accuracy |
| P300 | `lee2019erp_{eegnet,atcnet}` | Lee2019_ERP (OpenBMI): 54 subjects, 62 channels | cross-session | 108 | ROC AUC, balanced accuracy |
| SSVEP | `nakanishi2015_{eegnet,atcnet}` | Nakanishi2015: 9 subjects, 8 channels, 12 classes | within-session, 5-fold | 45 | accuracy, kappa |
| SSVEP | `lee2019ssvep_{eegnet,atcnet}` | Lee2019_SSVEP (OpenBMI): 54 subjects, 62 channels, 4 classes | cross-session | 108 | accuracy, kappa |

The protocols are:

- **cross-session:** per subject, test on each session and train on the others;
- **cross-subject:** test on each subject and train on all the others;
- **within-session:** per subject and session, stratified k-fold.

Runs with equal training-set sizes are trained together, in packs of
`train.pack_size`. The P300 recipes weight each member's loss by the inverse
class frequencies of its own training set (`train.class_weight: balanced`).

```bash
cd recipes/moabb
python train.py hparams/bnci2014001_eegnet.yaml                 # EEGNet, 300 epochs, seed 0
python train.py hparams/bnci2014009_atcnet.yaml --set train.seeds=[0,1,2]
python train.py hparams/nakanishi2015_eegnet.yaml --baseline --out results.csv
python summarize.py results/rtx4080super/*.csv                  # the table below, from its CSVs
python train.py hparams/bnci2014001_eegnet.yaml --set data.subjects=[1] train.epochs=3 --device cpu   # smoke test
python sweep.py hparams/lee2019mi_eegnet.yaml --out sweep.csv     # time and peak memory per pack size K
python plot_sweep.py results/h100-mig/sweep-*/*.csv --out results/h100-mig   # the plots in Benchmarks
```

`--baseline` trains every run a second time as a separate braindecode model,
from the same initial weights and with the same data order, so packed and
unpacked results can be compared run by run. `--out r.csv` writes one row per
run and method, every run's training loss per epoch to `r-losses.csv`, its
test logits to `r-logits.npz`, and the environment (GPU, library versions, git
commit) to `r-env.json`; `--wandb PROJECT` also logs them to W&B. The Lee2019
ERP and SSVEP recordings are about 100 GB and 60 GB:
[`cluster/drac`](cluster/drac) runs every recipe on a DRAC cluster.

`sweep.py` times a recipe's own training loop on its own data for pack sizes
from 1 up to the largest that fits: peak memory is about linear in K, so it
stops before a K predicted to fill more than 90% of the GPU.

### Speed options

The recipes compute the first BatchNorm's input moments once per training trial
(`input_moments`, exact). Against recomputing them every step, that made whole
recipes 1.17x (BNCI2014001 ATCNet), 1.28x (BNCI2014001 EEGNet), 1.34x
(Lee2019_MI ATCNet) and 1.64x (Lee2019_MI EEGNet) faster on the RTX 4080
SUPER, for n_trials x L^2 x 4 bytes of memory (L the first conv's length:
0.1-0.3 GiB here).

`train.py` and `sweep.py` take three options that trade exactness for speed or
memory. All are off by default:

- `--compile` trains the packed model through `torch.compile`, which fuses the
  elementwise chains at full time resolution. Dropout then draws its masks from
  Inductor's generator, so masks differ from eager mode.
- `--precision bf16` runs the model under bf16 autocast, with TF32 matmuls (or
  `tf32` for TF32 matmuls alone). The first BatchNorm's statistics stay in
  float32: `var = E[u^2] - mu^2` would lose too much in bf16.
- `--data-dtype float16` keeps the trials on the GPU in float16 and converts
  each batch, which halves the data's memory (6 GB for Lee2019_MI ATCNet).

Every option keeps the members independent. `benchmarks/approximation.py`
measures each option's numerical error against float64 on a recipe's own data,
with dropout off. After 20 epochs of BNCI2014001, the test predictions that
differ from float64's are:

| | EEGNet | ATCNet |
|---|---|---|
| float32 (the default) | 0.02-0.17% | 0.02-0.06% |
| `--compile`, `--precision tf32` or `--data-dtype float16` | 0.06-0.14% | 0.02-0.04% |
| `--precision bf16`, alone or with the others | 0.35% | 0.48-0.73% |

The float32 range is its own run-to-run spread (cuDNN is not deterministic), so
only bf16 is distinguishable from it. One bf16 step has about 1% relative error
in the logits and gradients, against 0.03-0.09% for float32.

Full recipe runs, with dropout, on the RTX 4080 SUPER, against the default
(float32), per run:

| recipe | runs | option | accuracy | difference per run | time (train + eval) | speedup |
|---|---|---|---|---|---|---|
| BNCI2014001 EEGNet | 54 | float32 | 0.669 ± 0.167 | | 26.9 s | 1.00x |
| | | float32 again | 0.669 ± 0.167 | +0.000 ± 0.001 | 26.6 s | 1.01x |
| | | `--compile` | 0.670 ± 0.164 | +0.001 ± 0.037 | 23.3 s | 1.15x |
| | | `--precision bf16` | 0.668 ± 0.166 | -0.002 ± 0.031 | 24.8 s | 1.08x |
| | | all three | 0.669 ± 0.166 | -0.000 ± 0.037 | 20.1 s | 1.34x |
| Lee2019_MI EEGNet | 108 | float32 | 0.696 ± 0.146 | | 45.4 s | 1.00x |
| | | float32 again | 0.696 ± 0.146 | +0.000 ± 0.000 | 44.6 s | 1.02x |
| | | `--compile` | 0.694 ± 0.148 | -0.001 ± 0.044 | 37.4 s | 1.21x |
| | | `--precision bf16` | 0.699 ± 0.144 | +0.004 ± 0.032 | 41.2 s | 1.10x |
| | | all three | 0.695 ± 0.148 | -0.001 ± 0.043 | 39.7 s | 1.14x |
| BNCI2014001 ATCNet | 54 | float32 | 0.688 ± 0.152 | | 274.5 s | 1.00x |
| | | float32 again | 0.687 ± 0.152 | -0.001 ± 0.004 | 281.6 s | 0.97x |
| | | `--compile` | 0.692 ± 0.144 | +0.005 ± 0.033 | 192.4 s | 1.43x |
| | | `--precision bf16` | 0.694 ± 0.149 | +0.006 ± 0.032 | 164.1 s | 1.67x |
| | | all three | 0.694 ± 0.143 | +0.006 ± 0.031 | 123.2 s | 2.23x |
| Lee2019_MI ATCNet | 108 | float32 | 0.676 ± 0.159 | | 310.0 s | 1.00x |
| | | `--compile` | 0.675 ± 0.162 | -0.002 ± 0.041 | 274.2 s | 1.13x |
| | | `--precision bf16` | 0.673 ± 0.161 | -0.004 ± 0.035 | 317.0 s | 0.98x |
| | | all three | 0.675 ± 0.160 | -0.002 ± 0.040 | 251.4 s | 1.23x |

Peak memory falls with every option: Lee2019_MI ATCNet (K=36) uses 9.5 GiB in
float32 and 5.1 GiB with all three, 3 GiB of it from float16 data. The
speedups depend on the recipe: bf16 alone is 1.67x on BNCI2014001 ATCNet but
0.98x on Lee2019_MI ATCNet, so time an option on a recipe before relying on it.

Rerunning float32 changes a run by at most 0.014. Each option changes runs by
about ±0.03-0.04, as much as different dropout masks do (packed against
braindecode in [Results](#results)), and the mean by at most 0.006, under 1.5
standard errors. So the options change which run lands where, not the
distribution of results. Use them for sweeps and exploration. For
comparisons that need matching runs, use float32, which is also what the
`--baseline` braindecode models are compared against.

### Convergence options

The recipes train with Adam at lr 0.001, cosine-annealed over the papers' 300
(EEGNet) or 500 (ATCNet) epochs. These options train in fewer epochs. They are
off by default, so the recipes' results stay as they are:

- `--optimizer schedule-free` uses schedule-free AdamW (Defazio et al., 2024;
  the `schedulefree` package) at a constant learning rate, set with
  `--set train.lr=...`. It needs no schedule, so the model after any epoch is
  usable, and one run gives every shorter budget. Its BatchNorm statistics are
  recomputed on the training set for the weights it evaluates. ATCNet's official
  L2 stays coupled, as with Adam.
- `--recalibrate-bn` recomputes the BatchNorm statistics with Adam too. EEGNet's
  BatchNorm momentum is 0.01, so its running statistics trail the weights by
  about 100 steps, about 20 epochs on BNCI2014001. That only matters for short
  budgets.
- `--warmup-epochs` adds a linear warmup. It didn't help here.
- `--eval-every N` evaluates every N epochs and writes the learning curves to
  `<out>-curves.csv` and to W&B. The time it takes is left out of
  `train_seconds`.

Both are exact: a pack trains like its members trained alone (float64 test).

On the RTX 4080 SUPER, with `--compile`. The difference is per run, against
the recipe, with its standard error:

| recipe | runs | training | accuracy | difference per run | reaches the recipe's accuracy at |
|---|---|---|---|---|---|
| BNCI2014001 EEGNet | 54 | Adam, lr 0.001, 300 epochs (the recipe) | 0.669 | | 300 epochs |
| | | Adam, lr 0.005, 150 epochs, `--recalibrate-bn` | 0.670 | +0.001 ± 0.005 | 150 epochs |
| | | schedule-free, lr 0.005, 300 epochs | 0.677 | +0.008 ± 0.005 | about 150 epochs |
| BNCI2014001 ATCNet | 54 | Adam, lr 0.001, 500 epochs (the recipe) | 0.692 | | 500 epochs |
| | | Adam, lr 0.0025, 250 epochs, `--recalibrate-bn` | 0.691 | -0.001 ± 0.005 | 250 epochs |
| | | schedule-free, lr 0.005, 500 epochs | 0.722 | +0.030 ± 0.005 | about 150-200 epochs |
| Lee2019_MI EEGNet | 108 | Adam, lr 0.001, 300 epochs (the recipe) | 0.694 | | 300 epochs |
| | | schedule-free, lr 0.005, 300 epochs | 0.709 | +0.015 ± 0.004 | about 100 epochs |
| Lee2019_MI ATCNet | 108 | Adam, lr 0.001, 500 epochs (the recipe) | 0.675 | | 500 epochs |
| | | schedule-free, lr 0.005, 500 epochs | 0.693 | +0.018 ± 0.006 | about 150 epochs |

Schedule-free reaches the recipe's accuracy in about a third of the epochs, and
ends higher when given the full budget. Its learning rate was chosen on
BNCI2014001's test sessions, among 0.001, 0.0025, 0.005 and 0.01. It was then
fixed in advance for Lee2019_MI, where it was not tuned. With Adam, lr 0.001
was simply too low: at the recipe's learning rate, halving the epochs costs
0.045 (EEGNet) and 0.030 (ATCNet). Each schedule-free epoch takes about as long
as an Adam one.

    python train.py hparams/bnci2014001_atcnet.yaml --optimizer schedule-free --set train.lr=0.005 train.epochs=200

### SpeechBrain-MOABB's protocol

`sb_search.py` runs the decoding protocol of SpeechBrain-MOABB (Borra,
Paissan and Ravanelli, 2024) with packed training. It reproduces their
preprocessing, their leave-one-session-out splits with class-balanced
validation sets, their training (Xavier init, Adam with a triangular cyclic
learning rate, class-weighted loss, four augmentations, averaging of the last
epochs' weights), and their two-step search: 50 TPE trials on the
preprocessing, architecture and training, then 50 on the augmentation, then
10 seeds over every fold. A trial trains all its folds as one pack.
`tests/test_sb_protocol.py` checks each piece against their code. Changes:

- Optuna's TPE replaces Orion's, with the same startup and candidate counts.
- BatchNorm momentum joins step 1's search (their EEGNet fixes it at 0.01).
- The model is braindecode's EEGNet with their hyperparameters.
- `--optimizer schedule-free` swaps in schedule-free AdamW. The number of
  epochs is then chosen on the validation sets (every 25 epochs from 250 to
  1000) instead of searched, with no checkpoint averaging. BatchNorm statistics
  are recomputed, so the momentum drops out.

Their tuned BNCI2014001 configuration, with one seed, scores 0.723 balanced
accuracy, against their 0.732 ± 0.004 over 10 seeds (one seed varies by
0.013). Its 18 folds trained in 616 s on the RTX 4080 SUPER. Random step-1
trials took 11-98 s, plus 12 s to preprocess.

    python sb_search.py hparams/sb_moabb/bnci2014001_eegnet.yaml --out results/sb/adam
    python sb_search.py hparams/sb_moabb/bnci2014001_eegnet.yaml --optimizer schedule-free --out results/sb/sf

### The improved protocol

[`recipes/moabb/PROTOCOL_IDEAS.md`](recipes/moabb/PROTOCOL_IDEAS.md) lists
what we'd change in their protocol, and why. `sb_search.py` implements most of
it behind options that are all off by default, so the commands above still
run SpeechBrain-MOABB's protocol. `hparams/sb_moabb/improved/bnci2014001_eegnet.yaml`
turns them on, with the reason for every choice next to it:

- **Steps from the yaml** (`steps`): step 0 searches only the preprocessing
  (band, window start and length, channels) on a fixed reference model,
  Lawhern et al.'s EEGNet-8,2; step 1 the architecture and training; step 2
  the augmentation. Temporal kernel and pool sizes can be in seconds.
- **EEG augmentation** (`augment: eeg`): braindecode's TimeReverse, SignFlip,
  FTSurrogate, FrequencyShift, BandstopFilter, SmoothTimeMask and
  ChannelsDropout (and Mixup), with the transforms and ranges of Rommel et
  al. (2022). Every random draw is the member's own (a hash of its seed, the
  step and the position in the batch), so a packed member is augmented exactly
  as it would be alone; the tests check that packed training equals training
  each member alone, and each transform equals braindecode's.
- **Cross-validation within the training session** (`protocol.inner_folds`):
  one validation fold per run (`runs`), or k folds; each (subject, test
  session, inner fold, seed) is a cell, a member of a pack.
- **Seeds per trial** (`seeds_per_trial`, `trial_seeds: per_trial`,
  `seed_per_inner_fold`), different in every trial; per-seed and per-fold
  scores and the objective's standard error are recorded.
- **Size**: every trial records one member's parameters and FLOPs;
  `selection: one_se` passes on the smallest configuration within one
  standard error of the best; `multi_objective: true` runs TPE on (validation,
  size).
- **Selection nested by test session** (`nested_by_test_session`): one
  search per test session, on the folds that test on it, so no session helps
  choose the configuration it's tested with (averaging validation over every
  fold, as their protocol does, lets it).
- **The search's own variance**: `--tpe-seed` and `--trial-seed` for
  replicate searches, `--rerun STEP` to retrain a study's best trials with new
  seeds, and `sb_analyze.py` for validation against size with Pareto fronts,
  fANOVA importances, and the spread of replicate searches.

BNCI2014001 is the development dataset: no default above was chosen from
accuracy on it, and results on it are exploratory until the protocol is
frozen and run on held-out datasets (PROTOCOL_IDEAS.md, "Development vs
held-out datasets").

    python sb_search.py hparams/sb_moabb/improved/bnci2014001_eegnet.yaml --optimizer schedule-free --out results/sb/imp
    python sb_search.py hparams/sb_moabb/improved/bnci2014001_eegnet.yaml --optimizer schedule-free --out results/sb/imp-r1 \
        --tpe-seed 1 --trial-seed 1   # a replicate search
    python sb_search.py hparams/sb_moabb/bnci2014001_eegnet.yaml --out results/sb/adam --rerun step1 \
        --rerun-top 10 --rerun-seeds 5   # their protocol's 10 best step-1 trials, 5 new seeds each
    uv run --with matplotlib python sb_analyze.py results/sb/imp results/sb/imp-r1 --out results/sb/analysis

A nested search's directory has one subdirectory per test session
(`test-0train/`, `test-1test/`); pass those to `sb_analyze.py`.
`cluster/drac/sb_improved.sbatch` runs each test session's search as its own
array task.

**Cost.** One trial on the RTX 4080 SUPER, all 9 subjects, schedule-free for
1000 epochs with an evaluation every 25 (another process shared the GPU, so
these are rough):

| model | protocol | cells | packs | training | peak memory |
|---|---|---|---|---|---|
| EEGNet-8,2 (2.4k parameters) | theirs | 18 | 1 | 29 s | 5.1 GiB |
| EEGNet-8,2 | improved, no nesting | 108 | 3 | 139 s | 5.6 GiB |
| EEGNet-8,2, EEG augmentation | theirs | 18 | 1 | 86 s | 1.7 GiB |
| EEGNet-8,2, EEG augmentation | improved, no nesting | 108 | 3 | 388 s | 2.7 GiB |
| their tuned EEGNet (146k) | theirs | 18 | 1 | 312 s | 10.3 GiB |
| their tuned EEGNet | improved, no nesting | 108 | 6 | 1911 s | 10.6 GiB |

The improved protocol's 108 cells are 6 validation folds (one per run) of the
18 subject-sessions; nested by test session, each of the two searches has 54.
Past about 18 cells a pack is compute-bound: 2.6-2.8 ms per cell and epoch for
EEGNet-8,2 from 18 to 108 cells, so cost grows about linearly with cells. Their
tuned model's packs of 36 ran out of memory and were split to 18 (153 s
lost); packs of 9, from splitting larger packs further, took 73 ms per cell and
epoch against 24 ms in packs of 18, which is why the improved yaml packs 36
cells rather than more. EEG augmentation makes small models' training about
2.8x slower (10 ms per batch of 32 x 36 members, mostly the FFTs and many small
kernels). Preprocessing a new band and window takes 13 s. Estimated per test
session's search on an H100 2g.20gb slice: about 9-10 h (step 0 1.2 h, step 1
about 5.5 h, step 2 2-3 h, final 0.3 h; the sbatch file has the reasoning).

### Version 2 of the protocol

[`recipes/moabb/PROTOCOL.md`](recipes/moabb/PROTOCOL.md) is version 2: the
protocol, the rules for changing it, and the log of what was run on
BNCI2014001. It fixes preprocessing by rule instead of searching it (step 0 is
gone), and applies unchanged to a new dataset from that dataset's metadata.
`hparams/sb_moabb/v2/bnci2014001_eegnet.yaml` turns it on; every option is off
by default, so the files above run as before.

- **Preprocessing by rule**: 4-40 Hz, 125 Hz (or the recording's rate, if
  lower), every EEG channel (`protocol.channels: all`), and the window from
  the cue's offset to the end of imagery, read from the hparams file's
  `paradigm` block (`protocol.window: paradigm`; 1.25-4.0 s after the cue on
  BNCI2014001). The block's counts and rate are checked against MOABB's
  metadata, and the search stops if they differ.
- **Splits by rule**: a subject with one session gets two pseudo-sessions
  (`protocol.outer_folds: rule`); one validation fold per run of the
  training data, at most 6, or 5 blocks in time with fewer than 3 runs
  (`protocol.inner_folds: rule`).
- **Step 2** starts with no augmentation (`first_trial: all_off`), and the
  one-standard-error rule counts the transforms on (`size: n_transforms`), so
  augmentation is kept only if it's better by more than one standard error.
- **The confound audit** (`sb_audit.py`), before the search: EEGNet-8,2 on
  the protocol's window, on the cue period alone, and on the window without
  the frontal channels, against the binomial chance threshold. Reported, never
  used for selection.

    python sb_audit.py hparams/sb_moabb/v2/bnci2014001_eegnet.yaml --out results/sb/v2-audit
    python sb_search.py hparams/sb_moabb/v2/bnci2014001_eegnet.yaml --optimizer schedule-free --out results/sb/v2

`cluster/drac/sb_v2.sbatch` runs the audit in task 0, then each test
session's search as its own array task.

**Cost.** On the RTX 4080 SUPER, all 9 subjects, schedule-free to 1000 epochs
with an evaluation every 25 (step 2 and the audit evaluate at the last epoch
only); one test session's search, as each task of the sbatch file runs it:

| trial | model | cells | packs | training | peak memory |
|---|---|---|---|---|---|
| version 1, improved, not nested (the table above) | EEGNet-8,2 | 108 | 3 | 139 s, and 13 s preprocessing | 5.6 GiB |
| version 2, step 1 | EEGNet-8,2 (2.1k parameters) | 54 | 2 | 60 s | 5.1 GiB |
| version 2, step 1 | middle of step 1's priors (14.7k) | 54 | 2 | 170 s | 8.7 GiB |
| version 2, step 2, all 7 transforms | EEGNet-8,2 | 54 | 2 | 178 s | 1.9 GiB |
| version 2, step 2, all 7 transforms | middle of the priors | 54 | 2 | 283 s | 4.7 GiB |
| the audit, 3 conditions | EEGNet-8,2 | 108 each | 3 each | 229 s (270 s with loading) | 5.1 GiB |

The trials are 343 samples (2.75 s) of 22 channels. The band is fixed, so
preprocessing runs once (13 s) instead of in every trial. The middle of the
priors is 34 temporal filters of 0.35 s, depth multiplier 2, F2 = 69, a
0.43 s separable kernel, pooling of 0.14 s, batches of 32. With every
transform off, step 2 costs as step 1; augmentation makes it about 3x
slower for EEGNet-8,2 and 1.7x for the larger model. Estimated per test
session's search: about 5.9 h here (step 1 2.4 h, step 2 3.2 h, the final
evaluation 0.2 h) and about 15 h on an H100 2g.20gb slice, from version 1's
step 0 there (job 65435553), which took about 2.6x this GPU's time; the
sbatch file has the reasoning.

### Results

On an RTX 4080 SUPER, packed and with `--baseline`, seeds 0, 1 and 2 (seed 0
only for Lee2019_MI). Times are training and evaluation:

| task | dataset | model | runs | metric | packed | braindecode | packed - braindecode, per run | packed time | braindecode time | speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| MI | Lee2019_MI | EEGNet | 108 | accuracy | 0.692 ± 0.150 | 0.695 ± 0.148 | -0.004 ± 0.040 | 64 s | 698 s | 11.0x |
| MI | Lee2019_MI | ATCNet | 108 | accuracy | 0.675 ± 0.159 | 0.671 ± 0.162 | +0.004 ± 0.044 | 455 s | 4699 s | 10.3x |
| P300 | BNCI2014_009 | EEGNet | 90 | roc auc | 0.948 ± 0.045 | 0.948 ± 0.045 | -0.000 ± 0.003 | 45 s | 299 s | 6.7x |
| P300 | BNCI2014_009 | ATCNet | 90 | roc auc | 0.744 ± 0.078 | 0.745 ± 0.079 | -0.001 ± 0.018 | 114 s | 966 s | 8.5x |
| SSVEP | Nakanishi2015 | EEGNet | 135 | accuracy | 0.842 ± 0.236 | 0.848 ± 0.228 | -0.007 ± 0.047 | 37 s | 215 s | 5.8x |
| SSVEP | Nakanishi2015 | ATCNet | 135 | accuracy | 0.958 ± 0.102 | 0.958 ± 0.099 | +0.001 ± 0.032 | 293 s | 4130 s | 14.1x |

(± is the standard deviation over runs.) In every recipe, the packed and
braindecode runs differ only in their dropout masks, so their per-run
difference is noise around zero. Packing makes training and evaluation 5.8x
to 14.1x faster.

On an H100 2g.20gb MIG slice (2/7 of the GPU), seed 0 with `--baseline`, and
`train.pack_size` set from [the sweep](#h100-mig-slices):

| task | dataset | model | runs | metric | packed | braindecode | packed - braindecode, per run | packed time | braindecode time | speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| MI | BNCI2014_001 | EEGNet | 18 | accuracy | 0.683 ± 0.172 | 0.686 ± 0.151 | -0.003 ± 0.033 | 18 s | 105 s | 5.7x |
| MI | BNCI2014_001 | ATCNet | 18 | accuracy | 0.690 ± 0.142 | 0.693 ± 0.140 | -0.003 ± 0.034 | 131 s | 861 s | 6.6x |
| MI | Lee2019_MI | EEGNet | 108 | accuracy | 0.692 ± 0.147 | 0.693 ± 0.144 | -0.001 ± 0.053 | 95 s | 1830 s | 19.3x |
| MI | Lee2019_MI | ATCNet | 108 | accuracy | 0.673 ± 0.160 | 0.676 ± 0.157 | -0.004 ± 0.049 | 614 s | 7829 s | 12.7x |
| P300 | BNCI2014_009 | EEGNet | 30 | roc auc | 0.949 ± 0.046 | 0.950 ± 0.045 | -0.000 ± 0.002 | 13 s | 89 s | 7.1x |
| P300 | BNCI2014_009 | ATCNet | 30 | roc auc | 0.754 ± 0.074 | 0.753 ± 0.076 | +0.001 ± 0.016 | 38 s | 276 s | 7.2x |
| SSVEP | Nakanishi2015 | EEGNet | 45 | accuracy | 0.833 ± 0.238 | 0.841 ± 0.216 | -0.009 ± 0.070 | 21 s | 66 s | 3.2x |
| SSVEP | Nakanishi2015 | ATCNet | 45 | accuracy | 0.955 ± 0.100 | 0.962 ± 0.090 | -0.007 ± 0.032 | 136 s | 1302 s | 9.6x |

Each speedup is 2-15% below the sweep's best speedup for training alone,
which leaves out evaluation. The CSVs are in
[`recipes/moabb/results/h100-mig/recipes-2g.20gb`](recipes/moabb/results/h100-mig/recipes-2g.20gb).

ATCNet was designed for motor imagery. On P300, the 0.8 s trials are shorter
than it needs, so it shrinks its kernels, pools and windows, and it does much
worse than EEGNet. These recipes show that packing gives the same results
across tasks; they are not tuned for accuracy.

**BNCI2014001 with ATCNet** uses ATCNet's defaults and the official code's
training choices that keep members independent: 4.5 s trials at 250 Hz (from
0.5 s before the cue, 1125 samples), max-norm on the conv kernels
(`conv_max_norm_const: 0.6`) and its L2 weight decay
(`PackedATCNet.source_optimizer_param_groups()`, as in braindecode's
`ATCNet`). Unlike the official code, it uses braindecode's preprocessing, a
cosine schedule over 500 epochs, and no model selection on the test session.
The other ATCNet recipes use the same model and training. `PackedATCNet`
implements `conv_max_norm_const` itself, but braindecode's `ATCNet` has it
only from braindecode 1.7 (Python >= 3.11), so `--baseline` with the ATCNet
recipes needs that version; on older braindecode, use
`--set model.conv_max_norm_const=null`, which drops the constraint from both.

**BNCI2014001**, seeds 0, 1 and 2 (54 runs), EEGNet:

| | accuracy | kappa | time |
|---|---|---|---|
| PackedEEGNet, packs of 18 | 0.669 ± 0.165 | 0.559 ± 0.221 | 29 s |
| braindecode EEGNet, one at a time | 0.670 ± 0.162 | 0.560 ± 0.216 | 131 s |

ATCNet:

| | accuracy | kappa | time |
|---|---|---|---|
| PackedATCNet, packs of 18 | 0.687 ± 0.152 | 0.583 ± 0.202 | 279 s |
| braindecode ATCNet, one at a time | 0.691 ± 0.146 | 0.588 ± 0.194 | 2926 s |

## Benchmarks

`benchmarks/bench_packing.py` measures seconds of training per model per epoch
on random data shaped like BNCI2014001 (22 electrodes, 4 classes, batch 64)
for three ways of training K models:

- **braindecode:** one braindecode model at a time.
- **torch.func:** PyTorch's ensembling recipe over K braindecode models (see
  [Why not torch.func?](#why-not-torchfunc)).
- **packed:** the brainfold model, for several pack sizes K.

```bash
python benchmarks/bench_packing.py                   # EEGNet, 4608 trials
python benchmarks/bench_packing.py --n-trials 288
python benchmarks/bench_packing.py --model atcnet
python benchmarks/bench_packing.py --repeats 5 --out eegnet.csv   # median of 5, every repeat to a CSV
```

4608 trials is a leave-one-subject-out training set and 288 a
leave-one-session-out one. The script currently times CUDA devices only. It
prints the GPU and the torch, CUDA and cuDNN versions, and with `--repeats` the
spread of the repeats, (max - min) / median.

The results below come from a workstation that also drives a display.
[`cluster/drac`](cluster/drac) runs this benchmark, and `sweep.py` on every
recipe, on H100 MIG slices of a DRAC cluster, with the environment recorded.

### RTX 4080 SUPER

EEGNet (513 samples):

| training set | EEGNet | braindecode | torch.func, K=18 | packed, K=4 | K=9 | K=18 | K=36 |
|---|---|---|---|---|---|---|---|
| 4608 trials | F1=4, D=2 | 0.111 | 0.091 (1.2x) | 0.062 | 0.028 | **0.017 (6.7x)** | 0.017 |
| | F1=8, D=2 | 0.115 | 0.190 (0.6x) | 0.062 | 0.028 | **0.027 (4.3x)** | 0.028 |
| 288 trials | F1=4, D=2 | 0.0075 | 0.0057 (1.3x) | 0.0044 | 0.0020 | **0.0011 (7.1x)** | 0.0011 |
| | F1=8, D=2 | 0.0076 | 0.0119 (0.6x) | 0.0044 | 0.0020 | **0.0017 (4.5x)** | 0.0017 |

ATCNet (defaults, 1125 samples):

| training set | braindecode | torch.func, K=18 | packed, K=1 | K=4 | K=9 | K=18 | K=36 |
|---|---|---|---|---|---|---|---|
| 4608 trials | 1.169 | 1.130 (1.0x) | 0.522 (2.2x) | 0.242 | 0.183 | **0.149 (7.8x)** | 0.138 (8.5x) |
| 288 trials | 0.080 | 0.073 (1.1x) | 0.043 (1.9x) | 0.016 | 0.012 | **0.011 (7.6x)** | 0.010 (8.2x) |

On this GPU, throughput stops improving at about K=18. The best pack size
depends on the GPU's compute, as the H100 results below show.

### H100 MIG slices

`sweep.py` on every recipe's own data and training loop, on two MIG slices of
an H100 80GB on DRAC's Fir cluster: 1g.10gb (1/7 of the GPU's compute, 10 GB)
and 2g.20gb (2/7, 20 GB). Training time per model per epoch against pack size
K, with one braindecode model dashed and each slice's elbow circled (the
smallest K within 10% of its best time per model):

![Time per model per epoch against pack size K, on H100 MIG slices](recipes/moabb/results/h100-mig/sweep-time.png)

| | BNCI2014_001 | BNCI2014_009 | Lee2019_MI | Nakanishi2015 |
|---|---|---|---|---|
| EEGNet: elbow on 1g.10gb, 2g.20gb | 12, 24 | 16, 32 | 8, 16 | 16, 32 |
| EEGNet: best speedup on 2g.20gb | 6.7x | 7.8x | 19.8x | 3.6x |
| ATCNet: elbow on 1g.10gb, 2g.20gb | 6, 12 | 24, 48 | 4, 8 | 8, 16 |
| ATCNet: best speedup on 2g.20gb | 6.9x | 8.3x | 13.0x | 10.6x |

- Packing is limited by compute, not memory: with twice the compute, every
  elbow is at twice the pack size, and the best time per model halves. Up to
  its elbow, every pack used less than 2 GiB on top of the data.
- Both methods use the same training loop with the data on the GPU, so the
  K=1 difference is the models' implementations. A pack of one is up to 8.7x
  faster than braindecode on Lee2019_MI, but slower on BNCI2014_009 and on
  Nakanishi2015 with EEGNet, where packing pays from K=2.

Each pack size's memory, predicted by `sweep.py` from the smaller ones before
it ran, against the memory it used (reserved by PyTorch, plus the CUDA context
and libraries); the dotted lines are 90% of each slice, where the sweep stops:

![Predicted against measured GPU memory per pack size](recipes/moabb/results/h100-mig/sweep-memory.png)

87% of the predictions are within 5%, but PyTorch reserves memory in steps,
so a few are up to 13% low: more than the 10% headroom. No pack size ran out
of memory. The CSVs, with each job's environment, are in
[`recipes/moabb/results/h100-mig`](recipes/moabb/results/h100-mig).

### Contributing results

Results from other GPUs are welcome. Run the commands above with
`--repeats 5 --out ...` and open a pull request adding a section for your
hardware, with the GPU, driver, CUDA, PyTorch and braindecode versions.

## Why not torch.func?

The standard way to train K copies of a model at once is `torch.func`:
`stack_module_state` on K models, then `vmap` over `functional_call`. On
braindecode's `EEGNet` it is exact: forward passes and BatchNorm running
statistics match K separate models in float64. It just isn't fast here (see
[Benchmarks](#benchmarks)): 1.2x with F1=4, and *slower* than one model at a
time with F1=8.

- **It batches the computation as written.** EEGNet's first block filters every
  electrode at full time resolution before mixing electrodes, and that work
  grows with K. `vmap` can only run it K times in parallel. brainfold mixes
  electrodes first (below), which does about 11x less work in that block.
- **Batched convolutions become grouped convolutions.** On CUDA these run in
  PyTorch's generic depthwise kernel (`conv_depthwise2d`), not cuDNN. In a
  profile of one K=18 training step, those kernels take more than half of the
  GPU time.
- braindecode's max-norm constraint uses `renorm`, which has no `vmap`
  batching rule and falls back to a loop over members (PyTorch warns about
  it). Replacing it with an equivalent batchable expression did not change
  the timing measurably, so it isn't the bottleneck.

Packing the later layers, which are small and launch-bound, into grouped
convolutions does the rest of brainfold's speedup.

## How it works

**Packing.** The K members sit side by side on the channel axis. Every conv is
grouped so that member k only sees its own channels. BatchNorm, dropout and
max-norm act per channel or per output row anyway. Repeated sub-networks within
a model, such as ATCNet's window branches, are packed the same way.

**Spatial-first first block.** EEGNet and ATCNet both start with a temporal
conv `w_f` (length L, F1 filters), its BatchNorm (gamma, beta) and a depthwise
spatial conv `s_g` (D per temporal filter, over C electrodes). All three are
linear in the input x. For spatial filter g reading temporal filter f:

```
z_g = a_f * (w_f * (s_g . x)) + (beta_f - a_f * mu_f) * sum(s_g),   a_f = gamma_f / sqrt(var_f + eps)
```

So the C electrodes are mixed into F1*D signals first (one matmul), and the
length-L filter then runs on F1*D signals instead of F1*C electrode channels:
11x less for C=22, D=2. The (F1, C, T) tensor is never built.

In training, BatchNorm needs the batch mean and variance of `u_f = w_f * x` over
(batch, electrodes, time). Both are quadratic in `w_f`:

```
mu_f = w_f . e,    E[u_f^2] = w_f^T M w_f,    var_f = E[u_f^2] - mu_f^2
```

Here `e` (length L) and `M` (L x L) are the window means and lag second moments
of the zero-padded input. `M` comes from the FFT autocorrelation of the input,
minus the products each window leaves out at the two edges, read off small
Gram matrices of the edge samples. Neither needs gradients. Both are means of
per-trial moments, so for a fixed dataset they can be computed once per trial
(`input_moments`) and averaged over each batch, instead of computed with FFTs
over every batch. Gradients reach
w, gamma, beta and s through `mu_f` and `var_f` exactly as through the original
BatchNorm, and the running statistics are updated the same way.

## Keeping members independent

A packed model is one `nn.Module` whose parameters hold all K members side by
side. Each member trains exactly as it would alone only if **every operation
in the training loop is separable across members**: it acts on each parameter
element on its own, or it is a sum of per-member terms. Anything that computes
one statistic over the whole pack (a norm, a maximum, a scale, a threshold, a
shape) couples the members, and one member's training then depends on what it
is packed with.

Separable, so safe:

- The summed loss `losses.sum()`. A *mean* over members also keeps them
  separate, but scales every gradient by 1/K, which changes SGD's effective
  learning rate and Adam's balance against `eps`.
- Elementwise optimizers: SGD (with momentum), Adam, AdamW, RMSprop. Weight
  decay, gradient accumulation, EMA and SWA weight averaging.
- Learning-rate schedules that depend only on the step (cosine, step, warmup).
- Data-parallel training (DDP averages gradients elementwise) and `torch.compile`.

Couples the members:

- **Global gradient clipping:** `clip_grad_norm_(model.parameters(), ...)` uses
  one norm for the whole pack. A member with large gradients shrinks everyone
  else's step. Clip each member's slice separately, or not at all.
- **Mixed-precision loss scaling:** `torch.amp.GradScaler` keeps one scale and
  skips the *whole* optimizer step when any gradient is inf or NaN, so one
  member's overflow skips a step for all of them. bf16 autocast needs no scaler
  and is fine.
- **Optimizers with per-tensor or cross-element statistics:** layer-wise trust
  ratios (LARS, LAMB), factored or matrix preconditioners (Adafactor, Shampoo,
  Muon), and SAM's global perturbation norm all look across a packed tensor or
  the whole model. Check that an optimizer's state is per element before using it.
- **Metric-driven schedules on an aggregate:** `ReduceLROnPlateau` or early
  stopping on the pack's mean validation loss ties every member to the others.
  Track metrics per member, and keep each member's best weights with
  `member_state_dict(k)`.
- **Re-initializing parameters** with `torch.nn.init` on the packed tensors:
  initializers that use the fan-out or the whole matrix (`xavier_*`,
  `orthogonal_`) see the packed shape, not one member's. Use `from_seeds` or
  `load_state_dicts` instead.
- **Regularizers or pruning over the whole model** that aren't sums of
  per-member terms, such as a penalty on the total weight norm (not squared)
  or a global magnitude-pruning threshold.

Shared by construction:

- **Hyperparameters.** All members share the architecture, optimizer settings,
  learning-rate schedule and batch size. Pack runs that differ only in seed
  and data (folds, subjects), not runs from a hyperparameter sweep.
- **Dropout's RNG** (see [Caveats](#caveats)).

## Caveats

These apply to every model; see [Model notes](#model-notes) for
model-specific ones.

- **Dropout shares one RNG across the pack.** A member's initial weights and
  data order depend only on its own seed, but its dropout masks depend on which
  runs it is packed with. Results are reproducible for a fixed pack
  composition, not per run.
- **Packed runs need equal training-set sizes:** every step draws one batch per
  member. Subsample to equal sizes or group runs by size.
- **Float32 variance in the first block:** `var = E[u^2] - mu^2` can lose
  precision when the temporal-filter output has a large mean relative to its
  spread. On roughly standardized inputs it matches braindecode to about 1e-4
  in float32 (tested), and the recipes' exponential moving standardization
  keeps inputs in that range. Be careful with raw, un-centered inputs.

## Roadmap

brainfold is for EEG decoding only. It is growing in three directions:

- **Hardware:** benchmark results on more GPUs, to show how the speedup and
  the best pack size depend on the device.
- **Datasets:** more recipes, covering more datasets, paradigms and evaluation
  protocols.
- **Models:** more braindecode architectures.

Backlog:

- **Per-member hyperparameters.** Adam and coupled weight decay act element by
  element, so the learning rate and the weight decay could be per-member
  tensors without coupling the members, and so could the dropout rate, as a
  per-group mask. One pack could then train a grid of hyperparameters. The
  natural grouping is still a single configuration over many seeds, sessions or
  subjects, which lowers the variance of its validation estimate; a grid would
  multiply the members needed.
- **Pruning finished members.** For early stopping, `subset(members)` would
  return a smaller pack, with the matching slices of the optimizer state.
  Freezing a member in place is not enough: Adam keeps moving it through its
  momentum and weight decay, and BatchNorm keeps updating its running
  statistics.

## Adding a model

A new packed model subclasses `PackedModel` (`src/brainfold/_packing.py`),
which provides the common interface from four methods: `_layout` (where each
member tensor lives in the packed parameters), `_counters`,
`braindecode_kwargs` and `_kwargs_from_braindecode`. To be merged, it needs:

- tests against K separate braindecode models in float64: forward passes,
  BatchNorm running statistics and Adam training steps, for the default
  configuration and the main variants;
- a `ValueError` for any option that would couple the members or that it
  doesn't support;
- an entry in `benchmarks/bench_packing.py` and the [Models](#models) table.

## License

MIT
