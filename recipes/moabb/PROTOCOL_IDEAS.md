# Improving SpeechBrain-MOABB's search protocol

Ideas for a better hyperparameter search protocol than SpeechBrain-MOABB's
(reimplemented in `sb_protocol.py` and `sb_search.py`), and what packed training
makes affordable now. The longer research plan for adaptive allocation is the
doc "Adaptive Subject and Seed Allocation for EEG Hyperparameter Search"
(https://claude.ai/artifact/GQvWNHe2SqdaProVMcPzMR).

The protocol itself, the rules for changing it and the log of variants run are
in `PROTOCOL.md` (version 2, draft). This file keeps the ideas and their
implementation notes.

Ideas 1-4, 6, 7 and 11 are implemented behind options that are off by default, so
`hparams/sb_moabb/bnci2014001_eegnet.yaml` still runs SpeechBrain-MOABB's
protocol; `hparams/sb_moabb/improved/bnci2014001_eegnet.yaml` turns them on.
The README's "The improved protocol" has the commands and measured costs.

## The problems

These are what we observed on BNCI2014001; they motivate the ideas, but no
default of the improved protocol is chosen from them (see "Development vs
held-out datasets").

- **The best configuration doesn't reproduce.** Each trial is scored once (one
  seed, 1986, the same for every trial) on a small validation split. The top of
  the search landscape is flatter than that noise, so the argmax is largely the
  luckiest trial, and TPE tunes to that one seed's draw of initial weights and
  batch order. Their 10-seed final evaluation measures the chosen
  configuration's variance, not the search's.
- **Nothing justifies the model's size.** Nothing in the search favours smaller
  models, so the winner's size is wherever the search landed. Their tuned
  BNCI2014001 EEGNet has 146k parameters and 84 MFLOPs per trial. Halfway
  through our step 1 (job 65405367, Adam), a trial with 11.8k parameters and
  20 MFLOPs reaches the same validation accuracy (0.803).
- **The validation split is small, and unlike the test.** Their validation set
  is 20% of each class in the training session, taken at equally spaced
  positions. That's about 58 trials per fold, interleaved in time with the
  training trials, while the test is a whole other session. In our
  reproduction of their tuned config, validation was 0.803 against 0.723 on
  test.
- **The augmentation is SpeechBrain's speech defaults** (CutCat, RandAmp,
  RandomShift, AddNoise), searched in step 2, and not chosen for EEG.
- **BatchNorm momentum is fixed at 0.01**, so the running statistics lag about
  100 steps (branch `convergence`). Our run searches it; Adam's best so far
  uses 0.87.

## Development vs held-out datasets

BNCI2014001 is the development dataset. We have used it for everything so
far (the recipes, the speed and convergence options, the reproduction), and
SpeechBrain-MOABB tuned their defaults on it. Results on it are exploratory:
they generate hypotheses and catch bugs, but they don't test the protocol.

- Every design decision of the improved protocol is taken from the
  literature, from SpeechBrain-MOABB's priors, from first principles or from
  cost (time and memory), and written down with its reason, in the hparams
  file or here. None is chosen from accuracy on BNCI2014001: not which
  hyperparameters go in step 0, not the augmentations or their ranges, not the
  one-standard-error threshold, not the number of inner folds or seeds.
  Local smoke tests measure cost and correctness only.
- Exception, written down: Rommel et al. (2022)'s motor-imagery results, from
  which the augmentations are chosen, were measured on BCI IV 2a, which is
  BNCI2014001. On BNCI2014001, the augmentation choice is therefore not
  independent of the data; on held-out datasets it is.
- Before any held-out dataset is run, the protocol is frozen: the code, the
  hparams template and this document, at a tagged commit. The held-out runs then
  only add an hparams file per dataset, changing only the dataset block and
  what the template marks as per dataset (sample rate, number of classes, run
  structure).
- Held-out motor-imagery candidates, none used by any of our choices so far:
  BNCI2014004, Cho2017, PhysionetMI, Zhou2016, Schirrmeister2017. Lee2019_MI is
  partly used: the convergence settings (schedule-free at lr 0.005) were
  confirmed on it, untuned, so it shouldn't count as held out for those.
- What a held-out dataset needs from the code: `Recordings` reads any MOABB
  motor-imagery dataset, and the cells come from MOABB's subject, session and
  run metadata, so nothing is BNCI2014001-specific. Two structural limits:
  leave-one-session-out needs at least two sessions per subject
  (PhysionetMI and Schirrmeister2017 have one session in MOABB, and would
  need an outer split within the session, not implemented); and
  `inner_folds: runs` needs at least two runs in the training session(s)
  (otherwise use `inner_folds: 5` with `group_by_run: false`, as the template
  says). Channel selection by adjacency steps from Cz needs a montage with Cz.
  Version 2 (`PROTOCOL.md`) has rules for all three: `protocol.outer_folds:
  rule`, `protocol.inner_folds: rule` and `protocol.channels: all`.
- Choosing the multi-fidelity or allocation algorithm (ideas 5 and 10) is part
  of this design, and follows the same rule: the candidates are compared on
  BNCI2014001 only, for the quality of the configuration they select against
  the compute they spend; the best or most efficient is picked, frozen, and its
  expected result written down; the held-out datasets then check that the
  ranking and the efficiency hold. That check is a reported result, not a
  step for re-tuning. The proposal doc's replay design makes the comparison
  cheap and fair: a dense table of cells for a pool of configurations (every
  fold and inner fold, several seeds), with validation scores at a fixed epoch
  grid, lets every candidate be replayed offline on identical data, like a
  tabular benchmark. Every trial now saves such a table
  (`trials/*.cells.npz`: each cell's validation and test scores at every
  evaluated epoch; with schedule-free, the `schedule_free.epochs` grid, with
  BatchNorm statistics recomputed before each evaluation).

## Ideas

| # | Idea | Status |
|---|---|---|
| 1 | Search preprocessing in its own step | implemented (`steps`, step 0); dropped in version 2, which fixes preprocessing (`PROTOCOL.md`) |
| 2 | EEG-specific augmentation | implemented (`augment: eeg`) |
| 3 | Validation by cross-validation within the training data | implemented (`protocol.inner_folds`) |
| 4 | Several seeds per trial, different between trials | implemented (`protocol.seeds_per_trial`, `trial_seeds`) |
| 5 | Adaptive allocation of folds and seeds (racing) | proposal doc; cells set up for it, not implemented |
| 6 | Choose for size as well as accuracy | implemented (`protocol.selection: one_se`, `multi_objective`) |
| 7 | Measure the search's own variance | implemented (`--tpe-seed`, `--trial-seed`, `--rerun`, `sb_analyze.py`) |
| 8 | Schedule-free: epochs chosen on validation, not searched | running (job 65405367, task 1) |
| 9 | Search BatchNorm momentum | running (job 65405367, task 0) |
| 10 | Multi-fidelity over epochs | idea; algorithm not chosen |
| 11 | Selection nested by test session | implemented (`protocol.nested_by_test_session`) |

### 1. Preprocessing as its own step

fmin, fmax, the time window and the electrode selection were searched jointly
with 13 architecture and training hyperparameters, in 50 trials.

Implemented: the yaml's `steps` (instead of `search`) lists the steps in order,
each with its priors, its augmentation, its own fixed values, and optionally its
own number of trials and selection rule. A later step starts from the values
earlier steps chose. The improved yaml has:

- Step 0: fmin and fmax (their priors), tmin (0-1 s after the cue: motor
  imagery's desynchronization starts within it), the window length `twindow`
  (their tmax prior, 1-4 s, as a length, cut at the trial's end) and the
  number of adjacency steps from Cz (their prior), on a fixed reference model:
  Lawhern et al. (2018)'s EEGNet-8,2 with their within-subject dropout (0.5),
  Adam's default learning rate and a batch of 32 (the middle of their prior).
  The sample rate can be searched too (`sample_rate`; per dataset, at most the
  recording's rate), but isn't by default.
- Step 1: the architecture and training, their priors.
- Temporal kernel and pool sizes can be given in seconds
  (`cnn_temporal_kernel_s`, `cnn_septemporal_kernel_s`,
  `cnn_septemporal_pool_s`; converted with the sample rate), so step 1 doesn't
  depend on step 0's sample rate. The improved yaml uses them, with their
  sample-unit priors converted at 125 Hz.

Not done, by the rule above: deciding whether step 0 is worth it from fANOVA
importances on BNCI2014001 runs. `sb_analyze.py` reports importances as an
analysis.

### 2. EEG-specific augmentation

Implemented: `augment: eeg` uses `sb_protocol.eeg_augment`, braindecode's
TimeReverse, SignFlip, FTSurrogate, FrequencyShift, BandstopFilter,
SmoothTimeMask and ChannelsDropout, and Mixup, on packed batches. Each
transform has a probability and a magnitude. Every random draw is a hash of
the member's seed, the optimizer step and the position in the batch, so a
member gets the same augmentation in any pack as alone (tested: packed
training equals training each member alone, in float64, and each transform
equals braindecode's functional given the same draws). BandstopFilter zeroes
the band's FFT bins instead of MNE's FIR notch filter (which runs on the CPU,
one example at a time). The augmented trials replace the originals, as
braindecode's AugmentedDataLoader, rather than SpeechBrain's added copies.

The improved step 2 searches, from Rommel et al. (2022, J. Neural Eng.,
"Data augmentation for learning predictive models on EEG: a systematic
comparison"), each transform on (probability 0.5, as in their experiments) or
off, and its magnitude over their range: FTSurrogate (phase noise 0-1 x 2pi),
FrequencyShift (0-3 Hz), BandstopFilter (0-2 Hz), SmoothTimeMask (0-2 s),
ChannelsDropout (p_drop 0-1), TimeReverse; SignFlip too (no effect there,
which they explain by their model squaring its activations; EEGNet doesn't).
Left out: ChannelsShuffle and ChannelsSymmetry (harmful for motor imagery
there), GaussianNoise (no robust gain there), SensorsRotation (not
significant; not implemented), Mixup (not in their comparison; available).

Dropped, by the rule above: testing whether augmentation helps at all on
BNCI2014001 as a decision criterion.

To do: keep "no augmentation" as a candidate in the augmentation step.
SpeechBrain-MOABB takes step 2's best even when it is worse than step 1's
best, which trained without augmentation. Our schedule-free run did exactly
that (job 65405367, task 1): step 1's best scored 0.801 on validation, step
2's best only 0.775, and the final, trained with step 2's augmentation, scored
0.767 on validation and 0.707 ± 0.003 on test, against Adam's 0.729. Each
transform can be off in the improved space, but all off at once is one corner
that TPE may never sample. Fix: enqueue an all-off trial first, or compare
step 2's best with the previous step's best by the selection rule and keep
the better one.

### 3. Cross-validation within the training data

Implemented: `protocol.inner_folds` k folds over each outer fold's training
trials (SpeechBrain-MOABB's training and validation sets together), each
trial validated once. With `group_by_run`, a fold's validation set is whole
runs (consecutive (session, run) groups of MOABB's metadata), so validation is
temporally separate from training, as the test session is; `inner_folds:
runs` is one fold per run. Without, a fold is every k-th trial of each class,
the k-fold version of their equally spaced split. Each (subject, test session,
inner fold, seed) is a cell, a member of a pack; the trial's objective is the
mean over cells. The final evaluation can train on every training trial
(`final_train: all`).

The improved yaml uses one fold per run (6 on BNCI2014001): whole runs, 5/6
of the training data in each fold (close to the final training on all of it),
every training trial validated once.

### 4. Seeds per trial

Implemented: `protocol.seeds_per_trial` seeds per trial, crossed with the
folds, packed together. With `trial_seeds: per_trial`, a trial's seeds are
drawn from (base seed, trial number) by NumPy's SeedSequence, so they differ
between trials and are reproducible. Per-seed and per-inner-fold scores and
the objective's standard error are trial user attrs and columns of
`<step>-summary.csv`. The standard error is over the (seed, inner fold)
replicates, each averaged over the outer folds: it measures seed and
validation-split noise, which is what comparing configurations on the same
subjects needs; replicates share subjects and data, so it's optimistic.

With `seed_per_inner_fold`, each inner fold trains from its own seed (derived
from the trial's), so k folds average over k seeds at the cost of one. The
improved yaml uses that, with one seed per trial: 6 seeds and 6 replicates
per trial on BNCI2014001. Not more seeds crossed with the folds, for cost: past
about 18 cells, a trial's time grows about linearly with its cells on the RTX
4080 SUPER (see the README's measurements).

### 5. Adaptive allocation of folds and seeds

From the proposal doc: score configurations on shared (paired) subjects,
folds and seeds; race each challenger against the incumbent; and spend more
cells only where a comparison near the top is undecided, with subject offsets
modelled rather than standardized away. Clearly worse configurations stop
after a few cells.

What packing changes: the cost is per pack, not per cell. Allocation should
come in pack-sized batches, and a cell that fills a pack costs less than one
that starts a new pack. Ideas 3 and 4 multiply the cost of each trial; racing
is what keeps that affordable.

Set up for it (not implemented): a trial's unit of work is now a cell, (outer
fold, inner fold, seed). `sb_search.make_cells` lists a trial's cells and
`train_cells` trains any list of them, in packs of cells with equal set sizes,
returning each cell's scores at every evaluated epoch. An allocator can ask for
a subset of a configuration's cells (e.g. one pack's worth), read their scores,
and ask for more. The `.cells.npz` tables are the data to replay such an
allocator offline.

### 6. Size as an objective

Implemented: every trial records one member's parameter count and forward
FLOPs (`torch.utils.flop_counter`). `protocol.selection: one_se` passes on the
smallest configuration (by `size`: params or flops) whose validation is within
one standard error of the best's (Breiman et al., 1984; Hastie et al., 2009,
section 7.10), the conventional one-SE rule, rather than a threshold of ours.
`multi_objective: true` runs Optuna's multi-objective TPE on (validation,
size); its studies are named `<step>-<optimizer>-mo`. `sb_analyze.py` plots
validation against both sizes with the Pareto front.

### 7. Measure the search's variance

Implemented:

- `--tpe-seed` and `--trial-seed` set TPE's seed and the base of the trials'
  seeds, so independent replicate searches can be submitted (each to its own
  `--out`).
- `--rerun STEP --rerun-top N --rerun-seeds M` retrains a finished study's N
  best trials with M new seeds each (seeds no search trial used), writing
  `rerun-<step>.csv` and `rerun-<step>-summary.csv`. It works on the
  reproduction's studies too (their context is rebuilt from `best-*.yaml`).
- `sb_analyze.py`, from one or more result directories: validation against
  parameters and FLOPs with the Pareto fronts, fANOVA importances, the chosen
  configuration of each replicate search and their spread, and, with reruns,
  the rank correlation between search and rerun scores and which of the top
  trials can't be told apart.

### 10. Multi-fidelity over epochs

Schedule-free's iterates are usable at any epoch (no schedule to finish), so
the number of epochs is an anytime fidelity: a configuration's validation at
epoch 250 is available on the way to epoch 1000. Multi-fidelity methods could
stop poor configurations early. The algorithm is not chosen.

Candidate families: successive halving, Hyperband and ASHA; model-based ones
such as BOHB and DEHB; learning-curve and freeze-thaw methods such as DyHPO
and ifBO; Optuna's pruners, including the WilcoxonPruner, which prunes on
paired per-instance comparisons and so relates to racing over folds (idea 5).

What the choice depends on:

- epochs as an anytime fidelity, against folds and seeds as exchangeable
  instances (racing), or both;
- granularity: a pack stops or resumes as a whole, so promotion happens per
  pack of cells, and resuming needs the schedule-free optimizer's state saved
  in `fit` (not implemented);
- one GPU (sequential) or many workers (asynchronous);
- how noisy validation is at each fidelity;
- how well rankings at few epochs predict rankings at the final epoch, which
  may be measured on the development dataset only.

As for every design decision, the candidates are compared on BNCI2014001 (from
the `.cells.npz` tables) and the choice is frozen before the held-out datasets
(see "Development vs held-out datasets").

### 11. Selection nested by test session

A correction in the spirit of the proposal doc's RQ4 (the validation should
stand in for the cross-session test). Their objective averages validation
over every leave-one-session-out fold. For a subject with sessions A and B,
the fold that tests on B validates on part of A, and the fold that tests on A
validates on part of B; both scores go into one objective, and so into the
configuration chosen (and into TPE's trajectory). Each test session therefore
helped choose the configuration it's tested with.

Implemented: `protocol.nested_by_test_session` runs one search per test
session, on the folds that test on it, whose training and validation data
exclude it; each goes to `<out>/test-<session>/`, and the final evaluation of
each covers its folds (together, every fold). With two sessions: two searches
of 9 folds, the same number of models as one search of 18 (half the cells
per trial, the same trials per search). `--test-session` runs one of them
alone, e.g. as separate jobs. Off by default; on in the improved yaml.

### 12. Penalizing the visual confound softly, not cutting it out

Version 2 starts the window after the cue ends. That is the cleanest test of
whether a model or protocol learns motor imagery, but not necessarily the best
setting for a BCI, which would use the earliest useful data. A softer option
would keep the early window and make the cue signal hard or costly to use:

- **Channel swapping:** during training, swap or shuffle the ocular and
  frontal channels (EOG, Fp/AF) between trials of different classes, so they
  can't carry the class. If EOG channels are recorded, regressing them out
  is the classical version.
- **A penalty on low tmin:** if the window is searched, a prior or cost on
  windows that overlap the cue period (as in πBO's user beliefs, or as a term
  in the objective), so the search takes the cue period only when it pays
  for it.
- **An adversarial head:** a gradient-reversal branch that tries to predict
  a cue-locked signal (the evoked response, or EOG), so the features learn
  to ignore it.
- **Masking the cue period:** a time mask on the cue period, applied with
  some probability during training.

Each of these changes what is being measured, so it would be a separate
protocol, compared against version 2's hard rule, and the audit would show
how much of the cue each one removes. Not planned for version 2.

## What makes this affordable now

- SpeechBrain-MOABB's README gives about 8 h on a V100 for one full training
  experiment (training and evaluating a fixed configuration), not for the
  search, which it says takes hours to days.
- brainfold trains all 18 folds of a trial as one pack: step-1 trials take
  18-300 s on a 2g.20gb slice, and their tuned config (862 epochs, with
  augmentation) takes 616 s on an RTX 4080 SUPER.
- Schedule-free AdamW reaches the recipes' accuracy in about a third of the
  epochs (branch `convergence`), which would cut every trial's cost further
  once the epoch count is chosen.
- Measured cost of the improved protocol (README, "The improved protocol"): on
  the RTX 4080 SUPER, past about 18 cells a pack is compute-bound, so a trial's
  cost grows about linearly with its cells. One validation fold per run
  multiplies the cells by 6 (EEGNet-8,2 at 1000 epochs: 29 s for their 18
  cells, 139 s for 108); EEG augmentation makes small models about 2.8x
  slower. With selection nested by test session, each session's search has 54
  cells per trial, and the two run in parallel: about 9-10 h each on a 2g.20gb
  slice (estimate). This is why the improved yaml gives each inner fold its
  own seed instead of crossing several seeds with the folds, and why
  allocation (idea 5) or multi-fidelity (idea 10) matters next.
