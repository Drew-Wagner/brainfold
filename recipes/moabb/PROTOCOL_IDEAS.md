# Improving SpeechBrain-MOABB's search protocol

Ideas for a better hyperparameter search protocol than SpeechBrain-MOABB's
(reimplemented in `sb_protocol.py` and `sb_search.py`), and what packed training
makes affordable now. The longer research plan for adaptive allocation is the
doc "Adaptive Subject and Seed Allocation for EEG Hyperparameter Search"
(https://claude.ai/artifact/GQvWNHe2SqdaProVMcPzMR).

## The problems

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

## Ideas

| # | Idea | Status |
|---|---|---|
| 1 | Search preprocessing in its own step | idea |
| 2 | EEG-specific augmentation | idea |
| 3 | Validation by cross-validation within the training data | `nested_search.py` (blocks of runs) |
| 4 | Several seeds per trial, different between trials | `nested_search.py` |
| 5 | Adaptive allocation of folds and seeds (racing) | proposal doc |
| 6 | Choose for size as well as accuracy | `nested_search.py` (one paired standard error) |
| 7 | Measure the search's own variance | idea |
| 8 | Schedule-free: epochs chosen on validation, not searched | running (job 65405367, task 1) |
| 9 | Search BatchNorm momentum | running (job 65405367, task 0) |

### 1. Preprocessing as its own step

fmin, fmax, the time window and the electrode selection are probably what matter
most. Today they are searched jointly with 13 architecture and training
hyperparameters, in 50 trials.

- Step 0: search preprocessing alone with a fixed, reasonable architecture (their
  default EEGNet), with several seeds per trial (idea 4). Preprocessing a new
  setting takes 11.5 s for BNCI2014001 (`Recordings`, cached), so the trials
  are as cheap as the model's training.
- The two steps interact: kernel and pooling sizes are in samples, so a shorter
  window or a different sample rate changes what they mean. Expressing temporal
  kernels in seconds would make the architecture step less dependent on step 0.
- Also worth searching: tmin (fixed at 0 now), and the resampling rate.
- Check with fANOVA importances on the current runs (Optuna's
  `get_param_importances`) before committing to it.

### 2. EEG-specific augmentation

Replace step 2's speech augmentations with EEG ones, e.g. from
`braindecode.augmentation`: frequency shift, FT surrogate, channel dropout,
smooth time mask, band-stop filtering, sign flip, mixup.
Rommel et al. (2022, "Data augmentation for learning predictive models on EEG:
a systematic comparison") compared these, motor imagery included; start from
the ones they found useful. Also test whether augmentation helps at all once
the validation is less noisy (idea 3); our step 2 will show what theirs adds.

### 3. Cross-validation within the training data

- k folds over the training session(s) instead of one 20% split, so every
  training trial is validated once: about 5x more validation trials, and 5x the
  training.
- Group the folds by run (BNCI2014001 has 6 runs per session) rather than by
  interleaved trials, so validation is temporally separate from training, as
  the test session is. This is the closest within-subject stand-in for the
  cross-session shift (the proposal doc's RQ4).
- With packing, k x 18 folds become one or a few packs. EEGNet-sized models
  packed 90 to a pack should cost far less than 5x in wall clock (to measure).

### 4. Seeds per trial

- Several seeds per trial, and a different seed set per trial, so TPE doesn't
  fit one seed's noise.
- Packing makes seeds nearly free until the pack fills the GPU: the marginal
  cost of a model is low while a pack is small (see the pack-size sweeps,
  `sweep.py`).

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

### 6. Size as an objective

- Select the smallest configuration whose validation is within one standard
  error of the best (needs ideas 3 or 4 for the standard error); or
- multi-objective search (accuracy, parameters or FLOPs) with Optuna's
  multi-objective samplers, reporting the Pareto front.

### 7. Measure the search's variance

- Run step 1 with several TPE seeds (each also changing the trial seeds), and
  check whether the best configurations cluster in configuration space and in
  test accuracy. Adam's step 1 takes about 1.2 h on an H100 2g.20gb slice.
- Rerun the top ~10 step-1 configurations with 5 seeds each (about 1 GPU-hour)
  to see whether they can be told apart at all.
- The proposal doc's RQ1 (variance split into subject, subject x configuration,
  seed and validation noise) generalises this.

### 8. Validate on spoken cues, and search the window's start (BCI Competition IV data set 1)

Drew's idea (November 2026), for after the first search on data set 1 (job 2178119),
which validates on the calibration session (visual cues), like the training data.

- **Validate on the evaluation session (spoken cues),** so that the search chooses
  configurations that carry over to a different cue. A visual response to the arrow
  can't help there; motor imagery can.
- **Make `tmin` free** (fixed at 0 in SpeechBrain-MOABB, and so far here), and see
  whether the search moves the window away from the cue onset towards the imagery
  itself: does it learn to "maximize" the motor imagery signal?
- **To work out first:**
  - Something must stay untouched for testing. Split the evaluation session's 4 runs:
    validate on some, test on others, or nest over them.
  - Search `tmin` with a duration (or `tmax > tmin`).
  - The interval is 0-4 s, and evaluation trials shorter than 4 s of imagery are
    dropped (`min_duration`). A later window needs a longer interval, and fewer
    trials last long enough (imagery lasts 1.5-8 s).

## What makes this affordable now

- SpeechBrain-MOABB reports about 8 h on a V100 for one full experiment
  (two search steps and the 10-seed evaluation).
- brainfold trains all 18 folds of a trial as one pack: step-1 trials take
  18-300 s on a 2g.20gb slice, and their tuned config (862 epochs, with
  augmentation) takes 616 s on an RTX 4080 SUPER.
- Schedule-free AdamW reaches the recipes' accuracy in about a third of the
  epochs (branch `convergence`), which would cut every trial's cost further
  once the epoch count is chosen.
