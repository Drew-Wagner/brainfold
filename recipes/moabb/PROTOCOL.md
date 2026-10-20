# The search protocol, version 2 (draft)

A hyperparameter search protocol for cross-session, within-subject
motor-imagery decoding, developed on BNCI2014001 and meant to be applied
unchanged to new datasets. It is meant to extend to cross-subject decoding,
as a later version; this one, and the held-out comparison, are cross-session
only.
On a new dataset the protocol runs from scratch: it searches again and
chooses its own configuration. What has to generalize is the procedure, not
the configuration it found on BNCI2014001.

Status: draft, 2026-10-18; the spec was first committed at d7e4159 and
implemented at 49331e0 and dd80594 (`hparams/sb_moabb/v2/bnci2014001_eegnet.yaml`,
`sb_audit.py`, `cluster/drac/sb_v2.sbatch`; the README's "Version 2 of the
protocol" has the commands and measured costs), then revised after a review
of the literature (2026-10-18, "Changes after review" below) before its first
run, job 65792547. The log below gives the commit each variant ran. Version 1 (the "improved
protocol", `hparams/sb_moabb/improved/bnci2014001_eegnet.yaml`) ran as
job 65435553. `PROTOCOL_IDEAS.md` keeps the ideas and their implementation
notes; this file is the protocol itself, the rules for changing it, and the
log of what we ran.

## Rules for changing the protocol

Every protocol variant we score on BNCI2014001 is one more trial of a search
over protocols. With one development dataset and noisy outcomes, keeping the
variant that scores best finds an improvement by chance (Combrisson and
Jerbi, 2015, make the same point for decoding accuracy against chance). So:

1. **A change is justified before it is run**, from the literature, the
   paradigm, or reasoning about the method, and written here with its reason.
   A score on BNCI2014001 is never the reason.
2. **BNCI2014001 is for checking, not ranking.** It shows that the protocol
   runs, what it costs, and where it fails (bugs, confounds). We don't run
   variants on it to pick the better one. A comparison we want becomes a
   pre-declared variant, judged on the held-out datasets.
3. **Every variant run on BNCI2014001 goes in the log below**, including
   abandoned ones, so the final comparison can say how many were tried.
4. **The protocol reads everything dataset-specific from the dataset's
   metadata**, by rules written here: timings, montage, runs, sessions, sample
   rate. Where a dataset lacks something, the protocol says what to do. Nothing
   is decided case by case.
5. **Before any held-out dataset is run**, the protocol, its code and the
   comparison below are frozen at a tagged commit. The held-out runs then only
   fill in each dataset's metadata table. Results that disappoint don't reopen
   the protocol; a fix after the freeze is a new version, and the held-out
   results of the old one are still reported.

## Inputs from each dataset

Filled in from the dataset's paradigm paper and MOABB's documentation, before
anything is run on it, and committed with the protocol.

| Input | BNCI2014001 | Used for |
|---|---|---|
| Cue onset, relative to MOABB's epoch start | 0 s (MOABB's interval is 2-6 s of the trial; the arrow appears at 2 s) | Window |
| Cue offset, or "persists" | 1.25 s | Window, audit |
| End of imagery | 4 s (fixation cross disappears at 6 s) | Window |
| Sessions per subject | 2 | Outer folds |
| Runs per session | 6 | Inner folds |
| Recording rate and hardware filters | 250 Hz, 0.5-100 Hz, 50 Hz notch | Checks the band and rate are possible |
| Montage positions | 22 EEG channels, 10-20 | Audit |
| EOG channels | 3 (dropped) | Dropped |

Implementation choice: the table is the hparams file's `paradigm` block
(`cue_onset`, `cue_offset` or `persists`, `imagery_end`, `sessions`,
`runs_per_session`, `recording_rate`, `hardware_band`, `montage`,
`eeg_channels`, `eog_channels`). Where MOABB has the same information
(sessions per subject, runs per session, recording rate, EEG and EOG channel
counts), the code compares them and stops on any difference, rather than
warning; a null entry isn't checked. The hardware filters and the montage's
system aren't in MOABB's metadata (BNCI2014001's raw files say 0-125 Hz), so
they come from the table only. The splits themselves are computed from
MOABB's subject, session and run metadata, never from the table.

Held-out candidates: motor-imagery datasets with at least two sessions per
subject in MOABB (cross-session): BNCI2014004 (9 subjects, 5 sessions),
BNCI2015001 (12, 2; some subjects 3), Zhou2016 (4, 3), Shin2017A (29, 3) and
Lee2019_MI (54, 2; it was used to confirm schedule-free AdamW, so it isn't
fully held out). One-session datasets (Cho2017, PhysionetMI,
Schirrmeister2017) wait for the cross-subject version. Their tables are
filled before the freeze, so a dataset the rules can't handle shows up before
any result does. Known so far, from a first reading of their papers and
MOABB 1.7.2 (to confirm when the tables are filled):

- Zhou2016 and Lee2019_MI: the cue persists through the imagery (flagged).
- BNCI2014004, BNCI2015001 and Lee2019_MI: some sessions or runs give visual
  feedback that follows the class during the imagery (a smiley, a bar, a
  moving fixation cross). The protocol needs a rule for these (for example,
  only runs without feedback, or flag the dataset).
- Shin2017A: MOABB's epoch starts at the task, after a 2 s arrow, so the cue
  period is before the epoch and `check_paradigm` stops. The protocol needs a
  rule for a cue before the epoch (extend the epoch, or skip the cue
  conditions).
- BNCI2014004's runs per session and BNCI2015001's sessions per subject vary,
  so those entries are null (not checked).

## Preprocessing: fixed, not searched

Version 1 searched preprocessing (step 0). Version 2 fixes it, for two
reasons about the method, not about BNCI2014001's scores:

- **Noise.** Each validation fold has 48 trials on BNCI2014001: at 70%
  accuracy the binomial standard deviation alone is about 0.066 per fold, and
  about 0.009 pooled over all 2,592 validation trials. Preprocessing settings
  near the top differ by less than that, so the search picks the luckiest.
- **Shortcuts.** A search maximizes validation accuracy by whatever
  discriminates the classes, including what isn't motor imagery: the evoked
  response to a cue that shows the class, eye movements toward it, slow
  drifts. Models prefer stable, high-variance components over the weaker task
  signal (Lin, Wu and Jung, 2026, show this for subject identity and the
  aperiodic 1/f background in EEG foundation models). Preprocessing is where
  those decisions are made, so it should be set from physiology, not
  optimized.

The rules:

- **Band: 4-30 Hz.** Motor imagery's event-related desynchronization and
  synchronization (Pfurtscheller and Lopes da Silva, 1999) are in the
  sensorimotor rhythms, at 8-14 Hz and around 20 Hz, usually taken as 7-30 Hz
  (Blankertz et al., 2008). The 4 Hz edge: eye movements have most of their
  power below it, and Schirrmeister et al. (2017) high-passed at 4 Hz so that
  their models couldn't use them, at little cost on BNCI2014001 (FBCSP 68.0%
  at 0-38 Hz, 67.8% at 4-38 Hz). The 30 Hz edge: muscle activity is greatest
  at 20-30 Hz frontally and 40-80 Hz temporally (Goncharova et al., 2003) and
  dominates above 30 Hz; no band edge removes it, since it reaches into the
  beta band. FBCSP's and EEGNet's 4-40 Hz (Ang et al., 2012; Lawhern et al.,
  2018) is a convention for covering the range with 4 Hz filters, not a
  physiological edge. MOABB's band-pass, as now.
- **Sample rate: 125 Hz** (SpeechBrain-MOABB's), over four times the band's
  upper edge. A dataset recorded below 125 Hz keeps its own rate.
- **Window: from 0.5 s after the cue's offset to the end of imagery.** The
  removal of a visual pattern evokes its own response: the ISCEV VEP standard
  leaves 400 ms after a pattern's offset before the next onset (Odom et al.,
  2016), and the band-pass spreads a sharp event by about 100-200 ms. If the
  cue persists through the imagery, from 0.5 s after cue onset, past the main
  visual evoked response, and the audit reports the dataset as cue-confounded.
  On BNCI2014001: 1.75-4.0 s after the cue (the arrow is shown for 1.25 s;
  Brunner et al., 2008), 2.25 s, about the length of the usual 0.5-2.5 s
  window. This window is a test of whether a model learns motor imagery, not
  the best window for a BCI, which would use the earliest useful data: motor
  imagery's desynchronization starts about 0.3 s after the cue (Pfurtscheller
  et al., 2008). We found no literature against ending at the end of imagery
  (FBCSP: 0.5-2.5 s and 0.5-4 s weren't significantly different;
  Schirrmeister et al., 2017).
- **Channels: every EEG channel**, EOG channels dropped. No channel-selection
  search: EEGNet's spatial filters weight the channels, and a rule based on
  Cz's neighbours depends on the montage.
- **Normalization:** each trial z-scored over time per channel, as now.

Implementation choices:

- The protocol stops before training if the band isn't possible: `fmax` at
  or above the Nyquist frequency of the sample rate, or the band outside the
  table's hardware filters. The spec only says the table "checks" them; a
  dataset that fails needs a protocol decision, not a silent fallback.
- "The recording rate if lower" is the lowest rate among the subjects' raw
  files.
- The window is cropped as SpeechBrain-MOABB's `crop_signals`: the nearest
  samples, `tmax` excluded (1.75-4.0 s is 280 samples at 125 Hz, 1.752-3.984
  s).

The cost is accuracy: excluding the cue period and the slow band will likely
score below SpeechBrain-MOABB on BNCI2014001. How much of their accuracy the
cue period explains is what the audit measures.

## The search

As version 1 except where noted.

- **Step 1, architecture and training:** SpeechBrain-MOABB's priors, with
  kernel and pool sizes in seconds. TPE as theirs: 50 trials, 20 random first,
  24 expected-improvement candidates.
- **Step 2, augmentation:** the EEG transforms of Rommel et al. (2022), each on
  (probability 0.5) or off. New: the first trial is the one with every
  transform off, and the one-standard-error rule counts complexity as the
  number of transforms on, so no augmentation wins unless augmentation is
  better by more than one standard error. (Their protocol kept step 2's best
  even when it was worse than training without augmentation.)

  Implementation choices: the enqueued trial sets every searched `aug_*_p`
  to 0 and lets the sampler draw the magnitudes, which then have no effect;
  it counts as one of the 20 random startup trials. A resumed search doesn't
  enqueue it again if a finished or waiting trial already has it (an
  interrupted one is redone). Ties in the number of transforms go to the
  higher validation score; any trial with every transform off counts as no
  augmentation.
- **Epochs:** schedule-free AdamW; the epoch is chosen on validation from
  250-1000 in steps of 25.
- **Seeds:** a different seed per inner fold and per trial.
- **Selection:** the smallest model (parameters) within one standard error of
  the best; nested by test session. New: the standard error is paired, of each
  trial's difference from the best over the inner folds, which every trial
  shares. Version 1's, the best trial's own standard error over the folds,
  counts how much harder some runs are than others as noise between trials,
  so it admitted too many.
- **Final evaluation:** 10 seeds, trained on all of the training data, at the
  chosen epoch. New: with per-trial seeds, the final seeds are drawn from
  (seed base, a final-evaluation trial number), so replicate searches don't
  share them (version 1's were the base + 0..9, so replicates 1 and 2 shared 9
  of 10).

## Splits, by rule

- **Outer folds:** with two or more sessions, leave one session out, with the
  search nested by test session. With one session: the runs split into two
  contiguous halves used as two pseudo-sessions; with one run, two contiguous
  halves in time. New in version 2: version 1 had no rule for one session.
- **Inner folds:** one per run of the training data, consecutive runs grouped
  so there are at most 6. With fewer than 3 runs: 5 contiguous blocks in time.
  New in version 2: version 1 fell back to interleaved folds, which aren't
  separate in time from training.

Implementation choices:

- One session: the pseudo-sessions are named `<session>-half1` and
  `<session>-half2`; with an odd number of runs (or trials, for one run), the
  first half has the extra one. Each subject is split on its own, so a
  dataset where some subjects have two sessions keeps theirs.
- Runs are counted over all the training data (each (session, run) pair),
  in recording order; grouping into at most 6 folds splits them as evenly as
  `numpy.array_split` (earlier folds get the extra run). The 5 blocks in time
  are `numpy.array_split` of the training trials in recording order (across
  the training sessions, if more than one).

## The confound audit

Run on every dataset before the search, reported, never used for selection.
A fixed reference model (EEGNet-8,2, Lawhern et al., 2018, with the step 1
defaults of version 1) is trained with the inner folds above on:

1. the protocol's window;
2. the cue period alone (cue onset to cue offset, or onset to 0.5 s if the cue
   persists), in the protocol's band;
3. the cue period below 4 Hz (from the hardware band's low edge, at least
   0.5 Hz), where visual evoked responses and eye movements have most of their
   power;
4. the protocol's window without the frontal channels (10-20 names starting
   Fp, AF or F);
5. the EOG channels alone, on the protocol's window, if the dataset has any.
   They're never an input to the search's models (BNCI2014001's description
   says they "must not be used for classification"; Brunner et al., 2008).

Condition 2 holds the cue and early motor imagery together: imagery's
desynchronization starts about 0.3 s after the cue (Pfurtscheller et al.,
2008, 2013), so accuracy above chance there is expected and isn't by itself a
confound. The confound tests are 5 (eye movements that follow the class:
the EOG channels alone decode BNCI2014001's four classes at 72%; Özkahraman
et al., 2024), 3 (slow cue-locked activity) and the drop from 1 to 4
(frontal, possibly ocular or muscle, activity). The report states each against
the binomial threshold for the number of trials (Combrisson and Jerbi, 2015).

Regressing the EOG out of the EEG (Gratton, Coles and Donchin, 1983; Schlögl
et al., 2007) isn't part of the fixed preprocessing: three of the five
held-out candidates have no EOG channels in MOABB, and regression also
removes some EEG (Jung et al., 2000). It's a candidate audit condition or a
later variant (`PROTOCOL_IDEAS.md`, idea 12).

Implementation choices (`sb_audit.py`):

- "The step 1 defaults of version 1" are version 1's `fixed` values, the
  reference model of its step 0: EEGNet-8,2, dropout 0.5, learning rate
  0.001, batches of 32, with schedule-free as the search uses.
- The model trains to a fixed epoch, the last of the schedule-free grid
  (1000), and is scored there: choosing the epoch on the same validation
  folds would bias the accuracy upwards, and the binomial threshold assumes
  no such choice.
- "Starting Fp, AF or F" is read as the 10-20 row, the letters before the
  number or `z`: Fp1, AF3, F3 and Fz are frontal; FC and FT (fronto-central
  and fronto-temporal rows) are kept. Listing Fp next to F only makes sense
  for rows (as string prefixes, F would cover Fp). On BNCI2014001 this drops
  Fz only, so condition 4 tests little there; the rows are
  `audit.frontal_rows` in the hparams file.
- It covers every outer fold (all subjects and test sessions, not nested),
  and each condition shares the seeds. The threshold's number of trials is
  the validation predictions per seed: every training trial once per outer
  fold it trains in, which is once with two sessions; with three or more,
  a trial is counted once per outer fold, which makes the threshold
  optimistic. Thresholds are given pooled and per subject; the drop from 1 to
  4 is the mean over subjects with its paired standard error.
- The EOG channels go through the same MOABB preprocessing as the EEG (typed
  as EEG for MOABB's pipeline, which keeps only EEG channels), with the same
  model. On BNCI2014001 they're monopolar, so they also carry some frontal
  EEG. Test-session
  scores, computed as in any trial, are dropped.
- It writes `audit.csv`, `audit-cells.csv` and `audit.md` to its `--out`;
  `cluster/drac/sb_v2.sbatch` runs it once per dataset, in task 0, into
  `<config>-audit/` next to the replicate searches' `<config>-r<n>/`.

## Comparison on the held-out datasets (to settle before the freeze)

A draft; nothing below is run yet.

- **Protocols compared:** SpeechBrain-MOABB's, and version 2.
  Open question: also SpeechBrain-MOABB's with version 2's fixed preprocessing,
  to separate the effect of the search from the effect of the window.
- **Replicates:** 3 independent searches per protocol and dataset (TPE and
  trial seeds 0, 1, 2).
- **Outcomes:** test accuracy (mean over the replicates, each the mean of 10
  seeds); its spread across replicates; how far the replicates' chosen
  configurations agree; the validation-test gap; parameters and FLOPs of the
  chosen models; GPU-hours.
- **Statistics:** to choose, with the number of variants in the log counted
  in.

## Log of variants run on BNCI2014001

| # | Variant | Commit | Job | Result | Notes |
|---|---|---|---|---|---|
| 1 | SpeechBrain-MOABB's protocol, Adam | 1e4ea9e (tag `protocol-sb-repro`) | 65405367_0 | 0.729 ± 0.007 test (10 seeds), 11.8k parameters, 3 h 02 min | Not exactly theirs: BatchNorm momentum searched (they fix 0.01) |
| 2 | SpeechBrain-MOABB's protocol, schedule-free | 1e4ea9e (tag `protocol-sb-repro`) | 65405367_1 | 0.707 ± 0.003 test, 53.7k parameters, 6 h 01 min | Step 2 kept augmentation that lowered validation |
| 3 | Version 1 (step 0 searched), replicate 0 | a0c5b53 (tag `protocol-v1`) | 65435553 | 0.769 ± 0.003 test (10 seeds, both test sessions), 33.7k and 16.1k parameters, 13 h 48 min and 15 h 39 min | Step 0 chose fmin 2.4-2.6 Hz, tmin 0.2 s, so the window includes most of the cue period. Both test sessions' step 2 chose FrequencyShift and BandstopFilter. One replicate; not a reason to change the protocol (rule 2) |
| 4 | Version 2 (revised after review), replicate 0 | a57f7f1 (task 0 resumed at b239140, which only adds a GPU check to the sbatch) | 65792547 (submitted 2026-10-18); task 0 resumed as 65914336 | 0.558 ± 0.009 test on 0train and 0.579 ± 0.005 on 1test (10 seeds each), mean 0.568; 7.2k and 15.2k parameters; 8 h 28 min (resumed) and 9 h 17 min | Task 0's first node (fc10715) couldn't give torch its GPU, so its audit and first 8 step-1 trials ran on the CPU (18 h 16 min, cancelled); the resumed search redid the trial that was running, so step 1 has 51 trials, 50 of them finished. Audit (EEGNet-8,2, 5,184 validation trials, threshold 0.260): 0.525 on the window, 0.571 on the cue period, 0.600 on the cue period below 4 Hz, 0.523 without the frontal rows, 0.317 on the EOG channels alone. A classical check off the record (scratch script, not a variant run: CSP+LDA, tangent space+LR, XDAWN+tangent space+LR on the same cells) gives 0.564, 0.607, 0.531 on the window and 0.471, 0.519, 0.686 on the cue period below 4 Hz, so the cue period's information is a phase-locked slow response that waveform models read and band-power models don't. Step 2 chose sign flip, FT surrogate, frequency shift and bandstop (1test) and time reverse, sign flip, FT surrogate and frequency shift (0train). Lower than version 1 and the SB-MOABB protocol, whose windows include the cue period; one replicate, not a reason to change the protocol (rule 2). The first draft's audit was run once locally (RTX 4080 SUPER, 2026-10-16), for its cost, on that draft's 1.25-4.0 s window and 4-40 Hz band: validation accuracy 0.570 on the window, 0.561 on the cue period alone, 0.571 without Fz (5,184 trials; binomial threshold 0.260); not a variant run |

Before these, BNCI2014001 was also used for the recipes, the speed options and
the convergence options (README), and SpeechBrain-MOABB tuned their defaults
on it.

## To implement

Done (49331e0, dd80594):

- The metadata table and the window rule (`tmin`, `tmax` from the cue timings):
  the `paradigm` block, `protocol.window: paradigm`.
- All EEG channels, without the adjacency selection: `protocol.channels: all`.
- Step 2's all-off first trial and the transform count as complexity: per
  step, `first_trial: all_off` and `size: n_transforms`.
- The one-session outer split and the contiguous-block inner folds:
  `protocol.outer_folds: rule`, `protocol.inner_folds: rule`.
- The audit, as a script that writes its report next to the search results:
  `sb_audit.py`.

Left: the paradigm tables of the held-out datasets, rules for feedback runs
and for a cue before MOABB's epoch (above), and the comparison above.

## Changes after review (2026-10-18, before the first run)

A review of the literature and the code, before version 2's first run.
Each change is justified by the literature or the method, not by a score
(rule 1):

- The window starts 0.5 s after the cue's offset, not at it: the offset
  evokes a visual response (Odom et al., 2016).
- The band is 4-30 Hz, not 4-40 Hz: 40 Hz is a convention, 30-40 Hz adds
  mostly muscle (Blankertz et al., 2008; Goncharova et al., 2003).
- The audit adds the EOG channels alone and the cue period below 4 Hz, and no
  longer reads the cue period's accuracy as a confound (Pfurtscheller et al.,
  2008, 2013).
- The one-standard-error rule uses a paired standard error.
- The final evaluation's seeds differ between replicates.
- The DRAC job runs the search even if the audit fails; W&B runs are named
  by protocol, replicate and test session.

## References

- Ang, Chin, Wang, Guan and Zhang (2012). Filter bank common spatial pattern
  algorithm on BCI Competition IV datasets 2a and 2b. Front. Neurosci.
  doi:10.3389/fnins.2012.00039
- Blankertz, Tomioka, Lemm, Kawanabe and Müller (2008). Optimizing spatial
  filters for robust EEG single-trial analysis. IEEE Signal Process. Mag.
  doi:10.1109/MSP.2008.4408441
- Brunner, Leeb, Müller-Putz, Schlögl and Pfurtscheller (2008). BCI
  Competition 2008 - Graz data set A.
  https://www.bbci.de/competition/iv/desc_2a.pdf
- Combrisson and Jerbi (2015). Exceeding chance level by chance: the caveat of
  theoretical chance levels in brain signal classification and statistical
  assessment of decoding accuracy. J. Neurosci. Methods.
  doi:10.1016/j.jneumeth.2015.01.010
- Goncharova, McFarland, Vaughan and Wolpaw (2003). EMG contamination of EEG:
  spectral and topographical characteristics. Clin. Neurophysiol.
  doi:10.1016/S1388-2457(03)00093-2
- Gratton, Coles and Donchin (1983). A new method for off-line removal of
  ocular artifact. Electroencephalogr. Clin. Neurophysiol.
  doi:10.1016/0013-4694(83)90135-9
- Jung et al. (2000). Removing electroencephalographic artifacts by blind
  source separation. Psychophysiology. doi:10.1111/1469-8986.3720163
- Lawhern et al. (2018). EEGNet: a compact convolutional neural network for
  EEG-based brain-computer interfaces. J. Neural Eng.
  doi:10.1088/1741-2552/aace8c
- Lin, Wu and Jung (2026). The identity trap in EEG foundation models: a
  diagnostic audit. arXiv:2606.06647
- Odom et al. (2016). ISCEV standard for clinical visual evoked potentials
  (2016 update). Doc. Ophthalmol. doi:10.1007/s10633-016-9553-y
- Özkahraman et al. (2024). Sensors. doi:10.3390/s25010120
- Pfurtscheller and Lopes da Silva (1999). Event-related EEG/MEG
  synchronization and desynchronization: basic principles. Clin.
  Neurophysiol. doi:10.1016/S1388-2457(99)00141-8
- Pfurtscheller et al. (2008). Eur. J. Neurosci.
  doi:10.1111/j.1460-9568.2008.06441.x
- Pfurtscheller et al. (2013). Front. Hum. Neurosci.
  doi:10.3389/fnhum.2013.00379
- Rommel et al. (2022). Data augmentation for learning predictive models on
  EEG: a systematic comparison. J. Neural Eng.
- Schirrmeister et al. (2017). Deep learning with convolutional neural
  networks for EEG decoding and visualization. Hum. Brain Mapp.
  doi:10.1002/hbm.23730
- Schlögl et al. (2007). A fully automated correction method of EOG artifacts
  in EEG recordings. Clin. Neurophysiol. doi:10.1016/j.clinph.2006.09.003
