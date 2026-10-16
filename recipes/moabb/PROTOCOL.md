# The search protocol, version 2 (draft)

A hyperparameter search protocol for within-subject motor-imagery decoding,
developed on BNCI2014001 and meant to be applied unchanged to new datasets.
On a new dataset the protocol runs from scratch: it searches again and
chooses its own configuration. What has to generalize is the procedure, not
the configuration it found on BNCI2014001.

Status: draft, 2026-10-16, not implemented. Version 1 (the "improved
protocol", `hparams/sb_moabb/improved/bnci2014001_eegnet.yaml`) is running as
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

Held-out candidates (from `PROTOCOL_IDEAS.md`): BNCI2014004, Cho2017,
PhysionetMI, Zhou2016, Schirrmeister2017. Their tables are filled before the
freeze, so a dataset the rules can't handle shows up before any result does.

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

- **Band: 4-40 Hz.** Motor imagery's event-related desynchronization and
  synchronization are in the mu (8-13 Hz) and beta (13-30 Hz) bands
  (Pfurtscheller and Lopes da Silva, 1999); FBCSP covers 4-40 Hz (Ang et al.,
  2008). Below 4 Hz, ocular activity, slow drifts and cue-evoked potentials
  dominate; above 40 Hz, muscle activity and line noise. MOABB's band-pass, as
  now.
- **Sample rate: 125 Hz** (SpeechBrain-MOABB's), over three times the band's
  upper edge. A dataset recorded below 125 Hz keeps its own rate.
- **Window: from the cue's offset to the end of imagery.** If the cue
  persists through the imagery, from 0.5 s after cue onset, past the main
  visual evoked response, and the audit reports the dataset as cue-confounded.
  On BNCI2014001: 1.25-4.0 s after the cue.
- **Channels: every EEG channel**, EOG channels dropped. No channel-selection
  search: EEGNet's spatial filters weight the channels, and a rule based on
  Cz's neighbours depends on the montage.
- **Normalization:** each trial z-scored over time per channel, as now.

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
- **Epochs:** schedule-free AdamW; the epoch is chosen on validation from
  250-1000 in steps of 25.
- **Seeds:** a different seed per inner fold and per trial.
- **Selection:** the smallest model (parameters) within one standard error of
  the best; nested by test session.
- **Final evaluation:** 10 seeds, trained on all of the training data, at the
  chosen epoch.

## Splits, by rule

- **Outer folds:** with two or more sessions, leave one session out, with the
  search nested by test session. With one session: the runs split into two
  contiguous halves used as two pseudo-sessions; with one run, two contiguous
  halves in time. New in version 2: version 1 had no rule for one session.
- **Inner folds:** one per run of the training data, consecutive runs grouped
  so there are at most 6. With fewer than 3 runs: 5 contiguous blocks in time.
  New in version 2: version 1 fell back to interleaved folds, which aren't
  separate in time from training.

## The confound audit

Run on every dataset before the search, reported, never used for selection.
A fixed reference model (EEGNet-8,2, Lawhern et al., 2018, with the step 1
defaults of version 1) is trained with the inner folds above on:

1. the protocol's window;
2. the cue period alone (cue onset to cue offset, or onset to 0.5 s if the cue
   persists);
3. the protocol's window without the frontal channels (10-20 names starting
   Fp, AF or F).

Accuracy well above chance on 2 says the cue's evoked response or eye
movements carry the class; a large drop from 1 to 3 says frontal (possibly
ocular) activity does. The report states both against the binomial threshold
for the number of trials (Combrisson and Jerbi, 2015).

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

| # | Variant | Job | Result | Notes |
|---|---|---|---|---|
| 1 | SpeechBrain-MOABB's protocol, Adam | 65405367_0 | 0.729 ± 0.007 test (10 seeds), 11.8k parameters, 3 h 02 min | Not exactly theirs: BatchNorm momentum searched (they fix 0.01) |
| 2 | SpeechBrain-MOABB's protocol, schedule-free | 65405367_1 | 0.707 ± 0.003 test, 53.7k parameters, 6 h 01 min | Step 2 kept augmentation that lowered validation |
| 3 | Version 1 (step 0 searched), replicate 0 | 65435553 | Running | Step 0 chose fmin 2.4-2.6 Hz, tmin 0.2 s |
| 4 | Version 2 | Not run | | This draft |

Before these, BNCI2014001 was also used for the recipes, the speed options and
the convergence options (README), and SpeechBrain-MOABB tuned their defaults
on it.

## To implement

- The metadata table and the window rule (`tmin`, `tmax` from the cue timings).
- All EEG channels, without the adjacency selection.
- Step 2's all-off first trial and the transform count as complexity.
- The one-session outer split and the contiguous-block inner folds.
- The audit, as a script that writes its report next to the search results.

## References

- Ang, Chin, Zhang and Guan (2008). Filter bank common spatial pattern
  (FBCSP) in brain-computer interface. IJCNN.
- Combrisson and Jerbi (2015). Exceeding chance level by chance: the caveat of
  theoretical chance levels in brain signal classification and statistical
  assessment of decoding accuracy. J. Neurosci. Methods.
  doi:10.1016/j.jneumeth.2015.01.010
- Lawhern et al. (2018). EEGNet: a compact convolutional neural network for
  EEG-based brain-computer interfaces. J. Neural Eng.
- Lin, Wu and Jung (2026). The identity trap in EEG foundation models: a
  diagnostic audit. arXiv:2606.06647
- Pfurtscheller and Lopes da Silva (1999). Event-related EEG/MEG
  synchronization and desynchronization: basic principles. Clin.
  Neurophysiol.
- Rommel et al. (2022). Data augmentation for learning predictive models on
  EEG: a systematic comparison. J. Neural Eng.
