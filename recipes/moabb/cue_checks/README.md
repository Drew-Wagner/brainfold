# Is the searched EEGNet decoding the cue?

Checks on the best step-1 configuration of each search of `nested_search.py`
(job 1666607: held-out session 0, trial 47; held-out session 1, trial 33; values in
`cue_check.CONFIGS`). Each trains on the other session and tests on the held-out
one: BNCI2014001, 9 subjects, 3 seeds, no augmentation. Times are seconds after
cue onset. MOABB's BNCI2014001 events mark the cue (t = 2 s of the trial), and
its default epochs start at the event; the arrow is shown from 0 to 1.25 s.

| Script | What it does | Retrained? |
| --- | --- | --- |
| `cue_check.py` | EEG and EOG, bands (fmin 4 and 8 Hz), EOG regressed out, 1 s windows, the second before the cue | yes, per variant |
| `cue_grid.py` | EEGNet's and Schirrmeister et al.'s windows x bands | yes, per cell |
| `cue_sliding.py` | 1 s windows every 0.25 s, -1 to 4 s, EEG and EOG | yes, per window |
| `cue_zeroshot.py` | the baseline models on altered test inputs (bands, EOG regression, 0.5 s blocks zeroed) | no |
| `cue_occlusion.py` | a 0.5 s block zeroed, stepped every 0.125 s; saves the models (`models-heldout<h>.pt` next to `--out`) | no |
| `cue_shuffle.py` | the same block shuffled among each subject's test trials, 50 ms cross-fades, 5 permutations; loads `cue_occlusion.py`'s models | no |

Run from `recipes/moabb`, e.g.

    python cue_checks/cue_check.py --out results/rtx4080/cue-checks/cue_check.csv
    python cue_checks/cue_occlusion.py --out results/rtx4080/cue-checks/cue_occlusion.csv
    python cue_checks/cue_shuffle.py --out results/rtx4080/cue-checks/cue_shuffle.csv

Each writes one row per subject, seed and variant (test balanced accuracy `test_acc`,
accuracy, F1 and kappa). Results from an RTX 4080 SUPER, 2 November 2026, are in
`results/rtx4080/cue-checks/`.
