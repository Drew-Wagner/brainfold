"""Cross-session baselines for BCI Competition IV data set 1, on the nested search's split.

Trained on the calibration session (visual cues), tested on the evaluation session (spoken cues),
as ``nested_search.py`` with ``hparams/sb_moabb/bciciv1_eegnet.yaml`` (held_out [1]):

- ``csp_lda``: CSP (6 components, log-variance) and LDA on 8-30 Hz, 0.5-4 s, every channel;
  deterministic, one score per subject.
- ``eegnet_fixed``: the config's ``fixed`` EEGNet (SpeechBrain-MOABB's, tuned for BNCI2014001, not
  for this dataset) trained as the nested search's final evaluation (augmented, every calibration
  trial), with the same 10 seeds.

    python bciciv1_baselines.py hparams/sb_moabb/bciciv1_eegnet.yaml --out results/rtx4080super/bciciv1-baselines
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline

import sb_protocol as protocol
from nested_search import chance_level, final_evaluation
from tracking import write_csv
from train import load_hparams


def csp_lda(recordings, cfg, held_out, fmin=8.0, fmax=30.0, tmin=0.5, tmax=4.0, n_components=6):
    """Rows of test scores per subject for CSP + LDA, trained on the other sessions."""
    X, y, subjects, sessions = recordings.trials(fmin, fmax)
    X = protocol.crop(X, cfg["sample_rate"], recordings.interval(), tmin, tmax).astype(np.float64)
    rows = []
    for s in np.unique(subjects):
        order = np.unique(sessions[subjects == s])
        test = (subjects == s) & (sessions == order[held_out])
        train = (subjects == s) & (sessions != order[held_out])
        model = make_pipeline(CSP(n_components=n_components, log=True), LinearDiscriminantAnalysis())
        model.fit(X[train], y[train])
        pred = model.predict(X[test])
        rows.append(dict(method="csp_lda", subject=int(s), seed="", n_train=int(train.sum()), n_test=int(test.sum()),
                         test_acc=balanced_accuracy_score(y[test], pred), test_accuracy=float((pred == y[test]).mean()),
                         chance_level=chance_level(int(test.sum()), cfg["n_classes"], cfg["nested"]["alpha"])))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hparams")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE")
    parser.add_argument("--subjects", type=int, nargs="+", help="a subset of subjects (smoke tests)")
    parser.add_argument("--epochs", type=int, help="override the EEGNet's epochs (smoke tests)")
    parser.add_argument("--skip-eegnet", action="store_true")
    parser.add_argument("--device", type=torch.device, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.optimizer, args.compile = "adam", False
    cfg = load_hparams(args.hparams, args.set)
    (held_out,) = cfg["nested"]["held_out"]
    torch.backends.cudnn.benchmark = True
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    recordings = protocol.Recordings(cfg["dataset"], cfg["n_classes"], cfg["sample_rate"], subjects=args.subjects,
                                     dataset_kwargs=cfg.get("dataset_kwargs"))
    print(f"{cfg['dataset']}: loaded in {time.perf_counter() - t0:.0f}s", flush=True)

    rows = csp_lda(recordings, cfg, held_out)
    write_csv(args.out / "baselines.csv", rows)
    if not args.skip_eegnet:
        hp = dict(cfg["fixed"])
        if args.epochs:
            hp["number_of_epochs"] = args.epochs
        seeds = [cfg["seed"] + i for i in range(cfg["final_seeds"])]
        t = time.perf_counter()
        final, _ = final_evaluation(hp, seeds, held_out, recordings, cfg, args)
        print(f"eegnet_fixed: {len(final)} models in {time.perf_counter() - t:.0f}s", flush=True)
        rows += [dict(method="eegnet_fixed", subject=int(r["fold"][3:5]), seed=r["seed"], n_train=r["n_train"],
                      n_test=r["n_test"], test_acc=r["test_acc"], test_accuracy=r["test_accuracy"],
                      chance_level=r["chance_level"]) for r in final]
        write_csv(args.out / "baselines.csv", rows)

    for method in dict.fromkeys(r["method"] for r in rows):
        means = []
        for s in sorted({r["subject"] for r in rows if r["method"] == method}):
            acc = [r["test_acc"] for r in rows if r["method"] == method and r["subject"] == s]
            means.append(np.mean(acc))
            print(f"{method} subject {s}: {np.mean(acc):.3f} ± {np.std(acc, ddof=1) if len(acc) > 1 else 0:.3f}"
                  f" ({len(acc)} runs)", flush=True)
        print(f"{method} mean: {np.mean(means):.3f}", flush=True)


if __name__ == "__main__":
    main()
