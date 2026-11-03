"""Is the nested search's best EEGNet decoding the cue (or eye movements) rather than motor imagery?

For each held-out session h of BNCI2014001, the best step-1 configuration of the
nested search for h (job 1666607) is trained on the other session and tested on h,
with variants of its input. All 9 subjects, 3 seeds, no augmentation (step 1's setting).
Times are relative to the cue's onset (MOABB's interval [2, 6] s of the trial).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # recipes/moabb
import sb_protocol as protocol  # noqa: E402
from nested_search import sign_flip_p  # noqa: E402
from sb_search import train_runs  # noqa: E402
from tracking import write_csv  # noqa: E402

CONFIGS = {  # the best step-1 trial of each search: heldout0 trial 47, heldout1 trial 33
    0: {'fmin': 0.11, 'fmax': 36.9, 'tmax': 1.9, 'n_steps_channel_selection': 3, 'avg_models': 12,
        'number_of_epochs': 816, 'lr': 0.0005, 'batch_size_exponent': 4, 'bn_momentum': 0.7506209009583701,
        'cnn_temporal_kernels': 35, 'cnn_temporal_kernelsize': 46, 'cnn_spatial_depth_multiplier': 2,
        'cnn_septemporal_point_kernels_ratio_': 7, 'cnn_septemporal_kernelsize_': 10, 'cnn_septemporal_pool': 2,
        'dropout': 0.16538341819398392},
    1: {'fmin': 0.26, 'fmax': 44.7, 'tmax': 3.8, 'n_steps_channel_selection': 3, 'avg_models': 10,
        'number_of_epochs': 740, 'lr': 0.001, 'batch_size_exponent': 4, 'bn_momentum': 0.36678841844953175,
        'cnn_temporal_kernels': 14, 'cnn_temporal_kernelsize': 42, 'cnn_spatial_depth_multiplier': 4,
        'cnn_septemporal_point_kernels_ratio_': 4, 'cnn_septemporal_kernelsize_': 15, 'cnn_septemporal_pool': 7,
        'dropout': 0.14705545852651586},
}
EOG = ["EOG1", "EOG2", "EOG3"]
SR = 125

# (name, channels, fmin or None for the config's, window (tmin, tmax) or None for (0, config tmax), regress EOG out)
VARIANTS = [("eeg", "eeg", None, None, False), ("eog", "eog", None, None, False),
            ("eeg_fmin4", "eeg", 4.0, None, False), ("eeg_fmin8", "eeg", 8.0, None, False),
            ("eog_fmin8", "eog", 8.0, None, False), ("eeg_eog_regressed", "eeg", None, None, True)]
for ch in ("eeg", "eog"):
    VARIANTS += [(f"{ch}_precue_-1-0", ch, None, (-1.0, 0.0), False)]
    VARIANTS += [(f"{ch}_window_{a}-{a + 1}", ch, None, (float(a), float(a + 1)), False) for a in range(4)]


class Loader:
    """MOABB's preprocessing (as sb_protocol.Recordings) of the EEG and EOG channels, filtered alike."""

    def __init__(self):
        self.rec = protocol.Recordings("BNCI2014_001", 4, SR)
        self.eeg_names = list(self.rec.info.ch_names)
        self.reference = self.rec.trials(CONFIGS[0]["fmin"], CONFIGS[0]["fmax"])[0]  # before relabelling EOG
        for sessions in self.rec.raws.values():  # MOABB filters picks="data", which excludes EOG
            for runs in sessions.values():
                for raw in runs.values():
                    raw.set_channel_types({c: "eeg" for c in EOG}, on_unit_change="ignore")
        self.cache = {}

    def trials(self, fmin, fmax, precue=False, window=None):
        """Trials over MOABB's interval [0, 4] s after the cue, [-1, 0] with ``precue``, or ``window`` (tmin, tmax)."""
        if precue:
            window = (-1.0, 0.0)
        key = (fmin, fmax, window)
        if key in self.cache:
            return self.cache[key]
        from moabb.paradigms import MotorImagery

        window = dict(tmin=window[0], tmax=window[1]) if window else {}
        paradigm = MotorImagery(n_classes=4, fmin=fmin, fmax=fmax, channels=self.eeg_names + EOG, resample=SR, **window)
        pipeline = paradigm.make_process_pipelines(self.rec.dataset)[0]
        to_labels = paradigm.make_labels_pipeline(self.rec.dataset)
        X, labels, subjects, sessions = [], [], [], []
        for subject, by_session in self.rec.raws.items():
            for session, by_run in by_session.items():
                for raw in by_run.values():
                    out = pipeline.transform(raw.copy())
                    if out is None:
                        continue
                    X.append(out["X"].astype(np.float32))
                    labels.append(to_labels.transform(out["events"]))
                    subjects += [subject] * len(out["X"])
                    sessions += [session] * len(out["X"])
        codes = np.array([self.rec.dataset.event_id[label] for label in np.concatenate(labels)])
        self.cache[key] = (np.concatenate(X), codes - codes.min(), np.array(subjects), np.array(sessions))
        return self.cache[key]


def regress_out_eog(eeg, eog, subjects, sessions, held_out):
    """EEG minus its least-squares fit on the EOG, fitted per subject on the training session only."""
    clean = eeg.copy()
    for subject in np.unique(subjects):
        names = np.unique(sessions[subjects == subject])
        mine = subjects == subject
        fit = mine & (sessions != names[held_out])
        A = eog[fit].transpose(0, 2, 1).reshape(-1, eog.shape[1]).astype(np.float64)
        B = np.linalg.lstsq(A, eeg[fit].transpose(0, 2, 1).reshape(-1, eeg.shape[1]).astype(np.float64), rcond=None)[0]
        clean[mine] = eeg[mine] - np.einsum("nkt,kc->nct", eog[mine], B).astype(np.float32)
    return clean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--only", nargs="+", help="variant names")
    args = parser.parse_args()
    device = torch.device("cuda")
    run_args = argparse.Namespace(device=device, optimizer="adam", compile=False)
    torch.backends.cudnn.benchmark = True
    loader = Loader()

    # check: the EEG part equals the protocol's own preprocessing
    X, *_ = loader.trials(CONFIGS[0]["fmin"], CONFIGS[0]["fmax"])
    assert np.array_equal(X[:, :len(loader.eeg_names)], loader.reference), "EEG differs from sb_protocol.Recordings"
    print(f"EEG matches sb_protocol.Recordings; EOG channels: {X.shape[1] - len(loader.eeg_names)}", flush=True)

    rows = []
    seeds = [1986 + s for s in range(args.seeds)]
    for name, channels, fmin, window, regress in VARIANTS:
        if args.only and name not in args.only:
            continue
        for held_out, cfg in CONFIGS.items():
            hp = dict(cfg, fmin=fmin or cfg["fmin"], augment=False, batch_size=2 ** cfg["batch_size_exponent"])
            precue = window is not None and window[0] < 0
            X, y, subjects, sessions = loader.trials(hp["fmin"], hp["fmax"], precue)
            n_eeg = len(loader.eeg_names)
            eeg = X[:, :n_eeg][:, loader.rec.channels(hp["n_steps_channel_selection"])]
            eog = X[:, n_eeg:]
            if regress:
                eeg = regress_out_eog(eeg, eog, subjects, sessions, held_out)
            x = eeg if channels == "eeg" else eog
            if not precue:  # the precue trials are already [-1, 0]
                tmin, tmax = window or (0.0, hp["tmax"])
                x = protocol.crop(x, SR, [0, 4], tmin, tmax)
            x = np.ascontiguousarray(x)
            xt = torch.from_numpy(x).to(device)
            data = dict(hp=hp, X=xt, Xn=protocol.standardize(xt), y=torch.from_numpy(y).to(device),
                        eval_epochs=[hp["number_of_epochs"]], aug=None,
                        kwargs=protocol.eegnet_kwargs(hp, x.shape[1], 4, x.shape[2]))
            folds = protocol.nested_folds(subjects, sessions, held_out, 3)
            runs = [(fold, seed, train, dict(test=test)) for seed in seeds for fold, train, test, _ in folds]
            t = time.perf_counter()
            result = train_runs(data, runs, len(runs), run_args, name)
            for c in result["curves"]:
                rows.append(dict(variant=name, held_out=held_out, subject=int(c["fold"][3:5]), seed=c["seed"],
                                 n_chans=x.shape[1], n_times=x.shape[2], **{k: v for k, v in c.items() if k.startswith("test_")}))
            acc = np.mean([r["test_acc"] for r in rows if r["variant"] == name and r["held_out"] == held_out])
            print(f"{name:22s} heldout{held_out}: test acc {acc:.3f}  ({x.shape[1]} ch x {x.shape[2]} samples, "
                  f"{time.perf_counter() - t:.0f}s)", flush=True)
            write_csv(args.out, rows)

    # per subject: the mean over held-out sessions and seeds; paired against "eeg"
    def per_subject(variant):
        return np.array([np.mean([r["test_acc"] for r in rows if r["variant"] == variant and r["subject"] == s])
                         for s in range(1, 10)])
    base = per_subject("eeg")
    print(f"\n{'variant':22s} {'acc':>6s} {'vs eeg':>7s} {'se':>6s} {'p':>6s}   per subject")
    for variant in dict.fromkeys(r["variant"] for r in rows):
        v = per_subject(variant)
        d = v - base
        p = sign_flip_p(-d) if variant != "eeg" else float("nan")  # one-sided: the variant is worse
        print(f"{variant:22s} {v.mean():6.3f} {d.mean():+7.3f} {d.std(ddof=1) / 3:6.3f} {p:6.3f}   "
              + " ".join(f"{a:.2f}" for a in v), flush=True)


if __name__ == "__main__":
    main()
