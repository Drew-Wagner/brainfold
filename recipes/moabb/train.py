"""EEGNet or ATCNet on a MOABB dataset, every (seed, fold) run trained at once.

One hparams file per dataset and model (see hparams/) sets the data, the
evaluation protocol, the model and the training. Runs are trained in packs of
`train.pack_size` runs with equal training-set sizes. Data loading and
preprocessing follow braindecode's MOABB tutorials; the preprocessed trials are
cached (see --cache-dir). Override any hyperparameter with --set.

    python train.py hparams/bnci2014001_eegnet.yaml
    python train.py hparams/bnci2014009_eegnet.yaml --set train.seeds=[0,1,2]
    python train.py hparams/nakanishi2015_atcnet.yaml --baseline   # also train every run alone with braindecode
    python train.py hparams/bnci2014001_eegnet.yaml --out results.csv   # one row per run and method
    python train.py hparams/bnci2014001_eegnet.yaml --set data.subjects=[1] train.epochs=3 --device cpu   # smoke test
    python train.py hparams/lee2019mi_eegnet.yaml --prepare-only   # download, preprocess and cache, no training

Evaluation protocols (`evaluation.protocol`):
    cross-session   per subject, test on each session, train on the others
    cross-subject   test on each subject, train on all the others
    within-session  per subject and session, stratified k-fold (`evaluation.n_folds`)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import warnings
from pathlib import Path

import braindecode
import mne
import moabb
import moabb.datasets
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    exponential_moving_standardize,
    preprocess,
)
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from brainfold import PackedATCNet, PackedEEGNet

warnings.filterwarnings("ignore", message=".*final_layer_with_constraint.*")
mne.set_log_level("WARNING")

ARCHITECTURES = {"EEGNet": PackedEEGNet, "ATCNet": PackedATCNet}


def load_data(dataset, subjects, sfreq, l_freq, h_freq, factor_new, init_block_size,
              trial_start_offset_seconds=0.0, trial_stop_offset_seconds=0.0, mapping=None, dataset_kwargs=None):
    """Trials (n, n_chans, n_times), labels, and each trial's subject and session.

    ``subjects=None`` loads every subject, one at a time. Trials span each event's
    annotation (MOABB's trial interval), extended by the two offsets: a negative
    start offset starts before the event.
    """
    if subjects is None:
        subjects = getattr(moabb.datasets, dataset)(**(dataset_kwargs or {})).subject_list
    X, y, subject, session = [], [], [], []
    for subject_id in subjects:  # one at a time: some datasets don't fit in memory at their raw rate
        recordings = MOABBDataset(dataset, subject_ids=[subject_id], dataset_kwargs=dataset_kwargs)
        preprocess(recordings, [
            Preprocessor("pick_types", eeg=True, meg=False, stim=False),
            Preprocessor(lambda data: data * 1e6),  # V -> uV
            Preprocessor("filter", l_freq=l_freq, h_freq=h_freq),
            Preprocessor("resample", sfreq=sfreq),
            Preprocessor(exponential_moving_standardize, factor_new=factor_new, init_block_size=init_block_size),
        ])
        rate = recordings.datasets[0].raw.info["sfreq"]
        windows = create_windows_from_events(
            recordings, trial_start_offset_samples=round(trial_start_offset_seconds * rate),
            trial_stop_offset_samples=round(trial_stop_offset_seconds * rate), mapping=mapping, preload=True)
        for recording in windows.datasets:
            for i in range(len(recording)):
                x, label = recording[i][:2]
                X.append(x.astype(np.float32))
                y.append(label)
            subject += [recording.description["subject"]] * len(recording)
            session += [str(recording.description["session"])] * len(recording)
    return np.stack(X), np.array(y), np.array(subject), np.array(session)


def load_cached(data, cache_dir):
    """``load_data(**data)``, cached in ``cache_dir`` by its arguments and the library versions."""
    if cache_dir is None:
        return load_data(**data)
    key = json.dumps([data, braindecode.__version__, moabb.__version__, mne.__version__], sort_keys=True)
    path = Path(cache_dir) / f"{data['dataset']}-{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = load_data(**data)
        np.savez(path.with_suffix(".tmp.npz"), *arrays)
        path.with_suffix(".tmp.npz").rename(path)
    with np.load(path) as f:
        return tuple(f[f"arr_{i}"] for i in range(4))


def fit(model, X, y, train_idx, seeds, epochs, lr, batch_size, weight_decay=None, class_weight=None):
    """Train every member with Adam and a cosine learning-rate schedule.

    ``model`` maps (batch, K, n_chans, n_times) to (batch, K, n_outputs). Member k
    trains on ``X[train_idx[k]]`` (equal sizes required), shuffled by ``seeds[k]`` alone.
    ``weight_decay`` (ATCNet only), {"conv": ..., "dense": ...}, is the official
    ATCNet code's L2 weight decay, through ``source_optimizer_param_groups``.
    ``class_weight="balanced"`` weights each member's loss by the inverse class
    frequencies of its own training set, as ``F.cross_entropy(weight=...)`` does.
    """
    generators = [torch.Generator(device=train_idx.device).manual_seed(seed) for seed in seeds]
    if weight_decay:
        params = model.source_optimizer_param_groups(weight_decay["conv"], weight_decay["dense"])
    else:
        params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=lr)
    n_train = train_idx.shape[1]
    if class_weight == "balanced":
        counts = F.one_hot(y[train_idx]).sum(1).float()  # (K, n_classes)
        weights = n_train / (counts.shape[1] * counts)
    elif class_weight is not None:
        raise ValueError(f"unknown class_weight {class_weight!r}")
    members = torch.arange(len(seeds), device=train_idx.device)
    for epoch in range(epochs):
        for group in optimizer.param_groups:
            group["lr"] = 0.5 * lr * (1 + math.cos(math.pi * epoch / epochs))
        model.train()
        order = torch.stack([torch.randperm(n_train, device=train_idx.device, generator=g) for g in generators])
        for i in range(0, n_train, batch_size):
            idx = train_idx.gather(1, order[:, i : i + batch_size]).T  # (batch, K)
            losses = F.cross_entropy(model(X[idx]).permute(0, 2, 1), y[idx], reduction="none")
            if class_weight is None:
                losses = losses.mean(0)
            else:
                w = weights[members, y[idx]]  # (batch, K)
                losses = (w * losses).sum(0) / w.sum(0)
            optimizer.zero_grad(set_to_none=True)
            # SUM of the member mean losses: each member gets exactly its own gradient,
            # and one Adam over the packed parameters == K separate Adams.
            losses.sum().backward()
            optimizer.step()
    return model


@torch.no_grad()
def predict(model, X, idx, batch_size=64):
    """(n, K, n_outputs) logits; column k is member k's output for ``X[idx[k]]``."""
    model.eval()
    return torch.cat([model(X[idx[:, i : i + batch_size].T]) for i in range(0, idx.shape[1], batch_size)])


class PackOfOne(torch.nn.Module):
    """A single braindecode model with the packed (batch, 1, ...) interface, for the baseline."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x[:, 0])[:, None]

    def source_optimizer_param_groups(self, *args):
        return self.model.source_optimizer_param_groups(*args)


def folds(protocol, y, subjects, sessions, n_folds=None):
    """(name, train indices, test indices) per fold of the evaluation protocol."""
    for subject in sorted(set(subjects.tolist())):
        is_subject = subjects == subject
        if protocol == "cross-subject":
            yield f"sub{subject:02d}", np.flatnonzero(~is_subject), np.flatnonzero(is_subject)
            continue
        for session in sorted(set(sessions[is_subject].tolist())):
            is_session = is_subject & (sessions == session)
            if protocol == "cross-session":
                yield f"sub{subject:02d}_test-{session}", np.flatnonzero(is_subject & ~is_session), np.flatnonzero(is_session)
            elif protocol == "within-session":
                trials = np.flatnonzero(is_session)
                splits = StratifiedKFold(n_folds, shuffle=True, random_state=0).split(trials, y[trials])
                for fold, (train, test) in enumerate(splits):
                    yield f"sub{subject:02d}_ses-{session}_fold{fold}", trials[train], trials[test]
            else:
                raise ValueError(f"unknown evaluation protocol {protocol!r}")


def packs(runs, pack_size):
    """Consecutive runs with equal training-set sizes, at most ``pack_size`` per pack."""
    by_size = {}
    for run in runs:
        by_size.setdefault(len(run[2]), []).append(run)
    for group in by_size.values():
        for start in range(0, len(group), pack_size):
            yield group[start : start + pack_size]


METRICS = {
    "accuracy": lambda labels, logits: (logits.argmax(1) == labels).mean(),
    "balanced_accuracy": lambda labels, logits: balanced_accuracy_score(labels, logits.argmax(1)),
    "kappa": lambda labels, logits: cohen_kappa_score(labels, logits.argmax(1)),
    "roc_auc": lambda labels, logits: roc_auc_score(labels, logits[:, 1] - logits[:, 0]),  # binary: class 1 positive
}


def scores(logits, labels, metrics):
    """{metric: value} per member of (n, K, n_outputs) logits and (n, K) labels."""
    logits, labels = logits.double().numpy(), labels.numpy()
    return [{m: float(METRICS[m](labels[:, k], logits[:, k])) for m in metrics} for k in range(labels.shape[1])]


def load_hparams(path, overrides):
    hparams = yaml.safe_load(Path(path).read_text())
    for override in overrides:
        key, value = override.split("=", 1)
        *parents, leaf = key.split(".")
        section = hparams
        for parent in parents:
            section = section[parent]
        if leaf not in section:
            raise KeyError(f"unknown hyperparameter {key}")
        section[leaf] = yaml.safe_load(value)
    return hparams


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hparams", help="hparams file; relative paths are tried here, then in this directory")
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE", help="e.g. train.epochs=100")
    parser.add_argument("--baseline", action="store_true",
                        help="also train each run alone as a braindecode model, from the same initial weights")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dir", default=Path.home() / ".cache" / "brainfold",
                        help="where preprocessed trials are cached; 'none' to disable")
    parser.add_argument("--out", type=Path, help="write one CSV row per run and method")
    parser.add_argument("--prepare-only", action="store_true", help="load and cache the trials, then exit")
    args = parser.parse_args()
    path = Path(args.hparams)
    hp = load_hparams(path if path.exists() else Path(__file__).parent / path, args.set)
    data, train, evaluation = hp["data"], hp["train"], hp["evaluation"]
    architecture, metrics = hp["architecture"], hp.get("metrics", ["accuracy", "kappa"])

    X, y, subjects, sessions = load_cached(data, None if str(args.cache_dir) == "none" else args.cache_dir)
    if args.prepare_only:
        print(f"{data['dataset']}: {len(X)} trials of {X.shape[1]} channels x {X.shape[2]} samples")
        return
    n_classes = len(np.unique(y))
    X, y = torch.from_numpy(X).to(args.device), torch.from_numpy(y).to(args.device)
    runs = [(name, seed, torch.from_numpy(train_idx), torch.from_numpy(test_idx))
            for seed in train["seeds"]
            for name, train_idx, test_idx in folds(evaluation["protocol"], y.cpu().numpy(), subjects, sessions,
                                                   evaluation.get("n_folds"))]

    print(f"{architecture} on {data['dataset']} ({evaluation['protocol']}): {len(runs)} runs, "
          f"{len(X)} trials of {X.shape[1]} channels x {X.shape[2]} samples, {n_classes} classes", flush=True)
    methods = ["packed", "braindecode"] if args.baseline else ["packed"]
    rows, seconds = [], dict.fromkeys(methods, 0.0)
    for pack in packs(runs, train["pack_size"]):
        seeds = [seed for _, seed, _, _ in pack]
        train_idx = torch.stack([idx for _, _, idx, _ in pack]).to(args.device)
        test_idx = torch.stack([idx for _, _, _, idx in pack]).to(args.device)
        labels = y[test_idx.T].cpu()
        model = ARCHITECTURES[architecture].from_seeds(
            seeds, n_chans=X.shape[1], n_outputs=n_classes, n_times=X.shape[2], **hp["model"]).to(args.device)
        singles = model.to_braindecode() if args.baseline else []  # copies of the initial weights
        settings = (train["epochs"], train["lr"], train["batch_size"], train.get("weight_decay"),
                    train.get("class_weight"))

        t0 = time.perf_counter()
        fit(model, X, y, train_idx, seeds, *settings)
        pack_scores = {"packed": scores(predict(model, X, test_idx).cpu(), labels, metrics)}
        run_seconds = {"packed": [(time.perf_counter() - t0) / len(pack)] * len(pack)}

        if args.baseline:
            logits, run_seconds["braindecode"] = [], []
            for k, single in enumerate(singles):
                t0 = time.perf_counter()
                single = fit(PackOfOne(single), X, y, train_idx[k : k + 1], seeds[k : k + 1], *settings)
                logits.append(predict(single, X, test_idx[k : k + 1]).cpu())
                run_seconds["braindecode"].append(time.perf_counter() - t0)
            pack_scores["braindecode"] = scores(torch.cat(logits, 1), labels, metrics)

        for k, (name, seed, _, _) in enumerate(pack):
            line = "  ".join(f"{m} " + " ".join(f"{metric} {pack_scores[m][k][metric]:.3f}" for metric in metrics)
                             for m in methods)
            print(f"seed {seed} {name}  {line}", flush=True)
            for m in methods:
                rows.append(dict(dataset=data["dataset"], architecture=architecture,
                                 protocol=evaluation["protocol"], method=m, seed=seed, fold=name,
                                 pack_size=len(pack), seconds=run_seconds[m][k], **pack_scores[m][k]))
        for m in methods:
            seconds[m] += sum(run_seconds[m])

    for m in methods:
        values = {metric: np.array([r[metric] for r in rows if r["method"] == m]) for metric in metrics}
        summary = ", ".join(f"{metric} {v.mean():.3f} ± {v.std():.3f}" for metric, v in values.items())
        print(f"{m:11s} {len(runs)} runs in {seconds[m]:.0f}s: {summary}")
    if args.baseline:
        first = metrics[0]
        diff = np.array([r[first] for r in rows if r["method"] == "packed"]) - np.array(
            [r[first] for r in rows if r["method"] == "braindecode"])
        print(f"packed - braindecode {first} per run: {diff.mean():+.3f} ± {diff.std():.3f}")
    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
