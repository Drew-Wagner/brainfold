"""Leave-one-session-out EEGNet on BNCI2014001, every (seed, fold) run trained at once.

Per subject, train on one session and test on the other: 9 subjects x 2 folds =
18 runs per seed. Runs are trained in packs of `train.pack_size`. Data loading
and preprocessing follow braindecode's BNCI2014001 tutorials.
Hyperparameters come from hparams.yaml; override any of them with --set.

    python train.py
    python train.py --set train.seeds=[0,1,2]
    python train.py --baseline       # also train every run alone as a braindecode EEGNet
    python train.py --set data.subjects=[1] train.epochs=3 --device cpu   # smoke test
"""

from __future__ import annotations

import argparse
import math
import time
import warnings
from pathlib import Path

import mne
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
from sklearn.metrics import cohen_kappa_score

from packed_eegnet import PackedEEGNet

warnings.filterwarnings("ignore", message=".*final_layer_with_constraint.*")
mne.set_log_level("WARNING")


def load_data(subjects, sfreq, l_freq, h_freq, factor_new, init_block_size):
    """Trials (n, n_chans, n_times), labels, and each trial's subject and session."""
    dataset = MOABBDataset("BNCI2014_001", subject_ids=subjects)
    preprocess(dataset, [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        Preprocessor(lambda data: data * 1e6),  # V -> uV
        Preprocessor("filter", l_freq=l_freq, h_freq=h_freq),
        Preprocessor("resample", sfreq=sfreq),
        Preprocessor(exponential_moving_standardize, factor_new=factor_new, init_block_size=init_block_size),
    ])
    windows = create_windows_from_events(dataset, preload=True)
    X, y, subject, session = [], [], [], []
    for recording in windows.datasets:
        for i in range(len(recording)):
            x, label = recording[i][:2]
            X.append(x)
            y.append(label)
        subject += [recording.description["subject"]] * len(recording)
        session += [recording.description["session"]] * len(recording)
    return np.stack(X).astype(np.float32), np.array(y), np.array(subject), np.array(session)


def fit(model, X, y, train_idx, seeds, epochs, lr, batch_size):
    """Train every member with Adam and a cosine learning-rate schedule.

    ``model`` maps (batch, K, n_chans, n_times) to (batch, K, n_outputs). Member k
    trains on ``X[train_idx[k]]`` (equal sizes required), shuffled by ``seeds[k]`` alone.
    """
    generators = [torch.Generator(device=train_idx.device).manual_seed(seed) for seed in seeds]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_train = train_idx.shape[1]
    for epoch in range(epochs):
        for group in optimizer.param_groups:
            group["lr"] = 0.5 * lr * (1 + math.cos(math.pi * epoch / epochs))
        model.train()
        order = torch.stack([torch.randperm(n_train, device=train_idx.device, generator=g) for g in generators])
        for i in range(0, n_train, batch_size):
            idx = train_idx.gather(1, order[:, i : i + batch_size]).T  # (batch, K)
            losses = F.cross_entropy(model(X[idx]).permute(0, 2, 1), y[idx], reduction="none").mean(0)
            optimizer.zero_grad(set_to_none=True)
            # SUM of the member mean losses: each member gets exactly its own gradient,
            # and one Adam over the packed parameters == K separate Adams.
            losses.sum().backward()
            optimizer.step()
    return model


@torch.no_grad()
def predict(model, X, idx, batch_size=64):
    """(n, K) predicted classes; column k is member k's prediction for ``X[idx[k]]``."""
    model.eval()
    return torch.cat([model(X[idx[:, i : i + batch_size].T]).argmax(-1) for i in range(0, idx.shape[1], batch_size)])


class PackOfOne(torch.nn.Module):
    """A single braindecode EEGNet with the packed (batch, 1, ...) interface, for the baseline."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x[:, 0])[:, None]


def folds(subjects, sessions):
    """(name, train mask, test mask): train on one session of a subject, test on the other."""
    for subject in sorted(set(subjects.tolist())):
        for session in sorted(set(sessions[subjects == subject].tolist())):
            test = (subjects == subject) & (sessions == session)
            train = (subjects == subject) & (sessions != session)
            yield f"sub{subject:02d}_test-{session}", train, test


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


def scores(preds, labels):
    """(accuracy, kappa) per column of (n, K) predictions."""
    return [((preds[:, k] == labels[:, k]).float().mean().item(), cohen_kappa_score(labels[:, k], preds[:, k]))
            for k in range(preds.shape[1])]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hparams", default=Path(__file__).with_name("hparams.yaml"))
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE", help="e.g. train.epochs=100")
    parser.add_argument("--baseline", action="store_true",
                        help="also train each run alone as a braindecode EEGNet, from the same initial weights")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    hp = load_hparams(args.hparams, args.set)
    train = hp["train"]

    X, y, subjects, sessions = load_data(**hp["data"])
    n_classes = len(np.unique(y))
    X, y = torch.from_numpy(X).to(args.device), torch.from_numpy(y).to(args.device)
    runs = [(name, seed, torch.from_numpy(train_mask.nonzero()[0]), torch.from_numpy(test_mask.nonzero()[0]))
            for seed in train["seeds"] for name, train_mask, test_mask in folds(subjects, sessions)]

    methods = ["packed", "braindecode"] if args.baseline else ["packed"]
    results, seconds = {m: [] for m in methods}, dict.fromkeys(methods, 0.0)
    for start in range(0, len(runs), train["pack_size"]):
        pack = runs[start : start + train["pack_size"]]
        seeds = [seed for _, seed, _, _ in pack]
        train_idx = torch.stack([idx for _, _, idx, _ in pack]).to(args.device)  # equal sizes required
        test_idx = torch.stack([idx for _, _, _, idx in pack]).to(args.device)
        labels = y[test_idx.T].cpu()
        model = PackedEEGNet.from_seeds(seeds, X.shape[1], n_classes, X.shape[2], **hp["model"]).to(args.device)
        eegnets = model.to_braindecode() if args.baseline else []  # copies of the initial weights

        t0 = time.perf_counter()
        fit(model, X, y, train_idx, seeds, train["epochs"], train["lr"], train["batch_size"])
        pack_scores = {"packed": scores(predict(model, X, test_idx).cpu(), labels)}
        seconds["packed"] += time.perf_counter() - t0

        if args.baseline:
            t0, preds = time.perf_counter(), []
            for k, eegnet in enumerate(eegnets):
                single = fit(PackOfOne(eegnet), X, y, train_idx[k : k + 1], seeds[k : k + 1],
                             train["epochs"], train["lr"], train["batch_size"])
                preds.append(predict(single, X, test_idx[k : k + 1]).cpu())
            pack_scores["braindecode"] = scores(torch.cat(preds, 1), labels)
            seconds["braindecode"] += time.perf_counter() - t0

        for k, (name, seed, _, _) in enumerate(pack):
            line = "  ".join(f"{m} acc {pack_scores[m][k][0]:.3f} kappa {pack_scores[m][k][1]:.3f}" for m in methods)
            print(f"seed {seed} {name}  {line}", flush=True)
        for m in methods:
            results[m] += pack_scores[m]

    for m in methods:
        acc, kappa = np.array(results[m]).T
        print(f"{m:11s} {len(runs)} runs in {seconds[m]:.0f}s: "
              f"acc {acc.mean():.3f} ± {acc.std():.3f}, kappa {kappa.mean():.3f} ± {kappa.std():.3f}")
    if args.baseline:
        diff = np.array(results["packed"])[:, 0] - np.array(results["braindecode"])[:, 0]
        print(f"packed - braindecode accuracy per run: {diff.mean():+.3f} ± {diff.std():.3f}")


if __name__ == "__main__":
    main()
