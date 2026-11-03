"""Label-permutation null for the cue checks: the test accuracy reached when training labels carry no information.

For each held-out session, its chosen configuration (cue_check.CONFIGS) is trained on the other session with
the training labels shuffled within each run (keeping each run's class balance and the session's slow
drift), then tested on the held-out session's true labels. One model per permutation, subject and held-out
session, each with its own seed. The 95th percentile per fold calibrates the binomial chance level (0.292
for 288 trials, 4 classes), and the mean over folds per permutation gives the group-level null.

    python cue_checks/cue_null.py --variant eog_fmin8 --permutations 100 --out results/.../cue_null_eog_fmin8.csv

Variants: "eeg" (the configuration's own band and window, as cue_check.py's baseline) and "eog_fmin8" (the 3
EOG channels, high-passed at 8 Hz: the borderline case).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from cue_check import CONFIGS, SR, Loader  # noqa: E402
import sb_protocol as protocol  # noqa: E402
from sb_search import train_runs  # noqa: E402
from tracking import write_csv  # noqa: E402

RUNS_PER_SESSION = 6  # BNCI2014001: 6 runs of 48 trials


class Tiled:
    """``t[idx % n]``: the trials, addressed by indices offset by n per permutation (labels differ, data don't)."""

    def __init__(self, t, n):
        self.t, self.n = t, n

    def __getitem__(self, idx):
        return self.t[idx % self.n]

    def __getattr__(self, name):  # device, dtype, shape: the trials'
        return getattr(self.t, name)


def shuffled_labels(y, subjects, sessions, g):
    """Labels permuted within each run (contiguous sixths of each subject's session, in time order)."""
    out = y.copy()
    for subject in np.unique(subjects):
        for session in np.unique(sessions[subjects == subject]):
            idx = np.flatnonzero((subjects == subject) & (sessions == session))
            assert len(idx) % RUNS_PER_SESSION == 0, len(idx)
            for run in np.array_split(idx, RUNS_PER_SESSION):
                out[run] = y[run][g.permutation(len(run))]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["eeg", "eog_fmin8"], required=True)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--pack-size", type=int, default=108)
    parser.add_argument("--epochs", type=int, help="override the configurations' epochs (smoke tests)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    run_args = argparse.Namespace(device=device, optimizer="adam", compile=False)
    torch.backends.cudnn.benchmark = True
    t0 = time.perf_counter()
    loader = Loader()
    print(f"data loaded in {time.perf_counter() - t0:.0f}s; variant {args.variant}, {args.permutations} permutations",
          flush=True)
    rows = []
    for held_out, cfg in CONFIGS.items():
        hp = dict(cfg, augment=False, batch_size=2 ** cfg["batch_size_exponent"])
        if args.epochs:
            hp["number_of_epochs"] = args.epochs
        fmin = 8.0 if args.variant == "eog_fmin8" else hp["fmin"]
        X, y, subjects, sessions = loader.trials(fmin, hp["fmax"])
        n_eeg = len(loader.eeg_names)
        x = X[:, n_eeg:] if args.variant == "eog_fmin8" else X[:, :n_eeg][:, loader.rec.channels(
            hp["n_steps_channel_selection"])]
        x = np.ascontiguousarray(protocol.crop(x, SR, [0, 4], 0.0, hp["tmax"]))
        N = len(y)
        g = np.random.default_rng(held_out)
        # block 0 holds the true labels, which the test indices (< N) address; block p + 1 permutation p
        labels = np.concatenate([y] + [shuffled_labels(y, subjects, sessions, g) for _ in range(args.permutations)])
        xt = torch.from_numpy(x).to(device)
        data = dict(hp=hp, X=None, Xn=Tiled(protocol.standardize(xt), N), y=torch.from_numpy(labels).to(device),
                    eval_epochs=[hp["number_of_epochs"]], aug=None,
                    kwargs=protocol.eegnet_kwargs(hp, x.shape[1], 4, x.shape[2]))
        folds = protocol.nested_folds(subjects, sessions, held_out, 3)
        runs = [(f"{fold}_perm{p:03d}", 1986 + p, train + (p + 1) * N, dict(test=test))
                for p in range(args.permutations) for fold, train, test, _ in folds]
        t = time.perf_counter()
        result = train_runs(data, runs, args.pack_size, run_args, f"null/{args.variant}")
        print(f"heldout{held_out}: {len(runs)} models in {time.perf_counter() - t:.0f}s, "
              f"peak {result['peak_allocated'] / 2**30:.1f} GiB", flush=True)
        for c in result["curves"]:
            fold, perm = c["fold"].rsplit("_perm", 1)
            rows.append(dict(variant=args.variant, held_out=held_out, permutation=int(perm), subject=int(fold[3:5]),
                             seed=c["seed"], **{k: v for k, v in c.items() if k.startswith("test_")}))
        write_csv(args.out, rows)
        acc = np.array([r["test_acc"] for r in rows if r["held_out"] == held_out])
        print(f"heldout{held_out}: null test acc mean {acc.mean():.3f}, 95th percentile per model "
              f"{np.percentile(acc, 95):.3f}", flush=True)


if __name__ == "__main__":
    main()
