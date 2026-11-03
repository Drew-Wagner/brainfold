"""Label-permutation null for the nested search's final models (nested_search.py), and their p-values.

For each held-out session of a finished nested search (--search, its final-hparams.yaml),
the chosen hyperparameters are trained as the final evaluation trains them (every
training session, with augmentation), but with the training labels shuffled within
each run, keeping each run's class balance and the session's slow drift; the models
are tested on the held-out session's true labels. One model per permutation and
subject, each with its own seed.

    python final_null.py hparams/sb_moabb/bciciv1_eegnet.yaml --search results/.../sb-nested --permutations 100

Writes to the search's directory: final-null.csv (one row per permutation, subject
and held-out session), and final-null-summary.csv: per subject, the null's 95th
percentile against the binomial chance level of final.csv and the subject's final score
(mean over seeds); per held-out session, the group-level null (mean over subjects per
permutation), its 95th percentile, and the p-value of the final mean,
(1 + #{null >= final}) / (1 + permutations).

Runs are contiguous equal blocks of each training session (``permutation_null.runs_per_session``
in the hparams file: BNCI2014001's 6, BCI Competition IV data set 1's calibration 2).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import sb_protocol as protocol
from sb_search import prepare, read_csv, train_runs
from tracking import write_csv
from train import load_hparams


class Tiled:
    """``t[idx % n]``: the trials, addressed by indices offset by n per permutation (labels differ, data don't)."""

    def __init__(self, t, n):
        self.t, self.n = t, n

    def __getitem__(self, idx):
        return self.t[idx % self.n]

    def __getattr__(self, name):  # device, dtype, shape: the trials'
        return getattr(self.t, name)


def shuffled_labels(y, subjects, sessions, train_sessions, runs, g):
    """Labels permuted within each run (contiguous equal blocks, in time order) of every training session."""
    out = y.copy()
    for subject in np.unique(subjects):
        for session in train_sessions[subject]:
            idx = np.flatnonzero((subjects == subject) & (sessions == session))
            for run in np.array_split(idx, runs):
                out[run] = y[run][g.permutation(len(run))]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hparams")
    parser.add_argument("--search", type=Path, required=True, help="a finished nested_search.py --out directory")
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE")
    parser.add_argument("--subjects", type=int, nargs="+", help="a subset of subjects (smoke tests)")
    parser.add_argument("--epochs", type=int, help="override the chosen epochs (smoke tests)")
    parser.add_argument("--device", type=torch.device, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.optimizer, args.compile = "adam", False
    cfg = load_hparams(args.hparams, args.set)
    runs_per_session = cfg["permutation_null"]["runs_per_session"]
    torch.backends.cudnn.benchmark = True
    chosen = yaml.safe_load((args.search / "final-hparams.yaml").read_text())

    t0 = time.perf_counter()
    recordings = protocol.Recordings(cfg["dataset"], cfg["n_classes"], cfg["sample_rate"], subjects=args.subjects,
                                     dataset_kwargs=cfg.get("dataset_kwargs"))
    print(f"{cfg['dataset']}: loaded in {time.perf_counter() - t0:.0f}s; {args.permutations} permutations of "
          f"{list(chosen)}", flush=True)
    rows = []
    out = args.search / "final-null.csv"
    for name, hp in chosen.items():
        held_out = int(name.removeprefix("heldout"))
        hp = dict(hp, augment=True)  # as final_evaluation
        if args.epochs:
            hp["number_of_epochs"] = args.epochs
        data = prepare(hp, recordings, cfg, args)
        subjects, sessions, y = data["subjects"], data["sessions"], data["labels"]
        N = len(y)
        train_sessions = {s: [x for i, x in enumerate(np.unique(sessions[subjects == s])) if i != held_out]
                          for s in np.unique(subjects)}
        g = np.random.default_rng(cfg["seed"] + held_out)
        # block 0 holds the true labels, which the test indices (< N) address; block p + 1 permutation p
        labels = np.concatenate([y] + [shuffled_labels(y, subjects, sessions, train_sessions, runs_per_session, g)
                                       for _ in range(args.permutations)])
        data.update(X=Tiled(data["X"], N), Xn=Tiled(data["Xn"], N), y=torch.from_numpy(labels).to(args.device))
        folds = protocol.nested_folds(subjects, sessions, held_out, cfg["nested"]["inner_folds"])
        runs = [(f"{fold}_perm{p:03d}", cfg["seed"] + 100_000 + p, train + (p + 1) * N, dict(test=test))
                for p in range(args.permutations) for fold, train, test, _ in folds]
        t = time.perf_counter()
        result = train_runs(data, runs, cfg["nested"]["pack_size"], args, f"null/{name}")
        print(f"{name}: {len(runs)} models in {time.perf_counter() - t:.0f}s, "
              f"peak {result['peak_allocated'] / 2**30:.1f} GiB", flush=True)
        epoch = data["hp"]["number_of_epochs"]
        for c in result["curves"]:
            if c["epoch"] != epoch:
                continue
            fold, perm = c["fold"].rsplit("_perm", 1)
            rows.append(dict(held_out=held_out, permutation=int(perm), subject=int(fold[3:5]), seed=c["seed"],
                             **{k: v for k, v in c.items() if k.startswith("test_")}))
        write_csv(out, rows)

    # summary against the final scores
    metric = cfg["metric"]
    final = read_csv(args.search / "final.csv")
    summary = []
    for held_out in sorted({r["held_out"] for r in rows}):
        mine = [r for r in rows if r["held_out"] == held_out]
        subjects = sorted({r["subject"] for r in mine})
        final_mean = {}
        for s in subjects:
            f = [r for r in final if int(r["held_out"]) == held_out and int(r["fold"][3:5]) == s]
            null = np.array([r[f"test_{metric}"] for r in mine if r["subject"] == s])
            final_mean[s] = float(np.mean([float(r[f"test_{metric}"]) for r in f]))
            summary.append(dict(held_out=held_out, subject=s, null_mean=float(null.mean()),
                                null_95=float(np.percentile(null, 95)), binomial=float(f[0]["chance_level"]),
                                final=final_mean[s], p=float((1 + (null >= final_mean[s]).sum()) / (1 + len(null)))))
        by = {(r["permutation"], r["subject"]): r[f"test_{metric}"] for r in mine}
        per_perm = np.array([np.mean([by[(p, s)] for s in subjects]) for p in range(args.permutations)])
        group_final = float(np.mean(list(final_mean.values())))
        summary.append(dict(held_out=held_out, subject="all", null_mean=float(per_perm.mean()),
                            null_95=float(np.percentile(per_perm, 95)), binomial=float("nan"), final=group_final,
                            p=float((1 + (per_perm >= group_final).sum()) / (1 + len(per_perm)))))
    write_csv(args.search / "final-null-summary.csv", summary)
    for row in summary:
        print(f"heldout{row['held_out']} subject {row['subject']}: final {row['final']:.3f}; null mean "
              f"{row['null_mean']:.3f}, 95th percentile {row['null_95']:.3f}; binomial {row['binomial']:.3f}; "
              f"p {row['p']:.3f}", flush=True)


if __name__ == "__main__":
    main()
