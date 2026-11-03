"""Zero-shot companion to cue_check.py: train the baseline models once, then test them on altered test inputs.

The models are cue_check.py's "eeg" variant (same configurations, seeds and folds).
Each alteration is applied to the inputs the trained models see at test time only:
a different band-pass (fmin 4 or 8 Hz), the EOG regressed out (weights fitted on the
training session), and occlusion in time (zeroing standardized samples, i.e. the
trial's mean, over a block). A drop measures how much these models rely on what
was removed; models never saw such inputs, so it also includes some
out-of-distribution penalty.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from cue_check import CONFIGS, SR, Loader, regress_out_eog  # noqa: E402
import sb_protocol as protocol  # noqa: E402  (cue_check put recipes/moabb on the path)
from nested_search import sign_flip_p  # noqa: E402
from tracking import write_csv  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    loader = Loader()
    seeds = [1986 + s for s in range(args.seeds)]
    rows = []

    for held_out, cfg in CONFIGS.items():
        hp = dict(cfg, augment=False, batch_size=2 ** cfg["batch_size_exponent"])
        picks = loader.rec.channels(hp["n_steps_channel_selection"])
        n_eeg = len(loader.eeg_names)

        def inputs(fmin, regress=False):
            X, y, subjects, sessions = loader.trials(fmin, hp["fmax"])
            eeg = X[:, :n_eeg][:, picks]
            if regress:
                eeg = regress_out_eog(eeg, X[:, n_eeg:], subjects, sessions, held_out)
            eeg = np.ascontiguousarray(protocol.crop(eeg, SR, [0, 4], 0.0, hp["tmax"]))
            return torch.from_numpy(eeg).to(device), y, subjects, sessions

        X, y, subjects, sessions = inputs(hp["fmin"])
        Xn = protocol.standardize(X)
        yt = torch.from_numpy(y).to(device)
        T = X.shape[2]
        tests = {"none": Xn}
        tests["fmin4"] = protocol.standardize(inputs(4.0)[0])
        tests["fmin8"] = protocol.standardize(inputs(8.0)[0])
        tests["eog_regressed"] = protocol.standardize(inputs(hp["fmin"], regress=True)[0])
        cue = round(1.25 * SR)  # the arrow is shown for the first 1.25 s
        for name, (a, b) in {"cue_only": (cue, T), "after_cue_only": (0, cue)}.items():
            occluded = Xn.clone()
            occluded[..., a:b] = 0
            tests[name] = occluded
        for start in np.arange(0, hp["tmax"], 0.5):
            a, b = round(start * SR), min(round((start + 0.5) * SR), T)
            occluded = Xn.clone()
            occluded[..., a:b] = 0
            tests[f"occlude_{start:.1f}-{start + 0.5:.1f}"] = occluded

        folds = protocol.nested_folds(subjects, sessions, held_out, 3)
        runs = [(fold, seed, train, test) for seed in seeds for fold, train, test, _ in folds]
        train_idx = torch.stack([torch.from_numpy(r[2]) for r in runs]).to(device)
        test_idx = torch.stack([torch.from_numpy(r[3]) for r in runs]).to(device)
        run_seeds = [r[1] for r in runs]
        t = time.perf_counter()
        model = protocol.build(protocol.eegnet_kwargs(hp, X.shape[1], 4, T), run_seeds, device)
        model, _ = protocol.fit(model, X, Xn, yt, train_idx, run_seeds, hp, optimizer="adam")
        print(f"heldout{held_out}: trained {len(runs)} models in {time.perf_counter() - t:.0f}s", flush=True)
        for name, Xt in tests.items():
            scored = protocol.scores(model, Xt, yt, test_idx)
            for (fold, seed, *_), s in zip(runs, scored):
                rows.append(dict(held_out=held_out, test_input=name, subject=int(fold[3:5]), seed=seed,
                                 **{f"test_{m}": v for m, v in s.items()}))
            acc = np.mean([s["acc"] for s in scored])
            print(f"  {name:20s} test acc {acc:.3f}", flush=True)
        write_csv(args.out, rows)

    # the 0.5 s occlusions differ in number between configurations; summarize what both have
    def per_subject(name):
        return np.array([np.mean([r["test_acc"] for r in rows if r["test_input"] == name and r["subject"] == s])
                         for s in range(1, 10)])
    common = [n for n in dict.fromkeys(r["test_input"] for r in rows)
              if {r["held_out"] for r in rows if r["test_input"] == n} == set(CONFIGS)]
    base = per_subject("none")
    print(f"\nboth held-out sessions: {'test input':20s} {'acc':>6s} {'vs none':>8s} {'se':>6s} {'p':>6s}")
    for name in common:
        v = per_subject(name)
        d = v - base
        p = sign_flip_p(-d) if name != "none" else float("nan")
        print(f"{'':23s}{name:20s} {v.mean():6.3f} {d.mean():+8.3f} {d.std(ddof=1) / 3:6.3f} {p:6.3f}", flush=True)


if __name__ == "__main__":
    main()
