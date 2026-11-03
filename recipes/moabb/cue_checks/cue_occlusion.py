"""Fine zero-shot occlusion: the baseline models (cue_zeroshot.py's) tested with a 0.5 s block of the
test trials zeroed (standardized, i.e. the trial's mean), the block stepped every 0.125 s across the window.
The trained models are saved (models-heldout<h>.pt) for later zero-shot tests.
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
from tracking import write_csv  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--block", type=float, default=0.5)
    parser.add_argument("--step", type=float, default=0.125)
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    loader = Loader()
    seeds = [1986 + s for s in range(args.seeds)]
    rows = []
    for held_out, cfg in CONFIGS.items():
        hp = dict(cfg, augment=False, batch_size=2 ** cfg["batch_size_exponent"])
        X, y, subjects, sessions = loader.trials(hp["fmin"], hp["fmax"])
        x = np.ascontiguousarray(protocol.crop(X[:, :len(loader.eeg_names)][:, loader.rec.channels(hp["n_steps_channel_selection"])], SR, [0, 4],
                                               0.0, hp["tmax"]))
        X = torch.from_numpy(x).to(device)
        Xn = protocol.standardize(X)
        yt = torch.from_numpy(y).to(device)
        T = X.shape[2]
        folds = protocol.nested_folds(subjects, sessions, held_out, 3)
        runs = [(fold, seed, train, test) for seed in seeds for fold, train, test, _ in folds]
        train_idx = torch.stack([torch.from_numpy(r[2]) for r in runs]).to(device)
        test_idx = torch.stack([torch.from_numpy(r[3]) for r in runs]).to(device)
        run_seeds = [r[1] for r in runs]
        t = time.perf_counter()
        model = protocol.build(protocol.eegnet_kwargs(hp, X.shape[1], 4, T), run_seeds, device)
        model, _ = protocol.fit(model, X, Xn, yt, train_idx, run_seeds, hp, optimizer="adam")
        torch.save(dict(state=model.state_dict(), runs=[(r[0], r[1]) for r in runs], hp=hp),
                   args.out.with_name(f"models-heldout{held_out}.pt"))
        print(f"heldout{held_out}: trained {len(runs)} models in {time.perf_counter() - t:.0f}s", flush=True)
        width = round(args.block * SR)
        for start in [None] + list(np.arange(0, hp["tmax"] - args.block + 1e-9, args.step)):
            Xt = Xn
            if start is not None:
                a = round(start * SR)
                Xt = Xn.clone()
                Xt[..., a:a + width] = 0
            scored = protocol.scores(model, Xt, yt, test_idx)
            for (fold, seed, *_), s in zip(runs, scored):
                rows.append(dict(held_out=held_out, start=start, end=None if start is None else start + args.block,
                                 subject=int(fold[3:5]), seed=seed, **{f"test_{m}": v for m, v in s.items()}))
            label = "none" if start is None else f"{start:.3f}-{start + args.block:.3f} s"
            print(f"  occluded {label:16s} test acc {np.mean([s['acc'] for s in scored]):.3f}", flush=True)
        write_csv(args.out, rows)


if __name__ == "__main__":
    main()
