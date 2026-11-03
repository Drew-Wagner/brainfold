"""Permutation occlusion: the saved baseline models (cue_occlusion.py) tested with a 0.5 s block of each
test trial replaced by the same block of another test trial of the same subject and session (a random
permutation), cross-faded over 50 ms outside each edge; block stepped every 0.125 s; 5 permutations each.
"""

import argparse
import sys
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
    parser.add_argument("--block", type=float, default=0.5)
    parser.add_argument("--step", type=float, default=0.125)
    parser.add_argument("--fade", type=float, default=0.05)
    parser.add_argument("--permutations", type=int, default=5)
    args = parser.parse_args()
    device = torch.device("cuda")
    loader = Loader()
    rows = []
    for held_out, cfg in CONFIGS.items():
        saved = torch.load(args.out.with_name(f"models-heldout{held_out}.pt"), weights_only=False)
        hp = saved["hp"]
        X, y, subjects, sessions = loader.trials(hp["fmin"], hp["fmax"])
        x = np.ascontiguousarray(protocol.crop(X[:, :len(loader.eeg_names)][:, loader.rec.channels(
            hp["n_steps_channel_selection"])], SR, [0, 4], 0.0, hp["tmax"]))
        Xn = protocol.standardize(torch.from_numpy(x).to(device))
        yt = torch.from_numpy(y).to(device)
        T = Xn.shape[2]
        folds = {fold: test for fold, _, test, _ in protocol.nested_folds(subjects, sessions, held_out, 3)}
        test_idx = torch.stack([torch.from_numpy(folds[fold]) for fold, _ in saved["runs"]]).to(device)
        model = protocol.build(protocol.eegnet_kwargs(hp, Xn.shape[1], 4, T), [s for _, s in saved["runs"]], device)
        model.load_state_dict(saved["state"])
        model.eval()
        base = protocol.scores(model, Xn, yt, test_idx)
        print(f"heldout{held_out}: unshuffled test acc {np.mean([s['acc'] for s in base]):.3f}", flush=True)
        for (fold, seed), s in zip(saved["runs"], base):
            rows.append(dict(held_out=held_out, start=None, permutation=None, subject=int(fold[3:5]), seed=seed,
                             **{f"test_{m}": v for m, v in s.items()}))
        width, fade = round(args.block * SR), round(args.fade * SR)
        g = torch.Generator().manual_seed(0)
        for start in np.arange(0, hp["tmax"] - args.block + 1e-9, args.step):
            a = round(start * SR)
            w = torch.zeros(T)
            w[a:a + width] = 1
            ramp = torch.linspace(0, 1, fade + 2)[1:-1]
            lo, hi = max(a - fade, 0), min(a + width + fade, T)
            w[lo:a] = ramp[fade - (a - lo):]
            w[a + width:hi] = ramp.flip(0)[:hi - a - width]
            w = w.to(device)
            accs = []
            for p in range(args.permutations):
                Xt = Xn.clone()
                for test in folds.values():  # within each subject's held-out session
                    idx = torch.from_numpy(test).to(device)
                    perm = idx[torch.randperm(len(idx), generator=g).to(device)]
                    Xt[idx] = Xn[idx] * (1 - w) + Xn[perm] * w
                scored = protocol.scores(model, Xt, yt, test_idx)
                accs.append(np.mean([s["acc"] for s in scored]))
                for (fold, seed), s in zip(saved["runs"], scored):
                    rows.append(dict(held_out=held_out, start=float(start), permutation=p, subject=int(fold[3:5]),
                                     seed=seed, **{f"test_{m}": v for m, v in s.items()}))
            print(f"  shuffled {start:.3f}-{start + args.block:.3f} s: test acc {np.mean(accs):.3f} "
                  f"(permutations {min(accs):.3f}-{max(accs):.3f})", flush=True)
        write_csv(args.out, rows)


if __name__ == "__main__":
    main()
