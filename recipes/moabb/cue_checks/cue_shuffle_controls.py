"""Controls for the permutation occlusion (cue_shuffle.py): is the dip the block's class information or an
artifact of replacing it?

On the saved baseline models (cue_occlusion.py), a 0.5 s block of each test trial is replaced by the same
block of another test trial of the same subject and session, with these variants:

- source "same": a trial of the same class. The edges and the swapped EEG are as in the others, but the class
  information is kept, so its drop is the artifact alone;
- source "different": a trial of another class (drawn with replacement), so its drop is artifact plus
  information;
- source "any" (cue_shuffle.py's permutation) with cross-fades of 0, 50 and 150 ms outside each edge, and
  "hann": a Hann taper over 1 s centred on the block, which replaces as much signal in total as the 0.5 s box.

    python cue_checks/cue_shuffle_controls.py --models <dir with models-heldout<h>.pt> \\
        --out results/rtx4080/cue-checks/cue_shuffle_controls.csv
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

VARIANTS = [("same", "fade50"), ("different", "fade50"), ("any", "fade0"), ("any", "fade150"), ("any", "hann")]


def weights(shape, a, width, T, device):
    """The blend weight of the swapped-in trial per sample: 1 over the block, with the given edges."""
    w = torch.zeros(T)
    if shape == "hann":  # 1 s Hann centred on the block: the same area as the 0.5 s box
        n = 2 * width
        lo = a + width // 2 - n // 2
        win = torch.hann_window(n, periodic=False)
        keep = slice(max(lo, 0), min(lo + n, T))
        w[keep] = win[keep.start - lo:keep.stop - lo]
        return w.to(device)
    fade = round(int(shape.removeprefix("fade")) / 1000 * SR)
    w[a:a + width] = 1
    if fade:
        ramp = torch.linspace(0, 1, fade + 2)[1:-1]
        lo, hi = max(a - fade, 0), min(a + width + fade, T)
        w[lo:a] = ramp[fade - (a - lo):]
        w[a + width:hi] = ramp.flip(0)[:hi - a - width]
    return w.to(device)


def donors(test, y, source, g):
    """For each trial of ``test``, the trial whose block it receives."""
    labels = y[test]
    if source == "any":
        return test[torch.randperm(len(test), generator=g).numpy()]
    out = np.empty_like(test)
    for i, c in enumerate(labels):
        pool = test[(labels == c) & (test != test[i])] if source == "same" else test[labels != c]
        out[i] = pool[torch.randint(len(pool), (1,), generator=g).item()]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True, help="directory with models-heldout<h>.pt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--block", type=float, default=0.5)
    parser.add_argument("--step", type=float, default=0.125)
    parser.add_argument("--permutations", type=int, default=5)
    args = parser.parse_args()
    device = torch.device("cuda")
    loader = Loader()
    rows = []
    for held_out in CONFIGS:
        saved = torch.load(args.models / f"models-heldout{held_out}.pt", weights_only=False)
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
        base = np.mean([s["acc"] for s in protocol.scores(model, Xn, yt, test_idx)])
        print(f"heldout{held_out}: unchanged test acc {base:.3f}", flush=True)
        width = round(args.block * SR)
        for source, shape in VARIANTS:
            g = torch.Generator().manual_seed(0)
            for start in np.arange(0, hp["tmax"] - args.block + 1e-9, args.step):
                w = weights(shape, round(start * SR), width, T, device)
                accs = []
                for p in range(args.permutations):
                    Xt = Xn.clone()
                    for test in folds.values():
                        idx = torch.from_numpy(test).to(device)
                        src = torch.from_numpy(donors(test, y, source, g)).to(device)
                        Xt[idx] = Xn[idx] * (1 - w) + Xn[src] * w
                    scored = protocol.scores(model, Xt, yt, test_idx)
                    accs.append(np.mean([s["acc"] for s in scored]))
                    for (fold, seed), s in zip(saved["runs"], scored):
                        rows.append(dict(held_out=held_out, source=source, shape=shape, start=float(start),
                                         permutation=p, subject=int(fold[3:5]), seed=seed, base=base,
                                         **{f"test_{m}": v for m, v in s.items()}))
                print(f"  {source:9s} {shape:7s} {start:.3f}-{start + args.block:.3f} s: test acc {np.mean(accs):.3f}",
                      flush=True)
            write_csv(args.out, rows)


if __name__ == "__main__":
    main()
