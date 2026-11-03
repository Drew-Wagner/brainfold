"""Time-resolved decoding: 1 s windows every 0.25 s, from 1 s before the cue to 4 s after it.

As cue_check.py (retrained per window; each held-out session's chosen configuration and
band; 9 subjects, 3 seeds, no augmentation), for the EEG and the 3 EOG channels. Every
window is cropped from the same epochs, -1 to 4 s after cue onset.
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

SPAN = (-1.0, 4.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    device = torch.device("cuda")
    run_args = argparse.Namespace(device=device, optimizer="adam", compile=False)
    torch.backends.cudnn.benchmark = True
    loader = Loader()
    seeds = [1986 + s for s in range(args.seeds)]
    starts = np.arange(-1.0, 3.01, 0.25)
    rows = []
    for held_out, cfg in CONFIGS.items():
        hp = dict(cfg, augment=False, batch_size=2 ** cfg["batch_size_exponent"])
        X, y, subjects, sessions = loader.trials(cfg["fmin"], cfg["fmax"], window=SPAN)
        n_eeg = len(loader.eeg_names)
        inputs = {"EEG": X[:, :n_eeg][:, loader.rec.channels(hp["n_steps_channel_selection"])], "EOG": X[:, n_eeg:]}
        folds = protocol.nested_folds(subjects, sessions, held_out, 3)
        for channels, full in inputs.items():
            for tmin in starts:
                x = np.ascontiguousarray(protocol.crop(full, SR, list(SPAN), tmin, tmin + 1.0))
                xt = torch.from_numpy(x).to(device)
                data = dict(hp=hp, X=xt, Xn=protocol.standardize(xt), y=torch.from_numpy(y).to(device),
                            eval_epochs=[hp["number_of_epochs"]], aug=None,
                            kwargs=protocol.eegnet_kwargs(hp, x.shape[1], 4, x.shape[2]))
                runs = [(fold, seed, train, dict(test=test)) for seed in seeds for fold, train, test, _ in folds]
                t = time.perf_counter()
                result = train_runs(data, runs, len(runs), run_args, f"{channels}/{tmin}")
                for c in result["curves"]:
                    rows.append(dict(held_out=held_out, channels=channels, tmin=float(tmin), tmax=float(tmin) + 1.0,
                                     subject=int(c["fold"][3:5]), seed=c["seed"], n_times=x.shape[2],
                                     **{k: v for k, v in c.items() if k.startswith("test_")}))
                acc = np.mean([c["test_acc"] for c in result["curves"]])
                print(f"heldout{held_out} {channels} {tmin:+.2f} to {tmin + 1:+.2f} s: test acc {acc:.3f} "
                      f"({x.shape[2]} samples, {time.perf_counter() - t:.0f}s)", flush=True)
                write_csv(args.out, rows)


if __name__ == "__main__":
    main()
