"""Windows x bands from the EEGNet and Schirrmeister papers, for the nested search's best configurations.

Retrained per cell, as cue_check.py: each held-out session's best step-1 configuration,
trained on the other session, tested on it; 9 subjects, 3 seeds, no augmentation.
Windows are in seconds after cue onset: the configuration's own (0 to its tmax),
EEGNet's (Lawhern et al., 2018: 0.5-2.5), Schirrmeister et al.'s (2017) FBCSP window
(0.5-4) and ConvNet input (-0.5-4). Bands: the configuration's own, EEGNet's 4-40 Hz,
Schirrmeister's 0-38 Hz (0 is the configuration's fmin: the recordings were
hardware-filtered from 0.5 Hz) and 4-38 Hz. Windows inside [0, 4] are cropped from the
same epochs as the reference; -0.5-4 is epoched separately.
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
    rows = []
    for held_out, cfg in CONFIGS.items():
        windows = {"own": (0.0, cfg["tmax"]), "eegnet": (0.5, 2.5), "schirr_fbcsp": (0.5, 4.0),
                   "schirr_convnet": (-0.5, 4.0)}
        bands = {"own": (cfg["fmin"], cfg["fmax"]), "eegnet_4-40": (4.0, 40.0), "schirr_0-38": (cfg["fmin"], 38.0),
                 "schirr_4-38": (4.0, 38.0)}
        hp = dict(cfg, augment=False, batch_size=2 ** cfg["batch_size_exponent"])
        picks = loader.rec.channels(hp["n_steps_channel_selection"])
        n_eeg = len(loader.eeg_names)
        for band, (fmin, fmax) in bands.items():
            for window, (tmin, tmax) in windows.items():
                if tmin < 0:
                    X, y, subjects, sessions = loader.trials(fmin, fmax, window=(tmin, tmax))
                    x = X[:, :n_eeg][:, picks]
                else:
                    X, y, subjects, sessions = loader.trials(fmin, fmax)
                    x = protocol.crop(X[:, :n_eeg][:, picks], SR, [0, 4], tmin, tmax)
                x = np.ascontiguousarray(x)
                xt = torch.from_numpy(x).to(device)
                data = dict(hp=hp, X=xt, Xn=protocol.standardize(xt), y=torch.from_numpy(y).to(device),
                            eval_epochs=[hp["number_of_epochs"]], aug=None,
                            kwargs=protocol.eegnet_kwargs(hp, x.shape[1], 4, x.shape[2]))
                folds = protocol.nested_folds(subjects, sessions, held_out, 3)
                runs = [(fold, seed, train, dict(test=test)) for seed in seeds for fold, train, test, _ in folds]
                t = time.perf_counter()
                result = train_runs(data, runs, len(runs), run_args, f"{band}/{window}")
                for c in result["curves"]:
                    rows.append(dict(held_out=held_out, band=band, fmin=fmin, fmax=fmax, window=window, tmin=tmin,
                                     tmax=tmax, subject=int(c["fold"][3:5]), seed=c["seed"], n_times=x.shape[2],
                                     **{k: v for k, v in c.items() if k.startswith("test_")}))
                acc = np.mean([c["test_acc"] for c in result["curves"]])
                print(f"heldout{held_out} band {band:12s} ({fmin}-{fmax} Hz) window {window:15s} ({tmin}-{tmax} s): "
                      f"test acc {acc:.3f}  ({time.perf_counter() - t:.0f}s)", flush=True)
                write_csv(args.out, rows)

    # mean over both held-out sessions, subjects and seeds
    print(f"\n{'band':14s}" + "".join(f"{w:>16s}" for w in ("own", "eegnet", "schirr_fbcsp", "schirr_convnet")))
    for band in ("own", "eegnet_4-40", "schirr_0-38", "schirr_4-38"):
        cells = [np.mean([r["test_acc"] for r in rows if r["band"] == band and r["window"] == w])
                 for w in ("own", "eegnet", "schirr_fbcsp", "schirr_convnet")]
        print(f"{band:14s}" + "".join(f"{c:16.3f}" for c in cells), flush=True)


if __name__ == "__main__":
    main()
