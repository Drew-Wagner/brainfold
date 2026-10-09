"""Training time and peak GPU memory per pack size K, on a recipe's own data and training loop.

    python sweep.py hparams/lee2019mi_eegnet.yaml --out sweep.csv
    python sweep.py hparams/bnci2014009_atcnet.yaml --wandb brainfold   # also log to W&B

Every member trains on the recipe's first fold (speed doesn't depend on which
fold), with seeds 0..K-1. Each K gets one warm-up epoch, then timed epochs until
both ``--epochs`` and ``--min-seconds`` are reached; peak memory is measured over
the timed epochs, after cudnn.benchmark has chosen its algorithms. One
braindecode model is timed the same way, as the K=1 baseline.

The largest K is estimated rather than found by running out of memory. Peak
memory is about linear in K: the shared data, plus each member's weights,
optimizer state and activations. Pack sizes run in increasing order; before
each one, memory is fitted on the largest ones measured so far, and the sweep
stops before a K predicted to fill more than 90% of the GPU.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from tracking import environment, finish_run, start_run, write_csv, write_environment
from train import ARCHITECTURES, PackOfOne, add_common_arguments, fit, folds, load_cached, load_hparams

GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
HEADROOM = 0.9  # fraction of the GPU's memory a pack may fill: fragmentation, workspaces


def predicted_memory(rows, K):
    """Memory K members would use: peak reserved memory fitted as a + b K on the three largest pack sizes
    measured, plus the memory outside PyTorch's allocator. None with fewer than two, or if the fit isn't increasing.

    Reserved memory grows slightly faster than linearly, so the fit follows the largest pack sizes and
    extrapolates one step.
    """
    rows = sorted(rows, key=lambda r: r["K"])[-3:]
    if len(rows) < 2:
        return None
    b, a = np.polyfit([r["K"] for r in rows], [r["peak_reserved"] for r in rows], 1)
    return a + b * K + max(r["overhead"] for r in rows) if b > 0 else None


def is_out_of_memory(error):
    """PyTorch's allocator raises OutOfMemoryError; cuBLAS and cuDNN raise RuntimeErrors."""
    return isinstance(error, torch.cuda.OutOfMemoryError) or any(s in str(error) for s in ("out of memory", "ALLOC_FAILED"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--epochs", type=int, default=2, help="least timed epochs per pack size, after one warm-up")
    parser.add_argument("--min-seconds", type=float, default=5.0, help="least timed seconds per pack size")
    parser.add_argument("--max-pack-size", type=int, default=GRID[-1])
    parser.add_argument("--out", type=Path, help="write one CSV row per measurement")
    args = parser.parse_args()
    hp = load_hparams(args.hparams, args.set)
    data, train, evaluation, architecture = hp["data"], hp["train"], hp["evaluation"], hp["architecture"]
    device = args.device
    torch.backends.cudnn.benchmark = True  # as in train.py

    X, y, subjects, sessions = load_cached(data, args.cache_dir)
    n_classes = len(np.unique(y))
    _, train_idx, _ = next(folds(evaluation["protocol"], y, subjects, sessions, evaluation.get("n_folds")))
    total = torch.cuda.mem_get_info(device)[1]  # under MIG, the slice's memory
    shape = dict(n_chans=X.shape[1], n_outputs=n_classes, n_times=X.shape[2])
    settings = dict(lr=train["lr"], batch_size=train["batch_size"], weight_decay=train.get("weight_decay"),
                    class_weight=train.get("class_weight"))
    common = dict(config=Path(args.hparams).stem, dataset=data["dataset"], architecture=architecture,
                  n_train=len(train_idx), batch_size=train["batch_size"], data_bytes=X.nbytes, gpu_memory=total,
                  **shape)
    env = environment(device)
    run = start_run(args.wandb, "sweep", args.hparams, hp, args, env)
    paths = [write_environment(args.out, env), args.out] if args.out else []
    print(f"{architecture} on {data['dataset']}: {len(train_idx)} training trials of {X.shape[1]} channels x "
          f"{X.shape[2]} samples, data {X.nbytes / 2**30:.2f} GiB, GPU {env.get('gpu')} {total / 2**30:.1f} GiB",
          flush=True)

    def measure(method, K):
        """One row: seconds per epoch and peak memory of training K members (or one braindecode model)."""
        seeds = list(range(K))
        row = dict(common, method=method, K=K)
        try:
            model = ARCHITECTURES[architecture].from_seeds(seeds, **shape, **hp["model"]).to(device)
            if method == "braindecode":
                model = PackOfOne(model.to_braindecode()[0]).to(device)
            idx = torch.from_numpy(train_idx).to(device).expand(K, -1)
            # Warm-up: cudnn.benchmark tries algorithms with large workspaces, so peak memory is measured after
            fit(model, X, y, idx, seeds, epochs=1, **settings)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            epochs, t0 = 0, time.perf_counter()
            while epochs < args.epochs or time.perf_counter() - t0 < args.min_seconds:
                fit(model, X, y, idx, seeds, epochs=1, **settings)  # returns after a sync (.cpu())
                epochs += 1
            seconds = (time.perf_counter() - t0) / epochs
            row.update(oom=False, epochs=epochs, seconds_per_epoch=seconds, seconds_per_model_epoch=seconds / K,
                       peak_allocated=torch.cuda.max_memory_allocated(device),
                       peak_reserved=torch.cuda.max_memory_reserved(device),
                       # memory in use outside PyTorch's allocator: CUDA context, cuDNN and cuBLAS handles
                       overhead=total - torch.cuda.mem_get_info(device)[0] - torch.cuda.memory_reserved(device))
        except RuntimeError as error:  # OutOfMemoryError is a RuntimeError
            if not is_out_of_memory(error):
                raise
            row.update(oom=True)
        model = idx = None
        torch.cuda.empty_cache()
        if row["oom"]:
            print(f"{method:>12s} K={K:<4d} out of memory", flush=True)
        else:
            print(f"{method:>12s} K={K:<4d} {row['seconds_per_model_epoch']:.4f} s/model/epoch, peak "
                  f"{row['peak_reserved'] / 2**30:.2f} GiB reserved", flush=True)
        return row

    rows = []
    try:
        X, y = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
    except torch.cuda.OutOfMemoryError:
        print("the data alone doesn't fit on this GPU")
        rows.append(dict(common, method="packed", K=0, oom=True))
    else:
        rows.append(measure("braindecode", 1))
        for K in GRID:
            if K > args.max_pack_size:
                break
            # Predicted before measuring, from the smaller pack sizes; includes the memory outside PyTorch
            predicted = predicted_memory([r for r in rows if r["method"] == "packed" and not r["oom"]], K)
            if predicted is not None and predicted > HEADROOM * total:
                break
            rows.append(dict(measure("packed", K), predicted_memory=predicted))
            if args.out:  # after every pack size, so a job that fails keeps what it measured
                write_csv(args.out, rows)
            if rows[-1]["oom"]:
                break  # the memory model was wrong: worth knowing, and larger K won't fit either
    if args.out:
        write_csv(args.out, rows)
    finish_run(run, rows, paths)


if __name__ == "__main__":
    main()
