"""Training time per model per epoch: braindecode EEGNet one at a time vs. PackedEEGNet.

Random data shaped like BNCI2014001 (22 electrodes, 513 samples, 4 classes).
The default 4608 trials is a leave-one-subject-out training set; use 288 for
leave-one-session-out.

    python benchmarks/bench_packing.py
    python benchmarks/bench_packing.py --n-trials 288
"""

from __future__ import annotations

import argparse
import time
import warnings

import torch
from braindecode.models import EEGNet

from packed_eegnet import PackedEEGNet, PackOfOne, packed_batches, packed_loss

warnings.filterwarnings("ignore", category=DeprecationWarning)


def seconds_per_epoch(model, K, args) -> float:
    """Mean seconds per training epoch of `model`, which maps (batch, K, C, T) -> (batch, K, n_outputs)."""
    X = torch.randn(args.n_trials, args.n_chans, args.n_times, device=args.device)
    y = torch.randint(4, (args.n_trials,), device=args.device)
    train_idx = torch.stack([torch.randperm(args.n_trials, device=args.device) for _ in range(K)])
    generators = [torch.Generator(device=args.device).manual_seed(k) for k in range(K)]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def epoch():
        model.train()
        for idx in packed_batches(train_idx, args.batch_size, generators):
            loss = packed_loss(model(X[idx]), y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    epoch()  # warmup: cuDNN autotuning, allocator
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.epochs):
        epoch()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / args.epochs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack-sizes", type=int, nargs="+", default=[1, 4, 9, 18, 36])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=4608)
    parser.add_argument("--n-chans", type=int, default=22)
    parser.add_argument("--n-times", type=int, default=513)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    shape = (args.n_chans, 4, args.n_times)

    print(f"{'F1':>3s} {'D':>2s} {'model':>18s} {'s/model/epoch':>14s} {'speedup':>8s}")
    for F1, D in [(4, 2), (8, 2)]:
        model = PackOfOne(EEGNet(*shape, F1=F1, D=D)).to(args.device)
        base = seconds_per_epoch(model, 1, args)
        print(f"{F1:3d} {D:2d} {'braindecode EEGNet':>18s} {base:14.4f} {1:8.1f}", flush=True)
        for K in args.pack_sizes:
            seconds = seconds_per_epoch(PackedEEGNet(K, *shape, F1=F1, D=D).to(args.device), K, args) / K
            print(f"{F1:3d} {D:2d} {f'PackedEEGNet K={K}':>18s} {seconds:14.4f} {base / seconds:8.1f}", flush=True)


if __name__ == "__main__":
    main()
