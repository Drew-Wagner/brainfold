"""Training time per model per epoch: braindecode EEGNet one at a time, torch.func
ensembling (vmap) of braindecode EEGNets, and PackedEEGNet.

Random data shaped like BNCI2014001 (22 electrodes, 513 samples, 4 classes).
The default 4608 trials is a leave-one-subject-out training set; use 288 for
leave-one-session-out.

    python benchmarks/bench_packing.py
    python benchmarks/bench_packing.py --n-trials 288
"""

from __future__ import annotations

import argparse
import copy
import time
import warnings

import torch
import torch.nn.functional as F
from braindecode.models import EEGNet
from torch.func import functional_call, stack_module_state, vmap

from packed_eegnet import PackedEEGNet

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*batching rule.*")  # vmap falls back to a loop for renorm


def seconds_per_epoch(forward, parameters, K, args) -> float:
    """Mean seconds per training epoch of K models; `forward` maps (batch, K, C, T) -> (batch, K, n_outputs)."""
    X = torch.randn(args.n_trials, args.n_chans, args.n_times, device=args.device)
    y = torch.randint(4, (args.n_trials,), device=args.device)
    optimizer = torch.optim.Adam(parameters, lr=1e-3)

    def epoch():
        order = torch.stack([torch.randperm(args.n_trials, device=args.device) for _ in range(K)])
        for i in range(0, args.n_trials, args.batch_size):
            idx = order[:, i : i + args.batch_size].T
            losses = F.cross_entropy(forward(X[idx]).permute(0, 2, 1), y[idx], reduction="none").mean(0)
            optimizer.zero_grad(set_to_none=True)
            losses.sum().backward()
            optimizer.step()

    epoch()  # warmup: cuDNN autotuning, allocator
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.epochs):
        epoch()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / args.epochs


def single(model):
    return (lambda x: model(x[:, 0])[:, None]), list(model.parameters())


def torch_func_ensemble(models):
    """The torch.func model-ensembling recipe: stacked parameters, vmap over functional_call."""
    params, buffers = stack_module_state(models)
    base = copy.deepcopy(models[0]).to("meta")
    call = vmap(lambda p, b, x: functional_call(base, (p, b), (x,)), in_dims=(0, 0, 1), out_dims=1,
                randomness="different")
    return (lambda x: call(params, buffers, x)), list(params.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack-sizes", type=int, nargs="+", default=[1, 4, 9, 18, 36])
    parser.add_argument("--vmap-size", type=int, default=18, help="K for the torch.func ensemble")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=4608)
    parser.add_argument("--n-chans", type=int, default=22)
    parser.add_argument("--n-times", type=int, default=513)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    shape = (args.n_chans, 4, args.n_times)

    print(f"{'F1':>3s} {'D':>2s} {'model':>24s} {'s/model/epoch':>14s} {'speedup':>8s}")
    for F1, D in [(4, 2), (8, 2)]:
        def row(name, forward, parameters, K):
            seconds = seconds_per_epoch(forward, parameters, K, args) / K
            print(f"{F1:3d} {D:2d} {name:>24s} {seconds:14.4f} {base / seconds:8.1f}", flush=True)

        base = seconds_per_epoch(*single(EEGNet(*shape, F1=F1, D=D).to(args.device)), 1, args)
        print(f"{F1:3d} {D:2d} {'braindecode EEGNet':>24s} {base:14.4f} {1:8.1f}", flush=True)
        K = args.vmap_size
        row(f"torch.func ensemble K={K}",
            *torch_func_ensemble([EEGNet(*shape, F1=F1, D=D).to(args.device) for _ in range(K)]), K)
        for K in args.pack_sizes:
            model = PackedEEGNet(K, *shape, F1=F1, D=D).to(args.device)
            row(f"PackedEEGNet K={K}", model, list(model.parameters()), K)


if __name__ == "__main__":
    main()
