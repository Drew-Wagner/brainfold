"""Training time per model per epoch: the braindecode model one at a time, torch.func
ensembling (vmap) of braindecode models, and the packed model.

Random data shaped like BNCI2014001 (22 electrodes, 4 classes): 513 samples for
EEGNet (128 Hz), 1125 for ATCNet (4.5 s at 250 Hz, its defaults). The default
4608 trials is a leave-one-subject-out training set; use 288 for
leave-one-session-out.

    python benchmarks/bench_packing.py
    python benchmarks/bench_packing.py --n-trials 288
    python benchmarks/bench_packing.py --model atcnet
    python benchmarks/bench_packing.py --repeats 5 --out eegnet.csv   # median of 5; one CSV row per measurement
"""

from __future__ import annotations

import argparse
import copy
import csv
import platform
import statistics
import time
import warnings

import torch
import torch.nn.functional as F
from braindecode.models import ATCNet, EEGNet
from torch.func import functional_call, stack_module_state, vmap

from brainfold import PackedATCNet, PackedEEGNet

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


# (name, braindecode model, packed model, the configurations benchmarked)
MODELS = {
    "eegnet": ("EEGNet", EEGNet, PackedEEGNet, [dict(F1=4, D=2), dict(F1=8, D=2)]),
    "atcnet": ("ATCNet", ATCNet, PackedATCNet, [dict()]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, default="eegnet")
    parser.add_argument("--pack-sizes", type=int, nargs="+", default=[1, 4, 9, 18, 36])
    parser.add_argument("--vmap-size", type=int, default=18, help="K for the torch.func ensemble (0: skip)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=4608)
    parser.add_argument("--n-chans", type=int, default=22)
    parser.add_argument("--n-times", type=int, help="default: 513 for EEGNet, 1125 for ATCNet")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=1, help="time each model this many times; report the median")
    parser.add_argument("--out", help="write one CSV row per measurement, with the environment")
    args = parser.parse_args()
    args.n_times = args.n_times or (513 if args.model == "eegnet" else 1125)
    torch.backends.cudnn.benchmark = True
    name, model_class, packed_class, configs = MODELS[args.model]
    shape = dict(n_chans=args.n_chans, n_outputs=4, n_times=args.n_times)

    environment = dict(host=platform.node(), gpu=torch.cuda.get_device_name(args.device), torch=torch.__version__,
                       cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    print(", ".join(f"{k} {v}" for k, v in environment.items()))
    print(f"{'config':>12s} {'model':>26s} {'s/model/epoch':>14s} {'spread':>8s} {'speedup':>8s}")
    rows = []
    for config in configs:
        label = " ".join(f"{k}={v}" for k, v in config.items()) or "default"

        def row(title, forward, parameters, K, base=None):
            """Median seconds per model per epoch over the repeats; spread is (max - min) / median."""
            times = [seconds_per_epoch(forward, parameters, K, args) / K for _ in range(args.repeats)]
            seconds = statistics.median(times)
            speedup = base / seconds if base else 1.0
            print(f"{label:>12s} {title:>26s} {seconds:14.4f} {(max(times) - min(times)) / seconds:8.1%} {speedup:8.1f}",
                  flush=True)
            rows.extend(dict(environment, model=name, config=label, method=title, K=K, n_trials=args.n_trials,
                             n_chans=args.n_chans, n_times=args.n_times, repeat=i, seconds=s)
                        for i, s in enumerate(times))
            return seconds

        base = row(f"braindecode {name}", *single(model_class(**shape, **config).to(args.device)), 1)
        if K := args.vmap_size:
            row(f"torch.func ensemble K={K}",
                *torch_func_ensemble([model_class(**shape, **config).to(args.device) for _ in range(K)]), K, base)
        for K in args.pack_sizes:
            model = packed_class(K, **shape, **config).to(args.device)
            row(f"Packed{name} K={K}", model, list(model.parameters()), K, base)
    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
