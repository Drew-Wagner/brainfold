"""Plots of sweep.py --out CSVs: time per model against pack size K, and predicted against measured memory.

    python plot_sweep.py results/h100-mig/sweep-*/*.csv --out results/h100-mig

Writes ``sweep-time.png``, one panel per configuration with a line per GPU, and
``sweep-memory.png``. The dashed lines are one braindecode model; the open
markers are each GPU's elbow, the smallest K within 10% of its best time per model.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

ELBOW = 1.10  # an elbow's time per model is within this factor of the best
PLAIN = FuncFormatter(lambda x, _: f"{x:g}")  # 6 rather than 6 x 10^0 on log axes


def plain(ax, x=True):
    """Ticks at 1, 2 and 5 times powers of 10, labelled as plain numbers; with x=False, only on the y axis."""
    for axis in (ax.xaxis, ax.yaxis) if x else (ax.yaxis,):
        axis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        axis.set_major_formatter(PLAIN)
        axis.set_minor_formatter(NullFormatter())


def read(path):
    """A CSV's rows, with the GPU from the -env.json next to it."""
    path = Path(path)
    env = path.with_name(f"{path.stem}-env.json")
    gpu = json.loads(env.read_text()).get("gpu", "") if env.exists() else ""
    return pd.read_csv(path).assign(gpu=gpu.replace("NVIDIA ", "").replace(" 80GB HBM3", ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--out", type=Path, default=Path("."))
    args = parser.parse_args()
    rows = pd.concat([read(path) for path in args.csv])
    rows["used"] = rows.peak_reserved + rows.overhead  # what predicted_memory predicts
    datasets = list(dict.fromkeys(rows.dataset))
    architectures = list(dict.fromkeys(rows.architecture))
    gpus = list(rows.sort_values("gpu_memory").gpu.unique())
    colors = dict(zip(gpus, plt.rcParams["axes.prop_cycle"].by_key()["color"]))

    fig, axes = plt.subplots(len(architectures), len(datasets), figsize=(3.6 * len(datasets), 3.2 * len(architectures)),
                             sharex=True, squeeze=False)
    for (dataset, architecture, gpu), df in rows.groupby(["dataset", "architecture", "gpu"]):
        ax = axes[architectures.index(architecture), datasets.index(dataset)]
        packed = df[(df.method == "packed") & ~df.oom].sort_values("K")
        ms = 1e3 * packed.seconds_per_model_epoch
        elbow = packed[ms.values <= ELBOW * ms.min()].iloc[0]
        ax.plot(packed.K, ms, "o-", ms=3, color=colors[gpu], label=gpu)
        ax.plot(elbow.K, 1e3 * elbow.seconds_per_model_epoch, "o", ms=9, mfc="none", color=colors[gpu])
        ax.axhline(1e3 * df[df.method == "braindecode"].seconds_per_model_epoch.iloc[0], ls="--", lw=1,
                   color=colors[gpu])
        ax.set(title=f"{dataset}, {architecture}", xscale="log", yscale="log")
        plain(ax, x=False)
        ax.set_xticks([1, 4, 16, 64, 256], ["1", "4", "16", "64", "256"])
        ax.xaxis.set_minor_formatter(NullFormatter())
    for ax in axes[-1]:
        ax.set_xlabel("pack size K")
    for ax in axes[:, 0]:
        ax.set_ylabel("ms per model per epoch")
    axes[0, 0].legend(title="GPU", fontsize="small")
    fig.tight_layout()
    fig.savefig(args.out / "sweep-time.png", dpi=120)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    predicted = rows[rows.predicted_memory.notna() & ~rows.oom]
    for gpu, df in predicted.groupby("gpu"):
        ax.plot(df.used / 2**30, df.predicted_memory / 2**30, "o", ms=3, color=colors[gpu], label=gpu)
        ax.axvline(0.9 * df.gpu_memory.iloc[0] / 2**30, ls=":", lw=1, color=colors[gpu])  # sweep.py's HEADROOM
    line = np.array([0.3, 20])
    ax.plot(line, line, "k-", lw=1)
    ax.fill_between(line, 0.9 * line, 1.1 * line, color="k", alpha=0.1, lw=0, label="±10%")
    ax.set(xscale="log", yscale="log", xlabel="measured GiB", ylabel="predicted GiB",
           title="Memory predicted before each K ran")
    plain(ax)
    ax.legend(title="GPU", fontsize="small")
    fig.tight_layout()
    fig.savefig(args.out / "sweep-memory.png", dpi=120)


if __name__ == "__main__":
    main()
