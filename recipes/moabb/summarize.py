"""Markdown table of train.py --out CSVs: one row per dataset and model, packed vs braindecode.

    python summarize.py results/*.csv

CSVs of the same dataset and model (e.g. one per seed) are combined. The
braindecode columns, the per-run difference and the times compare the runs
that have a braindecode baseline.
"""

import argparse

import pandas as pd

PARADIGMS = {"BNCI2014_001": "MI", "Lee2019_MI": "MI", "BNCI2014_009": "P300", "Lee2019_ERP": "P300",
             "Nakanishi2015": "SSVEP", "Lee2019_SSVEP": "SSVEP"}
METRICS = ("accuracy", "balanced_accuracy", "kappa", "roc_auc")  # train.py's


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="+")
    args = parser.parse_args()
    print("| task | dataset | model | runs | metric | packed | braindecode | packed - braindecode, per run "
          "| packed time | braindecode time | speedup |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    results = pd.concat([pd.read_csv(path) for path in args.csv])
    if "seconds" not in results:  # newer CSVs time training and evaluation separately
        results["seconds"] = results.train_seconds + results.eval_seconds
    for (dataset, model), df in results.groupby(["dataset", "architecture"], sort=False):
        metric = next(c for c in df.dropna(axis=1, how="all").columns if c in METRICS)  # the first metric
        packed, single = (df[df.method == m].set_index(["seed", "fold"]) for m in ("packed", "braindecode"))
        cells = [PARADIGMS.get(dataset, ""), dataset, model, len(packed), metric.replace("_", " "),
                 f"{packed[metric].mean():.3f} ± {packed[metric].std(ddof=0):.3f}"]
        if len(single):
            matched = packed.loc[single.index]
            diff = matched[metric] - single[metric]
            cells += [f"{single[metric].mean():.3f} ± {single[metric].std(ddof=0):.3f}",
                      f"{diff.mean():+.3f} ± {diff.std(ddof=0):.3f}", f"{matched.seconds.sum():.0f} s",
                      f"{single.seconds.sum():.0f} s", f"{single.seconds.sum() / matched.seconds.sum():.1f}x"]
        else:
            cells += ["", "", f"{packed.seconds.sum():.0f} s", "", ""]
        print("| " + " | ".join(str(c) for c in cells) + " |")


if __name__ == "__main__":
    main()
