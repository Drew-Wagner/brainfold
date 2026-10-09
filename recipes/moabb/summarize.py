"""Markdown table of train.py --out CSVs: one row per configuration, packed vs braindecode.

    python summarize.py results/*.csv
"""

import argparse

import pandas as pd

PARADIGMS = {"BNCI2014_001": "MI", "Lee2019_MI": "MI", "BNCI2014_009": "P300", "Lee2019_ERP": "P300",
             "Nakanishi2015": "SSVEP", "Lee2019_SSVEP": "SSVEP"}
NOT_METRICS = {"dataset", "architecture", "protocol", "method", "seed", "fold", "pack_size", "seconds"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="+")
    args = parser.parse_args()
    print("| task | dataset | model | runs | metric | packed | braindecode | packed - braindecode, per run "
          "| packed time | braindecode time | speedup |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for path in args.csv:
        df = pd.read_csv(path)
        metric = next(c for c in df.columns if c not in NOT_METRICS)  # the first metric
        packed, single = (df[df.method == m].set_index(["seed", "fold"]) for m in ("packed", "braindecode"))
        diff = (packed[metric] - single[metric]).dropna()
        dataset, model = df.dataset.iloc[0], df.architecture.iloc[0]
        cells = [PARADIGMS.get(dataset, ""), dataset, model, len(packed), metric.replace("_", " "),
                 f"{packed[metric].mean():.3f} ± {packed[metric].std(ddof=0):.3f}"]
        if len(single):
            cells += [f"{single[metric].mean():.3f} ± {single[metric].std(ddof=0):.3f}",
                      f"{diff.mean():+.3f} ± {diff.std(ddof=0):.3f}", f"{packed.seconds.sum():.0f} s",
                      f"{single.seconds.sum():.0f} s", f"{single.seconds.sum() / packed.seconds.sum():.1f}x"]
        else:
            cells += ["", "", f"{packed.seconds.sum():.0f} s", "", ""]
        print("| " + " | ".join(str(c) for c in cells) + " |")


if __name__ == "__main__":
    main()
