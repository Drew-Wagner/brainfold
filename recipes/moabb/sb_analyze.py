"""Analysis of sb_search.py result directories: size against validation, importances, replicate searches.

    uv run --with matplotlib python sb_analyze.py results/sb/imp results/sb/imp-r1 results/sb/imp-r2 --out results/sb/analysis
    uv run --with matplotlib python sb_analyze.py results/sb/bnci2014001-adam --step step1 --out results/sb/analysis-adam

For each step (or --step), from every directory's study.journal and files:

- <step>-size.png: each trial's validation score against its parameter count and
  forward FLOPs (one member), with the Pareto front (no smaller trial scores higher)
  per directory and the standard error as error bars where trials have one.
- <step>-importance.csv: Optuna's fANOVA importances of the searched values for
  the validation score, per directory (an analysis, not a criterion: with 50
  trials, they're noisy).
- <step>-searches.csv: per directory (replicate search), the chosen trial
  (best-<step>.yaml, else the best), its validation, standard error, test, size and
  values; with several directories, the spread of these is printed.
- With rerun-<step>-summary.csv (sb_search.py --rerun): the rank correlation
  between the search's validation scores and the reruns', and which of the
  rerun trials are within two standard errors of the best rerun.

Test scores are reported, never used to choose anything.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import optuna
import yaml


def studies(directory):
    """{study name: study} in a result directory."""
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(directory / "study.journal")))
    return {s.study_name: optuna.load_study(study_name=s.study_name, storage=storage)
            for s in optuna.get_all_study_summaries(storage)}


def trial_table(study):
    """One dict per finished trial: number, valid, se, test, params, flops, values."""
    rows = []
    for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
        a = t.user_attrs
        rows.append(dict(number=t.number, valid=t.values[0], se=a.get("se", float("nan")), test=a.get("test"),
                         params=a.get("params"), flops=a.get("flops"), epoch=a.get("epoch"),
                         train_seconds=a.get("train_seconds"), values=a.get("values", t.params)))
    return rows


def pareto(rows, size):
    """The trials no smaller trial beats on validation, by increasing size."""
    front, best = [], -math.inf
    for r in sorted((r for r in rows if r[size] is not None), key=lambda r: (r[size], -r["valid"])):
        if r["valid"] > best:
            front.append(r)
            best = r["valid"]
    return front


def plot_sizes(tables, step, out):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for (name, rows), color in zip(tables.items(), plt.cm.tab10.colors):
        for ax, size, scale, label in ((axes[0], "params", 1, "parameters"), (axes[1], "flops", 1e-6, "MFLOPs")):
            rows_sized = [r for r in rows if r[size] is not None]
            if not rows_sized:
                continue
            x = np.array([r[size] for r in rows_sized]) * scale
            y = np.array([r["valid"] for r in rows_sized])
            se = np.array([r["se"] for r in rows_sized], dtype=float)
            ax.errorbar(x, y, yerr=np.where(np.isfinite(se), se, 0), fmt="o", ms=3.5, alpha=0.6, color=color,
                        elinewidth=0.8, label=name)
            front = pareto(rows_sized, size)
            ax.step([r[size] * scale for r in front], [r["valid"] for r in front], where="post", color=color, lw=1.2)
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(f"{label} (one member)")
    axes[0].set_ylabel("validation score")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"{step}: validation against size, Pareto fronts as steps")
    fig.tight_layout()
    fig.savefig(out / f"{step}-size.png", dpi=150)
    plt.close(fig)


def importances(study):
    """fANOVA importances of the searched values for the validation score (the first objective)."""
    evaluator = optuna.importance.FanovaImportanceEvaluator(seed=0)
    return optuna.importance.get_param_importances(study, evaluator=evaluator, target=lambda t: t.values[0])


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def chosen(directory, step, rows):
    """The trial the search kept (best-<step>.yaml), else its best."""
    path = directory / f"best-{step}.yaml"
    number = yaml.safe_load(path.read_text())["trial"] if path.exists() else max(rows, key=lambda r: r["valid"])["number"]
    return next(r for r in rows if r["number"] == number)


def reruns(directory, step):
    path = directory / f"rerun-{step}-summary.csv"
    if not path.exists():
        return
    from scipy.stats import spearmanr

    with open(path, newline="") as f:
        rows = [{k: float(v) if k not in ("step",) else v for k, v in r.items() if v != ""} for r in csv.DictReader(f)]
    rho = spearmanr([r["search_valid"] for r in rows], [r["valid"] for r in rows])[0] if len(rows) > 2 else float("nan")
    best = max(rows, key=lambda r: r["valid"])
    tied = [r for r in rows
            if best["valid"] - r["valid"] <= 2 * math.hypot(best.get("se", 0) or 0, r.get("se", 0) or 0)]
    print(f"  reruns of {len(rows)} trials: Spearman rho(search valid, rerun valid) = {rho:.2f}; best rerun trial "
          f"{int(best['trial'])} ({best['valid']:.4f} ± {best.get('se', float('nan')):.4f}); "
          f"{len(tied)} within 2 combined SE of it: {[int(r['trial']) for r in tied]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", type=Path, nargs="+", help="sb_search.py --out directories")
    parser.add_argument("--step", help="only this step (default: every step)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    by_step = {}  # step -> {directory label: (study, rows)}
    labels = [d.name if [x.name for x in args.dirs].count(d.name) == 1 else str(d) for d in args.dirs]
    for directory, label in zip(args.dirs, labels):
        for name, study in studies(directory).items():
            step = name.split("-")[0]
            if args.step and step != args.step:
                continue
            by_step.setdefault(step, {})[f"{label}/{name.split('-', 1)[1]}"] = (directory, study, trial_table(study))

    for step, runs in by_step.items():
        print(f"{step}:")
        tables = {label: rows for label, (_, _, rows) in runs.items() if rows}
        plot_sizes(tables, step, args.out)
        imp_rows, search_rows = [], []
        for label, (directory, study, rows) in runs.items():
            if len(rows) < 2:
                continue
            try:
                imp = importances(study)
            except (RuntimeError, ValueError) as error:  # e.g. a single searched value
                print(f"  {label}: no importances ({error})")
                imp = {}
            imp_rows.append(dict(search=label, **imp))
            top = ", ".join(f"{k} {v:.2f}" for k, v in list(imp.items())[:6])
            print(f"  {label}: {len(rows)} trials; fANOVA importances: {top}")
            c = chosen(directory, step, rows)
            search_rows.append(dict(search=label, trial=c["number"], valid=c["valid"], se=c["se"], test=c["test"],
                                    params=c["params"], flops=c["flops"], epoch=c["epoch"], **c["values"]))
            reruns(directory, step)
        if imp_rows:
            write_csv(args.out / f"{step}-importance.csv", imp_rows)
        if search_rows:
            write_csv(args.out / f"{step}-searches.csv", search_rows)
        if len(search_rows) > 1:  # replicate searches: how much the chosen configuration moves
            for key in ("valid", "test", "params", "flops"):
                v = np.array([r[key] for r in search_rows if r[key] is not None], dtype=float)
                print(f"  chosen {key} over {len(v)} searches: mean {v.mean():.4g}, sd {v.std(ddof=1):.3g}, "
                      f"range {v.min():.4g} - {v.max():.4g}")
            for name in search_rows[0]:
                if name in ("search", "trial", "valid", "se", "test", "params", "flops", "epoch"):
                    continue
                v = [r.get(name) for r in search_rows]
                print(f"  chosen {name}: {v}")


if __name__ == "__main__":
    main()
