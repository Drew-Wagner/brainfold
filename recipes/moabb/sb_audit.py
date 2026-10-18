"""The confound audit of the search protocol's version 2 (PROTOCOL.md, "The confound audit").

    python sb_audit.py hparams/sb_moabb/v2/bnci2014001_eegnet.yaml --out results/sb/v2
    python sb_audit.py hparams/sb_moabb/v2/bnci2014001_eegnet.yaml --out /tmp/audit --subjects 1 2 \\
        --set schedule_free.epochs.last=50   # smoke test

Run on a dataset before its search; reported, never used for selection. A fixed
reference model (the hparams file's `fixed` values: Lawhern et al.'s EEGNet-8,2
with version 1's step-0 training) is trained with the protocol's inner folds, within
each outer fold's training data, on:

1. `window`: the protocol's window (0.5 s after the cue's offset to the end of imagery);
2. `cue`: the cue period alone (cue onset to offset, or to 0.5 s if the cue persists),
   in the protocol's band;
3. `cue_slow`: the cue period below 4 Hz (from the hardware band's low edge, at least
   0.5 Hz), where visual evoked responses and eye movements have most of their power;
4. `window_no_frontal`: the protocol's window without the frontal 10-20 rows
   (`audit.frontal_rows`: Fp, AF and F);
5. `window_eog`: the EOG channels alone on the protocol's window, if the dataset has
   any (they're never an input to the search's models).

Motor imagery's desynchronization starts about 0.3 s after the cue's onset
(Pfurtscheller et al., 2008), so 2 holds the cue and early imagery together: above
chance there is expected, and isn't by itself a confound. The confound tests are 5
(eye movements that follow the class, if well above chance), 3 (slow cue-locked
activity) and the drop from 1 to 4 (frontal, possibly ocular or muscle, activity).
Each is reported against the binomial chance threshold for its number of validation
trials (Combrisson and Jerbi, 2015), pooled and per subject.

The model trains to a fixed epoch (with schedule-free, the last of `schedule_free.epochs`;
with Adam, `number_of_epochs`), with no epoch chosen on validation, so the validation
accuracy is a held-out estimate the binomial threshold applies to. The three conditions
share their seeds. Test sessions are never evaluated for the report (their scores,
computed as in any trial, are dropped).

Writes to --out, next to the search results: audit.csv (per condition, pooled and per
subject), audit-cells.csv (per cell), audit.md (the report) and audit-env.json. With
--wandb, each condition is logged as it finishes (training loss per epoch, mean and per
member, the learning rate and seconds per epoch; accuracy, threshold, timings and peak
memory), then the report's rows as a table and the files as an artifact.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import sb_protocol as protocol
import sb_search
from tracking import environment, finish_run, start_run, write_csv, write_environment
from train import load_hparams


def conditions(cfg, n_eog):
    """[(name, hyperparameters to set, description)] of the audit's conditions (``n_eog``: the EOG channels)."""
    paradigm = cfg["paradigm"]
    (tmin, tmax), (cue_min, cue_max) = protocol.protocol_window(paradigm), protocol.cue_window(paradigm)
    rows = list(cfg.get("audit", {}).get("frontal_rows", ["Fp", "AF", "F"]))
    slow_min = max(0.5, (paradigm.get("hardware_band") or (0, None))[0])
    conds = [
        ("window", dict(tmin=tmin, tmax=tmax), f"the protocol's window, {tmin:g}-{tmax:g} s"),
        ("cue", dict(tmin=cue_min, tmax=cue_max),
         f"the cue period alone (the cue and early imagery), {cue_min:g}-{cue_max:g} s"),
        ("cue_slow", dict(tmin=cue_min, tmax=cue_max, fmin=slow_min, fmax=4.0),
         f"the cue period below 4 Hz, {cue_min:g}-{cue_max:g} s, {slow_min:g}-4 Hz"),
        ("window_no_frontal", dict(tmin=tmin, tmax=tmax, drop_rows=rows),
         f"the protocol's window without the rows {', '.join(rows)}"),
    ]
    if n_eog:
        conds.append(("window_eog", dict(tmin=tmin, tmax=tmax, modality="eog"),
                      f"the {n_eog} EOG channels alone on the protocol's window"))
    return conds


def subject_of(fold):
    return int(fold.split("_", 1)[0][3:])


def summarize(cells, n_classes, alpha, n_seeds=1):
    """Pooled and per-subject rows: validation trials, correct, accuracy, balanced accuracy, threshold.

    Each training trial is validated once per outer fold it trains in (once with two
    sessions); with ``n_seeds`` seeds per trial, the counts are per seed.
    """
    rows = []
    groups = {"all": cells}
    for c in cells:
        groups.setdefault(str(subject_of(c["fold"])), []).append(c)
    for scope, group in groups.items():
        n = sum(c["n_valid"] for c in group) / n_seeds
        correct = sum(c["valid_accuracy"] * c["n_valid"] for c in group) / n_seeds
        threshold = protocol.binomial_threshold(int(round(n)), n_classes, alpha)
        rows.append(dict(subject=scope, n_valid=int(round(n)), correct=int(round(correct)), accuracy=correct / n,
                         balanced_accuracy=float(np.mean([c["valid_acc"] for c in group])),
                         chance=1 / n_classes, threshold=threshold, above_chance=bool(correct / n > threshold)))
    return rows


def paired_drop(by_condition, a, b):
    """Mean over subjects of accuracy(a) - accuracy(b), its standard error, and the number of subjects."""
    acc = {name: {r["subject"]: r["accuracy"] for r in rows if r["subject"] != "all"}
           for name, rows in by_condition.items()}
    subjects = sorted(set(acc[a]) & set(acc[b]))
    d = np.array([acc[a][s] - acc[b][s] for s in subjects])
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return float(d.mean()), se, len(d)


def markdown(cfg, info, conds, by_condition, results, alpha):
    """The report."""
    paradigm = cfg["paradigm"]
    lines = [f"# Confound audit: {cfg['dataset']}", "",
             "Reported, never used for selection (PROTOCOL.md, \"The confound audit\").", "",
             f"- Reference model: EEGNet-8,2 (the hparams file's `fixed`), {info['optimizer']}, trained to epoch "
             f"{info['epochs']} (no epoch chosen on validation); {info['params']} parameters per model.",
             f"- Preprocessing: {info['fmin']:g}-{info['fmax']:g} Hz, {info['sample_rate']:g} Hz; every EEG channel.",
             f"- Validation: the protocol's inner folds ({info['inner_folds']}) within each outer fold's training "
             f"data; {info['n_cells']} cells per condition.",
             f"- Chance: {1 / cfg['n_classes']:.3f}; the threshold is the binomial one at p < {alpha} for each "
             "row's number of validation trials (Combrisson and Jerbi, 2015).", ""]
    if paradigm["cue_offset"] == "persists":
        lines += ["**The cue persists through the imagery: the dataset is cue-confounded** (PROTOCOL.md); the "
                  "window starts 0.5 s after the cue's onset.", ""]
    lines += ["The cue period holds the cue and early imagery together, so its accuracy above chance isn't by "
              "itself a confound. The confound tests are the EOG channels alone, the cue period below 4 Hz and "
              "the drop without the frontal rows.", ""]
    lines += ["| condition | channels | trials | accuracy | balanced accuracy | threshold | above chance | "
              "subjects above their threshold |", "|---|---|---|---|---|---|---|---|"]
    for name, _, description in conds:
        pooled = next(r for r in by_condition[name] if r["subject"] == "all")
        subjects = [r for r in by_condition[name] if r["subject"] != "all"]
        lines.append(f"| {name}: {description} | {results[name]['n_channels']} | {pooled['n_valid']} | "
                     f"{pooled['accuracy']:.3f} | {pooled['balanced_accuracy']:.3f} | {pooled['threshold']:.3f} | "
                     f"{'yes' if pooled['above_chance'] else 'no'} | "
                     f"{sum(r['above_chance'] for r in subjects)} of {len(subjects)} |")
    drop, se, n = paired_drop(by_condition, "window", "window_no_frontal")
    lines += [""] + [f"- {name}: accuracy {r['accuracy']:.3f} against a threshold of {r['threshold']:.3f} "
                     f"({r['accuracy'] - r['threshold']:+.3f})."
                     for name in ("cue", "cue_slow", "window_eog") if name in by_condition
                     for r in [next(r for r in by_condition[name] if r["subject"] == "all")]]
    lines += [f"- Window minus window without the frontal rows: {drop:+.3f} ± {se:.3f} (mean ± standard error over "
              f"{n} subjects, paired).", "",
              "Per subject:", "", "| subject | " + " | ".join(name for name, _, _ in conds) + " | trials |",
              "|---|" + "---|" * (len(conds) + 1)]
    subjects = [r["subject"] for r in by_condition["window"] if r["subject"] != "all"]
    for s in subjects:
        cells = []
        for name, _, _ in conds:
            r = next(r for r in by_condition[name] if r["subject"] == s)
            cells.append(f"{r['accuracy']:.3f}{'' if r['above_chance'] else ' (n.s.)'}")
        n_valid = next(r for r in by_condition["window"] if r["subject"] == s)["n_valid"]
        lines.append(f"| {s} | " + " | ".join(cells) + f" | {n_valid} |")
    lines += ["", "(n.s.: not above the subject's binomial threshold.)", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hparams")
    parser.add_argument("--out", type=Path, required=True, help="the search's directory: the report goes next to it")
    parser.add_argument("--optimizer", choices=["adam", "schedule-free"], default="schedule-free")
    parser.add_argument("--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                        help="override the hparams file, e.g. schedule_free.epochs.last=50")
    parser.add_argument("--subjects", type=int, nargs="+", help="a subset of subjects (smoke tests)")
    parser.add_argument("--device", type=torch.device, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true", help="train through torch.compile")
    parser.add_argument("--wandb", metavar="PROJECT", help="log to this W&B project (set WANDB_MODE=offline on clusters)")
    parser.add_argument("--trial-seed", type=int, help="the base of the seeds (default: `seed`)")
    args = parser.parse_args()
    cfg = load_hparams(args.hparams, args.set)
    settings = dict(sb_search.PROTOCOL, **cfg.get("protocol", {}))
    if settings["window"] != "paradigm" or not settings["inner_folds"]:
        raise SystemExit("the audit needs protocol.window: paradigm and inner folds (version 2's hparams)")
    torch.backends.cudnn.benchmark = True
    args.out.mkdir(parents=True, exist_ok=True)
    alpha = cfg.get("audit", {}).get("alpha", 0.05)
    seed_base = cfg["seed"] if args.trial_seed is None else args.trial_seed

    t0 = time.perf_counter()
    recordings = protocol.Recordings(cfg["dataset"], cfg["n_classes"], cfg["sample_rate"], subjects=args.subjects)
    rules = sb_search.apply_paradigm(cfg, settings, recordings)
    env = dict(environment(args.device), load_seconds=time.perf_counter() - t0, optimizer=args.optimizer,
               trial_seed=seed_base, **rules)
    paths = [write_environment(args.out / "audit", env)]
    run = start_run(args.wandb, "sb-audit", args.hparams, cfg, args, env,
                    **sb_search.run_names(args.hparams, args.out))
    log = sb_search.Logger(run, cfg["metric"])
    print(f"{cfg['dataset']}: recordings of {len(recordings.subjects)} subjects loaded in {env['load_seconds']:.0f}s; "
          f"{rules}", flush=True)

    hp = dict(cfg["fixed"], augment="none")
    if args.optimizer == "schedule-free":
        hp["chosen_epochs"] = cfg["schedule_free"]["epochs"]["last"]  # trained to it, evaluated there only
    epochs = hp.get("chosen_epochs", hp["number_of_epochs"])
    seeds = sb_search.trial_seeds(seed_base, settings["seeds_per_trial"], 0, settings["trial_seeds"] == "per_trial")
    conds = conditions(cfg, len(recordings.eog_names))
    cells, by_condition, results = [], {}, {}
    for name, values, description in conds:
        result = sb_search.run_trial(dict(hp, **values), seeds, recordings, cfg, args, f"audit/{name}", settings)
        results[name] = result
        condition_cells = [dict(condition=name, **{k: v for k, v in r.items() if not k.startswith("test_")})
                           for r in result["rows"]]
        cells += condition_cells
        by_condition[name] = summarize(condition_cells, cfg["n_classes"], alpha, len(seeds))
        pooled = by_condition[name][0]
        print(f"audit {name} ({description}): accuracy {pooled['accuracy']:.4f} on {pooled['n_valid']} validation "
              f"trials, threshold {pooled['threshold']:.4f} (p < {alpha}); {result['n_channels']} channels, "
              f"{result['n_times']} samples; {result['n_cells']} cells in {result['n_packs']} packs, "
              f"{result['train_seconds']:.0f}s training, {result['data_seconds']:.0f}s data, "
              f"{result['peak_allocated'] / 2**30:.1f} GiB", flush=True)
        n_train = int(result["rows"][0]["n_train"])
        log.trial(f"audit/{name}", result, {k: v for k, v in values.items() if k not in ("drop_rows", "modality")}, float("nan"),
                  float("nan"), sb_search.epoch_lr(hp, args.optimizer, n_train), (result["params"], result["flops"]))
        if run is not None:
            run.log({f"audit/{name}/accuracy": pooled["accuracy"], f"audit/{name}/threshold": pooled["threshold"],
                     f"audit/{name}/n_channels": result["n_channels"], f"audit/{name}/n_times": result["n_times"]})

    report_rows = [dict(condition=name, description=description, **r)
                   for name, _, description in conds for r in by_condition[name]]
    write_csv(args.out / "audit.csv", report_rows)
    write_csv(args.out / "audit-cells.csv", cells)
    first = results["window"]
    info = dict(optimizer=args.optimizer, epochs=epochs, params=first["params"], sample_rate=rules["sample_rate"],
                fmin=rules["fmin"], fmax=rules["fmax"], n_cells=first["n_cells"],
                inner_folds=settings["inner_folds"])
    text = markdown(cfg, info, conds, by_condition, results, alpha)
    (args.out / "audit.md").write_text(text)
    print(text, flush=True)
    paths += [args.out / name for name in ("audit.csv", "audit-cells.csv", "audit.md")]
    drop, se, _ = paired_drop(by_condition, "window", "window_no_frontal")
    summary = {f"audit/{name}/accuracy": by_condition[name][0]["accuracy"] for name, _, _ in conds}
    summary.update({"audit/frontal_drop": drop, "audit/frontal_drop_se": se})
    finish_run(run, report_rows, paths, summary)


if __name__ == "__main__":
    main()
