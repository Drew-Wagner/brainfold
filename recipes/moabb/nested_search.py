"""SpeechBrain-MOABB's search (sb_search.py), nested, cross-validated, and choosing for size.

    python nested_search.py hparams/sb_moabb/bnci2014001_eegnet.yaml --out results/sb-nested/adam
    python nested_search.py hparams/sb_moabb/bnci2014001_eegnet.yaml --out /tmp/smoke --subjects 1 2 3 \\
        --set trials=2 startup_trials=2 final_seeds=1 nested.seeds=1 fixed.number_of_epochs=3 \\
        --set-search step1.number_of_epochs="uniform(2, 4, discrete=True)"   # smoke test

Three changes to their protocol, in the `nested` section of the hparams file:

- Nested: one search per held-out session (``held_out``, by position, e.g. 0 for
  BNCI2014001's 0train). Its trials see only the other sessions, of every
  subject; its chosen hyperparameters are then trained on those sessions and
  tested on the held-out one. In sb_search.py, as in SpeechBrain-MOABB, the
  validation trials of one fold's training session are the test trials of the
  folds testing on that session, so the search chooses on test data. Here the
  reported score is that of the procedure (search, then train), which may choose
  different hyperparameters for each held-out session.
- Lower-variance trials: each trial's score is the mean over ``inner_folds``
  contiguous blocks of the training sessions (whole runs where they divide
  evenly; sb_protocol.nested_folds) and ``seeds`` seeds, different for every
  trial, instead of one 20% split and one seed shared by every trial.
- The simplest configuration within ``width`` spreads of the best (Breiman's
  one-standard-error rule). Trials are compared paired by subject: the gap of a
  trial to the best is the mean over subjects of their per-subject difference,
  and its spread the standard error of that mean (``se``) or the standard
  deviation over subjects (``sd``). Subjects are the independent units: a
  subject's folds and trials are correlated. "Simplest" is fewest ``flops``
  (one trial's forward) or ``params``; ``null`` chooses the best.

Each step also records, for every trial, a one-sided sign-flip permutation
p-value for the best being better than it, over subjects (exact up to 16
subjects). The final test scores carry Combrisson and Jerbi's (2015) binomial
chance level per fold (p < ``alpha``), since 1 / n_classes is too low for small
test sets. It treats test trials as independent, so it is a minimum: slow drift
within a session makes neighbouring trials' errors correlated.

Outputs in --out: study.journal (resumable as sb_search.py), and per held-out
session heldout<h>/: trials/, <step>-trials.csv (one row per inner fold and seed
of every trial), <step>-selection.csv (every trial's per-subject scores, gap,
bound, p-value and size), chosen-<step>.yaml and final.csv; then final.csv
(every held-out session) and final-hparams.yaml.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import optuna
import torch
import yaml
from scipy.stats import binom

import sb_protocol as protocol
from sb_search import (Logger, choose_epoch, epoch_lr, parse_prior, prepare, read_csv, stale_trials_failed, suggest,
                       train_runs)
from tracking import environment, finish_run, start_run, write_csv, write_environment
from train import load_hparams


# ---- choosing a trial --------------------------------------------------------------------------------

def complexity(kwargs):
    """Parameters and FLOPs (one trial's forward, multiply-adds counted twice) of one EEGNet."""
    from braindecode.models import EEGNet
    from torch.utils.flop_counter import FlopCounterMode

    model = EEGNet(**kwargs).eval()
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        model(torch.zeros(1, kwargs["n_chans"], kwargs["n_times"]))
    return dict(params=sum(p.numel() for p in model.parameters()), flops=int(counter.get_total_flops()))


def sign_flip_p(d, n_draws=100_000, seed=0):
    """One-sided p-value for mean(d) > 0, flipping the signs of d (exact for up to 16 values)."""
    n = len(d)
    if n <= 16:
        signs = 1 - 2 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1)
    else:
        signs = np.random.default_rng(seed).choice([-1, 1], (n_draws, n))
    return float(((signs * d).mean(1) >= d.mean() - 1e-12).mean())


def select(per_subject, sizes=None, width=1.0, spread="se"):
    """The index of the trial to choose, and a table row per trial.

    per_subject (n_trials, n_subjects) are the trials' validation scores, sizes
    their complexity (None to choose the best). A trial is within the bound when
    its mean paired gap to the best is at most ``width`` times the gap's standard
    error (``se``) or standard deviation over subjects (``sd``); the smallest of
    those is chosen, ties going to the higher score.
    """
    per_subject = np.asarray(per_subject, dtype=float)
    mean = per_subject.mean(1)
    best = int(np.argmax(mean))
    d = per_subject[best] - per_subject
    sd = d.std(1, ddof=1) if per_subject.shape[1] > 1 else np.zeros(len(d))
    bound = width * (sd / np.sqrt(per_subject.shape[1]) if spread == "se" else sd)
    gap = d.mean(1)
    within = gap <= bound + 1e-12
    if sizes is None:
        chosen = best
    else:
        chosen = min(np.flatnonzero(within), key=lambda i: (sizes[i], gap[i]))
    table = [dict(mean=float(mean[i]), gap=float(gap[i]), bound=float(bound[i]), within=bool(within[i]),
                  p_best_better=1.0 if i == best else sign_flip_p(d[i]), best=i == best, chosen=i == chosen)
             for i in range(len(mean))]
    return chosen, table


def chance_level(n, n_classes, alpha):
    """Combrisson and Jerbi's (2015) accuracy to exceed for p < alpha with n test trials (binomial)."""
    return float(binom.ppf(1 - alpha, n, 1 / n_classes) / n)


# ---- one trial and the final evaluation --------------------------------------------------------------

def search_trial(hp, seeds, held_out, recordings, cfg, args, label):
    """Every subject's inner folds of held-out session ``held_out``, with every seed; scores per subject."""
    nested = cfg["nested"]
    data = prepare(hp, recordings, cfg, args)
    folds = protocol.nested_folds(data["subjects"], data["sessions"], held_out, nested["inner_folds"])
    runs = [(f"{name}_inner{j}", seed, train, dict(valid=valid))
            for seed in seeds for name, _, _, inner in folds for j, (train, valid) in enumerate(inner)]
    result = choose_epoch(train_runs(data, runs, nested["pack_size"], args, label), cfg["metric"],
                          args.optimizer == "schedule-free", data["hp"])
    by_subject = {}
    for row in result["rows"]:
        by_subject.setdefault(row["fold"].split("_")[0], []).append(row[f"valid_{cfg['metric']}"])
    per_subject = [float(np.mean(by_subject[s])) for s in sorted(by_subject)]
    return dict(result, per_subject=per_subject, data_seconds=data["data_seconds"], **complexity(data["kwargs"]))


def final_evaluation(hp, seeds, held_out, recordings, cfg, args):
    """Train on every training session with ``hp`` and each seed; rows of test scores per subject and seed."""
    data = prepare(dict(hp, augment=True), recordings, cfg, args)
    folds = protocol.nested_folds(data["subjects"], data["sessions"], held_out, cfg["nested"]["inner_folds"])
    runs = [(name, seed, train, dict(test=test)) for seed in seeds for name, train, test, _ in folds]
    result = train_runs(data, runs, cfg["nested"]["pack_size"], args, f"heldout{held_out}/final")
    epoch = data["hp"]["number_of_epochs"]
    scored = {(c["seed"], c["fold"]): c for c in result["curves"] if c["epoch"] == epoch}
    rows = []
    for row in result["rows"]:
        c = scored[(row["seed"], row["fold"])]
        threshold = chance_level(row["n_test"], cfg["n_classes"], cfg["nested"]["alpha"])
        rows.append(dict(row, held_out=held_out, epoch=epoch, **{k: v for k, v in c.items() if k.startswith("test_")},
                         chance_level=threshold, above_chance=c["test_accuracy"] > threshold))
    return rows, result


# ---- the protocol ------------------------------------------------------------------------------------

def search(held_out, recordings, cfg, args, storage, log):
    """Both steps of the search for one held-out session; the chosen hyperparameters."""
    nested, metric = cfg["nested"], cfg["metric"]
    out = args.out / f"heldout{held_out}"
    trials_dir = out / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    sf = args.optimizer == "schedule-free"
    hp = dict(cfg["fixed"])
    for step_index, (step, space) in enumerate(cfg["search"].items()):
        hp["augment"] = step_index > 0  # their step 1 runs without augmentation
        sampler = optuna.samplers.TPESampler(n_startup_trials=cfg["startup_trials"],
                                             n_ei_candidates=cfg["ei_candidates"], seed=cfg["seed"] + held_out)
        study = optuna.create_study(study_name=f"heldout{held_out}-{step}-{args.optimizer}", storage=storage,
                                    sampler=sampler, direction="maximize", load_if_exists=True)
        stale_trials_failed(study)
        complete = lambda: study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        while len(complete()) < cfg["trials"]:
            trial = study.ask()
            values = suggest(trial, space)
            label = f"heldout{held_out}/{step}/trial{trial.number:03d}"
            seeds = [cfg["seed"] + 1000 * (1 + trial.number) + s for s in range(nested["seeds"])]
            result = search_trial({**hp, **values}, seeds, held_out, recordings, cfg, args, label)
            name = trials_dir / f"{step}-trial{trial.number:03d}"
            write_csv(name.with_suffix(".rows.csv"), [dict(step=step, trial=trial.number, **values, **r)
                                                      for r in result["rows"]])
            write_csv(name.with_suffix(".curves.csv"), [dict(step=step, trial=trial.number, **c)
                                                        for c in result["curves"]])
            np.save(name.with_suffix(".losses.npy"), result["losses"].numpy())
            for key in ("per_subject", "epoch", "train_seconds", "params", "flops"):
                trial.set_user_attr(key, result[key])
            trial.set_user_attr("values", values)
            study.tell(trial, float(np.mean(result["per_subject"])))
            print(f"{label}: valid {metric} {np.mean(result['per_subject']):.4f} (best {study.best_value:.4f}) "
                  f"at epoch {result['epoch']}; {result['params']} parameters, {result['flops'] / 1e6:.1f} MFLOPs; "
                  f"{result['train_seconds']:.0f}s training, {result['data_seconds']:.0f}s data, "
                  f"{result['peak_allocated'] / 2**30:.1f} GiB; {values}", flush=True)
            n_train = int(result["rows"][0]["n_train"])
            log.trial(f"heldout{held_out}/{step}", result, values, float("nan"), study.best_value,
                      epoch_lr({**hp, **values}, args.optimizer, n_train))

        trials = complete()
        measure = nested["select"].get(step)
        sizes = None if measure is None else [t.user_attrs[measure] for t in trials]
        chosen, table = select([t.user_attrs["per_subject"] for t in trials], sizes, nested["width"],
                               nested["spread"])
        rows = [dict(trial=t.number, **row, params=t.user_attrs["params"], flops=t.user_attrs["flops"],
                     epoch=t.user_attrs["epoch"], **{f"subject{s}": v for s, v in enumerate(t.user_attrs["per_subject"])},
                     **t.user_attrs["values"]) for t, row in zip(trials, table)]
        write_csv(out / f"{step}-selection.csv", rows)
        for kind in ("rows", "curves"):
            write_csv(out / f"{step}-{'trials' if kind == 'rows' else 'curves'}.csv",
                      [row for t in trials for row in read_csv(trials_dir / f"{step}-trial{t.number:03d}.{kind}.csv")])
        trial, row = trials[chosen], rows[chosen]
        hp.update(trial.user_attrs["values"])
        if sf:
            hp["chosen_epochs"] = trial.user_attrs["epoch"]
        best = rows[int(np.argmax([r["mean"] for r in rows]))]
        (out / f"chosen-{step}.yaml").write_text(yaml.safe_dump(
            dict(trial=trial.number, valid=row["mean"], gap=row["gap"], bound=row["bound"], params=row["params"],
                 flops=row["flops"], epoch=row["epoch"], n_within=sum(r["within"] for r in rows),
                 best=dict(trial=best["trial"], valid=best["mean"], params=best["params"], flops=best["flops"]),
                 values=trial.user_attrs["values"]), sort_keys=False))
        print(f"heldout{held_out}/{step}: chose trial {trial.number}, valid {metric} {row['mean']:.4f} "
              f"({row['params']} parameters, {row['flops'] / 1e6:.1f} MFLOPs), {sum(r['within'] for r in rows)} "
              f"trials within the bound; the best, trial {best['trial']}: {best['mean']:.4f} ({best['params']} "
              f"parameters, {best['flops'] / 1e6:.1f} MFLOPs)", flush=True)
        if log.run is not None:
            log.run.summary.update({f"heldout{held_out}/{step}/chosen_{k}": row[k]
                                    for k in ("trial", "mean", "gap", "bound", "params", "flops")})
    return hp


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hparams")
    parser.add_argument("--out", type=Path, required=True, help="directory for the studies and results")
    parser.add_argument("--optimizer", choices=["adam", "schedule-free"], default="adam")
    parser.add_argument("--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                        help="override the hparams file, e.g. trials=10 nested.seeds=3")
    parser.add_argument("--set-search", nargs="+", action="extend", default=[], metavar="STEP.NAME=PRIOR",
                        help='override or add a prior, e.g. step1.lr="choices([0.001])"')
    parser.add_argument("--subjects", type=int, nargs="+", help="a subset of subjects (smoke tests)")
    parser.add_argument("--device", type=torch.device, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true", help="train through torch.compile (recompiles per trial)")
    parser.add_argument("--wandb", metavar="PROJECT", help="log to this W&B project (set WANDB_MODE=offline on clusters)")
    args = parser.parse_args()
    cfg = load_hparams(args.hparams, args.set)
    for override in args.set_search:
        key, prior = override.split("=", 1)
        step, name = key.split(".", 1)
        parse_prior(prior)  # fail early
        cfg["search"][step][name] = prior
    if args.optimizer == "schedule-free":
        for space in cfg["search"].values():
            for name in cfg["schedule_free"]["not_searched"]:
                space.pop(name, None)
    torch.backends.cudnn.benchmark = True
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    recordings = protocol.Recordings(cfg["dataset"], cfg["n_classes"], cfg["sample_rate"], subjects=args.subjects,
                                     dataset_kwargs=cfg.get("dataset_kwargs"))
    env = dict(environment(args.device), load_seconds=time.perf_counter() - t0, optimizer=args.optimizer)
    paths = [write_environment(args.out / "results", env)]
    run = start_run(args.wandb, "sb-nested", args.hparams, cfg, args, env)
    log = Logger(run, cfg["metric"])
    _, _, subjects, sessions = recordings.trials(cfg["fixed"]["fmin"], cfg["fixed"]["fmax"])
    n_sessions = len(np.unique(sessions[subjects == subjects[0]]))
    held_outs = cfg["nested"]["held_out"]
    held_outs = list(range(n_sessions)) if held_outs is None else list(held_outs)
    print(f"{cfg['dataset']}: recordings of {len(recordings.subjects)} subjects loaded in {env['load_seconds']:.0f}s; "
          f"optimizer {args.optimizer}; a search per held-out session {held_outs}", flush=True)

    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(args.out / "study.journal")))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    metric = cfg["metric"]
    seeds = [cfg["seed"] + i for i in range(cfg["final_seeds"])]
    final_rows, chosen_hp = [], {}
    for held_out in held_outs:
        hp = search(held_out, recordings, cfg, args, storage, log)
        chosen_hp[f"heldout{held_out}"] = hp
        path = args.out / f"heldout{held_out}" / "final.csv"
        if path.exists():  # resumed after this session's final evaluation
            final_rows += read_csv(path)
            continue
        rows, result = final_evaluation(hp, seeds, held_out, recordings, cfg, args)
        write_csv(path, rows)
        final_rows += rows
        paths.append(path)
        mean = np.mean([r[f"test_{metric}"] for r in rows])
        print(f"heldout{held_out}/final: test {metric} {mean:.4f} over {len(seeds)} seeds; "
              f"{np.mean([r['above_chance'] for r in rows]):.0%} of folds above chance; "
              f"{result['train_seconds']:.0f}s training", flush=True)

    write_csv(args.out / "final.csv", final_rows)
    (args.out / "final-hparams.yaml").write_text(yaml.safe_dump(chosen_hp, sort_keys=False))
    paths += [args.out / "final.csv", args.out / "final-hparams.yaml"]
    paths += sorted(args.out.glob("heldout*/*.csv")) + sorted(args.out.glob("heldout*/chosen-*.yaml"))
    per_seed = [np.mean([float(r[f"test_{metric}"]) for r in final_rows if int(r["seed"]) == s]) for s in seeds]
    above = np.mean([str(r["above_chance"]) == "True" for r in final_rows])
    summary = {f"final/test_{metric}": float(np.mean(per_seed)), f"final/test_{metric}_std": float(np.std(per_seed)),
               "final/above_chance": float(above)}
    print(f"final: test {metric} {np.mean(per_seed):.4f} ± {np.std(per_seed):.4f} over {len(seeds)} seeds "
          f"(per seed: {', '.join(f'{v:.4f}' for v in per_seed)}); {above:.0%} of folds above chance", flush=True)
    if run is not None:
        for i, value in enumerate(per_seed):
            run.log({"final_seed": i, f"final/test_{metric}": value})
    finish_run(run, final_rows, paths, summary)


if __name__ == "__main__":
    main()
