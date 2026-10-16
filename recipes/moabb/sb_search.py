"""SpeechBrain-MOABB's hyperparameter search and final evaluation, with Optuna's TPE and packed training.

    python sb_search.py hparams/sb_moabb/bnci2014001_eegnet.yaml --out results/sb/bnci2014001-adam
    python sb_search.py hparams/sb_moabb/bnci2014001_eegnet.yaml --optimizer schedule-free --out results/sb/bnci2014001-sf
    python sb_search.py hparams/sb_moabb/bnci2014001_eegnet.yaml --out /tmp/smoke --set trials=2 final_seeds=1 \\
        fixed.number_of_epochs=3 --set-search step1.number_of_epochs="uniform(2, 4, discrete=True)"   # smoke test

    # the improved protocol (hparams/sb_moabb/improved/, see PROTOCOL_IDEAS.md), and a replicate search
    python sb_search.py hparams/sb_moabb/improved/bnci2014001_eegnet.yaml --optimizer schedule-free --out results/sb/imp
    python sb_search.py hparams/sb_moabb/improved/bnci2014001_eegnet.yaml --optimizer schedule-free --out results/sb/imp-r1 \\
        --tpe-seed 1 --trial-seed 1
    # rerun a finished study's 10 best step-1 trials with 5 new seeds each
    python sb_search.py hparams/sb_moabb/bnci2014001_eegnet.yaml --out results/sb/bnci2014001-adam --rerun step1 \\
        --rerun-top 10 --rerun-seeds 5

As their run_hparam_optimization.sh: each step of `search` runs `trials` trials
of TPE (`startup_trials` random ones first), maximizing the validation `metric`
averaged over every fold, with seed `seed`; the best trial's values are then
fixed for the next step. The final evaluation trains every fold with
`final_seeds` seeds and reports the test sessions' scores. One trial trains all
its folds as packs of `pack_size` (see sb_protocol.py for the protocol itself).

Their Orion TPE is replaced by Optuna's (TPESampler with their startup and
candidate counts). Test scores are recorded for every trial but never used to
choose anything.

Options that change the protocol (all off by default, so SpeechBrain-MOABB's runs
as before; the hparams files under hparams/sb_moabb/improved/ turn them on):

- `steps` instead of `search`: the steps in order, each with its priors, its
  `augment` (none, speechbrain or eeg), its own `fixed` values and optionally its
  own `trials` and `selection`, so preprocessing can be a step of its own. With schedule-free, a
  step trains to the epoch an earlier step chose, or with `epochs: grid` chooses
  it again.
- `protocol`: `inner_folds` validation folds within the training sessions
  (`group_by_run`: of whole runs), `seeds_per_trial` seeds (`trial_seeds:
  per_trial` for seeds that differ between trials), `selection: one_se` (the
  smallest model within one standard error of the best), `multi_objective`
  (TPE on validation and parameter count), `nested_by_test_session` (one
  search per test session, on the folds that test on it, so no session helps
  choose the configuration it's tested with; --test-session runs one of them),
  `final_train: all` and `final_augment`.
- Kernel and pool sizes in seconds (`cnn_temporal_kernel_s`, ...; sb_protocol.eegnet_kwargs),
  `sample_rate` and `twindow` (the window's length from `tmin`) as hyperparameters.

A trial's unit of work is a cell, one (outer fold, inner fold, seed): `make_cells`
lists a trial's cells and `train_cells` trains any list of them in packs, so an
allocator can ask for some cells of a trial at a time.

Everything goes to --out: the Optuna studies (study.journal; rerun the same
command to resume after a time limit, and the trial that was running is redone),
trials/ (per trial: one row per cell, the scores at every evaluated epoch, the
training loss per epoch and cell, and .cells.npz, every cell's validation and test
scores at every evaluated epoch), <step>-trials.csv, <step>-curves.csv and
<step>-summary.csv (one row per trial: objective, standard error, per-seed and
per-inner-fold scores, size), best-<step>.yaml, final.csv, final-curves.csv,
final-hparams.yaml and results-env.json. With --wandb, every trial is logged
as it finishes: its validation and test scores, timings, peak memory, size and
values, and its training loss per epoch (mean and per cell).
"""

from __future__ import annotations

import argparse
import copy
import math
import time
import traceback
from pathlib import Path

import numpy as np
import optuna
import torch
import yaml

import sb_protocol as protocol
from tracking import environment, finish_run, start_run, write_csv, write_environment
from train import load_hparams, peak_memory


# ---- search spaces in Orion's syntax ----------------------------------------------------------------

def uniform(low, high, discrete=False, precision=None):
    return dict(kind="int" if discrete else "float", low=low, high=high, precision=precision)


def loguniform(low, high, precision=None):
    return dict(kind="float", low=low, high=high, log=True, precision=precision)


def choices(options):
    return dict(kind="categorical", choices=list(options))


def parse_prior(text):
    """An Orion prior such as ``uniform(0.1, 5, precision=2)`` or ``choices([1, 2])``."""
    return eval(text, {"__builtins__": {}}, dict(uniform=uniform, loguniform=loguniform, choices=choices))


def suggest(trial, space):
    """{name: value} for an Optuna trial; ``precision`` rounds to significant digits, as Orion's."""
    values = {}
    for name, text in space.items():
        prior = parse_prior(text)
        if prior["kind"] == "int":
            values[name] = trial.suggest_int(name, prior["low"], prior["high"])
        elif prior["kind"] == "categorical":
            values[name] = trial.suggest_categorical(name, prior["choices"])
        else:
            value = trial.suggest_float(name, prior["low"], prior["high"], log=prior.get("log", False))
            values[name] = float(f"{value:.{prior['precision']}g}") if prior["precision"] else value
    return values


def steps_of(cfg):
    """{step: settings}: `steps` as given, or SpeechBrain-MOABB's from `search` (augmentation from step 2 on)."""
    if "steps" in cfg:
        return {name: dict(dict(augment="none", fixed={}), **step) for name, step in cfg["steps"].items()}
    return {name: dict(search=space, augment="none" if i == 0 else "speechbrain", fixed={})
            for i, (name, space) in enumerate(cfg["search"].items())}


PROTOCOL = dict(  # SpeechBrain-MOABB's
    inner_folds=0,  # 0: one validation set, `valid_ratio` of each class, equally spaced in time; "runs": one per run
    group_by_run=False,
    seeds_per_trial=1,
    trial_seeds="fixed",  # every trial uses seeds seed, seed + 1, ...; or "per_trial"
    seed_per_inner_fold=False,  # True: each inner fold trains from its own seed, derived from the trial's
    selection="best",  # or "one_se"
    size="params",  # or "flops": the size one_se and multi_objective minimize
    multi_objective=False,
    nested_by_test_session=False,  # True: one search per test session, on the folds that test on it
    final_train="train",  # the final evaluation trains on the trials it would have trained on; or "all"
    final_augment="speechbrain",
)


def trial_seeds(base, n, number, per_trial):
    """``n`` seeds for trial ``number``: base, base + 1, ...; or, per trial, drawn from (base, number)."""
    if not per_trial:
        return [base + i for i in range(n)]
    return [int(s) % 2**31 for s in np.random.SeedSequence([base, number]).generate_state(n)]


# ---- one trial: every cell, packed -----------------------------------------------------------------

def prepare(hp, recordings, cfg, args):
    """The trial's data on the device, its sample rate and its leave-one-session-out folds."""
    sample_rate = hp.get("sample_rate", cfg["sample_rate"])
    X, y, subjects, sessions = recordings.trials(hp["fmin"], hp["fmax"], sample_rate)
    runs = recordings.runs(hp["fmin"], hp["fmax"], sample_rate)
    interval = recordings.interval()
    tmax = min(hp["tmin"] + hp["twindow"], interval[1]) if "twindow" in hp else hp["tmax"]
    X = protocol.crop(X, sample_rate, interval, hp["tmin"], tmax)
    X = X[:, recordings.channels(hp["n_steps_channel_selection"])]
    folds = protocol.leave_one_session_out(y, subjects, sessions, cfg["valid_ratio"])
    if getattr(args, "test_session", None) is not None:  # a search nested by test session: its folds only
        folds = testing_on(folds, args.test_session)
    X = torch.from_numpy(X).to(args.device)
    return dict(X=X, Xn=protocol.standardize(X), y=torch.from_numpy(y).to(args.device), folds=folds,
                sessions=sessions, runs=runs, y_np=y, sample_rate=sample_rate)


def testing_on(folds, session):
    """The leave-one-session-out folds that test on ``session``; none of them trains or validates on it."""
    return [f for f in folds if f[0].split("_test-", 1)[1] == session]


def make_cells(data, seeds, settings, final=False):
    """Every cell of a trial, (outer fold, inner fold, seed), in the order they're packed.

    With ``inner_folds`` 0, the cells are SpeechBrain-MOABB's (fold, seed) runs; with k,
    each outer fold's training trials (theirs and their validation set's) are split k ways,
    and with ``seed_per_inner_fold`` each inner fold gets its own seed (so k folds and one
    seed per trial average over k seeds, at no extra cost).
    In the final evaluation with ``final_train: all``, a cell trains on all of them and has
    no validation set.
    """
    cells = []
    for seed in seeds:
        for name, train, valid, test in data["folds"]:
            if final and settings["final_train"] == "all":
                splits = [(np.union1d(train, valid), None)]
            elif settings["inner_folds"] and not final:
                splits = protocol.inner_folds(np.union1d(train, valid), data["y_np"], data["sessions"], data["runs"],
                                              settings["inner_folds"], settings["group_by_run"])
            else:
                splits = [(train, valid)]
            for j, (tr, va) in enumerate(splits):
                cell_seed = seed
                if settings.get("seed_per_inner_fold") and j > 0:  # inner fold j's own seed, from (seed, j)
                    cell_seed = int(np.random.SeedSequence([seed, j]).generate_state(1)[0] % 2**31)
                cells.append(dict(fold=name, inner=j, seed=cell_seed, train=torch.from_numpy(tr),
                                  valid=None if va is None else torch.from_numpy(va), test=torch.from_numpy(test)))
    return cells


def pack_cells(cells, pack_size):
    """Packs: cells with equal training- and test-set sizes, ``pack_size`` at most, then split by validation size."""
    by_size = {}
    for cell in cells:
        by_size.setdefault((len(cell["train"]), len(cell["test"])), []).append(cell)
    for group in by_size.values():
        for start in range(0, len(group), pack_size):
            pack = group[start : start + pack_size]
            n_valid = lambda c: -1 if c["valid"] is None else len(c["valid"])
            for size in dict.fromkeys(n_valid(c) for c in pack):
                yield [c for c in pack if n_valid(c) == size]


def augmentation(hp, mode, sample_rate):
    """fit()'s ``aug`` (SpeechBrain-MOABB's) and ``eeg_aug`` arguments for the augmentation ``mode``."""
    if mode in (True, "speechbrain"):
        shift = 1e-2 * hp["shift_delta_"]
        return dict(max_num_segments=hp["max_num_segments"], amp_delta=hp["amp_delta"],
                    min_shift=math.floor(-sample_rate * shift), max_shift=math.floor(sample_rate * shift),
                    snr_low=hp["snr_white_low"], snr_high=hp["snr_white_low"] + hp["snr_white_delta"]), None
    if mode == "eeg":  # every hp named aug_*: sb_protocol.eeg_augment's arguments and mixup's
        return None, {k[len("aug_"):]: v for k, v in hp.items() if k.startswith("aug_")}
    if mode in (False, None, "none"):
        return None, None
    raise ValueError(f"unknown augmentation {mode!r}")


def train_cells(cells, hp, data, cfg, args, label, eval_epochs):
    """Train ``cells`` (any of a trial's) in packs; their scores at every epoch in ``eval_epochs``.

    Returns curve rows (one per cell and evaluated epoch), rows (one per cell), the
    cells' training losses (epochs, n_cells) and timings. A pack that runs out of
    memory is split in two, and the packs after it are no larger than its halves
    (``oom_seconds`` is the time lost).
    """
    device, X, Xn, y = args.device, data["X"], data["Xn"], data["y"]
    aug, eeg_aug = augmentation(hp, hp.get("augment", "none"), data["sample_rate"])
    kwargs = protocol.eegnet_kwargs(hp, X.shape[1], cfg["n_classes"], X.shape[2], data["sample_rate"])
    rows, curves, losses, timing = [], [], [], dict(train_seconds=0.0, oom_seconds=0.0, peak_allocated=0, n_packs=0)

    limit = [len(cells)]  # the largest pack that fitted in memory so far

    def train_pack(pack):
        if len(pack) > limit[0]:
            for start in range(0, len(pack), limit[0]):
                train_pack(pack[start : start + limit[0]])
            return
        pack_seeds = [c["seed"] for c in pack]
        train_idx, test_idx = (torch.stack([c[key] for c in pack]).to(device) for key in ("train", "test"))
        valid_idx = None if pack[0]["valid"] is None else torch.stack([c["valid"] for c in pack]).to(device)
        evaluated = {}

        def on_eval(epoch, model):
            valid = protocol.scores(model, Xn, y, valid_idx) if valid_idx is not None else [{}] * len(pack)
            evaluated[epoch] = (valid, protocol.scores(model, Xn, y, test_idx))

        if args.compile:
            torch._dynamo.reset()
        peak_memory(device)
        t = time.perf_counter()
        try:
            model = protocol.build(kwargs, pack_seeds, device)
            _, member_losses = protocol.fit(model, X, Xn, y, train_idx, pack_seeds, hp, optimizer=args.optimizer,
                                            aug=aug, eval_epochs=eval_epochs, on_eval=on_eval, compile=args.compile,
                                            eeg_aug=eeg_aug, sample_rate=data["sample_rate"])
        except torch.OutOfMemoryError:
            if len(pack) == 1:
                raise
            out_of_memory = True
        else:
            out_of_memory = False
        if out_of_memory:  # outside the except block, whose traceback holds the failed pack's tensors
            model = None
            timing["oom_seconds"] += time.perf_counter() - t
            limit[0] = (len(pack) + 1) // 2  # and the packs after this one start at that size
            print(f"  out of memory with {len(pack)} runs per pack; splitting it", flush=True)
            train_pack(pack[: limit[0]])
            train_pack(pack[limit[0] :])
            return
        timing["train_seconds"] += time.perf_counter() - t
        timing["peak_allocated"] = max(timing["peak_allocated"], peak_memory(device)[0] or 0)
        timing["n_packs"] += 1
        losses.append(member_losses)
        for k, cell in enumerate(pack):
            for epoch, (valid, test) in sorted(evaluated.items()):
                curves.append(dict(label=label, seed=cell["seed"], fold=cell["fold"], inner=cell["inner"], epoch=epoch,
                                   **{f"valid_{m}": v for m, v in valid[k].items()},
                                   **{f"test_{m}": v for m, v in test[k].items()}))
            rows.append(dict(label=label, seed=cell["seed"], fold=cell["fold"], inner=cell["inner"],
                             n_train=train_idx.shape[1], n_valid=0 if valid_idx is None else valid_idx.shape[1],
                             n_test=test_idx.shape[1]))

    for pack in pack_cells(cells, cfg["pack_size"]):
        train_pack(pack)
    return curves, rows, torch.cat(losses, 1), timing


class InvalidConfiguration(Exception):
    """Hyperparameters that don't make a model, e.g. pools longer than the trial."""


def run_trial(hp, seeds, recordings, cfg, args, label, settings=PROTOCOL, final=False):
    """Train every cell of a trial with hyperparameters ``hp``; scores at each evaluated epoch.

    Returns rows (one per cell, final scores), curve rows (every evaluated epoch),
    the cells' training losses (epochs, n_cells), the objective (the mean validation
    score over cells), its standard error over the (seed, inner fold) replicates, the
    per-seed and per-inner-fold means, and timings.
    """
    hp = dict(hp, batch_size=2 ** hp["batch_size_exponent"])
    sf = args.optimizer == "schedule-free"
    if sf:
        epochs = cfg["schedule_free"]["epochs"]
        eval_epochs = list(range(epochs["first"], epochs["last"] + 1, epochs["every"]))
        if "chosen_epochs" in hp:  # the final evaluation (or a step after the first): train to the chosen epoch
            hp["number_of_epochs"], eval_epochs = hp["chosen_epochs"], [hp["chosen_epochs"]]
        else:
            hp["number_of_epochs"] = epochs["last"]
    else:
        eval_epochs = [hp["number_of_epochs"]]

    t0 = time.perf_counter()
    data = prepare(hp, recordings, cfg, args)
    try:
        size = protocol.model_size(protocol.eegnet_kwargs(hp, data["X"].shape[1], cfg["n_classes"], data["X"].shape[2],
                                                          data["sample_rate"]))
    except (ValueError, RuntimeError) as error:
        raise InvalidConfiguration(repr(error)) from error
    cells = make_cells(data, seeds, settings, final)
    data_seconds = time.perf_counter() - t0
    curves, rows, losses, timing = train_cells(cells, hp, data, cfg, args, label, eval_epochs)

    # the epoch to report: the last (Adam), or the best mean validation score (schedule-free)
    metric = cfg["metric"]
    by_epoch = {}
    for c in curves:
        if f"valid_{metric}" in c:
            by_epoch.setdefault(c["epoch"], []).append(c[f"valid_{metric}"])
    mean_valid = {epoch: float(np.mean(v)) for epoch, v in by_epoch.items()}
    chosen = max(mean_valid, key=lambda e: (mean_valid[e], -e)) if sf and mean_valid else hp["number_of_epochs"]
    final_curves = {(c["seed"], c["fold"], c["inner"]): c for c in curves if c["epoch"] == chosen}
    for row in rows:
        row.update({k: v for k, v in final_curves[(row["seed"], row["fold"], row["inner"])].items()
                    if k.startswith(("valid_", "test_"))}, epoch=chosen)
    summary = summarize(rows, metric)
    return dict(rows=rows, curves=curves, losses=losses, objective=mean_valid.get(chosen, float("nan")), epoch=chosen,
                mean_valid=mean_valid, data_seconds=data_seconds, n_cells=len(cells), params=size[0], flops=size[1],
                **summary, **timing)


def summarize(rows, metric):
    """Per-seed and per-inner-fold mean validation scores and the objective's standard error.

    The standard error is over the (seed, inner fold) replicates, each the mean over
    the outer folds (subjects and test sessions): it measures the noise of seeds and
    validation splits, as the one-standard-error rule needs, not the spread over
    subjects. Replicates share subjects and training trials, so it's optimistic.
    NaN with a single replicate.
    """
    key = f"valid_{metric}"
    if not rows or key not in rows[0]:
        return dict(se=float("nan"), per_seed={}, per_inner={})
    replicates, per_seed, per_inner = {}, {}, {}
    for r in rows:
        replicates.setdefault((r["seed"], r["inner"]), []).append(r[key])
        per_seed.setdefault(r["seed"], []).append(r[key])
        per_inner.setdefault(r["inner"], []).append(r[key])
    means = [np.mean(v) for v in replicates.values()]
    se = float(np.std(means, ddof=1) / math.sqrt(len(means))) if len(means) > 1 else float("nan")
    return dict(se=se, per_seed={s: float(np.mean(v)) for s, v in per_seed.items()},
                per_inner={j: float(np.mean(v)) for j, v in per_inner.items()})


def save_cells(path, curves, metrics):
    """Every cell's validation and test scores at every evaluated epoch, as arrays (cells, epochs, metrics)."""
    cells = list(dict.fromkeys((c["fold"], c["inner"], c["seed"]) for c in curves))
    epochs = sorted({c["epoch"] for c in curves})
    position = {cell: i for i, cell in enumerate(cells)}
    valid = np.full((len(cells), len(epochs), len(metrics)), np.nan, dtype=np.float32)
    test = valid.copy()
    for c in curves:
        i, e = position[(c["fold"], c["inner"], c["seed"])], epochs.index(c["epoch"])
        valid[i, e] = [c.get(f"valid_{m}", np.nan) for m in metrics]
        test[i, e] = [c[f"test_{m}"] for m in metrics]
    np.savez_compressed(path, fold=np.array([c[0] for c in cells]), inner=np.array([c[1] for c in cells]),
                        seed=np.array([c[2] for c in cells]), epoch=np.array(epochs), metric=np.array(metrics),
                        valid=valid, test=test)


# ---- the protocol ----------------------------------------------------------------------------------

def stale_trials_failed(study):
    """Mark trials left running by an interrupted job as failed, so they're redone."""
    for trial in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)


def read_csv(path):
    import csv

    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def select(trials, rule, size):
    """The trial to keep from a step's finished trials, and why.

    ``best``: the highest objective. ``one_se``: the smallest (``size``: params or
    flops) whose objective is within one standard error of the best's (Breiman et al.,
    1984; Hastie et al., 2009, section 7.10), the best when its standard error is unknown.
    """
    best = max(trials, key=lambda t: t.values[0])
    se = best.user_attrs.get("se", float("nan"))
    if rule == "best" or not math.isfinite(se):
        return best, "best"
    if rule != "one_se":
        raise ValueError(f"unknown selection {rule!r}")
    within = [t for t in trials if t.values[0] >= best.values[0] - se]
    chosen = min(within, key=lambda t: (t.user_attrs[size], -t.values[0]))
    return chosen, f"one_se: smallest {size} of {len(within)} trials within {se:.4f} of trial {best.number}"


class Logger:
    """W&B logging as the search runs: per trial, per epoch of every trial, per final seed."""

    def __init__(self, run, metric):
        self.run, self.metric, self.trials, self.epochs = run, metric, 0, 0
        if run is not None:
            run.define_metric("trial")
            run.define_metric("epoch")
            run.define_metric("train_loss*", step_metric="epoch")
            run.define_metric("lr", step_metric="epoch")
            run.define_metric("final_seed")
            run.define_metric("final/*", step_metric="final_seed")

    def trial(self, step, result, values, test, best, lr, size=None):
        if self.run is None:
            return
        self.run.define_metric(f"{step}/*", step_metric="trial")
        for epoch, loss in enumerate(result["losses"].tolist()):  # training loss per member and mean
            self.epochs += 1
            self.run.log({"epoch": self.epochs, "train_loss": float(np.mean(loss)), "lr": lr(epoch),
                          **{f"train_loss/member{k}": v for k, v in enumerate(loss)}})
        self.trials += 1
        m = self.metric
        extra = {}
        if math.isfinite(result.get("se", float("nan"))):
            extra[f"{step}/valid_{m}_se"] = result["se"]
        if len(result.get("per_seed", {})) > 1:
            extra.update({f"{step}/valid_{m}_seed/{i}": v for i, v in enumerate(result["per_seed"].values())})
        if len(result.get("per_inner", {})) > 1:
            extra.update({f"{step}/valid_{m}_inner/{j}": v for j, v in result["per_inner"].items()})
        if size is not None:
            extra.update({f"{step}/params": size[0], f"{step}/mflops": size[1] / 1e6})
        self.run.log({"trial": self.trials, f"{step}/valid_{m}": result["objective"], f"{step}/best_valid_{m}": best,
                      f"{step}/test_{m}": test, f"{step}/epoch": result["epoch"],
                      f"{step}/train_seconds": result["train_seconds"], f"{step}/data_seconds": result["data_seconds"],
                      f"{step}/seconds_per_epoch": result["train_seconds"] / len(result["losses"]),
                      f"{step}/peak_allocated": result["peak_allocated"], f"{step}/n_cells": result["n_cells"],
                      f"{step}/n_packs": result["n_packs"],
                      **{f"{step}/mean_valid_{m}_at/{e}": v for e, v in result["mean_valid"].items()},
                      **{f"{step}/param/{k}": v for k, v in values.items()}, **extra})


def epoch_lr(hp, optimizer, n_train):
    """The learning rate at the start of each epoch, for logging."""
    batch_size = 2 ** hp["batch_size_exponent"]
    if optimizer == "schedule-free":
        return lambda epoch: hp["lr"]
    schedule = protocol.cyclic_lr(1e-8, hp["lr"], round(5 * n_train / batch_size))
    return lambda epoch: schedule(epoch * math.ceil(n_train / batch_size), True)


def context_before(cfg, steps, step, out, optimizer):
    """The hyperparameters a step's trials started from, for studies run before trials recorded them."""
    hp = dict(cfg["fixed"])
    for name, settings in steps.items():
        if name == step:
            if settings.get("epochs") == "grid":
                hp.pop("chosen_epochs", None)
            return dict(hp, **settings["fixed"], augment=settings["augment"])
        best = yaml.safe_load((out / f"best-{name}.yaml").read_text())
        hp.update(best["values"])
        if optimizer == "schedule-free":
            hp["chosen_epochs"] = best["epoch"]
    raise KeyError(step)


def rerun(args, cfg, steps, settings, recordings, storage, paths, run):
    """Rerun the ``--rerun-top`` best trials of step ``--rerun`` with ``--rerun-seeds`` new seeds each."""
    step, metric = args.rerun, cfg["metric"]
    suffix = "-mo" if settings["multi_objective"] else ""
    study = optuna.load_study(study_name=f"{step}-{args.optimizer}{suffix}", storage=storage)
    done = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    top = sorted(done, key=lambda t: -t.values[0])[: args.rerun_top]
    base = args.trial_seed if args.trial_seed is not None else cfg["seed"]
    rows, summary = [], []
    for rank, trial in enumerate(top):
        hp = trial.user_attrs.get("hp") or {**context_before(cfg, steps, step, args.out, args.optimizer),
                                             **trial.user_attrs["values"]}
        seeds = trial_seeds(base + 1_000_003, args.rerun_seeds, trial.number, True)  # not any search trial's
        result = run_trial(hp, seeds, recordings, cfg, args, f"rerun-{step}/trial{trial.number:03d}", settings)
        rows += [dict(step=step, trial=trial.number, search_valid=trial.values[0], **r) for r in result["rows"]]
        test = float(np.mean([r[f"test_{metric}"] for r in result["rows"]]))
        summary.append(dict(step=step, trial=trial.number, rank=rank, search_valid=trial.values[0],
                            valid=result["objective"], se=result["se"], test=test, seeds=len(seeds),
                            **{f"valid_seed{i}": v for i, v in enumerate(result["per_seed"].values())}))
        print(f"rerun {step} trial {trial.number} (rank {rank}): search valid {trial.values[0]:.4f}, "
              f"rerun valid {result['objective']:.4f} ± {result['se']:.4f} over {len(seeds)} seeds, test {test:.4f}; "
              f"{result['train_seconds']:.0f}s training", flush=True)
        if run is not None:
            run.log({"rerun_rank": rank, f"rerun/{step}/search_valid": trial.values[0],
                     f"rerun/{step}/valid": result["objective"], f"rerun/{step}/valid_se": result["se"],
                     f"rerun/{step}/test": test, f"rerun/{step}/train_seconds": result["train_seconds"]})
        write_csv(args.out / f"rerun-{step}.csv", rows)
        write_csv(args.out / f"rerun-{step}-summary.csv", summary)
    paths += [args.out / f"rerun-{step}.csv", args.out / f"rerun-{step}-summary.csv"]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hparams")
    parser.add_argument("--out", type=Path, required=True, help="directory for the studies and results")
    parser.add_argument("--optimizer", choices=["adam", "schedule-free"], default="adam")
    parser.add_argument("--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                        help="override the hparams file, e.g. trials=10 fixed.number_of_epochs=5")
    parser.add_argument("--set-search", nargs="+", action="extend", default=[], metavar="STEP.NAME=PRIOR",
                        help='override or add a prior, e.g. step1.lr="choices([0.001])"')
    parser.add_argument("--no-search", action="store_true", help="only the final evaluation, with `fixed`")
    parser.add_argument("--subjects", type=int, nargs="+", help="a subset of subjects (smoke tests)")
    parser.add_argument("--device", type=torch.device, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true", help="train through torch.compile (recompiles per trial)")
    parser.add_argument("--wandb", metavar="PROJECT", help="log to this W&B project (set WANDB_MODE=offline on clusters)")
    parser.add_argument("--tpe-seed", type=int, help="TPE's seed (default: `seed`), for replicate searches")
    parser.add_argument("--trial-seed", type=int,
                        help="the base of the trials' and final evaluation's seeds (default: `seed`)")
    parser.add_argument("--rerun", metavar="STEP", help="rerun this step's best trials with new seeds, then stop")
    parser.add_argument("--rerun-top", type=int, default=10, help="with --rerun: how many of the best trials")
    parser.add_argument("--rerun-seeds", type=int, default=5, help="with --rerun: seeds per trial")
    parser.add_argument("--test-session",
                        help="only the folds that test on this session: one part of a search nested by test session")
    args = parser.parse_args()
    cfg = load_hparams(args.hparams, args.set)
    steps = steps_of(cfg)
    settings = dict(PROTOCOL, **cfg.get("protocol", {}))
    for override in args.set_search:
        key, prior = override.split("=", 1)
        step, name = key.split(".", 1)
        parse_prior(prior)  # fail early
        steps[step]["search"][name] = prior
    if args.optimizer == "schedule-free":
        for step in steps.values():
            for name in cfg["schedule_free"]["not_searched"]:
                step["search"].pop(name, None)
    torch.backends.cudnn.benchmark = True
    args.out.mkdir(parents=True, exist_ok=True)
    tpe_seed = cfg["seed"] if args.tpe_seed is None else args.tpe_seed
    seed_base = cfg["seed"] if args.trial_seed is None else args.trial_seed

    t0 = time.perf_counter()
    recordings = protocol.Recordings(cfg["dataset"], cfg["n_classes"], cfg["sample_rate"], subjects=args.subjects)
    env = dict(environment(args.device), load_seconds=time.perf_counter() - t0, optimizer=args.optimizer,
               tpe_seed=tpe_seed, trial_seed=seed_base)
    paths = [write_environment(args.out / "results", env)]
    run = start_run(args.wandb, "sb-protocol", args.hparams, cfg, args, env)
    log = Logger(run, cfg["metric"])
    metric = cfg["metric"]
    print(f"{cfg['dataset']}: recordings of {len(recordings.subjects)} subjects loaded in {env['load_seconds']:.0f}s; "
          f"optimizer {args.optimizer}", flush=True)

    sessions = [args.test_session]
    if args.test_session is None and settings["nested_by_test_session"]:
        sessions = sorted({session for by_session in recordings.raws.values() for session in by_session})
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rows, summary = [], {}
    for session in sessions:  # one search per held-out test session, or one over every fold
        session_args = copy.copy(args)
        session_args.test_session = session
        if session is not None and args.test_session is None:
            session_args.out = args.out / f"test-{session}"
        prefix = "" if session is None else f"test-{session}/"
        if session is not None:
            print(f"selection nested by test session: {session}, in {session_args.out}", flush=True)
        session_rows, session_summary = search(session_args, cfg, steps, settings, recordings, log, tpe_seed,
                                               seed_base, paths, prefix)
        rows += session_rows
        summary.update(session_summary)
    if len(sessions) > 1 and not args.rerun:  # every fold, each tested with a configuration chosen without its session
        per_seed = [np.mean([r[f"test_{metric}"] for r in rows if r["seed"] == s])
                    for s in dict.fromkeys(r["seed"] for r in rows)]
        summary.update({f"final/test_{metric}": float(np.mean(per_seed)),
                        f"final/test_{metric}_std": float(np.std(per_seed))})
        print(f"final, every test session: test {metric} {np.mean(per_seed):.4f} ± {np.std(per_seed):.4f} "
              f"over {len(per_seed)} seeds", flush=True)
        write_csv(args.out / "final.csv", rows)
        paths.append(args.out / "final.csv")
    finish_run(run, rows, paths, summary)


def search(args, cfg, steps, settings, recordings, log, tpe_seed, seed_base, paths, prefix=""):
    """The search's steps and the final evaluation, in ``args.out``; returns the final rows and summary.

    With ``args.test_session``, only the folds that test on that session: their training
    and validation data never include it, so no trial tested on it helped choose the
    configuration (selection nested by test session).
    """
    metric, sf, run = cfg["metric"], args.optimizer == "schedule-free", log.run
    trials_dir = args.out / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(args.out / "study.journal")))
    if args.rerun:
        return rerun(args, cfg, steps, settings, recordings, storage, paths, run), {}
    multi = settings["multi_objective"]
    hp = dict(cfg["fixed"])
    for step, step_settings in ([] if args.no_search else steps.items()):
        sampler = optuna.samplers.TPESampler(n_startup_trials=step_settings.get("startup_trials", cfg["startup_trials"]),
                                             n_ei_candidates=cfg["ei_candidates"], seed=tpe_seed)
        study = optuna.create_study(study_name=f"{step}-{args.optimizer}" + ("-mo" if multi else ""), storage=storage,
                                    sampler=sampler, directions=["maximize", "minimize"] if multi else ["maximize"],
                                    load_if_exists=True)
        stale_trials_failed(study)
        complete = lambda: study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        failed = 0
        while len(complete()) < step_settings.get("trials", cfg["trials"]):
            trial = study.ask()
            values = suggest(trial, step_settings["search"])
            trial_hp = {**hp, **step_settings["fixed"], **values, "augment": step_settings["augment"]}
            if step_settings.get("epochs") == "grid":  # schedule-free: choose the epoch again
                trial_hp.pop("chosen_epochs", None)
            label = f"{step}/trial{trial.number:03d}"
            seeds = trial_seeds(seed_base, settings["seeds_per_trial"], trial.number,
                                settings["trial_seeds"] == "per_trial")
            try:
                result = run_trial(trial_hp, seeds, recordings, cfg, args, label, settings)
            except InvalidConfiguration as error:  # the trial fails, and TPE moves on
                traceback.print_exc()
                trial.set_user_attr("error", str(error))
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                failed += 1
                if failed > step_settings.get("trials", cfg["trials"]):
                    raise RuntimeError(f"{step}: {failed} configurations failed to build") from error
                continue
            size = result["params"], result["flops"]
            test = float(np.mean([r[f"test_{metric}"] for r in result["rows"]]))
            name = trials_dir / f"{step}-trial{trial.number:03d}"
            write_csv(name.with_suffix(".rows.csv"), [dict(step=step, trial=trial.number, **values, **r)
                                                      for r in result["rows"]])
            write_csv(name.with_suffix(".curves.csv"), [dict(step=step, trial=trial.number, **c)
                                                        for c in result["curves"]])
            np.save(name.with_suffix(".losses.npy"), result["losses"].numpy())
            save_cells(name.with_suffix(".cells.npz"), result["curves"], list(protocol.METRICS))
            for key, value in dict(values=values, epoch=result["epoch"], test=test,
                                   train_seconds=result["train_seconds"], se=result["se"],
                                   per_seed={str(k): v for k, v in result["per_seed"].items()},
                                   per_inner={str(k): v for k, v in result["per_inner"].items()},
                                   seeds=seeds, params=size[0], flops=size[1], n_cells=result["n_cells"],
                                   peak_allocated=result["peak_allocated"], hp=trial_hp).items():
                trial.set_user_attr(key, value)
            study.tell(trial, [result["objective"], size[0 if settings["size"] == "params" else 1]] if multi
                       else result["objective"])
            best = max(t.values[0] for t in complete())
            se = f" ± {result['se']:.4f}" if math.isfinite(result["se"]) else ""
            print(f"{label}: valid {metric} {result['objective']:.4f}{se} (best {best:.4f}) at epoch {result['epoch']}, "
                  f"test {test:.4f}; {result['n_cells']} cells in {result['n_packs']} packs, "
                  f"{result['train_seconds']:.0f}s training, {result['data_seconds']:.0f}s data, "
                  f"{result['peak_allocated'] / 2**30:.1f} GiB; {size[0]} params, {size[1] / 1e6:.1f} MFLOPs; {values}",
                  flush=True)
            n_train = int(result["rows"][0]["n_train"])
            log.trial(prefix + step, result, values, test, best, epoch_lr(trial_hp, args.optimizer, n_train), size)
        chosen, why = select(complete(), step_settings.get("selection", settings["selection"]), settings["size"])
        hp.update(chosen.user_attrs["values"])
        if sf:
            hp["chosen_epochs"] = chosen.user_attrs["epoch"]
        (args.out / f"best-{step}.yaml").write_text(yaml.safe_dump(
            dict(trial=chosen.number, valid=chosen.values[0], epoch=chosen.user_attrs["epoch"],
                 values=chosen.user_attrs["values"], selection=why), sort_keys=False))
        print(f"{step}: chose trial {chosen.number} ({why}), valid {metric} {chosen.values[0]:.4f}: "
              f"{chosen.user_attrs['values']}", flush=True)
        # every finished trial of this step (earlier jobs' too), in one file each
        for kind in ("rows", "curves"):
            rows = [row for t in complete()
                    for row in read_csv(trials_dir / f"{step}-trial{t.number:03d}.{kind}.csv")]
            write_csv(args.out / f"{step}-{'trials' if kind == 'rows' else 'curves'}.csv", rows)
            paths.append(args.out / f"{step}-{'trials' if kind == 'rows' else 'curves'}.csv")
        write_csv(args.out / f"{step}-summary.csv", [trial_summary(t, metric) for t in complete()])
        paths.append(args.out / f"{step}-summary.csv")

    # Final evaluation: every fold with final_seeds seeds, on the test sessions, with augmentation
    hp["augment"] = settings["final_augment"]
    seeds = [seed_base + i for i in range(cfg["final_seeds"])]
    result = run_trial(hp, seeds, recordings, cfg, args, "final", settings, final=True)
    final_rows = [dict(step="final", **row) for row in result["rows"]]
    write_csv(args.out / "final.csv", final_rows)
    write_csv(args.out / "final-curves.csv", result["curves"])
    np.save(args.out / "final-losses.npy", result["losses"].numpy())
    (args.out / "final-hparams.yaml").write_text(yaml.safe_dump(hp, sort_keys=False))
    per_seed = [np.mean([r[f"test_{metric}"] for r in final_rows if r["seed"] == s]) for s in seeds]
    summary = {f"final/{prefix}test_{metric}": float(np.mean(per_seed)),
               f"final/{prefix}test_{metric}_std": float(np.std(per_seed)),
               f"final/{prefix}train_seconds": result["train_seconds"], f"final/{prefix}epoch": result["epoch"]}
    print(f"{prefix}final: test {metric} {np.mean(per_seed):.4f} ± {np.std(per_seed):.4f} over {len(seeds)} seeds "
          f"(per seed: {', '.join(f'{v:.4f}' for v in per_seed)}); {result['train_seconds']:.0f}s training", flush=True)
    if run is not None:
        for i, value in enumerate(per_seed):
            run.log({"final_seed": i, f"final/{prefix}test_{metric}": value})
    paths += [args.out / name for name in ("final.csv", "final-curves.csv", "final-hparams.yaml")]
    return final_rows, summary


def trial_summary(trial, metric):
    """One row per trial: its objective, standard error, per-seed and per-inner-fold scores, size and values."""
    attrs = trial.user_attrs
    return dict(trial=trial.number, **{f"valid_{metric}": trial.values[0]}, se=attrs.get("se"),
                **{f"test_{metric}": attrs.get("test")}, epoch=attrs.get("epoch"), params=attrs.get("params"),
                flops=attrs.get("flops"), train_seconds=attrs.get("train_seconds"),
                **{f"valid_seed{i}": v for i, v in enumerate(attrs.get("per_seed", {}).values())},
                **{f"valid_inner{j}": v for j, v in attrs.get("per_inner", {}).items()},
                **attrs.get("values", {}))


if __name__ == "__main__":
    main()
