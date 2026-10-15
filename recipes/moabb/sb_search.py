"""SpeechBrain-MOABB's hyperparameter search and final evaluation, with Optuna's TPE and packed training.

    python sb_search.py hparams/sb_moabb/bnci2014001_eegnet.yaml --out results/sb/bnci2014001-adam
    python sb_search.py hparams/sb_moabb/bnci2014001_eegnet.yaml --optimizer schedule-free --out results/sb/bnci2014001-sf
    python sb_search.py hparams/sb_moabb/bnci2014001_eegnet.yaml --out /tmp/smoke --set trials=2 final_seeds=1 \\
        fixed.number_of_epochs=3 --set-search step1.number_of_epochs="uniform(2, 4, discrete=True)"   # smoke test

As their run_hparam_optimization.sh: each step of `search` runs `trials` trials
of TPE (`startup_trials` random ones first), maximizing the validation `metric`
averaged over every fold, with seed `seed`; the best trial's values are then
fixed for the next step. The final evaluation trains every fold with
`final_seeds` seeds and reports the test sessions' scores. One trial trains all
its folds as packs of `pack_size` (see sb_protocol.py for the protocol itself).

Their Orion TPE is replaced by Optuna's (TPESampler with their startup and
candidate counts). Test scores are recorded for every trial but never used to
choose anything.

Everything goes to --out: the Optuna studies (study.journal; rerun the same
command to resume after a time limit, and the trial that was running is redone),
trials/ (per trial: one row per fold, the scores at every evaluated epoch, and
the training loss per epoch and fold), <step>-trials.csv and <step>-curves.csv
(every trial of a step), best-<step>.yaml, final.csv, final-curves.csv,
final-hparams.yaml and results-env.json. With --wandb, every trial is logged
as it finishes: its validation and test scores, timings, peak memory and
values, and its training loss per epoch (mean and per fold).
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import optuna
import torch
import yaml

import sb_protocol as protocol
from tracking import environment, finish_run, start_run, write_csv, write_environment
from train import load_hparams, packs, peak_memory


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


# ---- one trial: every fold, packed -----------------------------------------------------------------

def run_trial(hp, seeds, recordings, cfg, args, label):
    """Train every (seed, fold) with hyperparameters ``hp``; scores at each evaluated epoch.

    Returns rows (one per seed and fold, final scores), curve rows (every evaluated
    epoch), the members' training losses (epochs, n_runs) and timings.
    """
    hp = dict(hp, batch_size=2 ** hp["batch_size_exponent"])
    sf = args.optimizer == "schedule-free"
    if sf:
        epochs = cfg["schedule_free"]["epochs"]
        eval_epochs = list(range(epochs["first"], epochs["last"] + 1, epochs["every"]))
        if "chosen_epochs" in hp:  # the final evaluation: train to the chosen epoch only
            hp["number_of_epochs"], eval_epochs = hp["chosen_epochs"], [hp["chosen_epochs"]]
        else:
            hp["number_of_epochs"] = epochs["last"]
    else:
        eval_epochs = [hp["number_of_epochs"]]

    t0 = time.perf_counter()
    X, y, subjects, sessions = recordings.trials(hp["fmin"], hp["fmax"])
    X = protocol.crop(X, cfg["sample_rate"], recordings.interval(), hp["tmin"], hp["tmax"])
    X = X[:, recordings.channels(hp["n_steps_channel_selection"])]
    folds = protocol.leave_one_session_out(y, subjects, sessions, cfg["valid_ratio"])
    device = args.device
    X = torch.from_numpy(X).to(device)
    Xn = protocol.standardize(X)
    y = torch.from_numpy(y).to(device)
    data_seconds = time.perf_counter() - t0

    aug = None
    if hp.get("augment", False):
        shift = 1e-2 * hp["shift_delta_"]
        aug = dict(max_num_segments=hp["max_num_segments"], amp_delta=hp["amp_delta"],
                   min_shift=math.floor(-cfg["sample_rate"] * shift), max_shift=math.floor(cfg["sample_rate"] * shift),
                   snr_low=hp["snr_white_low"], snr_high=hp["snr_white_low"] + hp["snr_white_delta"])
    kwargs = protocol.eegnet_kwargs(hp, X.shape[1], cfg["n_classes"], X.shape[2])
    runs = [(name, seed, torch.from_numpy(train), torch.from_numpy(test), torch.from_numpy(valid))
            for seed in seeds for name, train, valid, test in folds]
    rows, curves, losses, timing = [], [], [], dict(train_seconds=0.0, peak_allocated=0)

    def train_pack(pack):
        """Train one pack and record its scores; a pack that runs out of memory is split in two."""
        pack_seeds = [r[1] for r in pack]
        train_idx, test_idx, valid_idx = (torch.stack([r[i] for r in pack]).to(device) for i in (2, 3, 4))
        evaluated = {}

        def on_eval(epoch, model):
            evaluated[epoch] = (protocol.scores(model, Xn, y, valid_idx), protocol.scores(model, Xn, y, test_idx))

        if args.compile:
            torch._dynamo.reset()
        peak_memory(device)
        t = time.perf_counter()
        try:
            model = protocol.build(kwargs, pack_seeds, device)
            _, member_losses = protocol.fit(model, X, Xn, y, train_idx, pack_seeds, hp, optimizer=args.optimizer,
                                            aug=aug, eval_epochs=eval_epochs, on_eval=on_eval, compile=args.compile)
        except torch.OutOfMemoryError:
            if len(pack) == 1:
                raise
            model = None
            print(f"  out of memory with {len(pack)} runs per pack; splitting it", flush=True)
            train_pack(pack[: len(pack) // 2])
            train_pack(pack[len(pack) // 2 :])
            return
        timing["train_seconds"] += time.perf_counter() - t
        timing["peak_allocated"] = max(timing["peak_allocated"], peak_memory(device)[0] or 0)
        losses.append(member_losses)
        for k, (name, seed, *_) in enumerate(pack):
            for epoch, (valid, test) in sorted(evaluated.items()):
                curves.append(dict(label=label, seed=seed, fold=name, epoch=epoch,
                                   **{f"valid_{m}": v for m, v in valid[k].items()},
                                   **{f"test_{m}": v for m, v in test[k].items()}))
            rows.append(dict(label=label, seed=seed, fold=name, n_train=train_idx.shape[1],
                             n_valid=valid_idx.shape[1], n_test=test_idx.shape[1]))

    # packs() groups runs by training- and test-set size; split those groups by validation size too
    for pack in packs(runs, cfg["pack_size"]):
        for n_valid in dict.fromkeys(len(r[4]) for r in pack):
            train_pack([r for r in pack if len(r[4]) == n_valid])
    losses = torch.cat(losses, 1)

    # the epoch to report: the last (Adam), or the best mean validation score (schedule-free)
    metric = cfg["metric"]
    by_epoch = {}
    for c in curves:
        by_epoch.setdefault(c["epoch"], []).append(c[f"valid_{metric}"])
    mean_valid = {epoch: float(np.mean(v)) for epoch, v in by_epoch.items()}
    chosen = max(mean_valid, key=lambda e: (mean_valid[e], -e)) if sf else hp["number_of_epochs"]
    final = {(c["seed"], c["fold"]): c for c in curves if c["epoch"] == chosen}
    for row in rows:
        row.update({k: v for k, v in final[(row["seed"], row["fold"])].items() if k.startswith(("valid_", "test_"))},
                   epoch=chosen)
    return dict(rows=rows, curves=curves, losses=losses, objective=mean_valid[chosen], epoch=chosen,
                mean_valid=mean_valid, data_seconds=data_seconds, **timing)


# ---- the protocol ----------------------------------------------------------------------------------

def stale_trials_failed(study):
    """Mark trials left running by an interrupted job as failed, so they're redone."""
    for trial in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)


def read_csv(path):
    import csv

    with open(path, newline="") as f:
        return list(csv.DictReader(f))


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

    def trial(self, step, result, values, test, best, lr):
        if self.run is None:
            return
        self.run.define_metric(f"{step}/*", step_metric="trial")
        for epoch, loss in enumerate(result["losses"].tolist()):  # training loss per member and mean
            self.epochs += 1
            self.run.log({"epoch": self.epochs, "train_loss": float(np.mean(loss)), "lr": lr(epoch),
                          **{f"train_loss/member{k}": v for k, v in enumerate(loss)}})
        self.trials += 1
        m = self.metric
        self.run.log({"trial": self.trials, f"{step}/valid_{m}": result["objective"], f"{step}/best_valid_{m}": best,
                      f"{step}/test_{m}": test, f"{step}/epoch": result["epoch"],
                      f"{step}/train_seconds": result["train_seconds"], f"{step}/data_seconds": result["data_seconds"],
                      f"{step}/seconds_per_epoch": result["train_seconds"] / len(result["losses"]),
                      f"{step}/peak_allocated": result["peak_allocated"],
                      **{f"{step}/mean_valid_{m}_at/{e}": v for e, v in result["mean_valid"].items()},
                      **{f"{step}/param/{k}": v for k, v in values.items()}})


def epoch_lr(hp, optimizer, n_train):
    """The learning rate at the start of each epoch, for logging."""
    batch_size = 2 ** hp["batch_size_exponent"]
    if optimizer == "schedule-free":
        return lambda epoch: hp["lr"]
    schedule = protocol.cyclic_lr(1e-8, hp["lr"], round(5 * n_train / batch_size))
    return lambda epoch: schedule(epoch * math.ceil(n_train / batch_size), True)


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
    args = parser.parse_args()
    cfg = load_hparams(args.hparams, args.set)
    for override in args.set_search:
        key, prior = override.split("=", 1)
        step, name = key.split(".", 1)
        parse_prior(prior)  # fail early
        cfg["search"][step][name] = prior
    sf = args.optimizer == "schedule-free"
    if sf:
        for space in cfg["search"].values():
            for name in cfg["schedule_free"]["not_searched"]:
                space.pop(name, None)
    torch.backends.cudnn.benchmark = True
    trials_dir = args.out / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    recordings = protocol.Recordings(cfg["dataset"], cfg["n_classes"], cfg["sample_rate"], subjects=args.subjects)
    env = dict(environment(args.device), load_seconds=time.perf_counter() - t0, optimizer=args.optimizer)
    paths = [write_environment(args.out / "results", env)]
    run = start_run(args.wandb, "sb-protocol", args.hparams, cfg, args, env)
    log = Logger(run, cfg["metric"])
    metric = cfg["metric"]
    print(f"{cfg['dataset']}: recordings of {len(recordings.subjects)} subjects loaded in {env['load_seconds']:.0f}s; "
          f"optimizer {args.optimizer}", flush=True)

    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(args.out / "study.journal")))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    hp = dict(cfg["fixed"])
    for step_index, (step, space) in enumerate([] if args.no_search else cfg["search"].items()):
        hp["augment"] = step_index > 0  # their step 1 runs with --repeat_augment 0
        sampler = optuna.samplers.TPESampler(n_startup_trials=cfg["startup_trials"],
                                             n_ei_candidates=cfg["ei_candidates"], seed=cfg["seed"])
        study = optuna.create_study(study_name=f"{step}-{args.optimizer}", storage=storage, sampler=sampler,
                                    direction="maximize", load_if_exists=True)
        stale_trials_failed(study)
        complete = lambda: study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        while len(complete()) < cfg["trials"]:
            trial = study.ask()
            values = suggest(trial, space)
            label = f"{step}/trial{trial.number:03d}"
            result = run_trial({**hp, **values}, [cfg["seed"]], recordings, cfg, args, label)
            test = float(np.mean([r[f"test_{metric}"] for r in result["rows"]]))
            name = trials_dir / f"{step}-trial{trial.number:03d}"
            write_csv(name.with_suffix(".rows.csv"), [dict(step=step, trial=trial.number, **values, **r)
                                                      for r in result["rows"]])
            write_csv(name.with_suffix(".curves.csv"), [dict(step=step, trial=trial.number, **c)
                                                        for c in result["curves"]])
            np.save(name.with_suffix(".losses.npy"), result["losses"].numpy())
            for key, value in dict(values=values, epoch=result["epoch"], test=test,
                                   train_seconds=result["train_seconds"]).items():
                trial.set_user_attr(key, value)
            study.tell(trial, result["objective"])
            best = study.best_value
            print(f"{label}: valid {metric} {result['objective']:.4f} (best {best:.4f}) at epoch {result['epoch']}, "
                  f"test {test:.4f}; {result['train_seconds']:.0f}s training, {result['data_seconds']:.0f}s data, "
                  f"{result['peak_allocated'] / 2**30:.1f} GiB; {values}", flush=True)
            n_train = int(result["rows"][0]["n_train"])
            log.trial(step, result, values, test, best, epoch_lr({**hp, **values}, args.optimizer, n_train))
        best = study.best_trial
        hp.update(best.user_attrs["values"])
        if sf:
            hp["chosen_epochs"] = best.user_attrs["epoch"]
        (args.out / f"best-{step}.yaml").write_text(yaml.safe_dump(
            dict(trial=best.number, valid=best.value, epoch=best.user_attrs["epoch"],
                 values=best.user_attrs["values"]), sort_keys=False))
        print(f"{step}: best trial {best.number}, valid {metric} {best.value:.4f}: {best.user_attrs['values']}",
              flush=True)
        # every finished trial of this step (earlier jobs' too), in one file each
        for kind in ("rows", "curves"):
            rows = [row for t in complete()
                    for row in read_csv(trials_dir / f"{step}-trial{t.number:03d}.{kind}.csv")]
            write_csv(args.out / f"{step}-{'trials' if kind == 'rows' else 'curves'}.csv", rows)
            paths.append(args.out / f"{step}-{'trials' if kind == 'rows' else 'curves'}.csv")

    # Final evaluation: every fold with final_seeds seeds, on the test sessions, with augmentation
    hp["augment"] = True
    seeds = [cfg["seed"] + i for i in range(cfg["final_seeds"])]
    result = run_trial(hp, seeds, recordings, cfg, args, "final")
    final_rows = [dict(step="final", **row) for row in result["rows"]]
    write_csv(args.out / "final.csv", final_rows)
    write_csv(args.out / "final-curves.csv", result["curves"])
    np.save(args.out / "final-losses.npy", result["losses"].numpy())
    (args.out / "final-hparams.yaml").write_text(yaml.safe_dump(hp, sort_keys=False))
    per_seed = [np.mean([r[f"test_{metric}"] for r in final_rows if r["seed"] == s]) for s in seeds]
    summary = {f"final/test_{metric}": float(np.mean(per_seed)), f"final/test_{metric}_std": float(np.std(per_seed)),
               "final/train_seconds": result["train_seconds"], "final/epoch": result["epoch"]}
    print(f"final: test {metric} {np.mean(per_seed):.4f} ± {np.std(per_seed):.4f} over {len(seeds)} seeds "
          f"(per seed: {', '.join(f'{v:.4f}' for v in per_seed)}); {result['train_seconds']:.0f}s training", flush=True)
    if run is not None:
        for i, value in enumerate(per_seed):
            run.log({"final_seed": i, f"final/test_{metric}": value})
    paths += [args.out / name for name in ("final.csv", "final-curves.csv", "final-hparams.yaml")]
    finish_run(run, final_rows, paths, summary)


if __name__ == "__main__":
    main()
