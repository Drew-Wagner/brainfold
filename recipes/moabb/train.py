"""EEGNet or ATCNet on a MOABB dataset, every (seed, fold) run trained at once.

One hparams file per dataset and model (see hparams/) sets the data, the
evaluation protocol, the model and the training. Runs are trained in packs of
`train.pack_size` runs with equal training-set sizes. Data loading and
preprocessing follow braindecode's MOABB tutorials; the preprocessed trials are
cached (see --cache-dir). Override any hyperparameter with --set.

    python train.py hparams/bnci2014001_eegnet.yaml
    python train.py hparams/bnci2014009_eegnet.yaml --set train.seeds=[0,1,2]
    python train.py hparams/nakanishi2015_atcnet.yaml --baseline   # also train every run alone with braindecode
    python train.py hparams/bnci2014001_eegnet.yaml --out results.csv   # one row per run and method
    python train.py hparams/bnci2014001_eegnet.yaml --set data.subjects=[1] train.epochs=3 --device cpu   # smoke test
    python train.py hparams/lee2019mi_eegnet.yaml --prepare-only   # download, preprocess and cache, no training
    python train.py hparams/bnci2014001_eegnet.yaml --out r.csv --wandb brainfold   # also log to W&B

With --out r.csv, every run's training loss per epoch goes to r-losses.csv and
its test logits and labels to r-logits.npz, so new metrics need no retraining.

Evaluation protocols (`evaluation.protocol`):
    cross-session   per subject, test on each session, train on the others
    cross-subject   test on each subject, train on all the others
    within-session  per subject and session, stratified k-fold (`evaluation.n_folds`)
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
import warnings
from pathlib import Path

import braindecode
import mne
import moabb
import moabb.datasets
import numpy as np
import schedulefree
import torch
import torch.nn.functional as F
import yaml
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    exponential_moving_standardize,
    preprocess,
)
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from brainfold import PackedATCNet, PackedEEGNet
from tracking import environment, finish_run, start_run, write_csv, write_environment

warnings.filterwarnings("ignore", message=".*final_layer_with_constraint.*")
mne.set_log_level("WARNING")

ARCHITECTURES = {"EEGNet": PackedEEGNet, "ATCNet": PackedATCNet}


def load_data(dataset, subjects, sfreq, l_freq, h_freq, factor_new, init_block_size,
              trial_start_offset_seconds=0.0, trial_stop_offset_seconds=0.0, mapping=None, dataset_kwargs=None):
    """Trials (n, n_chans, n_times), labels, and each trial's subject and session.

    ``subjects=None`` loads every subject, one at a time. Trials span each event's
    annotation (MOABB's trial interval), extended by the two offsets: a negative
    start offset starts before the event.
    """
    if subjects is None:
        subjects = getattr(moabb.datasets, dataset)(**(dataset_kwargs or {})).subject_list
    X, y, subject, session = [], [], [], []
    for subject_id in subjects:  # one at a time: some datasets don't fit in memory at their raw rate
        recordings = MOABBDataset(dataset, subject_ids=[subject_id], dataset_kwargs=dataset_kwargs)
        preprocess(recordings, [
            Preprocessor("pick_types", eeg=True, meg=False, stim=False),
            Preprocessor(lambda data: data * 1e6),  # V -> uV
            Preprocessor("filter", l_freq=l_freq, h_freq=h_freq),
            Preprocessor("resample", sfreq=sfreq),
            Preprocessor(exponential_moving_standardize, factor_new=factor_new, init_block_size=init_block_size),
        ])
        rate = recordings.datasets[0].raw.info["sfreq"]
        windows = create_windows_from_events(
            recordings, trial_start_offset_samples=round(trial_start_offset_seconds * rate),
            trial_stop_offset_samples=round(trial_stop_offset_seconds * rate), mapping=mapping, preload=True)
        for recording in windows.datasets:
            for i in range(len(recording)):
                x, label = recording[i][:2]
                X.append(x.astype(np.float32))
                y.append(label)
            subject += [recording.description["subject"]] * len(recording)
            session += [str(recording.description["session"])] * len(recording)
    return np.stack(X), np.array(y), np.array(subject), np.array(session)


def load_cached(data, cache_dir):
    """``load_data(**data)``, cached in ``cache_dir`` by its arguments and the library versions."""
    if cache_dir is None:
        return load_data(**data)
    key = json.dumps([data, braindecode.__version__, moabb.__version__, mne.__version__], sort_keys=True)
    path = Path(cache_dir) / f"{data['dataset']}-{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = load_data(**data)
        np.savez(path.with_suffix(".tmp.npz"), *arrays)
        path.with_suffix(".tmp.npz").rename(path)
    with np.load(path) as f:
        return tuple(f[f"arr_{i}"] for i in range(4))


def fit(model, X, y, train_idx, seeds, epochs, lr, batch_size, weight_decay=None, class_weight=None,
        compile=False, autocast_dtype=None, optimizer="adam", warmup_epochs=0, eval_every=None, on_eval=None):
    """Train every member with Adam and a cosine learning-rate schedule, or schedule-free AdamW.

    ``model`` maps (batch, K, n_chans, n_times) to (batch, K, n_outputs). Member k
    trains on ``X[train_idx[k]]`` (equal sizes required), shuffled by ``seeds[k]`` alone.
    ``weight_decay`` (ATCNet only), {"conv": ..., "dense": ...}, is the official
    ATCNet code's L2 weight decay, through ``source_optimizer_param_groups``.
    ``class_weight="balanced"`` weights each member's loss by the inverse class
    frequencies of its own training set, as ``F.cross_entropy(weight=...)`` does.
    ``compile`` trains through ``torch.compile(model)``; ``autocast_dtype`` (e.g.
    torch.bfloat16) runs the model under autocast. ``X`` may be stored in a lower
    precision than the model: batches are converted to the model's dtype.

    ``optimizer="schedule-free"`` is schedule-free AdamW (Defazio et al., 2024) at
    a constant ``lr``: no schedule, so the model is usable after any epoch. Its
    weight decay stays the coupled L2 above (added to the gradient, like Adam's),
    not AdamW's decoupled decay. Its weights are evaluated at the averaged point x
    while gradients are taken elsewhere, so the BatchNorm statistics are
    recomputed for x on the training set (see ``recalibrate_batch_norms``).
    ``warmup_epochs`` warms the learning rate up linearly, per step, for either.

    ``on_eval(epoch, model)`` is called, model ready to evaluate, after every
    ``eval_every`` epochs but the last; training continues unchanged after it.
    Returns the model and each member's mean training loss per epoch, (epochs, K).
    """
    generators = [torch.Generator(device=train_idx.device).manual_seed(seed) for seed in seeds]
    if weight_decay:
        groups = model.source_optimizer_param_groups(weight_decay["conv"], weight_decay["dense"])
    else:
        groups = [{"params": list(model.parameters())}]
    n_train = train_idx.shape[1]
    warmup_steps = round(warmup_epochs * math.ceil(n_train / batch_size))
    decayed = []  # (parameters, L2) that schedule-free decays itself
    if optimizer == "adam":
        opt = torch.optim.Adam(groups, lr=lr)
    elif optimizer == "schedule-free":
        decayed = [(g["params"], g.pop("weight_decay", 0)) for g in groups]
        decayed = [(params, l2) for params, l2 in decayed if l2]
        opt = schedulefree.AdamWScheduleFree(groups, lr=lr, warmup_steps=warmup_steps)
        opt.train()
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")
    if class_weight == "balanced":
        counts = F.one_hot(y[train_idx]).sum(1).float()  # (K, n_classes)
        weights = n_train / (counts.shape[1] * counts)
    elif class_weight is not None:
        raise ValueError(f"unknown class_weight {class_weight!r}")
    members = torch.arange(len(seeds), device=train_idx.device)
    # The first BatchNorm's input moments, once per training trial instead of every step;
    # train_pos[k, i] is train_idx[k, i]'s row in them
    dtype = next(model.parameters()).dtype
    moments = None
    if hasattr(model, "input_moments"):
        trials, train_pos = train_idx.unique(return_inverse=True)
        chunks = [model.input_moments(X[trials[i : i + 256]].to(dtype)) for i in range(0, len(trials), 256)]
        moments = tuple(torch.cat(parts) for parts in zip(*chunks))
    # Shares model's parameters. Static shapes, one graph per batch size (the last batch is smaller): with
    # dynamic batch sizes, Inductor's backward asserted the full batch's strides on Lee2019_MI (torch 2.14)
    forward = torch.compile(model, dynamic=False) if compile else model

    def logits(rows, forward=forward):
        """Training-mode logits for the trials train_idx[k, rows[:, k]], (batch, K, n_outputs)."""
        idx = train_idx.gather(1, rows).T  # (batch, K)
        with torch.autocast(X.device.type, autocast_dtype, enabled=autocast_dtype is not None):
            if moments is None:
                return forward(X[idx].to(dtype)), idx
            pos = train_pos.gather(1, rows).T
            return forward(X[idx].to(dtype), tuple(m[pos] for m in moments)), idx

    def recalibrate():
        positions = torch.arange(n_train, device=train_idx.device).expand(len(seeds), -1)
        recalibrate_batch_norms(model, lambda: (logits(positions[:, i : i + batch_size], model)
                                                for i in range(0, n_train, batch_size)))

    def evaluate(epoch):
        buffers = None
        if optimizer == "schedule-free":
            opt.eval()  # the parameters become x
            buffers = [b.clone() for b in model.buffers()]
            recalibrate()
        on_eval(epoch, model)
        if buffers is not None:
            for b, saved in zip(model.buffers(), buffers):
                b.copy_(saved)
            opt.train()

    epoch_losses = torch.zeros(epochs, len(seeds), device=train_idx.device)  # read once at the end: no sync per step
    step = 0
    for epoch in range(epochs):
        model.train()
        order = torch.stack([torch.randperm(n_train, device=train_idx.device, generator=g) for g in generators])
        for i in range(0, n_train, batch_size):
            if optimizer == "adam":  # cosine over epochs, times the warmup
                warmup = min(1, (step + 1) / warmup_steps) if warmup_steps else 1
                for group in opt.param_groups:
                    group["lr"] = 0.5 * lr * (1 + math.cos(math.pi * epoch / epochs)) * warmup
            out, idx = logits(order[:, i : i + batch_size])
            losses = F.cross_entropy(out.float().permute(0, 2, 1), y[idx], reduction="none")
            if class_weight is None:
                losses = losses.mean(0)
            else:
                w = weights[members, y[idx]]  # (batch, K)
                losses = (w * losses).sum(0) / w.sum(0)
            opt.zero_grad(set_to_none=True)
            # SUM of the member mean losses: each member gets exactly its own gradient,
            # and one Adam over the packed parameters == K separate Adams.
            losses.sum().backward()
            for params, l2 in decayed:  # at the point the gradient was taken, as Adam's weight_decay
                torch._foreach_add_([p.grad for p in params], params, alpha=l2)
            opt.step()
            step += 1
            epoch_losses[epoch] += losses.detach() * idx.shape[0] / n_train
        if on_eval and eval_every and (epoch + 1) % eval_every == 0 and epoch + 1 < epochs:
            evaluate(epoch + 1)
    if optimizer == "schedule-free":
        opt.eval()
        recalibrate()
    return model, epoch_losses.cpu()


@torch.no_grad()
def recalibrate_batch_norms(model, batches):
    """Set every BatchNorm's running statistics to their average over ``batches()``.

    ``batches()`` runs the model in training mode over the training set, as
    ``torch.optim.swa_utils.update_bn``: the statistics become the averages of
    the batches' (momentum=None), for the current weights. Dropout masks come
    from a forked RNG, so training continues as it would have without this.
    """
    norms = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    momenta = [bn.momentum for bn in norms]
    for bn in norms:
        bn.reset_running_stats()
        bn.momentum = None
    device = next(model.parameters()).device
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        model.train()
        for _ in batches():
            pass
    for bn, momentum in zip(norms, momenta):
        bn.momentum = momentum


@torch.no_grad()
def predict(model, X, idx, batch_size=64, autocast_dtype=None):
    """(n, K, n_outputs) logits; column k is member k's output for ``X[idx[k]]``."""
    model.eval()
    dtype = next(model.parameters()).dtype
    with torch.autocast(X.device.type, autocast_dtype, enabled=autocast_dtype is not None):
        return torch.cat([model(X[idx[:, i : i + batch_size].T].to(dtype)).float()
                          for i in range(0, idx.shape[1], batch_size)])


class PackOfOne(torch.nn.Module):
    """A single braindecode model with the packed (batch, 1, ...) interface, for the baseline."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x[:, 0])[:, None]

    def source_optimizer_param_groups(self, *args):
        return self.model.source_optimizer_param_groups(*args)


def folds(protocol, y, subjects, sessions, n_folds=None):
    """(name, train indices, test indices) per fold of the evaluation protocol."""
    for subject in sorted(set(subjects.tolist())):
        is_subject = subjects == subject
        if protocol == "cross-subject":
            yield f"sub{subject:02d}", np.flatnonzero(~is_subject), np.flatnonzero(is_subject)
            continue
        for session in sorted(set(sessions[is_subject].tolist())):
            is_session = is_subject & (sessions == session)
            if protocol == "cross-session":
                yield f"sub{subject:02d}_test-{session}", np.flatnonzero(is_subject & ~is_session), np.flatnonzero(is_session)
            elif protocol == "within-session":
                trials = np.flatnonzero(is_session)
                splits = StratifiedKFold(n_folds, shuffle=True, random_state=0).split(trials, y[trials])
                for fold, (train, test) in enumerate(splits):
                    yield f"sub{subject:02d}_ses-{session}_fold{fold}", trials[train], trials[test]
            else:
                raise ValueError(f"unknown evaluation protocol {protocol!r}")


def packs(runs, pack_size):
    """Consecutive runs with equal training- and test-set sizes, at most ``pack_size`` per pack."""
    by_size = {}
    for run in runs:
        by_size.setdefault((len(run[2]), len(run[3])), []).append(run)
    for group in by_size.values():
        for start in range(0, len(group), pack_size):
            yield group[start : start + pack_size]


METRICS = {
    "accuracy": lambda labels, logits: (logits.argmax(1) == labels).mean(),
    "balanced_accuracy": lambda labels, logits: balanced_accuracy_score(labels, logits.argmax(1)),
    "kappa": lambda labels, logits: cohen_kappa_score(labels, logits.argmax(1)),
    "roc_auc": lambda labels, logits: roc_auc_score(labels, logits[:, 1] - logits[:, 0]),  # binary: class 1 positive
}


def scores(logits, labels, metrics):
    """{metric: value} per member of (n, K, n_outputs) logits and (n, K) labels."""
    logits, labels = logits.double().numpy(), labels.numpy()
    return [{m: float(METRICS[m](labels[:, k], logits[:, k])) for m in metrics} for k in range(labels.shape[1])]


def load_hparams(path, overrides):
    """The hparams file (tried as given, then relative to this directory), with ``KEY=VALUE`` overrides."""
    path = Path(path)
    hparams = yaml.safe_load((path if path.exists() else Path(__file__).parent / path).read_text())
    for override in overrides:
        key, value = override.split("=", 1)
        *parents, leaf = key.split(".")
        section = hparams
        for parent in parents:
            section = section[parent]
        if leaf not in section:
            raise KeyError(f"unknown hyperparameter {key}")
        section[leaf] = yaml.safe_load(value)
    return hparams


def add_common_arguments(parser):
    """The arguments train.py and sweep.py share."""
    parser.add_argument("hparams", help="hparams file; relative paths are tried here, then in this directory")
    parser.add_argument("--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                        help="e.g. train.epochs=100; may be repeated")
    parser.add_argument("--device", type=torch.device, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dir", type=lambda s: None if s == "none" else Path(s),
                        default=str(Path.home() / ".cache" / "brainfold"),
                        help="where preprocessed trials are cached; 'none' to disable")
    parser.add_argument("--wandb", metavar="PROJECT", help="log to this W&B project (set WANDB_MODE=offline on clusters)")
    parser.add_argument("--compile", action="store_true", help="train packed models through torch.compile")
    parser.add_argument("--precision", choices=["float32", "tf32", "bf16"], default="float32",
                        help="tf32: TensorFloat-32 matmuls; bf16: also bf16 autocast (BatchNorm statistics stay float32)")
    parser.add_argument("--data-dtype", choices=["float32", "float16"], default="float32",
                        help="dtype of the trials kept on the device; batches are converted to float32")
    parser.add_argument("--optimizer", choices=["adam", "schedule-free"], default="adam",
                        help="adam: cosine-annealed (the recipes'); schedule-free: AdamW at a constant lr "
                             "(Defazio et al., 2024), e.g. with --set train.lr=0.0025")
    parser.add_argument("--warmup-epochs", type=float, default=0, help="linear learning-rate warmup")


def precision_settings(args):
    """Apply ``--precision`` globally; returns the autocast dtype for ``fit`` and ``predict``."""
    torch.backends.cuda.matmul.allow_tf32 = args.precision in ("tf32", "bf16")
    return torch.bfloat16 if args.precision == "bf16" else None


def warm_up(model, X, y, train_idx, test_idx, seeds, settings):
    """One throwaway epoch and prediction on a copy of ``model``.

    cudnn.benchmark searches for the fastest algorithms the first time it sees a
    shape, with workspaces as large as the free memory; this keeps that search out
    of the timings and peak memory. The RNG state is restored, so dropout masks
    don't change.
    """
    with torch.random.fork_rng(devices=[X.device]):
        warm = fit(copy.deepcopy(model), X, y, train_idx, seeds, **{**settings, "epochs": 1})[0]
        predict(warm, X, test_idx, autocast_dtype=settings["autocast_dtype"])


def train_and_predict(model, X, y, train_idx, test_idx, seeds, settings, on_eval=None):
    """Test logits (n, K, n_outputs), training losses (epochs, K), training and evaluation seconds.

    ``on_eval(epoch, test logits)`` gets the learning curve's points (``fit``'s
    ``eval_every``); their time is left out of the training seconds.
    """
    curve_seconds = 0.0

    def evaluate(epoch, model):
        nonlocal curve_seconds
        t = time.perf_counter()
        on_eval(epoch, predict(model, X, test_idx, autocast_dtype=settings["autocast_dtype"]).cpu())
        curve_seconds += time.perf_counter() - t

    t0 = time.perf_counter()
    model, losses = fit(model, X, y, train_idx, seeds, on_eval=evaluate if on_eval else None,
                        **settings)  # returns after a sync (.cpu())
    t1 = time.perf_counter()
    logits = predict(model, X, test_idx, autocast_dtype=settings["autocast_dtype"]).cpu()
    return logits, losses, t1 - t0 - curve_seconds, time.perf_counter() - t1


def peak_memory(device):
    """(allocated, reserved) bytes at their peak since the last call, which they're then reset from.

    Cached memory is released first, so the next peak doesn't start at this one's reserved memory.
    """
    if device.type != "cuda":
        return None, None
    peak = torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    return peak


def write_results(out, rows, losses, logits, curves):
    """``out`` (one row per run and method), ``<out>-losses.csv``, ``<out>-logits.npz`` and,
    with --eval-every, ``<out>-curves.csv``."""
    paths = [out, out.with_name(f"{out.stem}-losses.csv"), out.with_name(f"{out.stem}-logits.npz")]
    write_csv(paths[0], rows)
    write_csv(paths[1], losses)
    if curves:
        paths.append(out.with_name(f"{out.stem}-curves.csv"))
        write_csv(paths[3], curves)
    # Every run's test trials, concatenated; run_index says which row of `rows` each belongs to,
    # trial its index in the cached data
    np.savez_compressed(paths[2], logits=np.concatenate([run["logits"] for run in logits]),
                        labels=np.concatenate([run["labels"] for run in logits]),
                        trial=np.concatenate([run["trials"] for run in logits]),
                        run_index=np.concatenate([np.full(len(run["labels"]), i) for i, run in enumerate(logits)]),
                        method=np.array([r["method"] for r in rows]), seed=np.array([r["seed"] for r in rows]),
                        fold=np.array([r["fold"] for r in rows]))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--baseline", action="store_true",
                        help="also train each run alone as a braindecode model, from the same initial weights")
    parser.add_argument("--out", type=Path, help="write one CSV row per run and method, and the losses and logits")
    parser.add_argument("--prepare-only", action="store_true", help="load and cache the trials, then exit")
    parser.add_argument("--eval-every", type=int, metavar="EPOCHS",
                        help="also evaluate the packed models every EPOCHS epochs, for learning curves "
                             "(<out>-curves.csv); not counted in train_seconds")
    args = parser.parse_args()
    hp = load_hparams(args.hparams, args.set)
    data, train, evaluation = hp["data"], hp["train"], hp["evaluation"]
    architecture, metrics = hp["architecture"], hp.get("metrics", ["accuracy", "kappa"])
    # cuDNN's default heuristics pick slow convolutions with workspaces of GBs for small packed ATCNets
    torch.backends.cudnn.benchmark = True

    t0 = time.perf_counter()
    X, y, subjects, sessions = load_cached(data, args.cache_dir)
    if args.prepare_only:
        print(f"{data['dataset']}: {len(X)} trials of {X.shape[1]} channels x {X.shape[2]} samples")
        return
    n_classes, data_bytes = len(np.unique(y)), X.nbytes
    X = torch.from_numpy(X).to(args.device, getattr(torch, args.data_dtype))
    y = torch.from_numpy(y).to(args.device)
    data_seconds = time.perf_counter() - t0  # loading (or preprocessing) and copying to the device
    runs = [(name, seed, torch.from_numpy(train_idx), torch.from_numpy(test_idx))
            for seed in train["seeds"]
            for name, train_idx, test_idx in folds(evaluation["protocol"], y.cpu().numpy(), subjects, sessions,
                                                   evaluation.get("n_folds"))]
    env = dict(environment(args.device), data_seconds=data_seconds, data_bytes=data_bytes)
    run = start_run(args.wandb, "recipe", args.hparams, hp, args, env)
    if run is not None:  # per-epoch charts, one line per pack
        run.define_metric("epoch")
        run.define_metric("pack*", step_metric="epoch")
        run.define_metric("mean/*", step_metric="epoch")
    paths = [write_environment(args.out, env)] if args.out else []

    print(f"{architecture} on {data['dataset']} ({evaluation['protocol']}): {len(runs)} runs, "
          f"{len(X)} trials of {X.shape[1]} channels x {X.shape[2]} samples, {n_classes} classes", flush=True)
    methods = ["packed", "braindecode"] if args.baseline else ["packed"]
    settings = dict(epochs=train["epochs"], lr=train["lr"], batch_size=train["batch_size"],
                    weight_decay=train.get("weight_decay"), class_weight=train.get("class_weight"),
                    autocast_dtype=precision_settings(args), optimizer=args.optimizer,
                    warmup_epochs=args.warmup_epochs, eval_every=args.eval_every)
    rows, losses, logits, curves, seconds = [], [], [], [], dict.fromkeys(methods, 0.0)
    warmed = set()  # (method, members per model, training size, test size) that cudnn.benchmark has seen
    for pack_index, pack in enumerate(packs(runs, train["pack_size"])):
        K, seeds = len(pack), [seed for _, seed, _, _ in pack]
        train_idx = torch.stack([idx for _, _, idx, _ in pack]).to(args.device)
        test_idx = torch.stack([idx for _, _, _, idx in pack]).to(args.device)
        labels = y[test_idx.T].cpu()
        model = ARCHITECTURES[architecture].from_seeds(
            seeds, n_chans=X.shape[1], n_outputs=n_classes, n_times=X.shape[2], **hp["model"]).to(args.device)
        # (model, the pack's members it trains): one packed model, or one braindecode model per member
        # from the same initial weights
        models = {"packed": [(model, slice(0, K))]}
        if args.baseline:
            models["braindecode"] = [(PackOfOne(single), slice(k, k + 1))
                                     for k, single in enumerate(model.to_braindecode())]

        if args.compile:
            torch._dynamo.reset()  # a new model: drop the last pack's graphs (dynamo limits recompiles)
        results = {}
        for method in methods:
            members = K if method == "packed" else 1  # per model
            shape = (method, members, train_idx.shape[1], test_idx.shape[1])
            if args.device.type == "cuda" and shape not in warmed:
                first, part = models[method][0]
                warm_up(first, X, y, train_idx[part], test_idx[part], seeds[part],
                        dict(settings, compile=args.compile and method == "packed"))
                warmed.add(shape)
            peak_memory(args.device)  # reset the peaks
            # compiling each braindecode model would cost more than it saves
            method_settings = dict(settings, compile=args.compile and method == "packed")
            curve = {}  # epoch: the members' test scores; learning curves for the packed model only
            on_eval = (lambda epoch, out: curve.update({epoch: scores(out, labels, metrics)})) \
                if args.eval_every and method == "packed" else None
            outputs = [train_and_predict(m, X, y, train_idx[part], test_idx[part], seeds[part], method_settings,
                                         on_eval) for m, part in models[method]]
            results[method] = dict(
                logits=torch.cat([o[0] for o in outputs], 1), losses=torch.cat([o[1] for o in outputs], 1),
                # a packed model's time is shared equally by its members
                train_seconds=[o[2] / members for o in outputs for _ in range(members)],
                eval_seconds=[o[3] / members for o in outputs for _ in range(members)],
                peak=peak_memory(args.device), curve=curve)
            seconds[method] += sum(o[2] + o[3] for o in outputs)

        pack_scores = {m: scores(results[m]["logits"], labels, metrics) for m in methods}
        packed = results["packed"]
        if args.eval_every:
            packed["curve"][train["epochs"]] = pack_scores["packed"]
        if run is not None:
            for epoch, loss in enumerate(packed["losses"].mean(1).tolist(), 1):
                entry = {"epoch": epoch, f"pack{pack_index}/train_loss": loss}
                for metric in metrics if epoch in packed["curve"] else []:
                    entry[f"pack{pack_index}/test_{metric}"] = np.mean([s[metric] for s in packed["curve"][epoch]])
                run.log(entry)
        for k, (name, seed, _, _) in enumerate(pack):
            line = "  ".join(f"{m} " + " ".join(f"{metric} {pack_scores[m][k][metric]:.3f}" for metric in metrics)
                             for m in methods)
            print(f"seed {seed} {name}  {line}", flush=True)
            for m in methods:
                result = results[m]
                rows.append(dict(dataset=data["dataset"], architecture=architecture,
                                 protocol=evaluation["protocol"], method=m, seed=seed, fold=name,
                                 pack=pack_index, pack_size=K, train_seconds=result["train_seconds"][k],
                                 eval_seconds=result["eval_seconds"][k], peak_allocated=result["peak"][0],
                                 peak_reserved=result["peak"][1], **pack_scores[m][k]))
                losses.extend(dict(method=m, seed=seed, fold=name, epoch=epoch, loss=float(loss))
                              for epoch, loss in enumerate(result["losses"][:, k]))
                logits.append(dict(logits=result["logits"][:, k].numpy(), labels=labels[:, k].numpy(),
                                   trials=test_idx[k].cpu().numpy()))
                curves.extend(dict(method=m, seed=seed, fold=name, epoch=epoch, **member_scores[k])
                              for epoch, member_scores in sorted(result["curve"].items()))
        if args.out:  # after every pack, so a job that runs out of time keeps what it finished
            result_paths = write_results(args.out, rows, losses, logits, curves)

    if run is not None and curves:  # the mean learning curve over every run
        for epoch in sorted({c["epoch"] for c in curves}):
            points = [c for c in curves if c["epoch"] == epoch]
            run.log({"epoch": epoch, **{f"mean/test_{metric}": np.mean([c[metric] for c in points])
                                        for metric in metrics}})
    for m in methods:
        values = {metric: np.array([r[metric] for r in rows if r["method"] == m]) for metric in metrics}
        summary = ", ".join(f"{metric} {v.mean():.3f} ± {v.std():.3f}" for metric, v in values.items())
        print(f"{m:11s} {len(runs)} runs in {seconds[m]:.0f}s: {summary}")
    if args.baseline:
        first = metrics[0]
        diff = np.array([r[first] for r in rows if r["method"] == "packed"]) - np.array(
            [r[first] for r in rows if r["method"] == "braindecode"])
        print(f"packed - braindecode {first} per run: {diff.mean():+.3f} ± {diff.std():.3f}")
    finish_run(run, rows, paths + (result_paths if args.out else []), {f"{m}_seconds": s for m, s in seconds.items()})


if __name__ == "__main__":
    main()
