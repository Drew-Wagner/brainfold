"""SpeechBrain-MOABB's decoding protocol (Borra, Magosso and Ravanelli, 2024), with packed training.

Reimplements what one hyperparameter trial of SpeechBrain-MOABB
(github.com/speechbrain/benchmarks, benchmarks/MOABB) does, but trains every
fold of a trial as one PackedEEGNet. sb_search.py runs the search around it.

- Data: MOABB's paradigm preprocessing (IIR band-pass at the raw rate, epochs
  over the dataset's interval, resampling), run from raw recordings kept in
  memory, so a new band-pass costs seconds; then their crop to [tmin, tmax]
  and their channel selection, the electrodes within ``n_steps`` adjacency
  steps of Cz.
- Splits: leave-one-session-out; from each training session, 20% of each
  class's trials, equally spaced in time, are the validation set.
- Training: braindecode's EEGNet with SpeechBrain-MOABB's Xavier init, Adam
  with their triangular cyclic learning rate (base 1e-8, half cycle 5 epochs),
  their class-weighted NLL, each trial z-scored over time per channel, their
  four augmentations (step 2), and the average of the last ``avg_models``
  epochs' weights and BatchNorm statistics as the final model.
- Or schedule-free AdamW at a constant learning rate (``optimizer:
  schedule-free``): no cyclic schedule and no checkpoint averaging; BatchNorm
  statistics are recomputed on the (unaugmented) training set before every
  evaluation, so ``bn_momentum`` has no effect.

Differences from SpeechBrain-MOABB, besides packing: braindecode's EEGNet
instead of theirs (same layers; max-norm through a parametrization instead of
renormalizing the weights in place, and even kernels padded to T + 1), the
validation set is evaluated only when needed (it never changed training), and
per-member random draws for the augmentations.
"""

from __future__ import annotations

import math
import warnings
from concurrent.futures import ThreadPoolExecutor

import mne
import numpy as np
import schedulefree
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

from brainfold import PackedEEGNet
from train import predict, recalibrate_batch_norms

warnings.filterwarnings("ignore", message=".*standard_1005.*")
warnings.filterwarnings("ignore", message=".*standard_1020.*")
warnings.filterwarnings("ignore", message=".*final_layer_with_constraint.*")
mne.set_log_level("ERROR")


# ---- data -------------------------------------------------------------------------------------------

class Recordings:
    """A MOABB dataset's raw recordings, kept in memory, preprocessed as MOABB's paradigm would.

    ``trials(fmin, fmax)`` equals ``MotorImagery(n_classes, fmin=fmin, fmax=fmax,
    resample=sample_rate).get_data(dataset)`` (SpeechBrain-MOABB's prepare.py) without
    reloading the files; its last few results are cached.
    """

    def __init__(self, dataset, n_classes, sample_rate, subjects=None, threads=8):
        import moabb.datasets

        self.dataset = getattr(moabb.datasets, dataset)()
        self.subjects = subjects or self.dataset.subject_list
        self.n_classes, self.sample_rate, self.threads = n_classes, sample_rate, threads
        self.raws = self.dataset.get_data(self.subjects)
        first = next(iter(next(iter(self.raws.values())).values())).values()
        self.info = next(iter(first)).copy().pick("eeg").info
        self._cache = {}

    def trials(self, fmin, fmax):
        """Trials (n, n_chans, n_times) float32, labels from 0, subjects and sessions."""
        key = (fmin, fmax)
        if key not in self._cache:
            from moabb.paradigms import MotorImagery

            paradigm = MotorImagery(n_classes=self.n_classes, fmin=fmin, fmax=fmax, channels=None,
                                    resample=self.sample_rate)
            pipeline = paradigm.make_process_pipelines(self.dataset)[0]
            to_labels = paradigm.make_labels_pipeline(self.dataset)
            runs = [(subject, session, raw) for subject, sessions in self.raws.items()
                    for session, by_run in sessions.items() for raw in by_run.values()]
            with ThreadPoolExecutor(self.threads) as pool:  # MNE's filtering releases the GIL
                outputs = list(pool.map(lambda run: pipeline.transform(run[2].copy()), runs))
            X, labels, subjects, sessions = [], [], [], []
            for (subject, session, _), out in zip(runs, outputs):
                if out is None:  # no selected event in this run
                    continue
                X.append(out["X"].astype(np.float32))
                labels.append(to_labels.transform(out["events"]))
                subjects += [subject] * len(out["X"])
                sessions += [session] * len(out["X"])
            # SpeechBrain-MOABB's labels: the event codes, from 0
            codes = np.array([self.dataset.event_id[label] for label in np.concatenate(labels)])
            if len(self._cache) >= 4:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = (np.concatenate(X), codes - codes.min(), np.array(subjects), np.array(sessions))
        return self._cache[key]

    def channels(self, n_steps, seed_nodes=("Cz",)):
        """Indices of the electrodes SpeechBrain-MOABB keeps with ``n_steps_channel_selection``."""
        adjacency, names = mne.channels.find_ch_adjacency(self.info, ch_type="eeg")
        adjacency = adjacency.toarray()
        names = list(names)
        selected, nodes = [], list(seed_nodes)
        for _ in range(n_steps):  # their get_neighbour_channels
            linked = []
            for node in nodes:
                linked += [names[i] for i in np.flatnonzero(adjacency[names.index(node)] > 0)]
            nodes = linked
            selected += linked
        selected = set(selected)
        return np.array([i for i, name in enumerate(self.info.ch_names) if name in selected])

    def interval(self):
        """The dataset's trial interval, from 0 (SpeechBrain-MOABB's ``interval``)."""
        return list(np.array(self.dataset.interval) - np.min(self.dataset.interval))


def crop(X, sample_rate, interval_in, tmin, tmax):
    """SpeechBrain-MOABB's crop_signals: nearest samples to tmin (included) and tmax (excluded)."""
    if [tmin, tmax] == list(interval_in):
        return X
    time = np.arange(interval_in[0], interval_in[1], 1 / sample_rate)
    return X[..., np.argmin(np.abs(time - tmin)):np.argmin(np.abs(time - tmax))]


def leave_one_session_out(y, subjects, sessions, valid_ratio):
    """(name, train, valid, test) indices per subject and held-out session, as LeaveOneSessionOut."""
    folds = []
    for subject in sorted(set(subjects.tolist())):
        names = np.unique(sessions[subjects == subject])
        for test_session in names:
            train, valid = [], []
            for session in names[names != test_session]:
                idx = np.flatnonzero((subjects == subject) & (sessions == session))
                for c in range(y[idx].max() + 1):  # per class, equally spaced through the session
                    of_class = idx[y[idx] == c]
                    picks = np.floor(np.linspace(0, len(of_class) - 1, round(valid_ratio * len(of_class)))).astype(int)
                    valid.extend(of_class[picks])
                train.extend(idx)
            valid = np.array(valid)
            train = np.setdiff1d(np.array(train), valid)
            test = np.flatnonzero((subjects == subject) & (sessions == test_session))
            folds.append((f"sub{subject:02d}_test-{test_session}", train, valid, test))
    return folds


def standardize(x, eps=1e-6):
    """SpeechBrain's mean_std_norm over time (the last dim), per trial and channel."""
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + eps)


# ---- augmentation -----------------------------------------------------------------------------------

def augment(x, max_num_segments, amp_delta, min_shift, max_shift, snr_low, snr_high):
    """SpeechBrain-MOABB's four augmentations of a packed batch x (batch, K, C, T), unnormalized.

    As their Augmenter (parallel, fixed batch size, 4 of 4 augmentations, shuffled):
    the batch is cut into 4 contiguous chunks, and each member applies its own random
    order of CutCat, RandAmp, RandomShift and white-noise AddNoise to them, one per
    chunk. Draws that SpeechBrain makes once per call (CutCat's segment count,
    RandomShift's shift) are made once per member.
    """
    B, K, C, T = x.shape
    device = x.device
    bounds = torch.linspace(0, B, 5).to(torch.int).tolist()
    chunk = torch.repeat_interleave(torch.arange(4), torch.tensor(bounds).diff()).to(device)  # (B,)
    order = torch.stack([torch.randperm(4) for _ in range(K)]).to(device)  # (K, 4): member k, chunk j -> aug
    which = order[:, chunk].T  # (B, K)

    # CutCat: every other segment from the previous example of the same chunk (torch.roll within it)
    position = torch.arange(B)
    start = torch.tensor(bounds[:-1])[chunk.cpu()]
    length = torch.tensor(bounds).diff()[chunk.cpu()]
    previous = (start + (position - start - 1) % length).to(device)
    n_segments = torch.randint(2, max_num_segments + 1, (K,), device=device)
    t = torch.arange(T, device=device)
    # torch.linspace(0, T, n + 1, dtype=int)'s cuts; segment i covers [cut_i, cut_{i+1})
    segment = torch.stack([torch.bucketize(t, torch.linspace(0, T, n + 1).to(torch.int).to(device), right=True) - 1
                           for n in range(max_num_segments + 1)])  # (n + 1, T), row n for n segments
    odd = (segment[n_segments] % 2 == 1)  # (K, T)
    single = (length == 1).to(device)  # SpeechBrain skips CutCat for one example
    cutcat = torch.where(odd[None, :, None, :] & ~single[:, None, None, None], x[previous], x)

    # RandAmp: divide by the peak over time, scale by U(1 - delta, 1 + delta) per example
    amp = torch.empty(B, K, 1, 1, device=device).uniform_(1 - amp_delta, 1 + amp_delta)
    randamp = x / x.abs().amax(-1, keepdim=True) * amp

    # RandomShift: roll in time by one integer shift per member
    shift = torch.randint(min_shift, max_shift + 1, (K,), device=device)
    shifted = x.gather(-1, ((t - shift[:, None]) % T)[None, :, None].expand(B, K, C, T))

    # AddNoise (white): mix to an SNR of U(snr_low, snr_high) dB per example, by RMS per channel
    rms = x.pow(2).mean(-1, keepdim=True).sqrt()
    snr = torch.empty(B, K, 1, 1, device=device).uniform_(snr_low, snr_high)
    factor = 1 / (10 ** (snr / 20) + 1)
    noise = torch.randn_like(x)
    noise = noise * (factor * rms / (noise.pow(2).mean(-1, keepdim=True).sqrt() + 1e-14))
    noisy = x * (1 - factor) + noise

    out = cutcat
    for a, result in ((1, randamp), (2, shifted), (3, noisy)):
        out = torch.where((which == a)[:, :, None, None], result, out)
    return out


# ---- model ------------------------------------------------------------------------------------------

def eegnet_kwargs(hp, n_chans, n_outputs, n_times):
    """braindecode EEGNet arguments for SpeechBrain-MOABB's EEGNet hyperparameters."""
    F1, D = hp["cnn_temporal_kernels"], hp["cnn_spatial_depth_multiplier"]
    ratio = hp["cnn_septemporal_point_kernels_ratio_"] / 4
    pool1 = hp.get("cnn_spatial_pool", 4)
    return dict(
        n_chans=n_chans, n_outputs=n_outputs, n_times=n_times,
        F1=F1, D=D, F2=math.ceil(ratio * F1 * D + 1), kernel_length=hp["cnn_temporal_kernelsize"],
        depthwise_kernel_length=round(hp["cnn_septemporal_kernelsize_"] * hp.get("max_cnn_spatial_pool", 4) / pool1),
        pool1_kernel_size=pool1, pool2_kernel_size=hp["cnn_septemporal_pool"],
        conv_spatial_max_norm=hp.get("cnn_spatial_max_norm", 1.0), final_layer_with_constraint=True,
        norm_rate=hp.get("dense_max_norm", 0.25), drop_prob=hp["dropout"],
        batch_norm_momentum=hp.get("bn_momentum", 0.01), batch_norm_eps=1e-5,  # SpeechBrain's BatchNorm eps
    )


def init_like_speechbrain(model):
    """MOABBBrain.init_model: Xavier-uniform weights except normalizations' (1), zero biases."""
    for module in model.modules():
        if torch.nn.utils.parametrize.is_parametrized(module, "weight"):  # braindecode's max-norm layers
            weight = module.parametrizations.weight.original
        else:
            weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        with torch.no_grad():
            if isinstance(weight, torch.Tensor):
                if "Norm" in type(module).__name__:
                    weight.fill_(1)
                else:
                    torch.nn.init.xavier_uniform_(weight)
            if isinstance(bias, torch.Tensor):
                bias.zero_()
    return model


def build(kwargs, seeds, device):
    """A PackedEEGNet whose member k is braindecode's EEGNet initialized by SpeechBrain from seeds[k]."""
    from braindecode.models import EEGNet

    members = []
    with torch.random.fork_rng(devices=[]):
        for seed in seeds:
            torch.manual_seed(seed)
            members.append(init_like_speechbrain(EEGNet(**kwargs)))
    return PackedEEGNet.from_braindecode(members).to(device)


# ---- training ---------------------------------------------------------------------------------------

def cyclic_lr(base_lr, max_lr, step_size):
    """SpeechBrain's triangular CyclicLRScheduler, as MOABBBrain drives it.

    Returns lr(steps_done, first_of_epoch): the first step runs at the optimizer's
    lr (max_lr); after that, each step at clr(steps done), except an epoch's first,
    which the validation stage moved one iteration ahead.
    """
    def clr(iterations):
        cycle = math.floor(1 + iterations / (2 * step_size))
        x = abs(iterations / step_size - 2 * cycle + 1)
        return base_lr + (max_lr - base_lr) * max(0, 1 - x)

    def lr(steps_done, first_of_epoch):
        if steps_done == 0:
            return max_lr
        return clr(steps_done + 1) if first_of_epoch else clr(steps_done)

    return lr


def fit(model, X, Xn, y, train_idx, seeds, hp, optimizer="adam", aug=None, eval_epochs=(), on_eval=None,
        compile=False):
    """Train every member of ``model`` as SpeechBrain-MOABB trains one fold.

    X (n, C, T) are the trials, Xn the same standardized; member k trains on
    ``train_idx[k]``, shuffled by ``seeds[k]``. ``hp`` holds number_of_epochs, lr,
    batch_size and avg_models; ``aug`` the augmentation's arguments (``augment``) or
    None. ``on_eval(epoch, model)`` is called after each epoch in ``eval_epochs``,
    with the model ready to evaluate; training then continues unchanged.
    Returns the final model and the members' mean training loss per epoch, (epochs, K).
    """
    epochs, lr, batch_size = hp["number_of_epochs"], hp["lr"], hp["batch_size"]
    K, n_train = train_idx.shape
    device = train_idx.device
    generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
    counts = F.one_hot(y[train_idx], int(y.max()) + 1).sum(1).float()  # (K, n_classes)
    class_weights = counts.max(1, keepdim=True).values / counts  # their n_max / n_c
    members = torch.arange(K, device=device)

    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        schedule = cyclic_lr(1e-8, lr, round(5 * n_train / batch_size))
        avg_models = hp["avg_models"]
    elif optimizer == "schedule-free":
        opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0)
        opt.train()
        schedule, avg_models = None, 1
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")

    # Without augmentation, every input is a standardized trial: precompute the first BatchNorm's moments
    dtype = next(model.parameters()).dtype
    trials, train_pos = train_idx.unique(return_inverse=True)
    moments = None
    if aug is None:
        chunks = [model.input_moments(Xn[trials[i : i + 256]].to(dtype)) for i in range(0, len(trials), 256)]
        moments = tuple(torch.cat(parts) for parts in zip(*chunks))
    forward = torch.compile(model, dynamic=False) if compile else model

    def batch_logits(rows, forward=forward, augmented=aug is not None):
        idx = train_idx.gather(1, rows).T  # (batch, K)
        if not augmented:
            if moments is None:
                return forward(Xn[idx].to(dtype)), idx
            pos = train_pos.gather(1, rows).T
            return forward(Xn[idx].to(dtype), tuple(m[pos] for m in moments)), idx
        x = X[idx].to(dtype)
        x = torch.cat([standardize(x), standardize(augment(x, **aug))])  # originals, then the augmented copies
        return forward(x), torch.cat([idx, idx])

    def recalibrate():
        positions = torch.arange(n_train, device=device).expand(K, -1)
        recalibrate_batch_norms(model, lambda: (batch_logits(positions[:, i : i + batch_size], model, False)
                                                for i in range(0, n_train, batch_size)))

    def evaluate(epoch):
        buffers = None
        if optimizer == "schedule-free":
            opt.eval()
            buffers = [b.clone() for b in model.buffers()]
            recalibrate()
        model.eval()
        on_eval(epoch, model)
        if buffers is not None:
            for b, saved in zip(model.buffers(), buffers):
                b.copy_(saved)
            opt.train()

    average = None  # running sum of the last avg_models epochs' state
    epoch_losses = torch.zeros(epochs, K, device=device)
    step = 0
    for epoch in range(epochs):
        model.train()
        order = torch.stack([torch.randperm(n_train, device=device, generator=g) for g in generators])
        for i in range(0, n_train, batch_size):
            if schedule is not None:
                for group in opt.param_groups:
                    group["lr"] = schedule(step, i == 0)
            out, idx = batch_logits(order[:, i : i + batch_size])
            # their nll_loss with class weights: mean over the batch of w_y * NLL (not normalized by the weights)
            nll = F.cross_entropy(out.float().permute(0, 2, 1), y[idx], reduction="none")  # (batch, K)
            losses = (class_weights[members, y[idx]] * nll).mean(0)
            opt.zero_grad(set_to_none=True)
            losses.sum().backward()  # each member gets exactly its own gradient
            opt.step()
            step += 1
            epoch_losses[epoch] += losses.detach() * (idx.shape[0] / (2 if aug else 1)) / n_train
        if epoch >= epochs - avg_models:
            state = [t.detach().clone().float() for t in model.state_dict().values()]
            average = state if average is None else [a + s for a, s in zip(average, state)]
        if on_eval and epoch + 1 in eval_epochs and epoch + 1 < epochs:
            evaluate(epoch + 1)

    if avg_models > 1:  # their checkpoint average, of weights and BatchNorm statistics
        with torch.no_grad():
            for t, a in zip(model.state_dict().values(), average):
                t.copy_((a / avg_models).to(t.dtype))
    if optimizer == "schedule-free":
        opt.eval()
        recalibrate()
    model.eval()
    if on_eval and epochs in eval_epochs:
        on_eval(epochs, model)
    return model, epoch_losses.cpu()


METRICS = {
    "acc": lambda labels, pred: balanced_accuracy_score(labels, pred),  # SpeechBrain-MOABB's "acc"
    "f1": lambda labels, pred: f1_score(labels, pred, average="macro"),
    "accuracy": lambda labels, pred: (labels == pred).mean(),
    "kappa": lambda labels, pred: cohen_kappa_score(labels, pred),
}


def scores(model, Xn, y, idx):
    """{metric: value} for every member, on the trials ``idx`` (K, n)."""
    pred = predict(model, Xn, idx, batch_size=256).argmax(-1).cpu().numpy()  # (n, K)
    labels = y[idx.T].cpu().numpy()
    return [{m: float(f(labels[:, k], pred[:, k])) for m, f in METRICS.items()} for k in range(idx.shape[0])]
