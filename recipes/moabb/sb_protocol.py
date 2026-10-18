"""SpeechBrain-MOABB's decoding protocol (Borra, Paissan and Ravanelli, 2024), with packed training.

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
import time
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
    reloading the files; its last few results are cached. ``runs`` gives each trial's
    run, for validation folds of whole runs (``inner_folds``). ``modality="eog"`` gives the
    EOG channels instead, preprocessed the same way (the audit's EOG condition only).
    """

    def __init__(self, dataset, n_classes, sample_rate, subjects=None, threads=8):
        import moabb.datasets

        self.dataset = getattr(moabb.datasets, dataset)()
        self.subjects = subjects or self.dataset.subject_list
        self.n_classes, self.sample_rate, self.threads = n_classes, sample_rate, threads
        self.raws = self.dataset.get_data(self.subjects)
        first = next(iter(next(iter(self.raws.values())).values())).values()
        raw = next(iter(first))
        self.info = raw.copy().pick("eeg").info
        self.eog_names = raw.copy().pick("eog").ch_names
        self.channel_types = raw.get_channel_types()
        self._cache = {}

    def structure(self):
        """{subject: {session: [run, ...]}} in MOABB's (recording) order."""
        return {subject: {session: list(by_run) for session, by_run in sessions.items()}
                for subject, sessions in self.raws.items()}

    def recording_rates(self):
        """The raw recordings' sample rates (one value for most datasets)."""
        return sorted({raw.info["sfreq"] for sessions in self.raws.values() for by_run in sessions.values()
                       for raw in by_run.values()})

    def trials(self, fmin, fmax, sample_rate=None, modality="eeg"):
        """Trials (n, n_chans, n_times) float32, labels from 0, subjects and sessions.

        ``sample_rate`` resamples to another rate than the one given at construction.
        """
        return self._preprocessed(fmin, fmax, sample_rate, modality)[:4]

    def runs(self, fmin, fmax, sample_rate=None, modality="eeg"):
        """Each trial's run (MOABB's run name), aligned with ``trials``."""
        return self._preprocessed(fmin, fmax, sample_rate, modality)[4]

    def _preprocessed(self, fmin, fmax, sample_rate, modality="eeg"):
        sample_rate = sample_rate or self.sample_rate
        key = (fmin, fmax) if sample_rate == self.sample_rate else (fmin, fmax, sample_rate)
        if modality != "eeg":
            key += (modality,)
        if key not in self._cache:
            from moabb.paradigms import MotorImagery

            paradigm = MotorImagery(n_classes=self.n_classes, fmin=fmin, fmax=fmax, channels=None,
                                    resample=sample_rate)
            pipeline = paradigm.make_process_pipelines(self.dataset)[0]
            to_labels = paradigm.make_labels_pipeline(self.dataset)
            runs = [(subject, session, run, raw) for subject, sessions in self.raws.items()
                    for session, by_run in sessions.items() for run, raw in by_run.items()]
            with ThreadPoolExecutor(self.threads) as pool:  # MNE's filtering releases the GIL
                outputs = list(pool.map(lambda run: pipeline.transform(self._raw(run[3], modality)), runs))
            X, labels, subjects, sessions, run_names = [], [], [], [], []
            for (subject, session, run, _), out in zip(runs, outputs):
                if out is None:  # no selected event in this run
                    continue
                X.append(out["X"].astype(np.float32))
                labels.append(to_labels.transform(out["events"]))
                subjects += [subject] * len(out["X"])
                sessions += [session] * len(out["X"])
                run_names += [run] * len(out["X"])
            # SpeechBrain-MOABB's labels: the event codes, from 0
            codes = np.array([self.dataset.event_id[label] for label in np.concatenate(labels)])
            if len(self._cache) >= 4:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = (np.concatenate(X), codes - codes.min(), np.array(subjects), np.array(sessions),
                                np.array(run_names))
        return self._cache[key]

    def _raw(self, raw, modality):
        """A copy of ``raw`` for MOABB's pipeline, which keeps EEG channels: as is, or its EOG channels typed as EEG."""
        if modality == "eeg":
            return raw.copy()
        if modality != "eog":
            raise ValueError(f"unknown modality {modality!r}")
        raw = raw.copy().pick(["eog", "stim"])
        raw.set_channel_types({name: "eeg" for name in self.eog_names}, on_unit_change="ignore")
        return raw

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


# ---- version 2's rules from each dataset's metadata (PROTOCOL.md) ------------------------------------

def protocol_window(paradigm, margin=0.5):
    """(tmin, tmax) in seconds from MOABB's epoch start: ``margin`` s after the cue's offset to the end of imagery.

    ``paradigm`` is the hparams file's ``paradigm`` block: ``cue_onset``, ``cue_offset``
    (seconds, or "persists") and ``imagery_end``, all from MOABB's epoch start. The margin
    skips the visual response to the cue's removal (the ISCEV VEP standard leaves 400 ms for
    the pattern-offset response; Odom et al., 2016) and the band-pass filter's spread. If
    the cue persists through the imagery, the window starts ``margin`` s after its onset,
    past the main visual evoked response.
    """
    onset, offset, end = paradigm["cue_onset"], paradigm["cue_offset"], paradigm["imagery_end"]
    tmin = (onset if offset == "persists" else offset) + margin
    if not tmin < end:
        raise ValueError(f"empty window: from {tmin} s to the end of imagery at {end} s")
    return float(tmin), float(end)


def cue_window(paradigm, margin=0.5):
    """(tmin, tmax) of the cue period: onset to offset, or onset to ``margin`` s if the cue persists."""
    onset, offset = paradigm["cue_onset"], paradigm["cue_offset"]
    return float(onset), float(onset + margin if offset == "persists" else offset)


def check_paradigm(paradigm, recordings, sample_rate, fmin, fmax):
    """Raise if the ``paradigm`` block disagrees with MOABB's metadata, or the band or rate isn't possible.

    Checks what the block states (a missing or null entry isn't checked): sessions per
    subject, runs per session, the recording rate, the numbers of EEG and EOG channels;
    and that the window fits in MOABB's epoch, ``fmax`` is below the Nyquist frequency of
    ``sample_rate``, and the band is within the hardware filters (``hardware_band``).
    """
    problems = []
    structure = recordings.structure()

    def differs(key, observed):
        stated = paradigm.get(key)
        if stated is not None and set(observed) != {stated}:
            problems.append(f"{key}: the paradigm block says {stated}, MOABB has {sorted(set(observed))}")

    differs("sessions", [len(sessions) for sessions in structure.values()])
    differs("runs_per_session", [len(runs) for sessions in structure.values() for runs in sessions.values()])
    differs("recording_rate", recordings.recording_rates())
    differs("eeg_channels", [recordings.channel_types.count("eeg")])
    differs("eog_channels", [recordings.channel_types.count("eog")])
    length = float(np.ptp(recordings.dataset.interval))
    for name, (tmin, tmax) in (("window", protocol_window(paradigm)), ("cue period", cue_window(paradigm))):
        if not 0 <= tmin < tmax <= length:
            problems.append(f"the {name} {tmin}-{tmax} s isn't within MOABB's epoch of {length} s")
    if fmax >= sample_rate / 2:
        problems.append(f"fmax {fmax} Hz isn't below the Nyquist frequency of {sample_rate} Hz")
    low, high = paradigm.get("hardware_band") or (0, math.inf)
    if fmin < low or fmax > high:
        problems.append(f"the band {fmin}-{fmax} Hz isn't within the hardware filters' {low}-{high} Hz")
    if problems:
        raise ValueError("the paradigm block or the protocol's preprocessing doesn't fit the dataset:\n  "
                         + "\n  ".join(problems))


def electrode_row(name):
    """The 10-20 (10-10, 10-5) row of an electrode name: "Fp" for Fp1, "AF" for AFz, "F" for Fz, "FC" for FC3."""
    return name.rstrip("0123456789hzZ")  # 10-5 names such as FFC3h end in h


def without_rows(names, rows=("Fp", "AF", "F")):
    """Indices of the channels whose 10-20 row isn't one of ``rows`` (case-insensitive)."""
    rows = {r.lower() for r in rows}
    return np.array([i for i, name in enumerate(names) if electrode_row(name).lower() not in rows])


def pseudo_sessions(subjects, sessions, runs):
    """Sessions for leave-one-session-out, with every subject's single session split in two.

    A subject recorded in one session gets two pseudo-sessions, ``<session>-half1`` and
    ``<session>-half2``: its runs in two contiguous halves (the first half has the extra
    run of an odd number), or, with a single run, its trials in two contiguous halves in
    time. Subjects with two or more sessions keep them. Trials are in recording order.
    """
    out = np.array(sessions, dtype=object)
    for subject in np.unique(subjects):
        idx = np.flatnonzero(subjects == subject)
        names = np.unique(sessions[idx])
        if len(names) > 1:
            continue
        order = list(dict.fromkeys(runs[idx].tolist()))
        if len(order) > 1:
            first = set(np.array_split(np.array(order, dtype=object), 2)[0].tolist())
            half = np.where(np.isin(runs[idx], list(first)), 1, 2)
        else:
            half = np.repeat([1, 2], [len(a) for a in np.array_split(idx, 2)])
        out[idx] = [f"{names[0]}-half{h}" for h in half]
    return out.astype(str)


def outer_session_names(structure):
    """The test sessions of leave-one-session-out with ``pseudo_sessions``, from {subject: {session: runs}}."""
    names = set()
    for sessions in structure.values():
        names |= set(sessions) if len(sessions) > 1 else {f"{s}-half{h}" for s in sessions for h in (1, 2)}
    return sorted(names)


def binomial_threshold(n, n_classes, alpha=0.05):
    """The accuracy that n trials must exceed to be above chance at level ``alpha`` (Combrisson and Jerbi, 2015).

    Their binoinv(1 - alpha, n, 1 / n_classes) / n: an accuracy above it has P <= alpha
    under the binomial of guessing.
    """
    from scipy.stats import binom

    return float(binom.ppf(1 - alpha, n, 1 / n_classes)) / n


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
            valid = np.array(valid, dtype=int)
            train = np.setdiff1d(np.array(train, dtype=int), valid)
            test = np.flatnonzero((subjects == subject) & (sessions == test_session))
            folds.append((f"sub{subject:02d}_test-{test_session}", train, valid, test))
    return folds


def inner_folds(pool, y, sessions, runs, k, by_run, max_folds=6, min_runs=3, n_blocks=5):
    """k (train, valid) splits of the training trials ``pool``, each trial validated once.

    ``by_run``: each validation set is whole runs, consecutive in recording order
    ((session, run) groups split as evenly as np.array_split does), so validation is
    temporally separate from training, as a test session is; k "runs" is one fold per
    run. Otherwise, the k-fold version of SpeechBrain-MOABB's split: every k-th trial of
    each class, in time order.

    k "rule" is version 2's (PROTOCOL.md, ``by_run`` ignored): one fold per run of the
    training data, consecutive runs grouped so there are at most ``max_folds``; with fewer
    than ``min_runs`` runs, ``n_blocks`` contiguous blocks of trials in time.
    """
    pool = np.sort(pool)
    fold_of = np.empty(len(pool), dtype=int)
    if k == "rule":
        n_runs = len(dict.fromkeys((sessions[i], runs[i]) for i in pool))
        if n_runs < min_runs:
            fold_of[:] = np.repeat(np.arange(n_blocks), [len(b) for b in np.array_split(pool, n_blocks)])
            return [(pool[fold_of != j], pool[fold_of == j]) for j in range(n_blocks)]
        k, by_run = min(n_runs, max_folds), True
    if by_run:
        group = [(sessions[i], runs[i]) for i in pool]
        names = list(dict.fromkeys(group))
        k = len(names) if k == "runs" else k
        if len(names) < k:
            raise ValueError(f"{k} validation folds of whole runs, but the training data have {len(names)} runs")
        chunk = {name: j for j, chunk in enumerate(np.array_split(np.arange(len(names)), k)) for name in
                 (names[i] for i in chunk)}
        fold_of[:] = [chunk[g] for g in group]
    else:
        for c in np.unique(y[pool]):
            of_class = np.flatnonzero(y[pool] == c)
            fold_of[of_class] = np.arange(len(of_class)) % k
    return [(pool[fold_of != j], pool[fold_of == j]) for j in range(fold_of.max() + 1)]


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


# ---- EEG augmentation: braindecode.augmentation's transforms on packed batches -----------------------

_MASK32 = 0xFFFFFFFF


def _hash32(x):
    """Chris Wellons' lowbias32 hash, on int64 tensors of 32-bit values (the products wrap modulo 2^64)."""
    x = x & _MASK32
    x = x ^ (x >> 16)
    x = (x * 0x7FEB352D) & _MASK32
    x = x ^ (x >> 15)
    x = (x * 0x846CA68B) & _MASK32
    return x ^ (x >> 16)


def member_uniform(seeds, step, stream, shape, device):
    """U[0, 1) draws (shape[0], K, *shape[1:]), float64, for the K members of a pack.

    Member k's draws are a hash of (seeds[k], step, stream) and their position in
    ``shape``, so a member gets the same numbers in any pack, or alone; and one
    kernel draws for every member, where K generators would take K launches.
    """
    seeds = torch.as_tensor(seeds, device=device, dtype=torch.int64)
    key = _hash32(_hash32(_hash32(seeds) + step) + stream)
    n = torch.arange(math.prod(shape), device=device).reshape(shape[0], 1, *shape[1:])
    bits = _hash32(key.reshape(1, -1, *[1] * (len(shape) - 1)) ^ _hash32(n))
    return bits.double() / 2**32


def _member_normal(seeds, step, stream, shape, device):
    u1, u2 = (member_uniform(seeds, step, stream + i, shape, device) for i in (0, 1))
    return torch.sqrt(-2 * torch.log1p(-u1)) * torch.cos(2 * math.pi * u2)


def _member_gamma(seeds, step, stream, shape, a, device, rounds=8):
    """Gamma(a, 1) draws by Marsaglia and Tsang's method, with ``rounds`` tries (each accepted with p > 0.95)."""
    boost = a < 1  # Gamma(a) = Gamma(a + 1) U^(1 / a)
    d = a + boost - 1 / 3
    c = 1 / math.sqrt(9 * d)
    out = accepted = None
    for r in range(rounds):
        x = _member_normal(seeds, step, stream + 3 * r, shape, device)
        u = member_uniform(seeds, step, stream + 3 * r + 2, shape, device)
        v = (1 + c * x) ** 3
        ok = (v > 0) & (torch.log1p(-u) < 0.5 * x**2 + d - d * v + d * torch.log(v.clamp_min(1e-300)))
        g = d * v
        if out is None:
            out, accepted = g, ok
        else:
            out = torch.where(~accepted & ok, g, out)
            accepted = accepted | ok
    if boost:
        out = out * (1 - member_uniform(seeds, step, stream + 3 * rounds, shape, device)) ** (1 / a)
    return out


# the draws of each transform (any distinct integers)
_APPLY, _FT, _SHIFT, _BANDSTOP, _MASK, _DROPOUT, _MIX_PERM, _MIX_LAM = (1000 * i for i in range(1, 9))


def eeg_augment(x, seeds, step, sample_rate, time_reverse_p=0.0, sign_flip_p=0.0, ft_surrogate_p=0.0,
                ft_surrogate_phase=1.0, frequency_shift_p=0.0, frequency_shift_hz=2.0, bandstop_p=0.0,
                bandstop_hz=1.0, time_mask_p=0.0, time_mask_s=1.0, channels_dropout_p=0.0, channels_dropout_drop=0.2):
    """braindecode.augmentation's transforms on a packed batch x (batch, K, C, T), in this order.

    Each ``*_p`` is a transform's probability of applying to an example (braindecode's
    ``probability``), drawn per example and member; the other arguments are their
    magnitudes. Every draw is member k's own (``member_uniform`` with ``seeds[k]`` and
    the optimizer step), so a member is augmented exactly as it would be alone.

    - TimeReverse.
    - FTSurrogate(phase_noise_magnitude=ft_surrogate_phase, channel_indep=False) and
      BandstopFilter(bandwidth=bandstop_hz), one multiplication of the spectrum: the
      band's centre is U(1 + bw/2, fs/2 - 1 - bw/2) Hz as braindecode's, but its FFT bins
      are zeroed instead of MNE's FIR notch filter (which runs on the CPU, one example
      at a time).
    - FrequencyShift(max_delta_freq=frequency_shift_hz): one shift per example, of
      U(-max, max) Hz, via the analytic signal, as braindecode.
    - SignFlip, SmoothTimeMask(mask_len_samples=round(time_mask_s * sample_rate)) and
      ChannelsDropout(p_drop=channels_dropout_drop), one multiplication by their masks.

    The transforms are linear, so this order only matters through the draws.
    """
    B, K, C, T = x.shape
    device = x.device
    stream = iter(range(_APPLY, _APPLY + 100))

    def applied(p):  # (B, K, 1, 1): whether each example and member gets a transform
        return member_uniform(seeds, step, next(stream), (B, 1, 1), device) < p

    if time_reverse_p > 0:
        x = torch.where(applied(time_reverse_p), x.flip(-1), x)
    spectrum = None  # FTSurrogate's and BandstopFilter's factors, (B, K, 1, T // 2 + 1)
    if ft_surrogate_p > 0:
        n_random = (T - 1) // 2  # positive frequencies but the Nyquist one; it and the mean keep their phase
        phase = member_uniform(seeds, step, _FT, (B, 1, n_random), device)
        rotation = torch.where(applied(ft_surrogate_p), torch.exp(2j * math.pi * ft_surrogate_phase * phase), 1)
        ones = torch.ones(B, K, 1, 1, dtype=rotation.dtype, device=device)
        spectrum = torch.cat([ones, rotation, ones.expand(B, K, 1, T // 2 + 1 - n_random - 1)], -1)
    if bandstop_p > 0 and bandstop_hz > 0:
        u = member_uniform(seeds, step, _BANDSTOP, (B, 1, 1), device)
        centre = 1 + bandstop_hz / 2 + u * (sample_rate / 2 - 2 - bandstop_hz)
        freqs = torch.fft.rfftfreq(T, 1 / sample_rate, device=device, dtype=torch.float64)
        keep = torch.where(applied(bandstop_p), ((freqs - centre).abs() > bandstop_hz / 2).double(), 1.0)
        spectrum = keep if spectrum is None else spectrum * keep
    if spectrum is not None:
        f = torch.fft.rfft(x, dim=-1)
        x = torch.fft.irfft(f * spectrum.to(f.dtype), n=T, dim=-1)
    if frequency_shift_p > 0:
        delta = (2 * member_uniform(seeds, step, _SHIFT, (B,), device) - 1) * frequency_shift_hz
        x = torch.where(applied(frequency_shift_p), _frequency_shift(x, sample_rate, delta.to(x.dtype)), x)
    mask = None  # SignFlip's, SmoothTimeMask's and ChannelsDropout's, multiplied
    if sign_flip_p > 0:
        mask = torch.where(applied(sign_flip_p), -1.0, 1.0)
    if time_mask_p > 0:
        length = round(time_mask_s * sample_rate)
        start = member_uniform(seeds, step, _MASK, (B, 1, 1), device) * (T - length)
        t = torch.arange(T, device=device, dtype=torch.float64)
        s = 1000 / T
        smooth = torch.sigmoid(s * -(t - start)) + torch.sigmoid(s * (t - start - length))
        smooth = torch.where(applied(time_mask_p), smooth, 1.0)
        mask = smooth if mask is None else mask * smooth
    if channels_dropout_p > 0:
        u = member_uniform(seeds, step, _DROPOUT, (B, C), device)
        dropped = torch.sigmoid(1000 * (u - channels_dropout_drop))[..., None]  # braindecode's soft 0/1 mask
        dropped = torch.where(applied(channels_dropout_p), dropped, 1.0)
        mask = dropped if mask is None else mask * dropped
    if mask is not None:
        x = x * mask.to(x.dtype)
    return x


def _frequency_shift(x, sample_rate, delta):
    """braindecode's _frequency_shift with one shift per example and member: x (B, K, C, T), delta (B, K)."""
    T = x.shape[-1]
    n = 2 ** int(np.ceil(np.log2(T)))
    f = torch.fft.fft(F.pad(x, (0, n - T)), dim=-1)
    h = torch.zeros(n, device=x.device, dtype=x.dtype)
    h[0] = 1
    if n % 2 == 0:
        h[n // 2] = 1
        h[1 : n // 2] = 2
    else:
        h[1 : (n + 1) // 2] = 2
    analytic = torch.fft.ifft(f * h, dim=-1)
    t = torch.arange(n, device=x.device, dtype=x.dtype) / sample_rate
    return (analytic * torch.exp(2j * math.pi * delta[..., None, None] * t))[..., :T].real


def mixup_draws(seeds, step, B, K, p, alpha, device):
    """Mixup (Zhang et al., 2018), per member: each example is mixed, with probability p, with
    the example at a random permutation of the batch, by lam ~ Beta(alpha, alpha) (braindecode's
    Mixup with beta_per_sample=True); the others keep lam = 1. Returns (perm, lam), both (B, K)."""
    perm = member_uniform(seeds, step, _MIX_PERM, (B,), device).argsort(0)
    g1 = _member_gamma(seeds, step, _MIX_LAM, (B,), alpha, device)
    g2 = _member_gamma(seeds, step, _MIX_LAM + 100, (B,), alpha, device)
    mixed = member_uniform(seeds, step, _MIX_LAM + 200, (B,), device) < p
    return perm, torch.where(mixed, g1 / (g1 + g2), torch.ones_like(g1))


# ---- model ------------------------------------------------------------------------------------------

def eegnet_kwargs(hp, n_chans, n_outputs, n_times, sample_rate=None):
    """braindecode EEGNet arguments for SpeechBrain-MOABB's EEGNet hyperparameters.

    Temporal sizes can also be given in seconds, which take precedence and need
    ``sample_rate``: ``cnn_temporal_kernel_s`` (the first conv's kernel),
    ``cnn_septemporal_kernel_s`` and ``cnn_septemporal_pool_s`` (after the first
    pooling, so at sample_rate / cnn_spatial_pool), so that they keep their meaning
    when the sample rate is searched.
    """
    F1, D = hp["cnn_temporal_kernels"], hp["cnn_spatial_depth_multiplier"]
    ratio = hp["cnn_septemporal_point_kernels_ratio_"] / 4
    pool1 = hp.get("cnn_spatial_pool", 4)
    kernel = hp.get("cnn_temporal_kernelsize")
    depthwise = None
    if "cnn_septemporal_kernelsize_" in hp:
        depthwise = round(hp["cnn_septemporal_kernelsize_"] * hp.get("max_cnn_spatial_pool", 4) / pool1)
    pool2 = hp.get("cnn_septemporal_pool")
    in_samples = lambda seconds, rate: max(1, round(seconds * rate))
    if "cnn_temporal_kernel_s" in hp:
        kernel = in_samples(hp["cnn_temporal_kernel_s"], sample_rate)
    if "cnn_septemporal_kernel_s" in hp:
        depthwise = in_samples(hp["cnn_septemporal_kernel_s"], sample_rate / pool1)
    if "cnn_septemporal_pool_s" in hp:
        pool2 = in_samples(hp["cnn_septemporal_pool_s"], sample_rate / pool1)
    return dict(
        n_chans=n_chans, n_outputs=n_outputs, n_times=n_times,
        F1=F1, D=D, F2=math.ceil(ratio * F1 * D + 1), kernel_length=kernel,
        depthwise_kernel_length=depthwise, pool1_kernel_size=pool1, pool2_kernel_size=pool2,
        conv_spatial_max_norm=hp.get("cnn_spatial_max_norm", 1.0), final_layer_with_constraint=True,
        norm_rate=hp.get("dense_max_norm", 0.25), drop_prob=hp["dropout"],
        batch_norm_momentum=hp.get("bn_momentum", 0.01), batch_norm_eps=1e-5,  # SpeechBrain's BatchNorm eps
    )


def model_size(kwargs):
    """(parameters, forward FLOPs for one trial) of one member, braindecode's EEGNet built from ``kwargs``.

    Raises if the configuration can't be built or run (e.g. pools longer than the trial).
    """
    from braindecode.models import EEGNet
    from torch.utils.flop_counter import FlopCounterMode

    with torch.random.fork_rng(devices=[]):  # leaves the global RNG as it was
        model = EEGNet(**kwargs).eval()
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(torch.zeros(1, kwargs["n_chans"], kwargs["n_times"]))
    return sum(p.numel() for p in model.parameters()), counter.get_total_flops()


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
        compile=False, eeg_aug=None, sample_rate=None, epoch_seconds=None):
    """Train every member of ``model`` as SpeechBrain-MOABB trains one fold.

    X (n, C, T) are the trials, Xn the same standardized; member k trains on
    ``train_idx[k]``, shuffled by ``seeds[k]``. ``hp`` holds number_of_epochs, lr,
    batch_size and avg_models; ``aug`` the augmentation's arguments (``augment``) or
    None. ``eeg_aug`` instead: ``eeg_augment``'s arguments, plus ``mixup_p`` and
    ``mixup_alpha``; these replace the standardized trials rather than adding augmented
    copies (as braindecode's AugmentedDataLoader), with every draw member k's own, from
    ``seeds[k]``. ``on_eval(epoch, model)`` is called after each epoch in ``eval_epochs``,
    with the model ready to evaluate; training then continues unchanged. A list
    ``epoch_seconds`` gets each epoch's training time, without evaluations (from CUDA
    events on a GPU, so no synchronization per epoch).
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
    if eeg_aug is not None:
        eeg_aug = dict(eeg_aug)
        mixup_p, mixup_alpha = eeg_aug.pop("mixup_p", 0.0), eeg_aug.pop("mixup_alpha", 0.2)
    if aug is None and eeg_aug is None:
        chunks = [model.input_moments(Xn[trials[i : i + 256]].to(dtype)) for i in range(0, len(trials), 256)]
        moments = tuple(torch.cat(parts) for parts in zip(*chunks))
    forward = torch.compile(model, dynamic=False) if compile else model

    def batch_logits(rows, forward=forward, augmented=True, step=0):
        """Logits, the trials' indices (batch, K) and, with mixup, the mixed-in trials' indices and lam."""
        idx = train_idx.gather(1, rows).T  # (batch, K)
        if eeg_aug is not None and augmented:
            x = eeg_augment(Xn[idx].to(dtype), seeds, step, sample_rate, **eeg_aug)
            if mixup_p == 0:
                return forward(x), idx, None
            perm, lam = mixup_draws(seeds, step, *idx.shape, mixup_p, mixup_alpha, device)
            lam = lam.to(dtype)
            x = lam[..., None, None] * x + (1 - lam[..., None, None]) * x.gather(0, perm[..., None, None].expand_as(x))
            return forward(x), idx, (idx.gather(0, perm), lam)
        if aug is None or not augmented:
            if moments is None:
                return forward(Xn[idx].to(dtype)), idx, None
            pos = train_pos.gather(1, rows).T
            return forward(Xn[idx].to(dtype), tuple(m[pos] for m in moments)), idx, None
        x = X[idx].to(dtype)
        x = torch.cat([standardize(x), standardize(augment(x, **aug))])  # originals, then the augmented copies
        return forward(x), torch.cat([idx, idx]), None

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
    timer = _EpochTimer(device) if epoch_seconds is not None else None
    for epoch in range(epochs):
        if timer:
            timer.start()
        model.train()
        order = torch.stack([torch.randperm(n_train, device=device, generator=g) for g in generators])
        for i in range(0, n_train, batch_size):
            if schedule is not None:
                for group in opt.param_groups:
                    group["lr"] = schedule(step, i == 0)
            out, idx, mix = batch_logits(order[:, i : i + batch_size], step=step)
            # their nll_loss with class weights: mean over the batch of w_y * NLL (not normalized by the weights)
            nll = F.cross_entropy(out.float().permute(0, 2, 1), y[idx], reduction="none")  # (batch, K)
            weighted = class_weights[members, y[idx]] * nll
            if mix is not None:  # mixup: lam of the loss for the trial's label, 1 - lam for the mixed-in one's
                other, lam = mix
                nll_other = F.cross_entropy(out.float().permute(0, 2, 1), y[other], reduction="none")
                weighted = lam.float() * weighted + (1 - lam.float()) * class_weights[members, y[other]] * nll_other
            losses = weighted.mean(0)
            opt.zero_grad(set_to_none=True)
            losses.sum().backward()  # each member gets exactly its own gradient
            opt.step()
            step += 1
            epoch_losses[epoch] += losses.detach() * (idx.shape[0] / (2 if aug else 1)) / n_train
        if timer:
            timer.stop()
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
    if timer:
        epoch_seconds.extend(timer.seconds())
    return model, epoch_losses.cpu()


class _EpochTimer:
    """Each epoch's time: CUDA events read once at the end on a GPU, else the wall clock."""

    def __init__(self, device):
        self.cuda, self.marks = device.type == "cuda", []

    def _now(self):
        if not self.cuda:
            return time.perf_counter()
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def start(self):
        self.marks.append([self._now(), None])

    def stop(self):
        self.marks[-1][1] = self._now()

    def seconds(self):
        if not self.cuda:
            return [b - a for a, b in self.marks]
        torch.cuda.synchronize()
        return [a.elapsed_time(b) / 1000 for a, b in self.marks]


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
