"""Version 2 of the search protocol (recipes/moabb/PROTOCOL.md): the window rule from the paradigm block,
the split rules, every EEG channel and the frontal rows, and step 2's enqueued no-augmentation trial with the
one-SE rule counting transforms."""

import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("moabb")
optuna = pytest.importorskip("optuna")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipes" / "moabb"))
import sb_protocol as protocol  # noqa: E402
import sb_search  # noqa: E402
from train import load_hparams  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)

V2 = Path(__file__).resolve().parents[1] / "recipes" / "moabb" / "hparams" / "sb_moabb" / "v2" / "bnci2014001_eegnet.yaml"
BNCI2014001 = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP1",
               "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]


# ---- the window rule --------------------------------------------------------------------------------

def test_window_from_the_paradigm_block():
    paradigm = dict(cue_onset=0.0, cue_offset=1.25, imagery_end=4.0)  # BNCI2014001
    assert protocol.protocol_window(paradigm) == (1.25, 4.0)
    assert protocol.cue_window(paradigm) == (0.0, 1.25)
    persists = dict(cue_onset=1.0, cue_offset="persists", imagery_end=5.0)  # e.g. a cue shown through the imagery
    assert protocol.protocol_window(persists) == (1.5, 5.0)
    assert protocol.cue_window(persists) == (1.0, 1.5)
    with pytest.raises(ValueError):
        protocol.protocol_window(dict(cue_onset=0.0, cue_offset=3.0, imagery_end=3.0))


class FakeRecordings:
    """What apply_paradigm and prepare read from protocol.Recordings, on synthetic trials."""

    def __init__(self, structure, rate=250.0, interval=(2, 6), names=BNCI2014001, eog=3, per_run=8, sample_rate=125,
                 n_times=500):
        self._structure, self._rate = structure, rate
        self.dataset = SimpleNamespace(interval=list(interval))
        self.channel_types = ["eeg"] * len(names) + ["eog"] * eog + ["stim"]
        self.info = SimpleNamespace(ch_names=list(names))
        self.sample_rate = sample_rate
        subjects, sessions, runs = [], [], []
        for subject, by_session in structure.items():
            for session, run_names in by_session.items():
                for run in run_names:
                    subjects += [subject] * per_run
                    sessions += [session] * per_run
                    runs += [run] * per_run
        n = len(subjects)
        self.X = np.arange(n * len(names) * n_times, dtype=np.float32).reshape(n, len(names), n_times) % 7
        self.y = np.arange(n) % 4
        self.meta = np.array(subjects), np.array(sessions), np.array(runs)

    def structure(self):
        return self._structure

    def recording_rates(self):
        return [self._rate]

    def trials(self, fmin, fmax, sample_rate=None):
        return self.X, self.y, *self.meta[:2]

    def runs(self, fmin, fmax, sample_rate=None):
        return self.meta[2]

    def interval(self):
        return [0, float(np.ptp(self.dataset.interval))]


def bnci2014001_like(n_subjects=2):
    return {s: {"0train": [str(r) for r in range(6)], "1test": [str(r) for r in range(6)]}
            for s in range(1, n_subjects + 1)}


def test_apply_paradigm_sets_the_window_and_rate():
    cfg = load_hparams(V2, [])
    settings = dict(sb_search.PROTOCOL, **cfg["protocol"])
    recordings = FakeRecordings(bnci2014001_like())
    rules = sb_search.apply_paradigm(cfg, settings, recordings)
    assert rules["tmin"] == 1.25 and rules["tmax"] == 4.0 and rules["sample_rate"] == 125
    assert cfg["fixed"]["tmin"] == 1.25 and cfg["fixed"]["tmax"] == 4.0 and "twindow" not in cfg["fixed"]
    assert (cfg["fixed"]["fmin"], cfg["fixed"]["fmax"]) == (4.0, 40.0)
    # a dataset recorded below 125 Hz keeps its rate; the paradigm block must say so
    cfg = load_hparams(V2, ["paradigm.recording_rate=100"])
    recordings = FakeRecordings(bnci2014001_like(), rate=100.0)
    assert sb_search.apply_paradigm(cfg, settings, recordings)["sample_rate"] == 100.0
    assert recordings.sample_rate == 100.0 and cfg["sample_rate"] == 100.0
    # version 1's files keep their window from the hyperparameters
    v1 = dict(sb_search.PROTOCOL)
    assert sb_search.apply_paradigm(dict(fixed=dict(tmin=0.0, twindow=4.0)), v1, recordings) is None


@pytest.mark.parametrize("override, recordings_kwargs, message", [
    ("paradigm.sessions=3", {}, "sessions"),
    ("paradigm.runs_per_session=5", {}, "runs_per_session"),
    ("paradigm.recording_rate=512", {}, "recording_rate"),
    ("paradigm.eog_channels=0", {}, "eog_channels"),
    ("paradigm.imagery_end=5.0", {}, "window"),
    ("paradigm.hardware_band=[8,30]", {}, "hardware"),
    ("fixed.fmax=70.0", {}, "Nyquist"),
])
def test_check_paradigm_stops_on_a_mismatch(override, recordings_kwargs, message):
    cfg = load_hparams(V2, [override])
    settings = dict(sb_search.PROTOCOL, **cfg["protocol"])
    with pytest.raises(ValueError, match=message):
        sb_search.apply_paradigm(cfg, settings, FakeRecordings(bnci2014001_like(), **recordings_kwargs))


# ---- the split rules --------------------------------------------------------------------------------

def synthetic(structure, per_run=8):
    recordings = FakeRecordings(structure, per_run=per_run)
    subjects, sessions, runs = recordings.meta
    return recordings, subjects, sessions, runs


def contiguous(idx):
    idx = np.sort(idx)
    return np.all(np.diff(idx) == 1)


@pytest.mark.parametrize("n_runs, n_inner", [(12, 6), (2, 5), (1, 5)])
def test_splits_with_one_session(n_runs, n_inner):
    """One session: two pseudo-sessions of contiguous runs (or halves of the one run); inner folds by rule."""
    structure = {1: {"0": [f"run{r:02d}" for r in range(n_runs)]}, 2: {"0": [f"run{r:02d}" for r in range(n_runs)]}}
    recordings, subjects, sessions, runs = synthetic(structure)
    pseudo = protocol.pseudo_sessions(subjects, sessions, runs)
    assert sorted(set(pseudo)) == ["0-half1", "0-half2"] == protocol.outer_session_names(structure)
    for s in (1, 2):
        idx = np.flatnonzero(subjects == s)
        first, second = idx[pseudo[idx] == "0-half1"], idx[pseudo[idx] == "0-half2"]
        assert contiguous(first) and contiguous(second) and first.max() < second.min()
        assert abs(len(first) - len(second)) <= 8  # whole runs: halves differ by at most one run
        if n_runs > 1:
            assert set(runs[first]).isdisjoint(runs[second])
    folds = protocol.leave_one_session_out(recordings.y, subjects, pseudo, 0.2)
    assert len(folds) == 4
    for name, train, valid, test in folds:
        pool = np.union1d(train, valid)
        assert np.intersect1d(pool, test).size == 0
        inner = protocol.inner_folds(pool, recordings.y, pseudo, runs, "rule", True)
        assert len(inner) == n_inner
        assert sorted(np.concatenate([v for _, v in inner]).tolist()) == pool.tolist()  # each validated once
        for tr, va in inner:
            assert contiguous(va)  # whole consecutive runs, or a block in time
            if n_runs >= 6:  # 6 runs per pseudo-session: one fold per run
                assert len(set(runs[va])) == 1 and set(runs[tr]).isdisjoint(runs[va])


@pytest.mark.parametrize("n_sessions, n_runs, n_inner, runs_per_fold", [
    (2, 6, 6, 1),  # BNCI2014001: one fold per run
    (2, 12, 6, 2),  # at most 6: consecutive runs in pairs
    (3, 2, 4, 1),  # 4 runs in the two training sessions
    (2, 3, 3, 1),  # 3 runs: still one fold per run
    (2, 2, 5, None),  # fewer than 3 runs: 5 blocks in time
    (2, 1, 5, None),
])
def test_inner_folds_rule(n_sessions, n_runs, n_inner, runs_per_fold):
    structure = {1: {f"{s}": [str(r) for r in range(n_runs)] for s in range(n_sessions)}}
    recordings, subjects, sessions, runs = synthetic(structure)
    assert protocol.outer_session_names(structure) == [str(s) for s in range(n_sessions)]
    assert np.array_equal(protocol.pseudo_sessions(subjects, sessions, runs), sessions)  # several sessions: kept
    for name, train, valid, test in protocol.leave_one_session_out(recordings.y, subjects, sessions, 0.2):
        pool = np.union1d(train, valid)
        inner = protocol.inner_folds(pool, recordings.y, sessions, runs, "rule", False)
        assert len(inner) == n_inner
        assert sorted(np.concatenate([v for _, v in inner]).tolist()) == pool.tolist()
        groups = [list(dict.fromkeys(zip(sessions[va], runs[va]))) for _, va in inner]
        for (tr, va), group in zip(inner, groups):
            assert contiguous(va) or runs_per_fold is not None
            if runs_per_fold is not None:
                assert len(group) == runs_per_fold
                assert set(zip(sessions[tr], runs[tr])).isdisjoint(group)
        if runs_per_fold is None:
            assert [len(va) for _, va in inner] == [len(b) for b in np.array_split(pool, 5)]


def test_cells_with_the_rules():
    """make_cells with outer_folds and inner_folds by rule, through prepare, on a one-session dataset."""
    structure = {s: {"0": [str(r) for r in range(4)]} for s in (1, 2, 3)}
    recordings = FakeRecordings(structure)
    cfg = dict(sample_rate=125, n_classes=4, valid_ratio=0.2)
    settings = dict(sb_search.PROTOCOL, window="paradigm", channels="all", outer_folds="rule", inner_folds="rule")
    hp = dict(fmin=4.0, fmax=40.0, tmin=1.25, tmax=4.0)
    args = SimpleNamespace(device=torch.device("cpu"), test_session=None)
    data = sb_search.prepare(hp, recordings, cfg, args, settings)
    assert len(data["folds"]) == 6  # 3 subjects x 2 pseudo-sessions
    cells = sb_search.make_cells(data, [1], settings)
    assert len(cells) == 6 * 5  # 2 runs of training data: 5 blocks in time
    args.test_session = "0-half2"
    data = sb_search.prepare(hp, recordings, cfg, args, settings)
    assert [f[0] for f in data["folds"]] == [f"sub0{s}_test-0-half2" for s in (1, 2, 3)]


# ---- channels ---------------------------------------------------------------------------------------

def test_electrode_rows():
    assert [protocol.electrode_row(n) for n in ("Fp1", "Fpz", "AF3", "AFz", "F3", "Fz", "FC3", "FCz", "FT7", "Cz",
                                                "CPz", "POz", "FFC3h", "FP2")] == \
        ["Fp", "Fp", "AF", "AF", "F", "F", "FC", "FC", "FT", "C", "CP", "PO", "FFC", "FP"]
    names = ["Fp1", "FP2", "AF3", "F3", "Fz", "FC3", "FT7", "C3", "Cz", "Oz"]
    assert [names[i] for i in protocol.without_rows(names)] == ["FC3", "FT7", "C3", "Cz", "Oz"]
    assert [BNCI2014001[i] for i in protocol.without_rows(BNCI2014001)] == BNCI2014001[1:]  # only Fz is frontal


def test_prepare_channels():
    """`channels: all` keeps every EEG channel; drop_rows removes the frontal ones; adjacency is version 1's."""
    import mne

    recordings = FakeRecordings(bnci2014001_like(1))
    info = mne.create_info(BNCI2014001, 250.0, "eeg")
    info.set_montage("standard_1005")
    recordings.info = info
    recordings.channels = lambda n: protocol.Recordings.channels(recordings, n)
    cfg = dict(sample_rate=125, n_classes=4, valid_ratio=0.2)
    args = SimpleNamespace(device=torch.device("cpu"), test_session=None)
    hp = dict(fmin=4.0, fmax=40.0, tmin=1.25, tmax=4.0, n_steps_channel_selection=1)
    v2 = dict(sb_search.PROTOCOL, channels="all")
    data = sb_search.prepare(hp, recordings, cfg, args, v2)
    assert data["channels"] == BNCI2014001 and data["X"].shape[1] == 22
    assert data["X"].shape[2] == round((4.0 - 1.25) * 125) - 1  # SpeechBrain's crop: nearest samples, tmax excluded
    data = sb_search.prepare(dict(hp, drop_rows=["Fp", "AF", "F"]), recordings, cfg, args, v2)
    assert data["channels"] == BNCI2014001[1:]
    assert torch.equal(data["X"], sb_search.prepare(hp, recordings, cfg, args, v2)["X"][:, 1:])
    data = sb_search.prepare(hp, recordings, cfg, args, sb_search.PROTOCOL)  # version 1: channels around Cz
    assert data["channels"] == [BNCI2014001[i] for i in recordings.channels(1)] and len(data["channels"]) < 22


# ---- step 2: no augmentation first, and the one-SE rule on the number of transforms ------------------

def test_enqueued_no_augmentation_trial(tmp_path):
    cfg = load_hparams(V2, [])
    step2 = sb_search.steps_of(cfg)["step2"]
    values = sb_search.first_trial(step2)
    assert values == {k: 0.0 for k in step2["search"] if k.endswith("_p")} and len(values) == 7
    assert sb_search.first_trial(sb_search.steps_of(cfg)["step1"]) is None
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(tmp_path / "j")))
    study = optuna.create_study(storage=storage, study_name="step2", direction="maximize",
                                sampler=optuna.samplers.TPESampler(n_startup_trials=2, seed=0))
    assert sb_search.enqueue_first(study, values)
    assert not sb_search.enqueue_first(study, values)  # already waiting
    trial = study.ask()
    suggested = sb_search.suggest(trial, step2["search"])
    assert {k: suggested[k] for k in values} == values and sb_search.n_transforms(suggested) == 0
    assert set(suggested) == set(step2["search"])  # the magnitudes are sampled (and have no effect)
    study.tell(trial, 0.5)
    assert not sb_search.enqueue_first(study, values)  # done: a resumed search doesn't redo it
    trial = study.ask()  # the next trial is TPE's
    assert trial.number == 1
    with pytest.raises(ValueError):
        sb_search.first_trial(dict(first_trial="all_off", search=dict(lr="choices([0.1])")))


def test_one_se_counts_transforms():
    def trial(number, value, se, **p):
        values = dict(dict(aug_time_reverse_p=0.0, aug_sign_flip_p=0.0, aug_ft_surrogate_phase=0.7), **p)
        return SimpleNamespace(number=number, values=[value], user_attrs=dict(se=se, params=1000, values=values))

    none = trial(0, 0.70, 0.01)
    one = trial(1, 0.705, 0.01, aug_ft_surrogate_p=0.5)
    two = trial(2, 0.71, 0.01, aug_ft_surrogate_p=0.5, aug_time_reverse_p=0.5)
    assert [sb_search.n_transforms(t.user_attrs["values"]) for t in (none, one, two)] == [0, 1, 2]
    # augmentation better by no more than one SE: no augmentation is kept
    assert sb_search.select([none, one, two], "one_se", "n_transforms")[0].number == 0
    # better by more than one SE: the fewest transforms within one SE of the best
    better = trial(3, 0.74, 0.01, aug_ft_surrogate_p=0.5, aug_time_reverse_p=0.5, aug_sign_flip_p=0.5)
    fewer = trial(4, 0.731, 0.01, aug_ft_surrogate_p=0.5)
    assert sb_search.select([none, one, two, better, fewer], "one_se", "n_transforms")[0].number == 4
    assert sb_search.select([none, one, two, better, fewer], "best", "n_transforms")[0].number == 3
    # ties in the count go to the higher score
    tie = trial(5, 0.735, 0.01, aug_time_reverse_p=0.5)
    assert sb_search.select([none, better, fewer, tie], "one_se", "n_transforms")[0].number == 5


def test_v1_and_reproduction_files_keep_their_settings():
    """Version 1's and the reproduction's yaml files don't turn on version 2's options."""
    root = V2.parents[1]
    for path in (root / "bnci2014001_eegnet.yaml", root / "improved" / "bnci2014001_eegnet.yaml"):
        cfg = load_hparams(path, [])
        settings = dict(sb_search.PROTOCOL, **cfg.get("protocol", {}))
        assert (settings["window"], settings["channels"], settings["outer_folds"]) == ("hparams", "adjacency",
                                                                                        "sessions")
        assert settings["inner_folds"] != "rule"
        assert all(s.get("first_trial") is None and s.get("size") is None for s in sb_search.steps_of(cfg).values())


def test_binomial_threshold():
    """Combrisson and Jerbi (2015): about 0.27 for 4 classes and 1152 trials, above 0.5 for 2 classes and few."""
    from scipy.stats import binom

    for n, c in ((1152, 4), (48, 4), (40, 2), (100, 2)):
        t = protocol.binomial_threshold(n, c)
        assert binom.sf(round(t * n), n, 1 / c) <= 0.05 < binom.sf(round(t * n) - 1, n, 1 / c)
    assert 0.26 < protocol.binomial_threshold(1152, 4) < 0.28
    assert protocol.binomial_threshold(40, 2) > 0.6


def test_fit_times_each_epoch():
    """fit(epoch_seconds=[...]) gets one positive time per epoch, and training is unchanged."""
    torch.manual_seed(0)
    X = torch.randn(24, 3, 64)
    Xn = protocol.standardize(X)
    y = torch.arange(24) % 2
    hp = dict(cnn_temporal_kernels=4, cnn_spatial_depth_multiplier=1, cnn_septemporal_point_kernels_ratio_=4,
              dropout=0.0, cnn_temporal_kernelsize=8, cnn_septemporal_kernelsize_=4, cnn_septemporal_pool=2)
    kwargs = protocol.eegnet_kwargs(hp, 3, 2, 64)
    train_idx = torch.stack([torch.arange(16), torch.arange(8, 24)])
    settings = dict(number_of_epochs=3, lr=0.01, batch_size=8, avg_models=1)
    seconds = []
    _, timed = protocol.fit(protocol.build(kwargs, [1, 2], "cpu"), X, Xn, y, train_idx, [1, 2], settings,
                            "schedule-free", epoch_seconds=seconds)
    _, untimed = protocol.fit(protocol.build(kwargs, [1, 2], "cpu"), X, Xn, y, train_idx, [1, 2], settings,
                              "schedule-free")
    assert len(seconds) == 3 and all(s > 0 for s in seconds)
    torch.testing.assert_close(timed, untimed)
