"""recipes/moabb/sb_protocol.py against SpeechBrain-MOABB's own code (copied below from
speechbrain/benchmarks, benchmarks/MOABB, and speechbrain e602161), and packed training
against its members trained alone (float64)."""

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("moabb")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipes" / "moabb"))
import sb_protocol as protocol  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ---- their code, verbatim but for formatting ---------------------------------------------------

class CyclicLRScheduler:  # speechbrain.nnet.schedulers, mode "triangular"
    def __init__(self, base_lr, max_lr, step_size):
        self.base_lr, self.max_lr, self.step_size = base_lr, max_lr, step_size
        self.clr_iterations = 0.0

    def __call__(self, epoch):
        old_lr = self.current_lr
        new_lr = self.clr(self.clr_iterations + 1)
        return old_lr, new_lr

    def clr(self, clr_iterations):
        cycle = math.floor(1 + clr_iterations / (2 * self.step_size))
        x = abs(clr_iterations / self.step_size - 2 * cycle + 1)
        return self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))

    def on_batch_end(self, opt):
        self.clr_iterations += 1
        lr = self.clr(self.clr_iterations)
        current_lr = opt.param_groups[0]["lr"]
        for param_group in opt.param_groups:
            param_group["lr"] = lr
        self.current_lr = current_lr


def get_idx_train_valid_classbalanced(idx_train, valid_ratio, y):  # utils/dataio_iterators.py
    idx_train = np.array(idx_train)
    nclasses = y[idx_train].max() + 1
    idx_valid = []
    for c in range(nclasses):
        to_select_c = idx_train[np.where(y[idx_train] == c)[0]]
        idx = np.linspace(0, to_select_c.shape[0] - 1, round(valid_ratio * to_select_c.shape[0]))
        idx = np.floor(idx).astype(int)
        tmp_idx_valid_c = to_select_c[idx]
        idx_valid.extend(tmp_idx_valid_c)
    idx_valid = np.array(idx_valid)
    idx_train = np.setdiff1d(idx_train, idx_valid)
    return idx_train, idx_valid


def get_neighbour_channels(adjacency_mtx, ch_names, n_steps=1, seed_nodes=["Cz"]):  # dataio_iterators.py
    sel_channels = []
    for i in range(n_steps):
        tmp_sel_channels = []
        for node in seed_nodes:
            idx_node = np.where(node == np.array(ch_names))[0][0]
            idx_linked_nodes = np.where(adjacency_mtx[idx_node, :] > 0)[0]
            linked_channels = np.array(ch_names)[idx_linked_nodes]
            tmp_sel_channels.extend(list(linked_channels))
        seed_nodes = tmp_sel_channels
        sel_channels.extend(tmp_sel_channels)
    sel_channels = np.unique(sel_channels)
    return sel_channels


def cutcat(waveforms, num_segments):  # speechbrain.augment.time_domain.CutCat, (batch, time, ...)
    if waveforms.shape[0] > 1:
        waveforms_rolled = torch.roll(waveforms, shifts=1, dims=0)
        idx_cut = torch.linspace(0, waveforms.shape[1], num_segments + 1, dtype=torch.int)
        for i in range(idx_cut.shape[0] - 1):
            if i % 2 == 1:
                start = idx_cut[i]
                stop = idx_cut[i + 1]
                waveforms[:, start:stop, ...] = waveforms_rolled[:, start:stop, ...]
    return waveforms


def rand_amp(waveforms, amp):  # RandAmp with amp_low = amp_high = amp
    abs_max, _ = torch.max(torch.abs(waveforms), dim=1, keepdim=True)
    return waveforms / abs_max * amp


# ---- tests ---------------------------------------------------------------------------------------

def test_cyclic_lr_as_moabbbrain_drives_it():
    """MOABBBrain: on_batch_end after every batch, then the validation stage sets clr(n + 1)."""
    max_lr, step_size, steps_per_epoch = 0.01, 7, 5
    scheduler = CyclicLRScheduler(1e-8, max_lr, step_size)
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=max_lr)
    ours = protocol.cyclic_lr(1e-8, max_lr, step_size)
    step = 0
    for epoch in range(1, 7):
        for i in range(steps_per_epoch):
            assert ours(step, i == 0) == pytest.approx(opt.param_groups[0]["lr"], rel=1e-12, abs=1e-15)
            scheduler.on_batch_end(opt)
            step += 1
        _, new_lr = scheduler(epoch)
        opt.param_groups[0]["lr"] = new_lr


def test_leave_one_session_out_validation_split():
    rng = np.random.default_rng(0)
    subjects = np.repeat([1, 2], 120)
    sessions = np.tile(np.repeat(["0train", "1test", "2extra"], 40), 2)
    y = rng.integers(0, 3, 240)
    folds = protocol.leave_one_session_out(y, subjects, sessions, 0.2)
    assert len(folds) == 6
    for name, train, valid, test in folds:
        subject = int(name[3:5])
        test_session = name.split("test-")[1]
        their_train, their_valid = [], []
        for session in sorted({"0train", "1test", "2extra"} - {test_session}):
            tr, va = get_idx_train_valid_classbalanced(
                np.flatnonzero((subjects == subject) & (sessions == session)), 0.2, y)
            their_train.extend(tr)
            their_valid.extend(va)
        assert sorted(train) == sorted(their_train)
        assert sorted(valid) == sorted(their_valid)
        assert sorted(test) == list(np.flatnonzero((subjects == subject) & (sessions == test_session)))


@pytest.mark.parametrize("n_steps", [1, 2, 3])
def test_channel_selection(n_steps):
    import mne

    names = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP1",
             "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]  # BNCI2014001's
    info = mne.create_info(names, 250.0, "eeg")
    info.set_montage("standard_1005")
    recordings = protocol.Recordings.__new__(protocol.Recordings)
    recordings.info = info
    adjacency, adjacency_names = mne.channels.find_ch_adjacency(info, ch_type="eeg")
    theirs = get_neighbour_channels(adjacency.toarray(), list(adjacency_names), n_steps)
    assert [names[i] for i in recordings.channels(n_steps)] == [n for n in names if n in set(theirs)]


def test_augment_matches_speechbrain():
    """Every member puts each augmentation on one of the 4 chunks, as SpeechBrain's would compute it."""
    B, K, C, T = 18, 5, 3, 40  # chunks of 4, 5, 4, 5
    x = torch.randn(B, K, C, T, dtype=torch.float64)
    shift, snr = 3, 6.0  # and RandAmp's amplitude is 1 with amp_delta 0
    out = protocol.augment(x, max_num_segments=2, amp_delta=0.0, min_shift=shift, max_shift=shift,
                           snr_low=snr, snr_high=snr)
    bounds = [0, 4, 9, 13, 18]
    factor = 1 / (10 ** (snr / 20) + 1)
    for k in range(K):
        used = []
        for j in range(4):
            sb = x[bounds[j] : bounds[j + 1], k].transpose(1, 2).clone()  # (chunk, T, C), their layout
            got = out[bounds[j] : bounds[j + 1], k].transpose(1, 2)
            candidates = {
                "cutcat": cutcat(sb.clone(), 2),
                "randamp": rand_amp(sb.clone(), 1.0),
                "shift": torch.roll(sb, shift, dims=1),
            }
            match = [name for name, ref in candidates.items() if torch.allclose(got, ref)]
            if not match:  # white noise at a fixed SNR: what's left has rms factor * rms(x) per channel
                noise = got - sb * (1 - factor)
                torch.testing.assert_close(noise.pow(2).mean(1).sqrt(), factor * sb.pow(2).mean(1).sqrt())
                match = ["noise"]
            used += match
        assert sorted(used) == ["cutcat", "noise", "randamp", "shift"]


def test_init_like_speechbrain():
    from braindecode.models import EEGNet

    model = protocol.init_like_speechbrain(EEGNet(n_chans=8, n_outputs=3, n_times=200, final_layer_with_constraint=True))
    for name, tensor in model.state_dict().items():
        if name.endswith("bias") or name.endswith("running_mean"):
            assert (tensor == 0).all(), name
        elif "bnorm" in name and name.endswith("weight"):
            assert (tensor == 1).all(), name
    spatial = model.conv_spatial.parametrizations.weight.original  # (F1 * D, 1, C, 1): fan_in C, fan_out F1 * D * C
    bound = math.sqrt(6 / (8 + 16 * 8))
    assert spatial.abs().max() <= bound and spatial.abs().max() > 0.9 * bound


@pytest.mark.parametrize("optimizer, avg_models", [("adam", 3), ("adam", 1), ("schedule-free", 1)])
def test_pack_equals_members(optimizer, avg_models):
    g = torch.Generator().manual_seed(0)
    X = torch.randn(60, 6, 120, generator=g, dtype=torch.float64) * 5 + 1
    Xn = protocol.standardize(X)
    y = torch.randint(3, (60,), generator=g)
    train_idx = torch.stack([torch.arange(0, 30), torch.randperm(60, generator=g)[:30]])
    seeds = [3, 4]
    hp = dict(cnn_temporal_kernels=4, cnn_spatial_depth_multiplier=2, cnn_septemporal_point_kernels_ratio_=4,
              cnn_temporal_kernelsize=25, cnn_septemporal_kernelsize_=5, cnn_septemporal_pool=4, dropout=0.0,
              bn_momentum=0.1)
    kwargs = protocol.eegnet_kwargs(hp, 6, 3, 120)
    settings = dict(number_of_epochs=4, lr=5e-3, batch_size=8, avg_models=avg_models)
    seen = {}

    def on_eval(name):
        return lambda epoch, model: seen.setdefault(name, {}).update(
            {epoch: protocol.predict(model, Xn, train_idx[: model.n_models], batch_size=16)})

    packed = protocol.build(kwargs, seeds, "cpu").double()
    packed, losses = protocol.fit(packed, X, Xn, y, train_idx, seeds, settings, optimizer, eval_epochs=[2, 4],
                                  on_eval=on_eval("packed"))
    for k, seed in enumerate(seeds):
        alone = protocol.build(kwargs, [seed], "cpu").double()
        alone, alone_losses = protocol.fit(alone, X, Xn, y, train_idx[k : k + 1], [seed], settings, optimizer,
                                           eval_epochs=[2, 4], on_eval=on_eval(k))
        torch.testing.assert_close(losses[:, k : k + 1], alone_losses)
        for key, value in alone.member_state_dict(0).items():
            if not key.endswith("num_batches_tracked"):
                torch.testing.assert_close(packed.member_state_dict(k)[key], value, msg=key)
        for epoch in (2, 4):  # the evaluations train_idx[:1] of the pack-of-one see member 0's trials
            if k == 0:
                torch.testing.assert_close(seen["packed"][epoch][:, :1], seen[k][epoch])


# ---- the improved protocol's pieces (PROTOCOL_IDEAS.md) ---------------------------------------------

def test_hash_matches_python():
    def lowbias32(x):
        x &= 0xFFFFFFFF
        x ^= x >> 16
        x = (x * 0x7FEB352D) & 0xFFFFFFFF
        x ^= x >> 15
        x = (x * 0x846CA68B) & 0xFFFFFFFF
        return x ^ (x >> 16)

    values = torch.randint(0, 2**32, (2000,), dtype=torch.int64, generator=torch.Generator().manual_seed(0))
    assert protocol._hash32(values).tolist() == [lowbias32(v) for v in values.tolist()]


def test_member_draws_dont_depend_on_the_pack():
    seeds = [5, 9, 5, 1986]
    packed = protocol.member_uniform(seeds, 17, 3, (8, 4), "cpu")
    assert packed.shape == (8, 4, 4)
    for k, seed in enumerate(seeds):
        torch.testing.assert_close(packed[:, k], protocol.member_uniform([seed], 17, 3, (8, 4), "cpu")[:, 0])
    assert not torch.equal(packed[:, 0], packed[:, 1])  # other seeds, other draws
    assert not torch.equal(packed, protocol.member_uniform(seeds, 18, 3, (8, 4), "cpu"))  # and every step
    u = protocol.member_uniform(list(range(50)), 0, 0, (2000,), "cpu")
    assert 0 <= u.min() and u.max() < 1 and abs(u.mean() - 0.5) < 0.01 and abs(u.var() - 1 / 12) < 0.002


def test_beta_draws():
    alpha = 0.4
    g1 = protocol._member_gamma(list(range(20)), 3, 0, (5000,), alpha, "cpu")
    g2 = protocol._member_gamma(list(range(20)), 3, 500, (5000,), alpha, "cpu")
    lam = g1 / (g1 + g2)
    assert abs(g1.mean() - alpha) < 0.02  # Gamma(a, 1) has mean a
    assert abs(lam.mean() - 0.5) < 0.01 and abs(lam.var() - 1 / (4 * (2 * alpha + 1))) < 0.005


EEG_AUG = dict(time_reverse_p=0.5, sign_flip_p=0.5, ft_surrogate_p=0.5, ft_surrogate_phase=0.9, frequency_shift_p=0.5,
               frequency_shift_hz=2.7, bandstop_p=0.5, bandstop_hz=1.0, time_mask_p=0.5, time_mask_s=0.3,
               channels_dropout_p=0.5, channels_dropout_drop=0.4)


def test_eeg_augment_pack_equals_members():
    x = torch.randn(12, 4, 3, 101, dtype=torch.float64)
    seeds = [3, 4, 3, 8]
    packed = protocol.eeg_augment(x, seeds, 7, 50.0, **EEG_AUG)
    for k, seed in enumerate(seeds):
        alone = protocol.eeg_augment(x[:, k : k + 1], [seed], 7, 50.0, **EEG_AUG)
        torch.testing.assert_close(packed[:, k : k + 1], alone)
    changed = (packed - x).abs().flatten(2).amax(-1) > 1e-9  # (B, K): some transform applied
    assert changed.any() and not changed.all()


class _Draws(np.random.RandomState):
    """A RandomState whose uniform() returns given draws, to give braindecode ours."""

    def __init__(self, draws):
        super().__init__(0)
        self.draws = draws

    def uniform(self, low=0.0, high=1.0, size=None):
        return low + (high - low) * self.draws


@pytest.mark.parametrize("transform", ["time_reverse", "sign_flip", "ft_surrogate", "frequency_shift", "time_mask",
                                       "channels_dropout"])
def test_eeg_transforms_match_braindecode(transform, monkeypatch):
    """Each transform alone, applied to every example (p = 1), against braindecode's functional with our draws."""
    from braindecode.augmentation import functional as bf

    B, K, C, T, fs, step = 6, 3, 4, 100, 50.0, 11
    x = torch.randn(B, K, C, T)
    seeds = [1, 2, 3]
    y = torch.zeros(B)
    out = protocol.eeg_augment(x, seeds, step, fs, **{f"{transform}_p": 1.0}, ft_surrogate_phase=0.7,
                               frequency_shift_hz=2.5, time_mask_s=0.4, channels_dropout_drop=0.3)
    for k in range(K):
        if transform == "time_reverse":
            ref = bf.time_reverse(x[:, k], y)[0]
        elif transform == "sign_flip":
            ref = bf.sign_flip(x[:, k], y)[0]
        elif transform == "ft_surrogate":
            u = protocol.member_uniform(seeds, step, protocol._FT, (B, 1, (T - 1) // 2), "cpu")[:, k]
            phase = 2j * np.pi * u.numpy()
            full = lambda b, c, n, device, random_state: torch.from_numpy(np.concatenate(
                [np.zeros((b, c, 1)), phase, np.zeros((b, c, 1)), -phase[..., ::-1]], -1))  # T even
            monkeypatch.setitem(bf._new_random_fft_phase, 0, full)
            ref = bf.ft_surrogate(x[:, k], y, 0.7, channel_indep=False)[0]
        elif transform == "frequency_shift":
            u = protocol.member_uniform(seeds, step, protocol._SHIFT, (B,), "cpu")[:, k]
            ref = bf.frequency_shift(x[:, k], y, (2 * u - 1).float() * 2.5, fs)[0]
        elif transform == "time_mask":
            u = protocol.member_uniform(seeds, step, protocol._MASK, (B, 1, 1), "cpu")[:, k, 0, 0]
            length = round(0.4 * fs)
            ref = bf.smooth_time_mask(x[:, k], y, u * (T - length), length)[0]
        else:
            u = protocol.member_uniform(seeds, step, protocol._DROPOUT, (B, C), "cpu")[:, k]
            ref = bf.channels_dropout(x[:, k], y, 0.3, random_state=_Draws(u.numpy()))[0]
        torch.testing.assert_close(out[:, k], ref.float(), atol=2e-5, rtol=1e-4)


def test_bandstop_removes_the_band_only():
    B, K, C, T, fs = 5, 2, 3, 200, 100.0
    x = torch.randn(B, K, C, T, dtype=torch.float64)
    out = protocol.eeg_augment(x, [4, 5], 2, fs, bandstop_p=1.0, bandstop_hz=2.0)
    centre = 1 + 1.0 + protocol.member_uniform([4, 5], 2, protocol._BANDSTOP, (B, 1, 1), "cpu") * (fs / 2 - 2 - 2.0)
    freqs = torch.fft.rfftfreq(T, 1 / fs, dtype=torch.float64)
    band = ((freqs - centre).abs() <= 1.0).expand(B, K, C, -1)
    f_in, f_out = torch.fft.rfft(x), torch.fft.rfft(out)
    assert band.sum(-1).min() >= 3  # 2 Hz at 0.5 Hz resolution
    assert f_out.abs()[band].max() < 1e-9
    torch.testing.assert_close(f_out[~band], f_in[~band])


def test_mixup_matches_braindecode():
    from braindecode.augmentation import functional as bf

    B, K = 8, 3
    x = torch.randn(B, K, 2, 30)
    perm, lam = protocol.mixup_draws([1, 2, 3], 4, B, K, 0.5, 0.4, "cpu")
    assert (lam == 1).any() and (lam < 1).any()
    for k in range(K):
        assert sorted(perm[:, k].tolist()) == list(range(B))
        mixed = lam[:, k, None, None].float() * x[:, k] + (1 - lam[:, k, None, None].float()) * x[perm[:, k], k]
        torch.testing.assert_close(mixed, bf.mixup(x[:, k], torch.zeros(B), lam[:, k].float(), perm[:, k])[0])


@pytest.mark.parametrize("mixup_p", [0.0, 0.5])
def test_pack_equals_members_with_eeg_augmentation(mixup_p):
    g = torch.Generator().manual_seed(0)
    X = torch.randn(60, 6, 120, generator=g, dtype=torch.float64) * 5 + 1
    Xn = protocol.standardize(X)
    y = torch.randint(3, (60,), generator=g)
    train_idx = torch.stack([torch.arange(0, 30), torch.randperm(60, generator=g)[:30], torch.arange(30, 60)])
    seeds = [3, 4, 3]
    hp = dict(cnn_temporal_kernels=4, cnn_spatial_depth_multiplier=2, cnn_septemporal_point_kernels_ratio_=4,
              cnn_temporal_kernelsize=25, cnn_septemporal_kernelsize_=5, cnn_septemporal_pool=4, dropout=0.0,
              bn_momentum=0.1)
    kwargs = protocol.eegnet_kwargs(hp, 6, 3, 120)
    settings = dict(number_of_epochs=3, lr=5e-3, batch_size=8, avg_models=1)
    eeg_aug = dict(EEG_AUG, mixup_p=mixup_p, mixup_alpha=0.4)
    packed = protocol.build(kwargs, seeds, "cpu").double()
    packed, losses = protocol.fit(packed, X, Xn, y, train_idx, seeds, settings, "schedule-free", eeg_aug=eeg_aug,
                                  sample_rate=40.0)
    for k, seed in enumerate(seeds):
        alone = protocol.build(kwargs, [seed], "cpu").double()
        alone, alone_losses = protocol.fit(alone, X, Xn, y, train_idx[k : k + 1], [seed], settings, "schedule-free",
                                           eeg_aug=eeg_aug, sample_rate=40.0)
        torch.testing.assert_close(losses[:, k : k + 1], alone_losses)
        for key, value in alone.member_state_dict(0).items():
            if not key.endswith("num_batches_tracked"):
                torch.testing.assert_close(packed.member_state_dict(k)[key], value, msg=key)


def test_inner_folds():
    rng = np.random.default_rng(0)
    sessions = np.repeat(["0train", "1test"], 48)
    runs = np.tile(np.repeat(["0", "1", "2", "3", "4", "5"], 8), 2)
    y = rng.integers(0, 4, 96)
    pool = np.flatnonzero(sessions == "0train")
    for by_run, k in ((False, 5), (True, 3), (True, 4), (True, 6), (True, "runs")):
        folds = protocol.inner_folds(pool, y, sessions, runs, k, by_run)
        assert len(folds) == (6 if k == "runs" else k)
        assert sorted(np.concatenate([v for _, v in folds]).tolist()) == pool.tolist()  # each validated once
        for train, valid in folds:
            assert sorted(np.concatenate([train, valid]).tolist()) == pool.tolist()
            if by_run:  # whole runs, consecutive
                in_valid = sorted(set(runs[valid].tolist()))
                assert set(runs[train].tolist()).isdisjoint(in_valid)
                assert [int(r) for r in in_valid] == list(range(int(in_valid[0]), int(in_valid[-1]) + 1))
            else:  # every k-th trial of each class
                for c in range(4):
                    of_class = pool[y[pool] == c]
                    assert abs(np.isin(of_class, valid).sum() - len(of_class) / k) < 1
    with pytest.raises(ValueError):
        protocol.inner_folds(pool, y, sessions, runs, 7, True)


def test_kernels_in_seconds_and_model_size():
    from braindecode.models import EEGNet

    hp = dict(cnn_temporal_kernels=8, cnn_spatial_depth_multiplier=2, cnn_septemporal_point_kernels_ratio_=4,
              dropout=0.5, cnn_temporal_kernel_s=0.5, cnn_septemporal_kernel_s=0.5, cnn_septemporal_pool_s=0.25)
    for rate in (128, 250):
        kwargs = protocol.eegnet_kwargs(hp, 22, 4, 4 * rate, rate)
        assert kwargs["kernel_length"] == round(rate / 2)
        assert kwargs["depthwise_kernel_length"] == round(rate / 4 / 2)
        assert kwargs["pool2_kernel_size"] == round(rate / 4 / 4)
        params, flops = protocol.model_size(kwargs)
        assert params == sum(p.numel() for p in EEGNet(**kwargs).parameters()) and flops > 0
    state = torch.random.get_rng_state()
    protocol.model_size(kwargs)
    assert torch.equal(state, torch.random.get_rng_state())  # doesn't touch the global RNG
