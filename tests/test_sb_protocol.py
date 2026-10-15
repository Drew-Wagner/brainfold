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
