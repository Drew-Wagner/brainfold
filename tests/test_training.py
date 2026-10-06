"""A member trained in a pack == the same member trained alone (no dropout, float64)."""

import torch

from packed_eegnet import PackedEEGNet, PackOfOne, fit, packed_batches, predict, seeded_packed_eegnet

C, N_OUT, T = 4, 3, 64
SMALL = dict(F1=2, D=2, kernel_length=9, depthwise_kernel_length=5, pool1_kernel_size=2, pool2_kernel_size=2,
             drop_prob=0.0)


def assert_same_member(packed: PackedEEGNet, k: int, alone: PackedEEGNet):
    for key, value in alone.to_state_dicts()[0].items():
        torch.testing.assert_close(packed.to_state_dicts()[k][key], value, msg=key)


def test_seeded_init_ignores_pack():
    before = torch.random.get_rng_state()
    packed = seeded_packed_eegnet([3, 5], C, N_OUT, T, **SMALL)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert_same_member(packed, 1, seeded_packed_eegnet([5], C, N_OUT, T, **SMALL))


def test_batches_ignore_pack():
    train_idx = torch.stack([torch.arange(10), torch.arange(10, 20)])
    packed = list(packed_batches(train_idx, 4, [torch.Generator().manual_seed(s) for s in (3, 5)]))
    alone = list(packed_batches(train_idx[1:], 4, [torch.Generator().manual_seed(5)]))
    assert [b.shape for b in packed] == [(4, 2), (4, 2), (2, 2)]
    for a, b in zip(packed, alone):
        assert torch.equal(a[:, 1:], b)


def test_fit_and_predict_match_training_alone():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(40, C, T, dtype=torch.float64, generator=g)
    y = torch.randint(N_OUT, (40,), generator=g)
    train_idx = torch.stack([torch.arange(0, 20), torch.arange(10, 30)])
    test_idx = torch.stack([torch.arange(30, 40), torch.arange(20, 30)])
    seeds = [3, 5]
    kwargs = dict(epochs=3, lr=1e-2, batch_size=8)

    packed = fit(seeded_packed_eegnet(seeds, C, N_OUT, T, **SMALL).double(), X, y, train_idx, seeds, **kwargs)
    for k, seed in enumerate(seeds):
        alone = fit(seeded_packed_eegnet([seed], C, N_OUT, T, **SMALL).double(),
                    X, y, train_idx[k : k + 1], [seed], **kwargs)
        assert_same_member(packed, k, alone)
        torch.testing.assert_close(predict(packed, X, test_idx)[:, k], predict(alone, X, test_idx[k : k + 1])[:, 0])


def test_braindecode_baseline_matches_member():
    """fit(PackOfOne(braindecode EEGNet)) trains exactly like the packed member it was copied from."""
    g = torch.Generator().manual_seed(0)
    X = torch.randn(20, C, T, dtype=torch.float64, generator=g)
    y = torch.randint(N_OUT, (20,), generator=g)
    train_idx = torch.arange(20)[None]
    packed = seeded_packed_eegnet([3], C, N_OUT, T, **SMALL).double()
    eegnet = packed.to_braindecode()[0]
    fit(packed, X, y, train_idx, [3], epochs=2, lr=1e-2, batch_size=8)
    fit(PackOfOne(eegnet), X, y, train_idx, [3], epochs=2, lr=1e-2, batch_size=8)
    assert_same_member(packed, 0, PackedEEGNet.from_braindecode([eegnet]))
