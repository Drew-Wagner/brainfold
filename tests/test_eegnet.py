"""PackedEEGNet == K separate braindecode EEGNets (float64), and close in float32."""

import warnings

import pytest
import torch
from braindecode.models import EEGNet

from brainfold import PackedEEGNet

warnings.filterwarnings("ignore", category=DeprecationWarning)

K, C, N_OUT, T = 3, 22, 4, 257


def members(dtype=torch.float64, **kwargs):
    """K braindecode EEGNets with non-trivial BN parameters and active max-norms."""
    models = []
    for seed in range(K):
        torch.manual_seed(seed)
        model = EEGNet(C, N_OUT, T, drop_prob=0.0, **kwargs).to(dtype)
        with torch.no_grad():
            model.conv_spatial.parametrizations.weight.original.mul_(5)
            if hasattr(model.final_layer, "linearconstraint"):
                model.final_layer.linearconstraint.parametrizations.weight.original.mul_(100)
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    if module.affine:
                        module.weight.uniform_(0.5, 1.5)
                        module.bias.normal_(0, 0.1)
                    module.running_mean.normal_(0, 0.1)
                    module.running_var.uniform_(0.5, 1.5)
        models.append(model)
    return models


def inputs(seed=0, dtype=torch.float64, batch=8):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, K, C, T, dtype=torch.float64, generator=g)
    x = x * torch.linspace(0.5, 2, C, dtype=torch.float64)[:, None] + 0.3  # uneven scale, nonzero mean
    return x.to(dtype), torch.randint(N_OUT, (batch, K), generator=g)


CONFIGS = [
    dict(),
    dict(F1=4, D=2),
    dict(F1=3, D=3, F2=7, kernel_length=33),
    dict(F1=4, pool_mode="max", activation=torch.nn.GELU),
    dict(F1=4, final_layer_with_constraint=True, norm_rate=0.25),
]


@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("config", CONFIGS)
def test_forward_and_running_stats(config, train):
    models = members(**config)
    packed = PackedEEGNet.from_braindecode(models).train(train)
    x, _ = inputs()
    out = packed(x)
    assert out.shape == (len(x), K, N_OUT)
    for k, model in enumerate(models):
        model.train(train)
        torch.testing.assert_close(out[:, k], model(x[:, k]))
    for state, model in zip(packed.to_state_dicts(), models):
        for key, value in model.state_dict().items():
            torch.testing.assert_close(state[key], value, msg=key)


@pytest.mark.parametrize("config", CONFIGS)
def test_adam_training(config):
    models = members(**config)
    packed = PackedEEGNet.from_braindecode(models)
    optimizers = [torch.optim.Adam(m.parameters(), lr=1e-2) for m in models]
    packed_optimizer = torch.optim.Adam(packed.parameters(), lr=1e-2)
    for step in range(5):
        x, y = inputs(seed=step)
        for k, (model, optimizer) in enumerate(zip(models, optimizers)):
            optimizer.zero_grad()
            torch.nn.functional.cross_entropy(model(x[:, k]), y[:, k]).backward()
            optimizer.step()
        packed_optimizer.zero_grad()
        losses = torch.nn.functional.cross_entropy(packed(x).permute(0, 2, 1), y, reduction="none").mean(0)
        losses.sum().backward()  # sum of member mean losses: each member gets its own gradient
        packed_optimizer.step()
    for state, model in zip(packed.to_state_dicts(), models):
        for key, value in model.state_dict().items():
            torch.testing.assert_close(state[key], value, msg=key)


def test_from_seeds_ignores_pack():
    before = torch.random.get_rng_state()
    packed = PackedEEGNet.from_seeds([3, 5], C, N_OUT, T)
    assert torch.equal(torch.random.get_rng_state(), before)
    alone = PackedEEGNet.from_seeds([5], C, N_OUT, T)
    for key, value in alone.member_state_dict(0).items():
        torch.testing.assert_close(packed.member_state_dict(1)[key], value, rtol=0, atol=0, msg=key)


def test_member_state_dict():
    """Loading one member's state changes that member only, outputs included."""
    packed = PackedEEGNet.from_seeds([0, 1, 2], C, N_OUT, T).double()
    source = PackedEEGNet.from_seeds([7], C, N_OUT, T).double()
    with torch.no_grad():
        for bn in (source.bn_temporal, source.bn1, source.bn2):
            bn.running_mean.normal_(0, 0.1)
    before = packed.to_state_dicts()
    packed.load_member_state_dict(1, source.member_state_dict(0))
    after = packed.to_state_dicts()
    for key in before[0]:
        torch.testing.assert_close(after[0][key], before[0][key], rtol=0, atol=0)
        torch.testing.assert_close(after[2][key], before[2][key], rtol=0, atol=0)
        torch.testing.assert_close(after[1][key], source.member_state_dict(0)[key], rtol=0, atol=0)
    x, _ = inputs()
    torch.testing.assert_close(packed.eval()(x)[:, 1], source.eval()(x[:, 1:2])[:, 0])


def test_roundtrip_through_braindecode():
    packed = PackedEEGNet(K, C, N_OUT, T).double()
    again = PackedEEGNet.from_braindecode(packed.to_braindecode())
    for a, b in zip(packed.state_dict().values(), again.state_dict().values()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_float32_close():
    models = members(torch.float32)
    packed = PackedEEGNet.from_braindecode(models).train()
    x, _ = inputs(dtype=torch.float32)
    out = packed(x)
    for k, model in enumerate(models):
        torch.testing.assert_close(out[:, k], model.train()(x[:, k]), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("train", [False, True])
def test_batch_norm_without_affine(train):
    """braindecode 1.8.1 cannot build EEGNet(batch_norm_affine=False), so compare to gamma=1, beta=0."""
    affine = PackedEEGNet(K, C, N_OUT, T, drop_prob=0.0).double().train(train)
    plain = PackedEEGNet(K, C, N_OUT, T, drop_prob=0.0, batch_norm_affine=False).double().train(train)
    plain.load_state_dict(affine.state_dict(), strict=False)
    x, _ = inputs()
    torch.testing.assert_close(plain(x), affine(x))
    for key, value in plain.state_dict().items():  # same running statistics
        torch.testing.assert_close(value, affine.state_dict()[key], msg=key)
    assert len(plain.to_state_dicts()[0]) == len(affine.to_state_dicts()[0]) - 6


def test_braindecode_signal_arguments():
    chs_info = [dict(ch_name=f"C{i}") for i in range(C)]
    packed = PackedEEGNet(K, n_outputs=N_OUT, chs_info=chs_info, input_window_seconds=2.0, sfreq=128)
    assert (packed.n_chans, packed.n_times) == (C, 256)


@pytest.mark.parametrize("kwargs", [dict(final_conv_length=3), dict(activation=torch.nn.PReLU), dict(pool_mode="lp")])
def test_unsupported_options_raise(kwargs):
    with pytest.raises(ValueError):
        PackedEEGNet(K, C, N_OUT, T, **kwargs)
