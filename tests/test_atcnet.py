"""PackedATCNet == K separate braindecode ATCNets (float64), and close in float32."""

import inspect
import warnings

import pytest
import torch
from braindecode.models import ATCNet

from packed_eegnet import PackedATCNet

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*n_times.*smaller than the minimum.*")

ATCNET_PARAMS = inspect.signature(ATCNet).parameters
pytestmark = pytest.mark.skipif(
    "head_dim" not in ATCNET_PARAMS or not hasattr(ATCNet, "get_config"),
    reason="needs braindecode >= 1.4 (ATCNet's current arguments and get_config)",
)
needs_conv_max_norm = pytest.mark.skipif(
    "conv_max_norm_const" not in ATCNET_PARAMS, reason="conv_max_norm_const needs braindecode >= 1.7"
)

K, C, N_OUT = 3, 22, 4
NO_DROPOUT = dict(conv_block_dropout=0.0, att_drop_prob=0.0, tcn_drop_prob=0.0)
SMALL = dict(conv_block_n_filters=4, conv_block_pool_size_1=4, conv_block_pool_size_2=4)

# (n_times, ATCNet arguments)
CONFIGS = [
    pytest.param(1125, dict(), id="default"),
    pytest.param(257, dict(SMALL, n_windows=3, head_dim=4, num_heads=3, tcn_depth=3, tcn_kernel_size=3,
                           conv_block_kernel_length_1=33, conv_block_kernel_length_2=7), id="small"),
    pytest.param(257, dict(SMALL, concat=True, tcn_activation=torch.nn.GELU), id="concat-gelu"),
    pytest.param(257, dict(SMALL, concat=True, n_windows=1), id="concat-one-window"),
    pytest.param(257, dict(SMALL, n_windows=1, max_norm_const=1.0), id="one-window"),
    pytest.param(400, dict(), id="short-input"),  # ATCNet shrinks kernels, pools and windows
    pytest.param(257, dict(SMALL, conv_max_norm_const=0.6), id="conv-max-norm", marks=needs_conv_max_norm),
]


def members(n_times, dtype=torch.float64, **kwargs):
    """K braindecode ATCNets without dropout, non-trivial normalization parameters and active max-norms."""
    models = []
    for seed in range(K):
        torch.manual_seed(seed)
        model = ATCNet(n_chans=C, n_outputs=N_OUT, n_times=n_times, **NO_DROPOUT, **kwargs).to(dtype)
        for block in model.attention_blocks:
            block.drop.p = 0.0  # fixed at 0.3 in braindecode
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name.endswith("original"):
                    param.mul_(10)
            for module in model.modules():
                if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    module.weight.uniform_(0.5, 1.5)
                    module.bias.normal_(0, 0.1)
                    module.running_mean.normal_(0, 0.1)
                    module.running_var.uniform_(0.5, 1.5)
                elif isinstance(module, torch.nn.LayerNorm):
                    module.weight.uniform_(0.5, 1.5)
                    module.bias.normal_(0, 0.1)
        models.append(model)
    return models


def atcnet_forward(model, x):
    """``model(x)``, but with ``concat=True`` every window feeds the final layer, as documented.

    braindecode's forward zips the windows with ``final_layer``, which has one entry
    when ``concat=True``, so only the first window runs (and the final layer then
    fails on its input size when ``n_windows > 1``).
    """
    if not model.concat:
        return model(x)
    features = model.conv_block(model.dimshuffle(model.ensuredims(x))).view(-1, model.F2, model.Tc)
    windows = [tcn(attention(features[..., w : w + model.Tw]))[..., -1]
               for w, (attention, tcn) in enumerate(zip(model.attention_blocks, model.temporal_conv_nets))]
    return model.final_layer[0](torch.cat(windows, dim=1))


def pack(models):
    packed = PackedATCNet.from_braindecode(models)
    packed.att_block_drop.p = 0.0
    return packed


def inputs(n_times, seed=0, dtype=torch.float64, batch=8):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, K, C, n_times, dtype=torch.float64, generator=g)
    x = x * torch.linspace(0.5, 2, C, dtype=torch.float64)[:, None] + 0.3  # uneven scale, nonzero mean
    return x.to(dtype), torch.randint(N_OUT, (batch, K), generator=g)


def assert_same_state(packed, models):
    for state, model in zip(packed.to_state_dicts(), models):
        assert state.keys() == model.state_dict().keys()
        for key, value in model.state_dict().items():
            torch.testing.assert_close(state[key], value, msg=key)


@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("n_times, config", CONFIGS)
def test_forward_and_running_stats(n_times, config, train):
    models = members(n_times, **config)
    packed = pack(models).train(train)
    x, _ = inputs(n_times)
    out = packed(x)
    assert out.shape == (len(x), K, N_OUT)
    for k, model in enumerate(models):
        model.train(train)
        torch.testing.assert_close(out[:, k], atcnet_forward(model, x[:, k]))
    assert_same_state(packed, models)


@pytest.mark.parametrize("n_times, config", CONFIGS)
def test_adam_training(n_times, config):
    models = members(n_times, **config)
    packed = pack(models)
    optimizers = [torch.optim.Adam(m.parameters(), lr=1e-2) for m in models]
    packed_optimizer = torch.optim.Adam(packed.parameters(), lr=1e-2)
    for step in range(5):
        x, y = inputs(n_times, seed=step)
        for k, (model, optimizer) in enumerate(zip(models, optimizers)):
            optimizer.zero_grad()
            torch.nn.functional.cross_entropy(atcnet_forward(model, x[:, k]), y[:, k]).backward()
            optimizer.step()
        packed_optimizer.zero_grad()
        losses = torch.nn.functional.cross_entropy(packed(x).permute(0, 2, 1), y, reduction="none").mean(0)
        losses.sum().backward()  # sum of member mean losses: each member gets its own gradient
        packed_optimizer.step()
    assert_same_state(packed, models)


def test_short_input_shrinks_like_atcnet():
    with pytest.warns(UserWarning, match="smaller than the minimum"):
        packed = PackedATCNet(K, C, N_OUT, n_times=400)
    model = ATCNet(n_chans=C, n_outputs=N_OUT, n_times=400)
    assert (packed.L1, packed.L2, packed.P1, packed.P2, packed.n_windows, packed.tcn_kernel_size) == (
        model.conv_block_kernel_length_1, model.conv_block_kernel_length_2, model.conv_block_pool_size_1,
        model.conv_block_pool_size_2, model.n_windows, model.tcn_kernel_size)
    assert packed.braindecode_kwargs()["conv_block_kernel_length_1"] == 64  # the arguments as given


def test_from_seeds_ignores_pack():
    before = torch.random.get_rng_state()
    packed = PackedATCNet.from_seeds([3, 5], C, N_OUT, n_times=257, **SMALL)
    assert torch.equal(torch.random.get_rng_state(), before)
    alone = PackedATCNet.from_seeds([5], C, N_OUT, n_times=257, **SMALL)
    for key, value in alone.member_state_dict(0).items():
        torch.testing.assert_close(packed.member_state_dict(1)[key], value, rtol=0, atol=0, msg=key)


def test_member_state_dict():
    """Loading one member's state changes that member only, outputs included."""
    packed = PackedATCNet.from_seeds([0, 1, 2], C, N_OUT, n_times=257, **SMALL).double()
    source = PackedATCNet.from_seeds([7], C, N_OUT, n_times=257, **SMALL).double()
    with torch.no_grad():
        for module in source.modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                module.running_mean.normal_(0, 0.1)
    before = packed.to_state_dicts()
    packed.load_member_state_dict(1, source.member_state_dict(0))
    after = packed.to_state_dicts()
    for key in before[0]:
        torch.testing.assert_close(after[0][key], before[0][key], rtol=0, atol=0)
        torch.testing.assert_close(after[2][key], before[2][key], rtol=0, atol=0)
        torch.testing.assert_close(after[1][key], source.member_state_dict(0)[key], rtol=0, atol=0)
    x, _ = inputs(257)
    torch.testing.assert_close(packed.eval()(x)[:, 1], source.eval()(x[:, 1:2])[:, 0])


@pytest.mark.parametrize("config", [dict(), dict(concat=True)])
def test_roundtrip_through_braindecode(config):
    packed = PackedATCNet(K, C, N_OUT, n_times=257, **SMALL, **config).double()
    again = PackedATCNet.from_braindecode(packed.to_braindecode())
    for a, b in zip(packed.state_dict().values(), again.state_dict().values()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_initialization_matches_braindecode():
    """Same shapes, and the same init distributions up to sampling noise."""
    n = 8
    packed = PackedATCNet(n, C, N_OUT, n_times=1125).to_state_dicts()
    torch.manual_seed(0)
    models = [ATCNet(n_chans=C, n_outputs=N_OUT, n_times=1125).state_dict() for _ in range(n)]
    for key, value in models[0].items():
        assert packed[0][key].shape == value.shape, key
        if value.is_floating_point() and value.numel() > 100:
            ours, theirs = (torch.stack([state[key] for state in states]) for states in (packed, models))
            torch.testing.assert_close(ours.std(), theirs.std(), rtol=0.1, atol=1e-6, msg=key)


def test_float32_close():
    models = members(1125, torch.float32)
    packed = pack(models).train()
    x, _ = inputs(1125, dtype=torch.float32)
    out = packed(x)
    for k, model in enumerate(models):
        torch.testing.assert_close(out[:, k], model.train()(x[:, k]), rtol=1e-4, atol=1e-4)


def test_braindecode_signal_arguments():
    chs_info = [dict(ch_name=f"C{i}") for i in range(C)]
    packed = PackedATCNet(K, n_outputs=N_OUT, chs_info=chs_info, input_window_seconds=4.5)  # sfreq=250 default
    assert (packed.n_chans, packed.n_times) == (C, 1125)


def test_unsupported_options_raise():
    with pytest.raises(ValueError):
        PackedATCNet(K, C, N_OUT, n_times=1125, tcn_activation=torch.nn.PReLU)
