"""K independent EEGNets trained as one network, with a cheaper first block.

Plain PyTorch, one file, no dependencies beyond torch (braindecode is only
imported by the optional conversion helpers at the bottom).

Architecture per member: braindecode's ``EEGNet`` (Lawhern et al., 2018)
    temporal conv (F1 filters, length L) -> BN -> depthwise spatial conv
    (D per filter, max-norm) -> BN -> ELU -> pool -> dropout -> separable
    conv (depthwise + pointwise to F2) -> BN -> ELU -> pool -> dropout ->
    classifier over the remaining time (optionally max-norm).

Two tricks, both exact up to floating point:

1. Packing. The K members sit side by side on the channel axis; every conv is
   grouped so that member k only ever sees its own channels, and BatchNorm,
   dropout and max-norm act per channel or per output row anyway. One forward
   pass and one optimizer step train all K members. Use a *sum* of the members'
   mean losses so each member gets exactly its own gradient (Adam is
   elementwise, so one Adam over the packed parameters == K Adams).

2. Reordered first block. temporal conv -> BN -> spatial conv is linear, so
   for temporal filter w_f and spatial filter s_g that reads filter f,

       z_g = a_f * (w_f * (s_g . x)) + (beta_f - a_f * mu_f) * sum(s_g),
       a_f = gamma_f / sqrt(var_f + eps).

   Mixing the C electrodes first means the length-L temporal filter runs on
   F1 * D signals instead of F1 * C, and the full-resolution (F1, C, T)
   tensor is never built. In training, BN needs the batch mean and variance of
   u_f = w_f * x over (batch, electrodes, time); both are quadratic in w_f:

       mu_f = w_f . e,     E[u_f^2] = w_f^T M w_f,

   with e and M the window means and L x L lag second moments of the padded
   input (computed below from an FFT autocorrelation plus edge corrections).
   Gradients flow through these to w, gamma, beta and s exactly as through the
   original BatchNorm.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# Model arguments of braindecode's EEGNet that PackedEEGNet takes with the same name and meaning.
EEGNET_ARGS = (
    "final_conv_length", "pool_mode", "F1", "D", "F2", "kernel_length", "depthwise_kernel_length",
    "pool1_kernel_size", "pool2_kernel_size", "conv_spatial_max_norm", "activation", "batch_norm_momentum",
    "batch_norm_affine", "batch_norm_eps", "drop_prob", "norm_rate",
)


class PackedEEGNet(nn.Module):
    """``n_models`` EEGNets. Input (batch, n_models, n_chans, n_times) -> (batch, n_models, n_outputs).

    Member k sees ``x[:, k]``. Every argument after ``n_models`` is braindecode
    ``EEGNet``'s, with the same default and meaning, so ``PackedEEGNet(K, **kwargs)``
    packs K ``EEGNet(**kwargs)``. Not supported: cropped decoding (a
    ``final_conv_length`` shorter than the remaining time) and activations with
    parameters (they would be shared across the pack).
    """

    def __init__(
        self,
        n_models: int,
        n_chans: int | None = None,
        n_outputs: int | None = None,
        n_times: int | None = None,
        final_conv_length: str | int = "auto",
        pool_mode: str = "mean",
        F1: int = 8,
        D: int = 2,
        F2: int | None = None,
        kernel_length: int = 64,
        *,
        depthwise_kernel_length: int = 16,
        pool1_kernel_size: int = 4,
        pool2_kernel_size: int = 8,
        conv_spatial_max_norm: float = 1,
        activation: type[nn.Module] = nn.ELU,
        batch_norm_momentum: float = 0.01,
        batch_norm_affine: bool = True,
        batch_norm_eps: float = 1e-3,
        drop_prob: float = 0.25,
        final_layer_with_constraint: bool = False,
        norm_rate: float = 0.25,
        chs_info: list[dict] | None = None,
        input_window_seconds: float | None = None,
        sfreq: float | None = None,
    ):
        super().__init__()
        # Signal parameters are inferred as in braindecode's EEGModuleMixin.
        if n_chans is None and chs_info is not None:
            n_chans = len(chs_info)
        if n_times is None and input_window_seconds is not None and sfreq is not None:
            n_times = round(input_window_seconds * sfreq)
        if n_chans is None or n_outputs is None or n_times is None:
            raise ValueError("need n_outputs, n_chans (or chs_info) and n_times (or input_window_seconds and sfreq)")
        if pool_mode not in ("mean", "max"):
            raise ValueError(f"pool_mode must be 'mean' or 'max', not {pool_mode!r}")
        if any(True for _ in activation().parameters()):
            raise ValueError(f"{activation.__name__} has parameters, which would be shared across the pack")
        F2 = F1 * D if F2 is None else F2
        t = n_times + 2 * (kernel_length // 2) - kernel_length + 1
        t = t // pool1_kernel_size + 2 * (depthwise_kernel_length // 2) - depthwise_kernel_length + 1
        n_final = t // pool2_kernel_size
        if final_conv_length not in ("auto", n_final):
            raise ValueError(f"final_conv_length must be 'auto' or {n_final}; cropped decoding is not supported")

        self.n_models, self.n_chans, self.n_outputs, self.n_times = n_models, n_chans, n_outputs, n_times
        self.chs_info, self.input_window_seconds, self.sfreq = chs_info, input_window_seconds, sfreq
        self.final_conv_length, self.pool_mode = n_final, pool_mode
        self.F1, self.D, self.F2, self.kernel_length = F1, D, F2, kernel_length
        self.depthwise_kernel_length = depthwise_kernel_length
        self.pool1_kernel_size, self.pool2_kernel_size = pool1_kernel_size, pool2_kernel_size
        self.conv_spatial_max_norm, self.activation = conv_spatial_max_norm, activation
        self.batch_norm_momentum, self.batch_norm_affine = batch_norm_momentum, batch_norm_affine
        self.batch_norm_eps, self.drop_prob = batch_norm_eps, drop_prob
        self.final_layer_with_constraint, self.norm_rate = final_layer_with_constraint, norm_rate
        K = n_models
        bn = dict(eps=batch_norm_eps, momentum=batch_norm_momentum, affine=batch_norm_affine)

        # First block, used through `_first_block` only. Init matches braindecode:
        # PyTorch's default conv init for the temporal conv, xavier for the spatial one.
        self.temporal_weight = nn.Parameter(
            torch.empty(K * F1, kernel_length).uniform_(-1, 1) / math.sqrt(kernel_length)
        )
        self.bn_temporal = nn.BatchNorm1d(K * F1, **bn)  # holds gamma, beta and running stats
        bound = math.sqrt(6 / (n_chans + F1 * D * n_chans))
        self.spatial_weight = nn.Parameter(torch.empty(K * F1 * D, n_chans).uniform_(-bound, bound))

        self.bn1 = nn.BatchNorm1d(K * F1 * D, **bn)
        self.act = activation()
        self.pool = F.avg_pool1d if pool_mode == "mean" else F.max_pool1d
        self.sep_depth = nn.Conv1d(K * F1 * D, K * F1 * D, depthwise_kernel_length,
                                   padding=depthwise_kernel_length // 2, groups=K * F1 * D, bias=False)
        self.sep_point = nn.Conv1d(K * F1 * D, K * F2, 1, groups=K, bias=False)
        self.bn2 = nn.BatchNorm1d(K * F2, **bn)
        self.drop = nn.Dropout(drop_prob)
        # braindecode's conv classifier, or its flatten + linear layer, which is the same map.
        self.classifier = nn.Conv1d(K * F2, K * n_outputs, n_final, groups=K)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: Tensor) -> Tensor:
        h = self._first_block(x)  # (batch, K * F1 * D, time)
        h = self.drop(self.pool(self.act(self.bn1(h)), self.pool1_kernel_size))
        h = self.bn2(self.sep_point(self.sep_depth(h)))
        h = self.drop(self.pool(self.act(h), self.pool2_kernel_size))
        weight = self.classifier.weight
        if self.final_layer_with_constraint:
            weight = weight.renorm(2, 0, self.norm_rate)
        out = F.conv1d(h, weight, self.classifier.bias, groups=self.n_models)
        return out.view(len(x), self.n_models, self.n_outputs)

    def _first_block(self, x: Tensor) -> Tensor:
        """spatial(BN(temporal(x))) computed spatial-first; x is (batch, K, C, T)."""
        K, F1, D, L = self.n_models, self.F1, self.D, self.kernel_length
        B, _, C, T = x.shape
        P = L // 2
        w = self.temporal_weight  # (K * F1, L)
        s = self.spatial_weight.renorm(2, 0, self.conv_spatial_max_norm).view(K, F1 * D, C)
        bn = self.bn_temporal

        if bn.training:
            e, M = _lag_moments(F.pad(x, (P, P)), L)  # (K, L), (K, L, L); no gradient needed
            w_k = w.view(K, F1, L)
            mean = torch.einsum("kfl,kl->kf", w_k, e).reshape(-1)
            var = torch.einsum("kfl,klm,kfm->kf", w_k, M, w_k).reshape(-1) - mean**2
            with torch.no_grad():  # same running-stat update as nn.BatchNorm
                n = B * C * (T + 2 * P - L + 1)
                bn.num_batches_tracked.add_(1)
                bn.running_mean.lerp_(mean, bn.momentum)
                bn.running_var.lerp_(var * n / (n - 1), bn.momentum)
        else:
            mean, var = bn.running_mean, bn.running_var
        scale = torch.rsqrt(var + bn.eps)
        if bn.affine:
            scale = bn.weight * scale
        shift = bn.bias - mean * scale if bn.affine else -mean * scale

        # Mix electrodes: one batched matmul per member over all (batch, time) columns.
        mixed = torch.bmm(s, x.permute(1, 2, 0, 3).reshape(K, C, B * T))
        mixed = mixed.view(K * F1 * D, B, T).transpose(0, 1)
        # Spatial filter g reads temporal filter g // D; filter each mixed signal.
        filtered = F.conv1d(mixed, w.repeat_interleave(D, 0)[:, None], padding=P, groups=K * F1 * D)
        scale, shift = scale.repeat_interleave(D), shift.repeat_interleave(D)
        return scale[:, None] * filtered + (shift * s.reshape(-1, C).sum(1))[:, None]

    # ---- conversion to and from K separate braindecode EEGNets --------------------

    def eegnet_kwargs(self) -> dict:
        """Keyword arguments for one braindecode ``EEGNet`` equal to a member."""
        kwargs = dict(n_chans=self.n_chans, n_outputs=self.n_outputs, n_times=self.n_times,
                      final_layer_with_constraint=self.final_layer_with_constraint)
        kwargs.update({name: getattr(self, name) for name in EEGNET_ARGS})
        for name in ("chs_info", "input_window_seconds", "sfreq"):
            if getattr(self, name) is not None:
                kwargs[name] = getattr(self, name)
        return kwargs

    def _layout(self):
        """(braindecode key, packed tensor, member shape) for every member tensor."""
        F1, D, F2, C, L = self.F1, self.D, self.F2, self.n_chans, self.kernel_length
        n_out, n_final = self.n_outputs, self.final_conv_length
        items = [
            ("conv_temporal.weight", self.temporal_weight, (F1, 1, 1, L)),
            ("conv_spatial.parametrizations.weight.original", self.spatial_weight, (F1 * D, 1, C, 1)),
            ("conv_separable_depth.weight", self.sep_depth.weight, (F1 * D, 1, 1, self.depthwise_kernel_length)),
            ("conv_separable_point.weight", self.sep_point.weight, (F2, F1 * D, 1, 1)),
        ]
        if self.final_layer_with_constraint:
            items += [
                ("final_layer.linearconstraint.parametrizations.weight.original", self.classifier.weight,
                 (n_out, F2 * n_final)),
                ("final_layer.linearconstraint.bias", self.classifier.bias, (n_out,)),
            ]
        else:
            items += [
                ("final_layer.conv_classifier.weight", self.classifier.weight, (n_out, F2, 1, n_final)),
                ("final_layer.conv_classifier.bias", self.classifier.bias, (n_out,)),
            ]
        names = ("weight", "bias", "running_mean", "running_var") if self.batch_norm_affine else ("running_mean", "running_var")
        for prefix, bn in self._batch_norms():
            for name in names:
                tensor = getattr(bn, name)
                items.append((f"{prefix}.{name}", tensor, (len(tensor) // self.n_models,)))
        return items

    def _batch_norms(self):
        return ("bnorm_temporal", self.bn_temporal), ("bnorm_1", self.bn1), ("bnorm_2", self.bn2)

    def member_state_dict(self, k: int) -> dict[str, Tensor]:
        """Member k's weights and BatchNorm statistics as a braindecode ``EEGNet`` state dict (a copy).

        Use it to keep a member's best epoch, e.g. for per-member early stopping.
        """
        state = {}
        for key, tensor, shape in self._layout():
            state[key] = tensor.detach().chunk(self.n_models)[k].reshape(shape).clone()
        for prefix, bn in self._batch_norms():  # shared by the pack
            state[f"{prefix}.num_batches_tracked"] = bn.num_batches_tracked.clone()
        return state

    @torch.no_grad()
    def load_member_state_dict(self, k: int, state: dict[str, Tensor]) -> None:
        """Overwrite member k with a braindecode ``EEGNet`` state dict; the other members are untouched.

        ``num_batches_tracked`` is shared by the pack and is not loaded.
        """
        for key, tensor, _ in self._layout():
            chunk = tensor.chunk(self.n_models)[k]
            chunk.copy_(state[key].reshape(chunk.shape))

    def to_state_dicts(self) -> list[dict[str, Tensor]]:
        """K state dicts in braindecode ``EEGNet`` format."""
        return [self.member_state_dict(k) for k in range(self.n_models)]

    @torch.no_grad()
    def load_state_dicts(self, states: list[dict[str, Tensor]]) -> None:
        """Load K braindecode ``EEGNet`` state dicts (num_batches_tracked from the first)."""
        if len(states) != self.n_models:
            raise ValueError(f"expected {self.n_models} state dicts, got {len(states)}")
        for k, state in enumerate(states):
            self.load_member_state_dict(k, state)
        for prefix, bn in self._batch_norms():
            bn.num_batches_tracked.copy_(states[0][f"{prefix}.num_batches_tracked"])

    @classmethod
    def from_seeds(cls, seeds, *args, **kwargs) -> PackedEEGNet:
        """A pack whose member k is initialized from ``seeds[k]`` alone.

        Member k's initial weights do not depend on the rest of the pack:
        ``PackedEEGNet.from_seeds(seeds, ...)`` member k equals
        ``PackedEEGNet.from_seeds([seeds[k]], ...)``. Arguments after ``seeds`` are
        ``PackedEEGNet``'s, without ``n_models``. The global RNG state is unchanged.
        """
        states = []
        with torch.random.fork_rng(devices=[]):
            for seed in seeds:
                torch.manual_seed(seed)
                states.append(cls(1, *args, **kwargs).member_state_dict(0))
            packed = cls(len(seeds), *args, **kwargs)
        packed.load_state_dicts(states)
        return packed

    @classmethod
    def from_braindecode(cls, models) -> PackedEEGNet:
        """Pack braindecode ``EEGNet``s that share their hyperparameters."""
        m = models[0]
        packed = cls(
            len(models), m.n_chans, m.n_outputs, m.n_times,
            final_layer_with_constraint=hasattr(m.final_layer, "linearconstraint"),
            **{name: getattr(m, name) for name in EEGNET_ARGS},
        )
        reference = next(m.parameters())
        packed.to(device=reference.device, dtype=reference.dtype)
        packed.load_state_dicts([model.state_dict() for model in models])
        return packed.train(m.training)

    def to_braindecode(self) -> list:
        """The members as separate braindecode ``EEGNet``s (copies)."""
        from braindecode.models import EEGNet

        reference = self.temporal_weight
        models = []
        for state in self.to_state_dicts():
            model = EEGNet(**self.eegnet_kwargs()).to(device=reference.device, dtype=reference.dtype)
            model.load_state_dict(state)
            models.append(model.train(self.training))
        return models


def _gram(segment: Tensor) -> Tensor:
    """sum over batch and electrodes of segment[t] * segment[u]: (K, n, n) for a (B, K, C, n) segment."""
    return torch.einsum("bkct,bkcu->ktu", segment, segment)


def _lag_moments(xpad: Tensor, L: int) -> tuple[Tensor, Tensor]:
    """Window means e (K, L) and lag second moments M (K, L, L) of the padded input.

    A length-L conv's output position t reads xpad[..., t + l] for l < L. Moments
    average over batch, electrodes and the T_out = Tp - L + 1 output positions:
        e[l]     = mean_t xpad[t + l]
        M[l, l'] = mean_t xpad[t + l] * xpad[t + l']
    """
    B, K, C, Tp = xpad.shape
    T_out = Tp - L + 1
    N = B * C * T_out
    lags = torch.arange(L, device=xpad.device)

    cumsum = F.pad(xpad.sum((0, 2)).cumsum(-1), (1, 0))
    e = (cumsum[:, lags + T_out] - cumsum[:, lags]) / N

    # M[l, l + k] is the lag-k product sum over the window [l, l + T_out). Take the
    # full autocorrelation R[k] = sum_t xpad[t] xpad[t + k] (FFT; n >= Tp + L - 1
    # avoids wraparound, a multiple of 64 keeps the FFT fast) and subtract the
    # products the window leaves out at the head (t < l) and tail (t >= l + T_out).
    n = -(-(Tp + L - 1) // 64) * 64
    spectrum = torch.fft.rfft(xpad, n=n)
    R = torch.fft.irfft((spectrum.real**2 + spectrum.imag**2).sum((0, 2)), n=n)[:, :L]
    t, k = lags[: L - 1, None], lags[None, :]
    head = _gram(xpad[..., : 2 * L - 2])[:, t, t + k]  # P[t, k] for t < L - 1
    tail = F.pad(_gram(xpad[..., T_out:]), (0, L))[:, t, t + k]  # zero past the end
    head = F.pad(head.cumsum(1), (0, 0, 1, 0))  # sum over t < l
    tail = F.pad(tail.flip(1).cumsum(1).flip(1), (0, 0, 0, 1))  # sum over t >= l + T_out
    band = (R[:, None, :] - head - tail) / N  # band[l, k] = M[l, l + k]

    first = torch.minimum(lags[:, None], lags[None, :])
    lag = (lags[:, None] - lags[None, :]).abs()
    return e, band[:, first, lag]
