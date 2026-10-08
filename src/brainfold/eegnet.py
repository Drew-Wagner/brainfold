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

from ._packing import PackedModel, spatial_first_block


# Model arguments of braindecode's EEGNet that PackedEEGNet takes with the same name and meaning.
EEGNET_ARGS = (
    "final_conv_length", "pool_mode", "F1", "D", "F2", "kernel_length", "depthwise_kernel_length",
    "pool1_kernel_size", "pool2_kernel_size", "conv_spatial_max_norm", "activation", "batch_norm_momentum",
    "batch_norm_affine", "batch_norm_eps", "drop_prob", "norm_rate",
)


class PackedEEGNet(PackedModel):
    """``n_models`` EEGNets. Input (batch, n_models, n_chans, n_times) -> (batch, n_models, n_outputs).

    Member k sees ``x[:, k]``. Every argument after ``n_models`` is braindecode
    ``EEGNet``'s, with the same default and meaning, so ``PackedEEGNet(K, **kwargs)``
    packs K ``EEGNet(**kwargs)``. Not supported: cropped decoding (a
    ``final_conv_length`` shorter than the remaining time) and activations with
    parameters (they would be shared across the pack).
    """

    braindecode_name = "EEGNet"

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
        s = self.spatial_weight.renorm(2, 0, self.conv_spatial_max_norm)
        P = self.kernel_length // 2
        return spatial_first_block(x, self.temporal_weight, s.view(self.n_models, -1, self.n_chans),
                                   self.bn_temporal, (P, P))

    # ---- conversion to and from K separate braindecode EEGNets --------------------

    def braindecode_kwargs(self) -> dict:
        """Keyword arguments for one braindecode ``EEGNet`` equal to a member."""
        kwargs = dict(n_chans=self.n_chans, n_outputs=self.n_outputs, n_times=self.n_times,
                      final_layer_with_constraint=self.final_layer_with_constraint)
        kwargs.update({name: getattr(self, name) for name in EEGNET_ARGS})
        for name in ("chs_info", "input_window_seconds", "sfreq"):
            if getattr(self, name) is not None:
                kwargs[name] = getattr(self, name)
        return kwargs

    eegnet_kwargs = braindecode_kwargs

    @classmethod
    def _kwargs_from_braindecode(cls, model) -> dict:
        return dict(n_chans=model.n_chans, n_outputs=model.n_outputs, n_times=model.n_times,
                    final_layer_with_constraint=hasattr(model.final_layer, "linearconstraint"),
                    **{name: getattr(model, name) for name in EEGNET_ARGS})

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

    def _counters(self):
        return [(f"{prefix}.num_batches_tracked", bn.num_batches_tracked) for prefix, bn in self._batch_norms()]
