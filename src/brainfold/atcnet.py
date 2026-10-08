"""K independent ATCNets trained as one network.

Architecture per member: braindecode's ``ATCNet`` (Altaheri et al., 2022)
    conv block: temporal conv (F1 filters, length L1, 'same') -> BN -> depthwise
    spatial conv (D per filter) -> BN -> ELU -> pool P1 -> channel dropout ->
    temporal conv (F2 = F1 * D to F2, length L2, 'same') -> BN -> ELU -> pool P2
    -> channel dropout. Then n_windows overlapping windows of the pooled
    sequence, each with its own weights: attention block (LayerNorm -> multi-head
    self-attention -> dropout, plus the input) -> TCN (tcn_depth residual
    blocks of two causal dilated convs) -> last time step -> max-norm linear
    layer; the windows' logits are averaged (or, with ``concat``, one max-norm
    linear layer reads all the windows).

The tricks of eegnet.py, plus one, all exact up to floating point:

1. Packing. The K members sit side by side on the channel axis and every op
   is grouped by member, as in PackedEEGNet.
2. The conv block's temporal conv -> BN -> spatial conv is EEGNet's first
   block and is computed spatial-first, as in PackedEEGNet.
3. Packed windows. The windows' branches share nothing but their input, so
   they are packed like members: K * n_windows attention blocks and TCNs run as
   one grouped op each, where braindecode loops over the windows.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ._packing import PackedModel, spatial_first_block

# Model arguments of braindecode's ATCNet that PackedATCNet takes with the same name and meaning.
ATCNET_ARGS = (
    "conv_block_n_filters", "conv_block_kernel_length_1", "conv_block_kernel_length_2", "conv_block_pool_size_1",
    "conv_block_pool_size_2", "conv_block_depth_mult", "conv_block_dropout", "n_windows", "head_dim", "num_heads",
    "att_drop_prob", "tcn_depth", "tcn_kernel_size", "tcn_drop_prob", "tcn_activation", "concat", "max_norm_const",
    "conv_max_norm_const",
)


class PackedATCNet(PackedModel):
    """``n_models`` ATCNets. Input (batch, n_models, n_chans, n_times) -> (batch, n_models, n_outputs).

    Member k sees ``x[:, k]``. Every argument after ``n_models`` is braindecode
    ``ATCNet``'s, with the same default and meaning, so ``PackedATCNet(K, **kwargs)``
    packs K ``ATCNet(**kwargs)``, including ATCNet's shrinking of kernels, pools
    and windows for short inputs. Not supported: a ``tcn_activation`` with
    parameters (it would be shared across the pack).
    """

    braindecode_name = "ATCNet"

    def __init__(
        self,
        n_models: int,
        n_chans: int | None = None,
        n_outputs: int | None = None,
        input_window_seconds: float | None = None,
        sfreq: float | None = 250.0,
        conv_block_n_filters: int = 16,
        conv_block_kernel_length_1: int = 64,
        conv_block_kernel_length_2: int = 16,
        conv_block_pool_size_1: int = 8,
        conv_block_pool_size_2: int = 7,
        conv_block_depth_mult: int = 2,
        conv_block_dropout: float = 0.3,
        n_windows: int = 5,
        head_dim: int = 8,
        num_heads: int = 2,
        att_drop_prob: float = 0.5,
        tcn_depth: int = 2,
        tcn_kernel_size: int = 4,
        tcn_drop_prob: float = 0.3,
        tcn_activation: type[nn.Module] = nn.ELU,
        concat: bool = False,
        max_norm_const: float = 0.25,
        conv_max_norm_const: float | None = None,
        chs_info: list[dict] | None = None,
        n_times: int | None = None,
    ):
        super().__init__()
        self.config = {name: value for name, value in locals().items() if name in ATCNET_ARGS}
        # Signal parameters are inferred as in braindecode's EEGModuleMixin.
        if n_chans is None and chs_info is not None:
            n_chans = len(chs_info)
        if n_times is None and input_window_seconds is not None and sfreq is not None:
            n_times = round(input_window_seconds * sfreq)
        if n_chans is None or n_outputs is None or n_times is None:
            raise ValueError("need n_outputs, n_chans (or chs_info) and n_times (or input_window_seconds and sfreq)")
        if any(True for _ in tcn_activation().parameters()):
            raise ValueError(f"{tcn_activation.__name__} has parameters, which would be shared across the pack")

        # ATCNet shrinks the model for inputs too short for its TCN and windows.
        min_len_tcn = (tcn_kernel_size - 1) * 2 ** (tcn_depth - 1) + 1
        min_n_times = (n_windows + min_len_tcn - 1) * conv_block_pool_size_1 * conv_block_pool_size_2
        if n_times < min_n_times:
            scale = n_times / min_n_times
            warnings.warn(f"n_times ({n_times}) is smaller than the minimum required ({min_n_times}) for "
                          f"ATCNet's parameters; scaling kernels, pools and windows by {scale:.2f}, as ATCNet does.")
            conv_block_kernel_length_1 = max(1, int(conv_block_kernel_length_1 * scale))
            conv_block_kernel_length_2 = max(1, int(conv_block_kernel_length_2 * scale))
            conv_block_pool_size_1 = max(1, int(conv_block_pool_size_1 * scale))
            conv_block_pool_size_2 = max(1, int(conv_block_pool_size_2 * scale))
            n_windows = max(1, int(n_windows * scale))
            tcn_kernel_size = max(2, int(tcn_kernel_size * scale))

        self.n_models, self.n_chans, self.n_outputs, self.n_times = n_models, n_chans, n_outputs, n_times
        self.chs_info, self.input_window_seconds, self.sfreq = chs_info, input_window_seconds, sfreq
        # The model as built, after any shrinking (self.config holds the arguments as given).
        self.F1, self.D, self.F2 = conv_block_n_filters, conv_block_depth_mult, conv_block_n_filters * conv_block_depth_mult
        self.L1, self.L2 = conv_block_kernel_length_1, conv_block_kernel_length_2
        self.P1, self.P2 = conv_block_pool_size_1, conv_block_pool_size_2
        self.n_windows, self.head_dim, self.num_heads = n_windows, head_dim, num_heads
        self.tcn_kernel_size, self.concat = tcn_kernel_size, concat
        self.max_norm_const, self.conv_max_norm_const = max_norm_const, conv_max_norm_const
        self.Tc = n_times // conv_block_pool_size_1 // conv_block_pool_size_2
        self.Tw = self.Tc - n_windows + 1
        if self.Tw < 1:
            raise ValueError(f"n_times={n_times} leaves {self.Tc} pooled steps for {n_windows} windows")

        K, W, F1, F2, C = n_models, n_windows, self.F1, self.F2, n_chans
        G, E = K * W, num_heads * head_dim  # attention and TCN groups: windows of each member
        bn = dict(eps=1e-4)  # the conv block's BatchNorms; the TCN's use PyTorch's defaults

        # Conv block. Init matches braindecode: PyTorch's default conv init.
        self.temporal_weight = _uniform(1 / math.sqrt(self.L1), K * F1, self.L1)
        self.bn_temporal = nn.BatchNorm1d(K * F1, **bn)
        self.spatial_weight = _uniform(1 / math.sqrt(C), K * F2, C)
        self.bn_spatial = nn.BatchNorm1d(K * F2, **bn)
        self.refine_weight = _uniform(1 / math.sqrt(F2 * self.L2), K * F2, F2, self.L2)
        self.bn_refine = nn.BatchNorm1d(K * F2, **bn)
        self.conv_drop = nn.Dropout1d(conv_block_dropout)  # braindecode's Dropout2d: whole channels

        # Attention blocks, one per (member, window); q, k and v projections stacked.
        self.ln_weight = nn.Parameter(torch.ones(G, F2))
        self.ln_bias = nn.Parameter(torch.zeros(G, F2))
        self.qkv_weight = _uniform(1 / math.sqrt(F2), G, 3 * E, F2)
        self.qkv_bias = _uniform(1 / math.sqrt(F2), G, 3 * E)
        self.out_weight = _uniform(1 / math.sqrt(E), G, F2, E)
        self.out_bias = _uniform(1 / math.sqrt(E), G, F2)
        self.att_drop = nn.Dropout(att_drop_prob)
        self.att_block_drop = nn.Dropout(0.3)  # fixed in braindecode's _AttentionBlock

        self.tcn = nn.ModuleList(_PackedTCNBlock(G, F2, self.tcn_kernel_size, 2**i) for i in range(tcn_depth))
        self.tcn_act = tcn_activation()
        self.tcn_drop = nn.Dropout(tcn_drop_prob)

        # Max-norm linear layers: one per (member, window), or one per member on the
        # concatenated windows. braindecode stores the weight divided by the max-norm scale.
        n_in = W * F2 if concat else F2
        self.classifier_weight = _uniform(1 / math.sqrt(n_in), K if concat else G, n_outputs, n_in)
        self.classifier_bias = _uniform(1 / math.sqrt(n_in), K if concat else G, n_outputs)
        with torch.no_grad():
            self.classifier_weight.div_(_max_norm_scale(self.classifier_weight, max_norm_const))

    def forward(self, x: Tensor) -> Tensor:
        B, K, W, F2 = len(x), self.n_models, self.n_windows, self.F2
        h = self._conv_block(x)  # (batch, K * F2, Tc)
        # Window w is h[..., w : w + Tw]; windows become groups, member-major: (batch, K * W, Tw, F2).
        h = h.view(B, K, F2, self.Tc).unfold(-1, self.Tw, 1).permute(0, 1, 3, 4, 2).reshape(B, K * W, self.Tw, F2)
        h = self._attention(h).transpose(2, 3).reshape(B, K * W * F2, self.Tw)
        for block in self.tcn:
            h = block(h, self.tcn_act, self.tcn_drop, self.conv_max_norm_const)
        features = h[..., -1].view(B, K, W, F2)  # the TCNs' last (causal) step

        weight = self.classifier_weight * _max_norm_scale(self.classifier_weight, self.max_norm_const)
        if self.concat:
            return torch.einsum("bkn,kon->bko", features.reshape(B, K, W * F2), weight) + self.classifier_bias
        logits = torch.einsum("bkwf,kwof->bkwo", features, weight.view(K, W, self.n_outputs, F2))
        return (logits + self.classifier_bias.view(K, W, self.n_outputs)).mean(2)

    def _conv_block(self, x: Tensor) -> Tensor:
        K, max_norm = self.n_models, self.conv_max_norm_const
        w, s, r = self.temporal_weight, self.spatial_weight, self.refine_weight
        if max_norm is not None:
            w, s, r = w.renorm(2, 0, max_norm), s.renorm(2, 0, max_norm), r.renorm(2, 0, max_norm)
        h = spatial_first_block(x, w, s.view(K, self.F2, self.n_chans), self.bn_temporal, _same(self.L1))
        h = self.conv_drop(F.avg_pool1d(F.elu(self.bn_spatial(h)), self.P1))
        h = F.conv1d(F.pad(h, _same(self.L2)), r, groups=K)
        return self.conv_drop(F.avg_pool1d(F.elu(self.bn_refine(h)), self.P2))

    def _attention(self, x: Tensor) -> Tensor:
        """x + attention(LayerNorm(x)) for x (batch, groups, Tw, F2), each group with its own weights."""
        B, G, T, F2 = x.shape
        h = F.layer_norm(x, (F2,), eps=1e-6) * self.ln_weight[:, None] + self.ln_bias[:, None]
        qkv = torch.einsum("bgtf,gef->bgte", h, self.qkv_weight) + self.qkv_bias[:, None]
        q, k, v = qkv.view(B, G, T, 3, self.num_heads, self.head_dim).permute(3, 0, 1, 4, 2, 5)
        h = F.scaled_dot_product_attention(q, k, v).transpose(2, 3).reshape(B, G, T, -1)  # heads concatenated
        h = torch.einsum("bgte,gfe->bgtf", h, self.out_weight) + self.out_bias[:, None]
        return x + self.att_block_drop(self.att_drop(h))

    def source_optimizer_param_groups(self, conv_weight_decay: float = 0.009,
                                      dense_weight_decay: float = 0.5) -> list[dict]:
        """braindecode ``ATCNet.source_optimizer_param_groups``: the official code's L2 weight decay.

        Conv and TCN kernels decay by ``conv_weight_decay``, the final layer's weights
        by ``dense_weight_decay``, everything else not at all. Coupled weight decay
        (Adam, SGD) is elementwise, so the members stay independent.
        """
        conv = [self.temporal_weight, self.spatial_weight, self.refine_weight]
        conv += [weight for block in self.tcn for weight, _, _ in block.layers()]
        decayed = {id(p) for p in conv} | {id(self.classifier_weight)}
        return [
            {"params": conv, "weight_decay": conv_weight_decay},
            {"params": [self.classifier_weight], "weight_decay": dense_weight_decay},
            {"params": [p for p in self.parameters() if id(p) not in decayed], "weight_decay": 0.0},
        ]

    # ---- conversion to and from K separate braindecode ATCNets --------------------

    def braindecode_kwargs(self) -> dict:
        """Keyword arguments for one braindecode ``ATCNet`` equal to a member."""
        kwargs = dict(n_chans=self.n_chans, n_outputs=self.n_outputs, n_times=self.n_times, sfreq=self.sfreq,
                      **self.config)
        if kwargs["conv_max_norm_const"] is None:
            del kwargs["conv_max_norm_const"]  # not an argument before braindecode 1.7
        for name in ("chs_info", "input_window_seconds"):
            if getattr(self, name) is not None:
                kwargs[name] = getattr(self, name)
        return kwargs

    @classmethod
    def _kwargs_from_braindecode(cls, model) -> dict:
        # ATCNet's attributes hold the shrunk sizes for short inputs; get_config has the arguments.
        config = model.get_config()
        kwargs = {name: config.get(name) for name in ATCNET_ARGS}
        kwargs["tcn_activation"] = model.tcn_activation
        return dict(n_chans=model.n_chans, n_outputs=model.n_outputs, n_times=model.n_times, sfreq=model.sfreq,
                    **kwargs)

    def _layout(self):
        K, W, F1, F2, C, E = self.n_models, self.n_windows, self.F1, self.F2, self.n_chans, self.num_heads * self.head_dim
        L1, L2, n_out = self.L1, self.L2, self.n_outputs

        def conv(name):  # braindecode's key for a conv weight, max-norm parametrized or not
            return f"{name}.parametrizations.weight.original" if self.conv_max_norm_const is not None else f"{name}.weight"

        items = [
            (conv("conv_block.conv1"), self.temporal_weight, (F1, 1, L1, 1)),
            (conv("conv_block.conv2"), self.spatial_weight, (F2, 1, 1, C)),
            (conv("conv_block.conv3"), self.refine_weight, (F2, F2, L2, 1)),
        ]
        for i, bn in enumerate((self.bn_temporal, self.bn_spatial, self.bn_refine), 1):
            items += _batch_norm_items(f"conv_block.bn{i}", bn, lambda t: t, K)

        for w in range(W):
            def window(t, w=w):  # window w's slice of a (K * W, ...) tensor, member-major
                return t.view(K, W, -1)[:, w]

            prefix = f"attention_blocks.{w}"
            items += [(f"{prefix}.ln.weight", window(self.ln_weight), (F2,)),
                      (f"{prefix}.ln.bias", window(self.ln_bias), (F2,))]
            for j, name in enumerate("qkv"):
                items += [
                    (f"{prefix}.mha.fc_{name}.weight", self.qkv_weight.view(K, W, 3, E * F2)[:, w, j], (E, F2)),
                    (f"{prefix}.mha.fc_{name}.bias", self.qkv_bias.view(K, W, 3, E)[:, w, j], (E,)),
                ]
            items += [(f"{prefix}.mha.fc_o.weight", window(self.out_weight), (F2, E)),
                      (f"{prefix}.mha.fc_o.bias", window(self.out_bias), (F2,))]

            for i, block in enumerate(self.tcn):
                prefix = f"temporal_conv_nets.{w}.{i}"
                for j, (weight, bias, bn) in enumerate(block.layers(), 1):
                    items += [(conv(f"{prefix}.conv{j}"), window(weight), (F2, F2, self.tcn_kernel_size)),
                              (f"{prefix}.conv{j}.bias", window(bias), (F2,))]
                    items += _batch_norm_items(f"{prefix}.bn{j}", bn, window, K)

            if not self.concat:
                items += [(f"final_layer.{w}.parametrizations.weight.original", window(self.classifier_weight),
                           (n_out, F2)),
                          (f"final_layer.{w}.bias", window(self.classifier_bias), (n_out,))]
        if self.concat:
            items += [("final_layer.0.parametrizations.weight.original", self.classifier_weight, (n_out, W * F2)),
                      ("final_layer.0.bias", self.classifier_bias, (n_out,))]
        return items

    def _counters(self):
        counters = [(f"conv_block.bn{i}.num_batches_tracked", bn.num_batches_tracked)
                    for i, bn in enumerate((self.bn_temporal, self.bn_spatial, self.bn_refine), 1)]
        for w in range(self.n_windows):
            for i, block in enumerate(self.tcn):
                for j, (_, _, bn) in enumerate(block.layers(), 1):
                    counters.append((f"temporal_conv_nets.{w}.{i}.bn{j}.num_batches_tracked", bn.num_batches_tracked))
        return counters


class _PackedTCNBlock(nn.Module):
    """``groups`` of braindecode's _TCNResidualBlock: two causal dilated convs, F2 -> F2 channels each."""

    def __init__(self, groups: int, F2: int, kernel_size: int, dilation: int):
        super().__init__()
        self.groups, self.dilation = groups, dilation
        fan_in = F2 * kernel_size  # braindecode: kaiming_uniform_ weights, PyTorch's default bias init
        self.conv1_weight = _uniform(math.sqrt(6 / fan_in), groups * F2, F2, kernel_size)
        self.conv1_bias = _uniform(1 / math.sqrt(fan_in), groups * F2)
        self.bn1 = nn.BatchNorm1d(groups * F2)
        self.conv2_weight = _uniform(math.sqrt(6 / fan_in), groups * F2, F2, kernel_size)
        self.conv2_bias = _uniform(1 / math.sqrt(fan_in), groups * F2)
        self.bn2 = nn.BatchNorm1d(groups * F2)

    def layers(self):
        return (self.conv1_weight, self.conv1_bias, self.bn1), (self.conv2_weight, self.conv2_bias, self.bn2)

    def forward(self, x: Tensor, activation: nn.Module, dropout: nn.Module, max_norm: float | None) -> Tensor:
        h = x
        for weight, bias, bn in self.layers():
            if max_norm is not None:
                weight = weight.renorm(2, 0, max_norm)
            h = F.pad(h, ((weight.shape[-1] - 1) * self.dilation, 0))  # causal
            h = dropout(activation(bn(F.conv1d(h, weight, bias, dilation=self.dilation, groups=self.groups))))
        return activation(x + h)


def _uniform(bound: float, *shape: int) -> nn.Parameter:
    return nn.Parameter(torch.empty(*shape).uniform_(-bound, bound))


def _same(L: int) -> tuple[int, int]:
    """PyTorch's padding='same' for a length-L kernel: the extra zero goes on the right."""
    return (L - 1) // 2, L - 1 - (L - 1) // 2


def _max_norm_scale(weight: Tensor, max_norm: float, eps: float = 1e-5) -> Tensor:
    """braindecode's MaxNorm (MaxNormLinear) for (rows, n_out, n_in) weights: weight * scale is the constrained weight.

    The norm runs over the outputs of each input feature, and the scale applies at any norm.
    """
    denom = weight.norm(2, dim=1, keepdim=True).clamp(min=max_norm / 2)
    return denom.clamp(max=max_norm) / (denom + eps)


def _batch_norm_items(prefix: str, bn: nn.BatchNorm1d, select, n_models: int):
    """Layout items for a packed BatchNorm; ``select`` picks a member-major part of a packed tensor."""
    items = []
    for name in ("weight", "bias", "running_mean", "running_var"):
        tensor = select(getattr(bn, name))
        items.append((f"{prefix}.{name}", tensor, (tensor.numel() // n_models,)))
    return items
