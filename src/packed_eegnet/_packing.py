"""What every packed model shares: per-member state dicts in braindecode's format,
and the spatial-first temporal conv -> BN -> spatial conv block (see eegnet.py)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PackedModel(nn.Module):
    """K independent braindecode models packed into one network.

    Subclasses define ``_layout``, ``_counters``, ``braindecode_kwargs`` and
    ``_kwargs_from_braindecode``; this class converts to and from K braindecode models.
    """

    braindecode_name: str  # the model's class in braindecode.models
    n_models: int

    def _layout(self) -> list[tuple[str, Tensor, tuple[int, ...]]]:
        """(braindecode key, packed tensor, member shape) for every member tensor.

        The packed tensor is member-major along dim 0, so ``tensor.chunk(n_models)[k]``
        holds member k. It may be a view of a parameter or buffer.
        """
        raise NotImplementedError

    def _counters(self) -> list[tuple[str, Tensor]]:
        """(braindecode key, tensor) for BatchNorm's num_batches_tracked, shared by the pack."""
        raise NotImplementedError

    def braindecode_kwargs(self) -> dict:
        """Keyword arguments for one braindecode model equal to a member."""
        raise NotImplementedError

    @classmethod
    def _kwargs_from_braindecode(cls, model) -> dict:
        """Arguments after ``n_models`` that pack models like ``model``."""
        raise NotImplementedError

    def member_state_dict(self, k: int) -> dict[str, Tensor]:
        """Member k's weights and BatchNorm statistics as a braindecode state dict (a copy).

        Use it to keep a member's best epoch, e.g. for per-member early stopping.
        """
        state = {}
        for key, tensor, shape in self._layout():
            state[key] = tensor.detach().chunk(self.n_models)[k].reshape(shape).clone()
        for key, tensor in self._counters():
            state[key] = tensor.clone()
        return state

    @torch.no_grad()
    def load_member_state_dict(self, k: int, state: dict[str, Tensor]) -> None:
        """Overwrite member k with a braindecode state dict; the other members are untouched.

        ``num_batches_tracked`` is shared by the pack and is not loaded.
        """
        for key, tensor, _ in self._layout():
            chunk = tensor.chunk(self.n_models)[k]
            chunk.copy_(state[key].reshape(chunk.shape))

    def to_state_dicts(self) -> list[dict[str, Tensor]]:
        """K state dicts in braindecode format."""
        return [self.member_state_dict(k) for k in range(self.n_models)]

    @torch.no_grad()
    def load_state_dicts(self, states: list[dict[str, Tensor]]) -> None:
        """Load K braindecode state dicts (num_batches_tracked from the first)."""
        if len(states) != self.n_models:
            raise ValueError(f"expected {self.n_models} state dicts, got {len(states)}")
        for k, state in enumerate(states):
            self.load_member_state_dict(k, state)
        for key, tensor in self._counters():
            tensor.copy_(states[0][key])

    @classmethod
    def from_seeds(cls, seeds, *args, **kwargs):
        """A pack whose member k is initialized from ``seeds[k]`` alone.

        Member k's initial weights do not depend on the rest of the pack:
        ``cls.from_seeds(seeds, ...)`` member k equals ``cls.from_seeds([seeds[k]], ...)``.
        Arguments after ``seeds`` are the class's, without ``n_models``. The global
        RNG state is unchanged.
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
    def from_braindecode(cls, models):
        """Pack braindecode models that share their hyperparameters."""
        packed = cls(len(models), **cls._kwargs_from_braindecode(models[0]))
        reference = next(models[0].parameters())
        packed.to(device=reference.device, dtype=reference.dtype)
        packed.load_state_dicts([model.state_dict() for model in models])
        return packed.train(models[0].training)

    def to_braindecode(self) -> list:
        """The members as separate braindecode models (copies)."""
        import braindecode.models

        model_class = getattr(braindecode.models, self.braindecode_name)
        reference = next(self.parameters())
        models = []
        for state in self.to_state_dicts():
            model = model_class(**self.braindecode_kwargs()).to(device=reference.device, dtype=reference.dtype)
            model.load_state_dict(state)
            models.append(model.train(self.training))
        return models


def spatial_first_block(x: Tensor, w: Tensor, s: Tensor, bn: nn.BatchNorm1d, padding: tuple[int, int]) -> Tensor:
    """spatial(BN(temporal(x))) computed spatial-first (see eegnet.py's docstring).

    x is (batch, K, C, T); w (K * F1, L) the temporal filters, padded by ``padding``
    (left, right); s (K, F1 * D, C) the spatial filters, filter g reading temporal
    filter g // D; ``bn`` the BatchNorm1d(K * F1) after the temporal conv.
    Returns (batch, K * F1 * D, T + left + right - L + 1).
    """
    B, K, C, T = x.shape
    L = w.shape[1]
    F1 = w.shape[0] // K
    D = s.shape[1] // F1

    if bn.training:
        e, M = _lag_moments(F.pad(x, padding), L)  # (K, L), (K, L, L); no gradient needed
        w_k = w.view(K, F1, L)
        mean = torch.einsum("kfl,kl->kf", w_k, e).reshape(-1)
        var = torch.einsum("kfl,klm,kfm->kf", w_k, M, w_k).reshape(-1) - mean**2
        with torch.no_grad():  # same running-stat update as nn.BatchNorm
            n = B * C * (T + sum(padding) - L + 1)
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
    left, right = padding
    if left != right:
        mixed, left = F.pad(mixed, padding), 0
    filtered = F.conv1d(mixed, w.repeat_interleave(D, 0)[:, None], padding=left, groups=K * F1 * D)
    scale, shift = scale.repeat_interleave(D), shift.repeat_interleave(D)
    return scale[:, None] * filtered + (shift * s.reshape(-1, C).sum(1))[:, None]


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
