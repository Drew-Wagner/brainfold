"""Training K independent runs as one PackedEEGNet.

A run is one member of the pack: its own initial weights, its own training
indices and its own shuffle order. Every helper here keeps member k's
computation independent of what else is in the pack (apart from dropout,
which draws from one shared RNG).
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .model import PackedEEGNet


def seeded_packed_eegnet(seeds: Sequence[int], n_chans: int, n_outputs: int, n_times: int, **kwargs) -> PackedEEGNet:
    """A PackedEEGNet whose member k is initialized from ``seeds[k]`` alone.

    Member k gets the same initial weights as ``seeded_packed_eegnet([seeds[k]], ...)``,
    whatever it is packed with. The global RNG state is left unchanged.
    """
    states = []
    with torch.random.fork_rng(devices=[]):
        for seed in seeds:
            torch.manual_seed(seed)
            states += PackedEEGNet(1, n_chans, n_outputs, n_times, **kwargs).to_state_dicts()
        model = PackedEEGNet(len(seeds), n_chans, n_outputs, n_times, **kwargs)
    model.load_state_dicts(states)
    return model


def packed_batches(train_idx: Tensor, batch_size: int, generators: Sequence[torch.Generator]) -> Iterator[Tensor]:
    """One epoch of (batch, K) index tensors; column k is a batch of ``train_idx[k]``.

    ``train_idx`` is (K, n_train): every member needs the same number of
    training examples. Member k is shuffled with ``generators[k]`` only.
    """
    n_train = train_idx.shape[1]
    order = torch.stack([torch.randperm(n_train, device=train_idx.device, generator=g) for g in generators])
    for i in range(0, n_train, batch_size):
        yield train_idx.gather(1, order[:, i : i + batch_size]).T


def packed_loss(logits: Tensor, y: Tensor) -> Tensor:
    """Sum over members of each member's mean cross-entropy.

    ``logits`` is (batch, K, n_outputs) and ``y`` is (batch, K). Summing (not
    averaging) the member losses gives each member exactly the gradient it
    would get alone, so one Adam over the packed parameters == K separate Adams.
    """
    return F.cross_entropy(logits.permute(0, 2, 1), y, reduction="none").mean(0).sum()


class PackOfOne(nn.Module):
    """One ordinary model, (batch, n_chans, n_times) -> (batch, n_outputs), as a pack of K = 1.

    Lets ``fit`` and ``predict`` train a single model such as a braindecode
    ``EEGNet``, for example as an unpacked baseline.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x[:, 0])[:, None]


def fit(
    model: nn.Module,
    X: Tensor,
    y: Tensor,
    train_idx: Tensor,
    seeds: Sequence[int],
    epochs: int,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> nn.Module:
    """Train every member with Adam and a cosine learning-rate schedule.

    ``model`` maps (batch, K, n_chans, n_times) to (batch, K, n_outputs): a
    ``PackedEEGNet``, or a single model wrapped in ``PackOfOne``.

    ``X`` is (n, n_chans, n_times) and ``y`` is (n,), shared by all members;
    ``train_idx`` (K, n_train) picks each member's training examples, and
    ``seeds[k]`` seeds member k's shuffle order.
    """
    generators = [torch.Generator(device=train_idx.device).manual_seed(seed) for seed in seeds]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for group in optimizer.param_groups:
            group["lr"] = 0.5 * lr * (1 + math.cos(math.pi * epoch / epochs))
        model.train()
        for idx in packed_batches(train_idx, batch_size, generators):
            loss = packed_loss(model(X[idx]), y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def predict(model: nn.Module, X: Tensor, idx: Tensor, batch_size: int = 64) -> Tensor:
    """(n, K) predicted classes; column k is member k's prediction for ``X[idx[k]]``.

    ``idx`` is (K, n), so every member is evaluated on the same number of examples.
    """
    model.eval()
    return torch.cat([model(X[idx[:, i : i + batch_size].T]).argmax(-1) for i in range(0, idx.shape[1], batch_size)])
