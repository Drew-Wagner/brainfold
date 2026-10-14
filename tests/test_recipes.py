"""The recipes' training loop: precomputed input moments change nothing (float64)."""

import sys
import warnings
from pathlib import Path

import pytest
import torch

pytest.importorskip("moabb")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipes" / "moabb"))
from train import fit  # noqa: E402

from brainfold import PackedATCNet, PackedEEGNet  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*n_times.*smaller than the minimum.*")


class Recompute(torch.nn.Module):
    """Hides ``input_moments``, so ``fit`` recomputes the moments from every batch."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


@pytest.mark.parametrize("cls, kwargs", [
    (PackedEEGNet, dict(n_times=129, F1=4, drop_prob=0.0)),
    (PackedATCNet, dict(n_times=257, conv_block_n_filters=4, conv_block_pool_size_1=4, conv_block_pool_size_2=4,
                        conv_block_dropout=0.0, att_drop_prob=0.0, tcn_drop_prob=0.0)),
])
def test_fit_with_precomputed_moments(cls, kwargs):
    g = torch.Generator().manual_seed(0)
    X = torch.randn(40, 8, kwargs["n_times"], generator=g, dtype=torch.float64) + 0.3
    y = torch.randint(3, (40,), generator=g)
    # Overlapping training sets, in different orders: fit maps each to the shared per-trial moments
    train_idx = torch.stack([torch.arange(0, 24), torch.arange(30, 6, -1), torch.randperm(40, generator=g)[:24]])
    seeds = [0, 1, 2]
    model = cls.from_seeds(seeds, n_chans=8, n_outputs=3, **kwargs).double()
    if cls is PackedATCNet:
        model.att_block_drop.p = 0.0
    reference = Recompute(cls.from_seeds(seeds, n_chans=8, n_outputs=3, **kwargs).double())
    reference.model.load_state_dict(model.state_dict())
    if cls is PackedATCNet:
        reference.model.att_block_drop.p = 0.0
    settings = dict(epochs=3, lr=1e-2, batch_size=10)
    _, losses = fit(model, X, y, train_idx, seeds, **settings)
    _, reference_losses = fit(reference, X, y, train_idx, seeds, **settings)
    torch.testing.assert_close(losses, reference_losses)
    for key, value in reference.model.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, msg=key)
