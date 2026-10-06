"""Train many EEGNets at once as one packed network."""

from .model import PackedEEGNet
from .training import PackOfOne, fit, packed_batches, packed_loss, predict, seeded_packed_eegnet

__all__ = ["PackOfOne", "PackedEEGNet", "fit", "packed_batches", "packed_loss", "predict", "seeded_packed_eegnet"]
