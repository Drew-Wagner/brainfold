"""Train many EEGNets at once as one packed network."""

from .model import PackedEEGNet

__all__ = ["PackedEEGNet"]
