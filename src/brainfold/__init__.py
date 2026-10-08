"""Train many EEGNets or ATCNets at once as one packed network."""

from .atcnet import PackedATCNet
from .eegnet import PackedEEGNet

__all__ = ["PackedATCNet", "PackedEEGNet"]
