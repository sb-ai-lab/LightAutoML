"""Set of loss functions for different machine learning algorithms."""

from .base import _valid_str_metric_names
from .cb import CBLoss
from .lgb import LGBLoss
from .xgb import XGBLoss
from .sklearn import SKLoss
from .torch import TORCHLoss
from .torch import TorchLossWrapper


__all__ = [
    "XGBLoss",
    "LGBLoss",
    "TORCHLoss",
    "SKLoss",
    "CBLoss",
    "_valid_str_metric_names",
    "TorchLossWrapper",
]
