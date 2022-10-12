"""Set of loss functions for different machine learning algorithms."""

from lightautoml.tasks.losses.gpu.torch_gpu import TORCHLoss_gpu

from .base import _valid_str_metric_names
from .cb import CBLoss
from .lgb import LGBLoss
from .sklearn import SKLoss
from .torch import TORCHLoss, TorchLossWrapper

__all__ = [
    "LGBLoss",
    "TORCHLoss",
    "SKLoss",
    "CBLoss",
    "_valid_str_metric_names",
    "TorchLossWrapper",
    "TORCHLoss_gpu",
    "CUMLLoss",
    "XGBLoss_gpu",
]
