"""Set of loss functions for different machine learning algorithms."""

from lightautoml.tasks.losses.gpu.cuml import CUMLLoss
from lightautoml.tasks.losses.gpu.torch_gpu import TORCHLoss_gpu
from lightautoml.tasks.losses.gpu.xgb_gpu import XGBLoss_gpu

from ..base import _valid_str_metric_names

__all__ = [
    "_valid_str_metric_names",
    "TORCHLoss_gpu",
    "CUMLLoss",
    "XGBLoss_gpu",
]
