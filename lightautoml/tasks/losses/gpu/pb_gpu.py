"""Metrics and loss functions for pyboost on GPU."""

import logging
from functools import partial
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

from lightautoml.tasks.gpu.common_metric_gpu import (
    _valid_str_multiclass_metric_names_gpu,
)
from lightautoml.tasks.losses.base import Loss
from lightautoml.tasks.gpu.utils_gpu import infer_gib_gpu

logger = logging.getLogger(__name__)


_pb_binary_metrics_dict = {
    "auc": "auc",
    "accuracy": "accuracy",
}

_pb_reg_metrics_dict = {
    "mae": "rmse",
    "mse": "rmse",
    "r2": "r2",
    "rmsle": "rmsle",
}

_pb_multiclass_metrics_dict_gpu = {
    "auc": "auc",
    "crossentropy": "crossentropy",
    "accuracy": "accuracy",
}

_pb_multilabel_metric_dict_gpu = _pb_multiclass_metrics_dict_gpu

_pb_multireg_metric_dict_gpu = _pb_reg_metrics_dict

_pb_metrics_dict_gpu = {
    "binary": _pb_binary_metrics_dict,
    "reg": _pb_reg_metrics_dict,
    "multiclass": _pb_multiclass_metrics_dict_gpu,
    "multilabel" : _pb_multilabel_metric_dict_gpu,
    "multi:reg" : _pb_multireg_metric_dict_gpu,
}

class PBLoss(Loss):
    """Loss used for LightGBM."""

    def __init__(
        self,
        loss: Union[str, Callable],
        loss_params: Optional[Dict] = None,
        fw_func: Optional[Callable] = None,
        bw_func: Optional[Callable] = None,
    ):
        """

        TBA
        """
        if type(loss) is str:
            if loss == 'mae':
                loss = 'mse'
            self.fobj_name = loss
            self.fobj = None
        else:
            self.fobj_name = None
            self.fobj = loss

        # set forward and backward transformations
        if fw_func is not None:
            self._fw_func = fw_func
        if bw_func is not None:
            self._bw_func = bw_func

        self.metric = None
        self.metric_name = None

    def set_callback_metric(
        self,
        metric: Union[str, Callable],
        greater_is_better: Optional[bool] = None,
        metric_params: Optional[Dict] = None,
        task_name: Optional[str] = None,
    ):
        """Callback metric setter.

        Args:
            metric: Callback metric.
            greater_is_better: Whether or not higher value is better.
            metric_params: Additional metric parameters.
            task_name: Name of task.

        Note:
            Value of ``task_name`` should be one of following options:

            - `'binary'`
            - `'reg'`
            - `'multiclass'`

        """
        
        self.metric_params = {}

        self.metric = metric
        if type(metric) is str:

            if metric_params is not None:
                self.metric_params = metric_params

            _metric_dict = _pb_metrics_dict_gpu[task_name]
            _metric = _metric_dict.get(metric)
            if type(_metric) is str:
                self.metric_name = _metric
            else:
                self.metric_name = None

        else:
            self.metric_name = None
