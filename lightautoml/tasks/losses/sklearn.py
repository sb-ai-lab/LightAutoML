"""Metrics and loss functions for scikit-learn models."""

import logging

from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np

from .base import Loss


logger = logging.getLogger(__name__)


def fw_rmsle(x, y):
    return np.log1p(x), y


_sk_loss_mapping = {"rmsle": ("mse", fw_rmsle, np.expm1)}

_sk_force_metric = {
    "rmsle": ("mse", None, None),
}


class SKLoss(Loss):
    """Loss used for scikit-learn."""

    def __init__(
        self,
        loss: str,
        loss_params: Optional[Dict] = None,
        fw_func: Optional[Callable] = None,
        bw_func: Optional[Callable] = None,
    ):
        """

        Args:
            loss: One of default loss function.
              Valid are: 'logloss', 'mse', 'crossentropy', 'rmsle'.
            loss_params: Addtional loss parameters.
            fw_func: Forward transformation.
              Used for transformation of target and item weights.
            bw_func: backward transformation.
              Used for predict values transformation.

        """
        assert loss in [
            "logloss",
            "mse",
            "crossentropy",
            "rmsle",
        ], "Not supported in sklearn in general case."
        self.flg_regressor = loss in ["mse", "rmsle"]

        if loss in _sk_loss_mapping:
            self.loss, fw_func, bw_func = _sk_loss_mapping[loss]
        else:
            self.loss = loss
            # set forward and backward transformations
            if fw_func is not None:
                self._fw_func = fw_func
            if bw_func is not None:
                self._bw_func = bw_func

        self.loss_params = loss_params

    def set_callback_metric(
        self,
        metric: Union[str, Callable],
        greater_is_better: Optional[bool] = None,
        metric_params: Optional[Dict] = None,
        task_name: Optional[str] = None,
    ):
        """
        Callback metric setter.

        Uses default callback of parent class `Loss`.

        Args:
            metric: Callback metric.
            greater_is_better: Whether or not higher value is better.
            metric_params: Additional metric parameters.
            task_name: Name of task.

        """
        if self.loss in _sk_force_metric:
            metric, greater_is_better, metric_params = _sk_force_metric[self.loss]
            logger.info2("For sklearn {0} callback metric switched to {1}".format(self.loss, metric))

        super().set_callback_metric(metric, greater_is_better, metric_params, task_name)
