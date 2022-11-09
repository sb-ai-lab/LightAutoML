"""Metrics and loss functions for cuml models."""

import logging
from functools import partial
from typing import Callable, Dict, Optional, Union

import cupy as cp

from lightautoml.tasks.gpu.common_metric_gpu import _valid_str_metric_names_gpu
from lightautoml.tasks.gpu.utils_gpu import infer_gib_gpu
from lightautoml.tasks.losses.base import Loss, MetricFunc

logger = logging.getLogger(__name__)


def fw_rmsle(x, y):
    return cp.log1p(x), y


_cuml_loss_mapping = {"rmsle": ("mse", fw_rmsle, cp.expm1)}

_cuml_force_metric = {
    "rmsle": ("mse", None, None),
}


class CUMLLoss(Loss):
    """Loss used for cuml."""

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
        ], "Not supported in cuml in general case."
        self.flg_regressor = loss in ["mse", "rmsle"]

        if loss in _cuml_loss_mapping:
            self.loss, fw_func, bw_func = _cuml_loss_mapping[loss]
        else:
            self.loss = loss
            # set forward and backward transformations
            if fw_func is not None:
                self._fw_func = fw_func
            if bw_func is not None:
                self._bw_func = bw_func

        self.loss_params = loss_params

    def metric_wrapper(
        self,
        metric_func: Callable,
        greater_is_better: Optional[bool],
        metric_params: Optional[Dict] = None,
    ) -> Callable:
        """Customize metric.

        Args:
            metric_func: Callable metric.
            greater_is_better: Whether or not higher value is better.
            metric_params: Additional metric parameters.

        Returns:
            Callable metric.

        """
        if greater_is_better is None:

            greater_is_better = infer_gib_gpu(metric_func)

        m = 2 * float(greater_is_better) - 1

        if metric_params is not None:
            metric_func = partial(metric_func, **metric_params)

        return MetricFunc(metric_func, m, self._bw_func)

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

        if self.loss in _cuml_force_metric:
            metric, greater_is_better, metric_params = _cuml_force_metric[self.loss]
            logger.warning(
                "For cuml {0} callback metric switched to {1}".format(self.loss, metric)
            )

        assert task_name in [
            "binary",
            "reg",
            "multi:reg",
            "multiclass",
            "multilabel",
        ], "Incorrect task name: {}".format(task_name)

        self.metric = metric

        if metric_params is None:
            metric_params = {}

        metric_dict = None
        if type(metric) is str:

            metric_dict = _valid_str_metric_names_gpu[task_name]

            self.metric_func = self.metric_wrapper(
                metric_dict[metric], greater_is_better, metric_params
            )
            self.metric_name = metric
        else:
            self.metric_func = self.metric_wrapper(
                metric, greater_is_better, metric_params
            )
            self.metric_name = None
