"""Base classes for metric and loss functions."""

from functools import partial
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

from ..common_metric import _valid_str_metric_names
from ..utils import infer_gib


class MetricFunc:
    """
    Wrapper for metric.
    """

    def __init__(self, metric_func, m, bw_func):
        """

        Args:
            metric_func: Callable metric function.
            m: Multiplier for metric value.
            bw_func: Backward function.

        """
        self.metric_func = metric_func
        self.m = m
        self.bw_func = bw_func

    def __call__(self, y_true, y_pred, sample_weight=None) -> float:
        y_pred = self.bw_func(y_pred)

        try:
            val = self.metric_func(y_true, y_pred, sample_weight=sample_weight)
        except TypeError:
            val = self.metric_func(y_true, y_pred)

        return val * self.m


class Loss:
    """Loss function with target transformation."""

    @staticmethod
    def _fw_func(target: Any, weights: Any) -> Tuple[Any, Any]:
        """Forward transformation.

        Args:
            target: Ground truth target values.
            weights: Item weights.

        Returns:
            Tuple (target, weights) without transformation.

        """
        return target, weights

    @staticmethod
    def _bw_func(pred: Any) -> Any:
        """Backward transformation for predicted values.

        Args:
            pred: Predicted target values.

        Returns:
            Pred without transformation.

        """
        return pred

    @property
    def fw_func(self):
        """Forward transformation for target values and item weights.

        Returns:
            Callable transformation.

        """
        return self._fw_func

    @property
    def bw_func(self):
        """Backward transformation for predicted values.

        Returns:
            Callable transformation.

        """
        return self._bw_func

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
            greater_is_better = infer_gib(metric_func)

        m = 2 * float(greater_is_better) - 1

        if metric_params is not None:
            metric_func = partial(metric_func, **metric_params)

        return MetricFunc(metric_func, m, self._bw_func)

    def set_callback_metric(
        self,
        metric: Union[str, Callable],
        greater_is_better: Optional[bool] = None,
        metric_params: Optional[Dict] = None,
        task_name: Optional[Dict] = None,
    ):
        """Callback metric setter.

        Args:
            metric: Callback metric
            greater_is_better: Whether or not higher value is better.
            metric_params: Additional metric parameters.
            task_name: Name of task.

        Note:
            Value of ``task_name`` should be one of following options:

            -  `'binary'`
            - `'reg'`
            - `'multiclass'`

        """

        assert task_name in [
            "binary",
            "reg",
            "multiclass",
        ], "Incorrect task name: {}".format(task_name)
        self.metric = metric

        if metric_params is None:
            metric_params = {}

        if type(metric) is str:
            metric_dict = _valid_str_metric_names[task_name]
            self.metric_func = self.metric_wrapper(metric_dict[metric], greater_is_better, metric_params)
            self.metric_name = metric
        else:
            self.metric_func = self.metric_wrapper(metric, greater_is_better, metric_params)
            self.metric_name = None
