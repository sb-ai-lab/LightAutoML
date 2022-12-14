"""Metrics and loss functions for Torch based models."""

from functools import partial
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union

from lightautoml_gpu.tasks.gpu.common_metric_gpu import _valid_str_metric_names_gpu
from lightautoml_gpu.tasks.gpu.utils_gpu import infer_gib_gpu
from lightautoml_gpu.tasks.losses.base import Loss
from lightautoml_gpu.tasks.losses.base import MetricFunc
from lightautoml_gpu.tasks.losses.torch import TorchLossWrapper
from lightautoml_gpu.tasks.losses.torch import _torch_loss_dict


class TORCHLossGPU(Loss):
    """Loss used for PyTorch."""

    def __init__(self, loss: Union[str, Callable], loss_params: Optional[Dict] = None):
        """

        Args:
            loss: name or callable objective function.
            loss_params: additional loss parameters.

        """
        self.loss_params = {}
        if loss_params is not None:
            self.loss_params = loss_params

        if loss in ["mse", "mae", "logloss", "crossentropy"]:
            self.loss = TorchLossWrapper(*_torch_loss_dict[loss], **self.loss_params)
        elif type(loss) is str:
            self.loss = partial(_torch_loss_dict[loss][0], **self.loss_params)
        else:
            self.loss = partial(loss, **self.loss_params)

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
            "multi:reg",
            "multiclass",
            "multilabel",
        ], "Incorrect task name: {}".format(task_name)
        self.metric = metric

        if metric_params is None:
            metric_params = {}

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
