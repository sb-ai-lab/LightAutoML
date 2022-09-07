"""Metrics and loss functions for LightGBM."""

import logging

from functools import partial
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import lightgbm as lgb
import numpy as np

from ..common_metric import _valid_str_multiclass_metric_names
from ..utils import infer_gib
from .base import Loss
from .lgb_custom import lgb_f1_loss_multiclass  # , F1Factory
from .lgb_custom import softmax_ax1


logger = logging.getLogger(__name__)


def fw_rmsle(x, y):
    return np.log1p(x), y


_lgb_binary_metrics_dict = {
    "auc": "auc",
    "logloss": "binary_logloss",
    "accuracy": "binary_error",
}

_lgb_reg_metrics_dict = {
    "mse": "mse",
    "mae": "mae",
    "r2": "mse",
    "rmsle": "mse",
    "mape": "mape",
    "quantile": "quantile",
    "huber": "huber",
    "fair": "fair",
}

_lgb_multiclass_metrics_dict = {
    "auc": _valid_str_multiclass_metric_names["auc"],
    "auc_mu": _valid_str_multiclass_metric_names["auc_mu"],
    "crossentropy": "multi_logloss",
    "accuracy": "multi_error",
    "f1_macro": _valid_str_multiclass_metric_names["f1_macro"],
    "f1_micro": _valid_str_multiclass_metric_names["f1_micro"],
    "f1_weighted": _valid_str_multiclass_metric_names["f1_weighted"],
}


_lgb_metrics_dict = {
    "binary": _lgb_binary_metrics_dict,
    "reg": _lgb_reg_metrics_dict,
    "multiclass": _lgb_multiclass_metrics_dict,
}

_lgb_loss_mapping = {
    "logloss": ("binary", None, None),
    "mse": ("regression", None, None),
    "mae": ("l1", None, None),
    "mape": ("mape", None, None),
    "crossentropy": ("multiclass", None, None),
    "rmsle": ("mse", fw_rmsle, np.expm1),
    "quantile": ("quantile", None, None),
    "huber": ("huber", None, None),
    "fair": ("fair", None, None),
    "f1": (lgb_f1_loss_multiclass, None, softmax_ax1),
}

_lgb_loss_params_mapping = {
    "quantile": {"q": "alpha"},
    "huber": {"a": "alpha"},
    "fair_c": {"c": "fair_c"},
}

_lgb_force_metric = {
    "rmsle": ("mse", None, None),
}


class LGBFunc:
    """
    Wrapper of metric function for LightGBM.
    """

    def __init__(self, metric_func, greater_is_better, bw_func):
        self.metric_func = metric_func
        self.greater_is_better = greater_is_better
        self.bw_func = bw_func

    def __call__(self, pred: np.ndarray, dtrain: lgb.Dataset) -> Tuple[str, float, bool]:

        label = dtrain.get_label()

        weights = dtrain.get_weight()

        if label.shape[0] != pred.shape[0]:
            pred = pred.reshape((label.shape[0], -1), order="F")
            label = label.astype(np.int32)

        label = self.bw_func(label)
        pred = self.bw_func(pred)

        # for weighted case
        try:
            val = self.metric_func(label, pred, sample_weight=weights)
        except TypeError:
            val = self.metric_func(label, pred)

        # TODO: what if grouped case

        return "Opt metric", val, self.greater_is_better


class LGBLoss(Loss):
    """Loss used for LightGBM."""

    def __init__(
        self,
        loss: Union[str, Callable],
        loss_params: Optional[Dict] = None,
        fw_func: Optional[Callable] = None,
        bw_func: Optional[Callable] = None,
    ):
        """

        Args:
            loss: Objective to optimize.
            loss_params: additional loss parameters.
              Format like in :mod:`lightautoml.tasks.custom_metrics`.
            fw_func: forward transformation.
              Used for transformation of target and item weights.
            bw_func: backward transformation.
              Used for predict values transformation.

        Note:
            Loss can be one of the types:

                - Str: one of default losses
                  ('auc', 'mse', 'mae', 'logloss', 'accuray', 'r2',
                  'rmsle', 'mape', 'quantile', 'huber', 'fair')
                  or another lightgbm objective.
                - Callable: custom lightgbm style objective.

        """
        if loss in _lgb_loss_mapping:
            fobj, fw_func, bw_func = _lgb_loss_mapping[loss]
            if type(fobj) is str:
                self.fobj_name = fobj
                self.fobj = None
            else:
                self.fobj_name = None
                self.fobj = fobj
            # map param name for known objectives
            if self.fobj_name in _lgb_loss_params_mapping:
                param_mapping = _lgb_loss_params_mapping[self.fobj_name]
                loss_params = {param_mapping[x]: loss_params[x] for x in loss_params}

        else:
            # set lgb style objective
            if type(loss) is str:
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

        self.fobj_params = {}
        if loss_params is not None:
            self.fobj_params = loss_params

        self.metric = None

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
            Callable metric, that returns ('Opt metric', value, greater_is_better).

        """
        if greater_is_better is None:
            greater_is_better = infer_gib(metric_func)

        if metric_params is not None:
            metric_func = partial(metric_func, **metric_params)

        return LGBFunc(metric_func, greater_is_better, self._bw_func)

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
        # force metric if special loss
        # what about task_name? in this case?
        if self.fobj_name in _lgb_force_metric:
            metric, greater_is_better, metric_params = _lgb_force_metric[self.fobj_name]
            logger.info2(
                "For lgbm {0} callback metric switched to {1}".format(self.fobj_name, metric),
                UserWarning,
            )

        self.metric_params = {}

        # set lgb style metric
        self.metric = metric
        if type(metric) is str:

            if metric_params is not None:
                self.metric_params = metric_params

            _metric_dict = _lgb_metrics_dict[task_name]
            _metric = _metric_dict.get(metric)
            if type(_metric) is str:
                self.metric_name = _metric
                self.feval = None
            else:
                self.metric_name = None
                # _metric = CustomWrapper(_metric)
                self.feval = self.metric_wrapper(_metric, greater_is_better, {})

        else:
            self.metric_name = None
            # metric = CustomWrapper(metric)
            self.feval = self.metric_wrapper(metric, greater_is_better, self.metric_params)
