"""Metrics and loss functions for Catboost."""

from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np

from .base import Loss


def cb_str_loss_wrapper(name: str, **params: Optional[Dict]):
    """CatBoost loss name wrapper, if it has keyword args.

    Args:
        name: One of CatBoost loss names.
        **params: Additional parameters.

    Returns:
        Wrapped CatBoost loss name.

    """
    return name + ":" + ";".join([k + "=" + str(v) for (k, v) in params.items()])


def fw_rmsle(x, y):
    return np.log1p(x), y


_cb_loss_mapping = {
    "mse": ("RMSE", None, None),
    "mae": ("MAE", None, None),
    "logloss": ("Logloss", None, None),
    "rmsle": ("RMSE", fw_rmsle, np.expm1),
    "mape": ("MAPE", None, None),
    "quantile": ("Quantile", None, None),
    "fair": ("FairLoss", None, None),
    "huber": ("Huber", None, None),
    "crossentropy": ("MultiClass", None, None),
}

_cb_loss_params_mapping = {
    "quantile": {"q": "alpha"},
    "huber": {"a": "delta"},
    "fair": {"c": "smoothness"},
}

_cb_binary_metrics_dict = {
    "auc": "AUC",
    "logloss": "Logloss",
    "accuracy": "Accuracy",
}

_cb_reg_metrics_dict = {
    "mse": "RMSE",
    "mae": "MAE",
    "r2": "R2",
    "rmsle": "MSLE",
    "mape": "MAPE",
    "quantile": "Quantile",
    "fair": "FairLoss",
    "huber": "Huber",
}

_cb_multiclass_metrics_dict = {
    "auc": "AUC:type=Mu",  # for overfitting detector
    "auc_mu": "AUC:type=Mu",
    "accuracy": "Accuracy",
    "crossentropy": "MultiClass",
    "f1_macro": "TotalF1:average=Macro",
    "f1_micro": "TotalF1:average=Micro",
    "f1_weighted": "TotalF1:average=Weighted",
}

_cb_metrics_dict = {
    "binary": _cb_binary_metrics_dict,
    "reg": _cb_reg_metrics_dict,
    "multiclass": _cb_multiclass_metrics_dict,
}


_cb_metric_params_mapping = {
    "quantile": {"q": "alpha"},
    "huber": {"a": "delta"},
    "fair": {"c": "smoothness"},
}


class CBLoss(Loss):
    """Loss used for CatBoost."""

    def __init__(
        self,
        loss: Union[str, Callable],
        loss_params: Optional[Dict] = None,
        fw_func: Optional[Callable] = None,
        bw_func: Optional[Callable] = None,
    ):
        """

        Args:
            loss: String with one of default losses.
            loss_params: additional loss parameters.
              Format like in :mod:`lightautoml.tasks.custom_metrics`.
            fw_func: Forward transformation.
              Used for transformation of target and item weights.
            bw_func: Backward transformation.
              Used for predict values transformation.

        """
        self.loss_params = {}
        if loss_params is not None:
            self.loss_params = loss_params

        if type(loss) is str:
            if loss in _cb_loss_mapping:
                loss_name, fw_func, bw_func = _cb_loss_mapping[loss]
                if loss in _cb_loss_params_mapping:
                    mapped_params = {_cb_loss_params_mapping[loss][k]: v for (k, v) in self.loss_params.items()}
                    self.fobj = None
                    self.fobj_name = cb_str_loss_wrapper(loss_name, **mapped_params)

                else:
                    self.fobj = None
                    self.fobj_name = loss_name
            else:
                raise ValueError("Unexpected loss for catboost")
                # special loss for catboost, that is not defined in _cb_loss_mapping
                # self.fobj = None
                # self.fobj_name = loss
        else:
            # custom catboost objective
            self.fobj = loss
            self.fobj_name = None

        if fw_func is not None:
            self._fw_func = fw_func

        if bw_func is not None:
            self._bw_func = bw_func

        self.fobj_params = {}
        if loss_params is not None:
            self.fobj_params = loss_params

        self.metric = None
        self.metric_name = None

    def set_callback_metric(
        self,
        metric: Union[str, Callable],
        greater_is_better: Optional[bool] = None,
        metric_params: Optional[Dict] = None,
        task_name: str = None,
    ):
        """
        Callback metric setter.

        Args:
            metric: Callback metric.
            greater_is_better: Whether or not higher value is better.
            metric_params: Additional metric parameters.
            task_name: Name of task. For now it omitted.

        """
        # TODO: for what cb_utils
        # How to say that this metric is special class if there any task type?

        assert task_name in [
            "binary",
            "reg",
            "multiclass",
        ], "Unknown task name: {}".format(task_name)

        self.metric_params = {}
        if metric_params is not None:
            self.metric_params = metric_params

        if type(metric) is str:
            self.metric = None
            _metric_dict = _cb_metrics_dict[task_name]
            if metric in _cb_metric_params_mapping:
                metric_params = {_cb_metric_params_mapping[metric][k]: v for (k, v) in self.metric_params.items()}
                self.metric_name = cb_str_loss_wrapper(_metric_dict[metric], **metric_params)
            else:
                self.metric_name = _metric_dict[metric]

        else:
            # TODO: Check it later
            self.metric_name = self.fobj_name
            self.metric_params = self.fobj_params
            self.metric = None
