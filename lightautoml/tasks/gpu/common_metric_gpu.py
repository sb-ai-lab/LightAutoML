"""Bunch of metrics with unified interface (GPU version)."""

from functools import partial
from typing import Callable, Optional

import cudf
import cupy as cp
import dask.array as da
import dask_cudf
from cuml.metrics import accuracy_score, log_loss, roc_auc_score
from cuml.metrics.regression import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)
from dask_ml.metrics import accuracy_score as dask_accuracy_score
from dask_ml.metrics import mean_absolute_error as dask_mean_absolute_error


def log_loss_gpu(y_true, y_pred, sample_weight=None, eps: float = 1e-15) -> float:

    res = None
    if isinstance(y_true, da.Array):

        # res = da.map_blocks(log_loss, y_true, y_pred,
        #                sample_weight=sample_weight, eps=eps,
        #                meta=cp.array((), dtype=cp.float32), drop_axis=1)

        # res = cp.array(res.compute()).mean()

        res = log_loss(
            y_true.compute(), y_pred.compute(), sample_weight=sample_weight, eps=eps
        )
    else:
        res = log_loss(y_true, y_pred, sample_weight=sample_weight, eps=eps)
    return res


def r2_score_gpu(y_true, y_pred) -> float:

    if isinstance(y_true, da.Array):
        output = da.map_blocks(
            r2_score, y_true, y_pred, meta=cp.array((), dtype=cp.float32), drop_axis=1
        )
        res = cp.array(output.compute()).mean()
    else:
        res = r2_score(y_true, y_pred)
    return res


def roc_auc_score_gpu(y_true, y_pred, sample_weight=None) -> float:

    if isinstance(y_true, da.Array):
        output = da.map_blocks(
            roc_auc_score,
            y_true,
            y_pred,
            meta=cp.array((), dtype=cp.float32),
            drop_axis=1,
        )
        res = cp.array(output.compute()).mean()
    else:
        res = roc_auc_score(y_true, y_pred)
    return res


def mean_squared_error_gpu(y_true, y_pred, sample_weight=None) -> float:

    """Computes Mean Squared Error for Multi-GPU data.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        sample_weight: Specify weighted mean (currently not used).

    Returns:
        metric value.

    """
    if isinstance(y_true, da.Array):
        err = y_pred - y_true
        err_sq = da.multiply(err, err)
        mean = err_sq.mean().compute()
    else:
        err_sq = cp.square(y_pred - y_true)
        mean = err_sq.mean()

    return mean


def mean_absolute_error_gpu(y_true, y_pred, sample_weight=None):

    if isinstance(y_true, da.Array):
        return dask_mean_absolute_error(y_true, y_pred, sample_weight)
    else:
        return mean_absolute_error(y_true, y_pred, sample_weight)


def mean_quantile_error_gpu(y_true, y_pred, sample_weight=None, q: float = 0.9) -> float:
    """Computes Mean Quantile Error for Multi-GPU data.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        sample_weight: Specify weighted mean.
        q: Metric coefficient.

    Returns:
        metric value.

    """

    if isinstance(y_true, da.Array):
        err = y_pred - y_true
        s = da.sign(err)
        err = da.where(s > 0, q * err, (q - 1) * err)
        if sample_weight is not None:
            return ((err * sample_weight).mean() / sample_weight.mean()).compute()
        return err.mean().compute()
    else:
        err = y_pred - y_true
        s = cp.sign(err)
        err = cp.abs(err)
        err = cp.where(s > 0, q, 1 - q) * err
        if sample_weight is not None:
            return (err * sample_weight).mean() / sample_weight.mean()
        return err.mean()


def mean_huber_error_gpu(y_true, y_pred, sample_weight=None, a: float = 0.9) -> float:
    """Computes Mean Huber Error for Multi-GPU data.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        sample_weight: Specify weighted mean.
        a: Metric coefficient.

    Returns:
        Metric value.

    """
    assert a >= 0, "a cannot be negative"

    if isinstance(y_true, da.Array):
        err = y_pred - y_true
        s = da.where(err < 0, err > -a, err < a)
        abs_err = da.where(err > 0, ebestclassmulticlassrr, -err)
        err = da.where(s, 0.5 * (err ** 2), a * abs_err - 0.5 * (a ** 2))
        if sample_weight is not None:
            return ((err * sample_weight).mean() / sample_weight.mean()).compute()
        return err.mean().compute()
    else:
        err = y_pred - y_true
        s = cp.abs(err) < a
        err = cp.where(s, 0.5 * (err ** 2), a * cp.abs(err) - 0.5 * (a ** 2))
        if sample_weight is not None:
            return (err * sample_weight).mean() / sample_weight.mean()
        return err.mean()


def mean_fair_error_gpu(y_true, y_pred, sample_weight=None, c: float = 0.9) -> float:
    """Computes Mean Fair Error for Multi-GPU data.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        sample_weight: Specify weighted mean.
        c: Metric coefficient.

    Returns:
        Metric value.

    """
    if isinstance(y_true, da.Array):
        err = y_pred - y_true
        x = da.where(err > 0, err, -err) / c
        err = c ** 2 * (x - da.log(x + 1))
        if sample_weight is not None:
            return ((err * sample_weight).mean() / sample_weight.mean()).compute()
        return err.mean().compute()
    else:
        x = cp.abs(y_pred - y_true) / c
        err = c ** 2 * (x - cp.log(x + 1))
        if sample_weight is not None:
            return (err * sample_weight).mean() / sample_weight.mean()
        return err.mean()


def mean_absolute_percentage_error_gpu(y_true, y_pred, sample_weight=None) -> float:
    """Computes Mean Absolute Percentage error for Mulit-GPU data.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        sample_weight: Specify weighted mean.

    Returns:
        Metric value.

    """
    if isinstance(y_true, da.Array):
        err = (y_true - y_pred) / y_true
        err = da.where(err > 0, err, -err)
        if sample_weight is not None:
            return ((err * sample_weight).mean() / sample_weight.mean()).compute()
        return err.mean().compute()
    else:
        err = (y_true - y_pred) / y_true
        err = cp.abs(err)
        if sample_weight is not None:
            return (err * sample_weight).mean() / sample_weight.mean()
        return err.mean()


def roc_auc_ovr_gpu(y_true, y_pred, sample_weight=None):
    """ROC-AUC One-Versus-Rest for Multi-GPU data.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        sample_weight: Weights of samples.

    Returns:
        Metric values.

    """
    if isinstance(y_true, da.Array):
        res = da.map_blocks(
            roc_auc_ovr_gpu,
            y_true,
            y_pred,
            sample_weight,
            meta=cp.array((), dtype=cp.float32),
            drop_axis=1,
        )
        return cp.array(res.compute()).mean()
    else:
        if isinstance(y_true, (cudf.Series, cudf.DataFrame)):
            y_pred = y_pred.values
            y_true = y_true.values
        n_classes = y_pred.shape[1]
        res = 0.0
        for i in range(n_classes):
            res += roc_auc_score(cp.where(y_true == i, 1, 0), y_pred[:, i])
        return res / n_classes


def rmsle_gpu(y_true, y_pred, sample_weight=None):
    """Root mean squared log error for Multi-GPU data.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        sample_weight: Weights of samples.

    Returns:
        Metric values.


    """

    if isinstance(y_true, da.Array):
        output_errors = da.subtract(da.log1p(y_true), da.log1p(y_pred))
        output_errors = da.multiply(output_errors, output_errors)
        if sample_weight is not None:
            output_errors = da.multiply(output_errors, sample_weight)
            output_errors = da.divide(da.sum(output_errors), sample_weight.sum())
        else:
            output_errors = da.mean(output_errors)
        return cp.sqrt(output_errors.compute())
    else:
        return mean_squared_log_error(
            y_true, y_pred, sample_weight=sample_weight, squared=False
        )


def auc_mu_gpu(
    y_true: cp.ndarray,
    y_pred: cp.ndarray,
    sample_weight: Optional[cp.ndarray] = None,
    class_weights: Optional[cp.ndarray] = None,
) -> float:
    """Compute multi-class metric AUC-Mu.

    We assume that confusion matrix full of ones, except diagonal elements.
    All diagonal elements are zeroes.
    By default, for averaging between classes scores we use simple mean.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        sample_weight: Not used.
        class_weights: The between classes weight matrix. If ``None``,
            the standard mean will be used. It is expected to be a lower
            triangular matrix (diagonal is also full of zeroes).
            In position (i, j), i > j, there is a partial positive score
            between i-th and j-th classes. All elements must sum up to 1.

    Returns:
        Metric value.

    Note:
        Code was refactored from https://github.com/kleimanr/auc_mu/blob/master/auc_mu.py

    """
    if isinstance(y_true, da.Array):
        raise NotImplementedError

    if not isinstance(y_pred, cp.ndarray):
        raise TypeError("Expected y_pred to be cp.ndarray, got: {}".format(type(y_pred)))
    if not y_pred.ndim == 2:
        raise ValueError("Expected array with predictions be a 2-dimentional array")
    if not isinstance(y_true, cp.ndarray):
        raise TypeError("Expected y_true to be cp.ndarray, got: {}".format(type(y_true)))
    if not y_true.ndim == 1:
        raise ValueError("Expected array with ground truths be a 1-dimentional array")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            "Expected number of samples in y_true and y_pred be same,"
            " got {} and {}, respectively".format(y_true.shape[0], y_pred.shape[0])
        )

    uniq_labels = cp.unique(y_true)
    n_samples, n_classes = y_pred.shape

    if not cp.all(uniq_labels == cp.arange(n_classes)):
        raise ValueError("Expected classes encoded values 0, ..., N_classes-1")

    if class_weights is None:
        class_weights = cp.tri(n_classes, k=-1)
        class_weights /= class_weights.sum()

    if not isinstance(class_weights, cp.ndarray):
        raise TypeError(
            "Expected class_weights to be cp.ndarray, got: {}".format(
                type(class_weights)
            )
        )
    if not class_weights.ndim == 2:
        raise ValueError("Expected class_weights to be a 2-dimentional array")
    if not class_weights.shape == (n_classes, n_classes):
        raise ValueError(
            "Expected class_weights size: {}, got: {}".format(
                (n_classes, n_classes), class_weights.shape
            )
        )
    # check sum?
    confusion_matrix = cp.ones((n_classes, n_classes)) - cp.eye(n_classes)
    auc_full = 0.0

    for class_i in range(n_classes):
        preds_i = y_pred[y_true == class_i]
        n_i = preds_i.shape[0]
        for class_j in range(class_i):
            preds_j = y_pred[y_true == class_j]
            n_j = preds_j.shape[0]
            n = n_i + n_j
            tmp_labels = cp.zeros((n,), dtype=cp.int32)
            tmp_labels[n_i:] = 1
            tmp_pres = cp.vstack((preds_i, preds_j))
            v = confusion_matrix[class_i, :] - confusion_matrix[class_j, :]
            scores = cp.dot(tmp_pres, v)
            score_ij = roc_auc_score(tmp_labels, scores)
            auc_full += class_weights[class_i, class_j] * score_ij

    return auc_full


# TODO: add the support for F1 score
# class F1Factory:
#     """
#     Wrapper for :func:`~sklearn.metrics.f1_score` function.
#     """
#
#     def __init__(self, average: str = 'micro'):
#         """
#
#         Args:
#             average: Averaging type ('micro', 'macro', 'weighted').
#
#         """
#         self.average = average
#
#     def __call__(self, y_true: cp.ndarray, y_pred: cp.ndarray,
#                  sample_weight: Optional[cp.ndarray] = None) -> float:
#         """Compute metric.
#
#         Args:
#             y_true: Ground truth target values.
#             y_pred: Estimated target values.
#             sample_weight: Sample weights.
#
#         Returns:
#             F1 score of the positive class in binary classification
#             or weighted average of the F1 scores of each class
#             for the multiclass task.
#
#         """
#         return f1_score(y_true, y_pred, sample_weight=sample_weight, average=self.average)


class BestClassBinaryWrapper_gpu:
    """Metric wrapper to get best class prediction instead of probs.

    There is cut-off for prediction by ``0.5``.

    """

    def __init__(self, func: Callable):
        """

        Args:
            func: Metric function. Function format:
               func(y_pred, y_true, weights, \*\*kwargs).

        """
        self.func = func

    def __call__(
        self,
        y_true: cp.ndarray,
        y_pred: cp.ndarray,
        sample_weight: Optional[cp.ndarray] = None,
        **kwargs
    ):
        y_pred = (y_pred > 0.5).astype(cp.float32)

        return self.func(y_true, y_pred)  # , sample_weight=sample_weight, **kwargs)


class AccuracyScoreWrapper:
    def __call__(
        self,
        y_true: cp.ndarray,
        y_pred: cp.ndarray,
        sample_weight: Optional[cp.ndarray] = None,
        **kwargs
    ):
        if type(y_pred) == cp.ndarray:
            return accuracy_score(
                y_true, y_pred
            )  # , sample_weight=sample_weight, **kwargs)
        elif type(y_pred) == da.Array:
            res = dask_accuracy_score(
                y_true, y_pred
            )  # , sample_weight=sample_weight, **kwargs)
            return res


class BestClassMulticlassWrapper_gpu:
    """Metric wrapper to get best class prediction instead of probs for multiclass.

    Prediction provides by argmax.

    """

    def __init__(self, func):
        """

        Args:
            func: Metric function. Function format:
               func(y_pred, y_true, weights, \*\*kwargs)

        """
        self.func = func

    def __call__(
        self,
        y_true: cp.ndarray,
        y_pred: cp.ndarray,
        sample_weight: Optional[cp.ndarray] = None,
        **kwargs
    ):

        if type(y_pred) == cp.ndarray:

            y_pred = (y_pred.argmax(axis=1)).astype(cp.float32)

        elif type(y_pred) == da.Array:

            def dask_argmax_gpu(data):
                res = cp.copy(data)
                res[:, 0] = data.argmax(axis=1).astype(cp.float32)
                return res

            y_pred = da.map_blocks(
                dask_argmax_gpu, y_pred, meta=cp.array((), dtype=cp.float32)
            )[:, 0]
            if y_true.ndim == 2:
                y_true = y_true[:, 0]

        elif isinstance(y_true, (cudf.Series, cudf.DataFrame)):
            y_pred = (y_pred.values.argmax(axis=1)).astype(cp.float32)
            y_true = y_true.values

        else:

            raise NotImplementedError

        return self.func(y_true, y_pred, sample_weight=sample_weight, **kwargs)


_valid_str_binary_metric_names_gpu = {
    "auc": roc_auc_score_gpu,
    "logloss": partial(log_loss_gpu, eps=1e-7),
    "accuracy": BestClassBinaryWrapper_gpu(AccuracyScoreWrapper),
}

_valid_str_reg_metric_names_gpu = {
    "r2": r2_score_gpu,
    "mse": mean_squared_error_gpu,
    "mae": mean_absolute_error_gpu,
    "rmsle": rmsle_gpu,
    "fair": mean_fair_error_gpu,
    "huber": mean_huber_error_gpu,
    "quantile": mean_quantile_error_gpu,
    "mape": mean_absolute_percentage_error_gpu,
}

_valid_str_multiclass_metric_names_gpu = {
    "auc_mu": auc_mu_gpu,
    "auc": roc_auc_ovr_gpu,
    "crossentropy": partial(log_loss_gpu, eps=1e-7),
    "accuracy": BestClassMulticlassWrapper_gpu(AccuracyScoreWrapper()),
    # TODO: uncomment after f1 score support is added
    # 'f1_macro': BestClassMulticlassWrapper_gpu(F1Factory('macro')),
    # 'f1_micro': BestClassMulticlassWrapper_gpu(F1Factory('micro')),
    # 'f1_weighted': BestClassMulticlassWrapper_gpu(F1Factory('weighted')),
}

_valid_str_metric_names_gpu = {
    "binary": _valid_str_binary_metric_names_gpu,
    "reg": _valid_str_reg_metric_names_gpu,
    "multiclass": _valid_str_multiclass_metric_names_gpu,
}
