"""Bunch of metrics with unified interface."""

from functools import partial
from typing import Callable
from typing import Optional

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score


def mean_quantile_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    q: float = 0.9,
) -> float:
    """Computes Mean Quantile Error.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        sample_weight: Specify weighted mean.
        q: Metric coefficient.

    Returns:
        metric value.

    """
    err = y_pred - y_true
    s = np.sign(err)
    err = np.abs(err)
    err = np.where(s > 0, q, 1 - q) * err
    if sample_weight is not None:
        return (err * sample_weight).mean() / sample_weight.mean()

    return err.mean()


def mean_huber_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    a: float = 0.9,
) -> float:
    """Computes Mean Huber Error.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        sample_weight: Specify weighted mean.
        a: Metric coefficient.

    Returns:
        Metric value.

    """
    err = y_pred - y_true
    s = np.abs(err) < a
    err = np.where(s, 0.5 * (err ** 2), a * np.abs(err) - 0.5 * (a ** 2))

    if sample_weight is not None:
        return (err * sample_weight).mean() / sample_weight.mean()

    return err.mean()


def mean_fair_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    c: float = 0.9,
) -> float:
    """Computes Mean Fair Error.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        sample_weight: Specify weighted mean.
        c: Metric coefficient.

    Returns:
        Metric value.

    """
    x = np.abs(y_pred - y_true) / c
    err = c ** 2 * (x - np.log(x + 1))

    if sample_weight is not None:
        return (err * sample_weight).mean() / sample_weight.mean()

    return err.mean()


def mean_absolute_percentage_error(
    y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None
) -> float:
    """Computes Mean Absolute Percentage error.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        sample_weight: Specify weighted mean.

    Returns:
        Metric value.

    """
    err = (y_true - y_pred) / y_true
    err = np.abs(err)

    if sample_weight is not None:
        return (err * sample_weight).mean() / sample_weight.mean()

    return err.mean()


def roc_auc_ovr(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None):
    """ROC-AUC One-Versus-Rest.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        sample_weight: Weights of samples.

    Returns:
        Metric values.

    """

    return roc_auc_score(y_true, y_pred, sample_weight=sample_weight, multi_class="ovr")


def rmsle(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None):
    """Root mean squared log error.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        sample_weight: Weights of samples.

    Returns:
        Metric values.


    """
    return np.sqrt(mean_squared_log_error(y_true, y_pred, sample_weight=sample_weight))


def auc_mu(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    class_weights: Optional[np.ndarray] = None,
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

    if not isinstance(y_pred, np.ndarray):
        raise TypeError("Expected y_pred to be np.ndarray, got: {}".format(type(y_pred)))
    if not y_pred.ndim == 2:
        raise ValueError("Expected array with predictions be a 2-dimentional array")
    if not isinstance(y_true, np.ndarray):
        raise TypeError("Expected y_true to be np.ndarray, got: {}".format(type(y_true)))
    if not y_true.ndim == 1:
        raise ValueError("Expected array with ground truths be a 1-dimentional array")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            "Expected number of samples in y_true and y_pred be same,"
            " got {} and {}, respectively".format(y_true.shape[0], y_pred.shape[0])
        )

    uniq_labels = np.unique(y_true)
    n_samples, n_classes = y_pred.shape

    if not np.all(uniq_labels == np.arange(n_classes)):
        raise ValueError("Expected classes encoded values 0, ..., N_classes-1")

    if class_weights is None:
        class_weights = np.tri(n_classes, k=-1)
        class_weights /= class_weights.sum()

    if not isinstance(class_weights, np.ndarray):
        raise TypeError("Expected class_weights to be np.ndarray, got: {}".format(type(class_weights)))
    if not class_weights.ndim == 2:
        raise ValueError("Expected class_weights to be a 2-dimentional array")
    if not class_weights.shape == (n_classes, n_classes):
        raise ValueError("Expected class_weights size: {}, got: {}".format((n_classes, n_classes), class_weights.shape))
    # check sum?
    confusion_matrix = np.ones((n_classes, n_classes)) - np.eye(n_classes)
    auc_full = 0.0

    for class_i in range(n_classes):
        preds_i = y_pred[y_true == class_i]
        n_i = preds_i.shape[0]
        for class_j in range(class_i):
            preds_j = y_pred[y_true == class_j]
            n_j = preds_j.shape[0]
            n = n_i + n_j
            tmp_labels = np.zeros((n,), dtype=np.int32)
            tmp_labels[n_i:] = 1
            tmp_pres = np.vstack((preds_i, preds_j))
            v = confusion_matrix[class_i, :] - confusion_matrix[class_j, :]
            scores = np.dot(tmp_pres, v)
            score_ij = roc_auc_score(tmp_labels, scores)
            auc_full += class_weights[class_i, class_j] * score_ij

    return auc_full


class F1Factory:
    """
    Wrapper for :func:`~sklearn.metrics.f1_score` function.
    """

    def __init__(self, average: str = "micro"):
        """

        Args:
            average: Averaging type ('micro', 'macro', 'weighted').

        """
        self.average = average

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """Compute metric.

        Args:
            y_true: Ground truth target values.
            y_pred: Estimated target values.
            sample_weight: Sample weights.

        Returns:
            F1 score of the positive class in binary classification
            or weighted average of the F1 scores of each class
            for the multiclass task.

        """
        return f1_score(y_true, y_pred, sample_weight=sample_weight, average=self.average)


class BestClassBinaryWrapper:
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

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None, **kwargs):
        y_pred = (y_pred > 0.5).astype(np.float32)

        return self.func(y_true, y_pred, sample_weight=sample_weight, **kwargs)


class BestClassMulticlassWrapper:
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

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None, **kwargs):
        y_pred = (y_pred.argmax(axis=1)).astype(np.float32)

        return self.func(y_true, y_pred, sample_weight=sample_weight, **kwargs)


# TODO: Add custom metrics - precision/recall/fscore at K. Fscore at best co
# TODO: Move to other module


_valid_str_binary_metric_names = {
    "auc": roc_auc_score,
    "logloss": partial(log_loss, eps=1e-7),
    "accuracy": BestClassBinaryWrapper(accuracy_score),
}

_valid_str_reg_metric_names = {
    "r2": r2_score,
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "rmsle": rmsle,
    "fair": mean_fair_error,
    "huber": mean_huber_error,
    "quantile": mean_quantile_error,
    "mape": mean_absolute_percentage_error,
}

_valid_str_multiclass_metric_names = {
    "auc_mu": auc_mu,
    "auc": roc_auc_ovr,
    "crossentropy": partial(log_loss, eps=1e-7),
    "accuracy": BestClassMulticlassWrapper(accuracy_score),
    "f1_macro": BestClassMulticlassWrapper(F1Factory("macro")),
    "f1_micro": BestClassMulticlassWrapper(F1Factory("micro")),
    "f1_weighted": BestClassMulticlassWrapper(F1Factory("weighted")),
}

_valid_str_metric_names = {
    "binary": _valid_str_binary_metric_names,
    "reg": _valid_str_reg_metric_names,
    "multiclass": _valid_str_multiclass_metric_names,
}

_valid_metric_args = {"quantile": ["q"], "huber": ["a"], "fair": ["c"]}
