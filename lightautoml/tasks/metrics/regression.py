"""Regression metrics with row weights and uniform averaging over outputs."""

import numpy as np

from scipy.stats import rankdata
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def _arrays(y_true, y_pred, sample_weight=None):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    if y_true.ndim != 2 or y_true.shape != y_pred.shape or not all(y_true.shape):
        raise ValueError("Targets and predictions must have the same nonempty shape (rows, outputs).")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Targets and predictions must be finite.")
    weights = _weights(sample_weight, len(y_true))
    return y_true, y_pred, weights


def _weights(sample_weight, size):
    weights = np.ones(size) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if weights.shape != (size,) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("sample_weight must contain one finite nonnegative weight per row.")
    if weights.sum() <= 0:
        raise ValueError("sample_weight must have positive total weight.")
    return weights


def _mean(values, weights):
    return float(np.average(values, axis=0, weights=weights).mean())


def rmse(y_true, y_pred, sample_weight=None):
    """Return the mean of per-output RMSEs, as in scikit-learn."""
    return float(
        np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight, multioutput="raw_values")).mean()
    )


def mape(y_true, y_pred, sample_weight=None):
    """Return relative MAPE with the scikit-learn float64 epsilon convention."""
    target, pred, weights = _arrays(y_true, y_pred, sample_weight)
    return _mean(np.abs(target - pred) / np.maximum(np.abs(target), np.finfo(float).eps), weights)


def median_absolute_percentage_error(y_true, y_pred, sample_weight=None):
    """Return median absolute percentage error (0--100 scale), averaged over outputs."""
    target, pred, weights = _arrays(y_true, y_pred, sample_weight)
    errors = np.abs(target - pred) / np.maximum(np.abs(target), np.finfo(float).eps)
    if sample_weight is None:
        return float(np.median(errors, axis=0).mean() * 100)
    medians = []
    for column in errors.T:
        order = np.argsort(column, kind="mergesort")
        cumulative = np.cumsum(weights[order])
        medians.append(column[order[np.searchsorted(cumulative, weights.sum() / 2)]])
    return float(np.mean(medians) * 100)


def smape(y_true, y_pred, sample_weight=None):
    """Return symmetric MAPE in [0, 200]; a zero/zero pair contributes zero."""
    target, pred, weights = _arrays(y_true, y_pred, sample_weight)
    denominator = np.abs(target) + np.abs(pred)
    errors = np.divide(np.abs(target - pred), denominator, out=np.zeros_like(target), where=denominator != 0)
    return 200 * _mean(errors, weights)


def _relative_mae(y_true, y_pred, sample_weight, absolute):
    target, pred, weights = _arrays(y_true, y_pred, sample_weight)
    denominator = np.average(np.abs(target) if absolute else target, axis=0, weights=weights)
    if (denominator <= 0).any():
        raise ValueError("Relative MAE requires a positive target mean for every output.")
    return float((np.average(np.abs(target - pred), axis=0, weights=weights) / denominator).mean())


def mae_to_mean(y_true, y_pred, sample_weight=None):
    """Return MAE divided by mean target; require a positive mean for every output."""
    return _relative_mae(y_true, y_pred, sample_weight, False)


def mae_to_module_mean(y_true, y_pred, sample_weight=None):
    """Return WAPE: MAE divided by mean absolute target, averaged over outputs."""
    return _relative_mae(y_true, y_pred, sample_weight, True)


def one_minus_wape(y_true, y_pred, sample_weight=None):
    """Return one minus WAPE; the denominator uses absolute targets."""
    return 1 - mae_to_module_mean(y_true, y_pred, sample_weight)


def mape_without_zeros(y_true, y_pred, sample_weight=None):
    """Return relative MAPE excluding zero targets separately for each output."""
    target, pred, weights = _arrays(y_true, y_pred, sample_weight)
    values = []
    for column, prediction in zip(target.T, pred.T):
        mask = (column != 0) & (weights > 0)
        if not mask.any():
            raise ValueError("MAPE without zeros requires a nonzero target with positive weight in every output.")
        values.append(np.average(np.abs((column[mask] - prediction[mask]) / column[mask]), weights=weights[mask]))
    return float(np.mean(values))


def adjusted_r2(y_true, y_pred, sample_weight=None, n_features=None):
    """Return adjusted R2 using an explicit feature count and positive-weight row count."""
    target, pred, weights = _arrays(y_true, y_pred, sample_weight)
    n = np.count_nonzero(weights)
    if isinstance(n_features, (bool, np.bool_)) or not isinstance(n_features, (int, np.integer)) or n_features < 0:
        raise ValueError("n_features must be a nonnegative integer.")
    if n <= n_features + 1:
        raise ValueError("Adjusted R2 requires more than n_features + 1 positive-weight observations.")
    return float(1 - (1 - r2_score(target, pred, sample_weight=weights)) * (n - 1) / (n - n_features - 1))


def spearman(y_true, y_pred, sample_weight=None):
    """Return mean Spearman correlation; weights apply to the Pearson correlation of ranks."""
    target, pred, weights = _arrays(y_true, y_pred, sample_weight)
    mask = weights > 0
    values = []
    for column, prediction in zip(target[mask].T, pred[mask].T):
        x, y = rankdata(column), rankdata(prediction)
        x -= np.average(x, weights=weights[mask])
        y -= np.average(y, weights=weights[mask])
        denominator = np.sqrt(np.dot(weights[mask], x * x) * np.dot(weights[mask], y * y))
        if denominator == 0:
            raise ValueError("Spearman correlation is undefined for constant targets or predictions.")
        values.append(np.dot(weights[mask], x * y) / denominator)
    return float(np.mean(values))


def powerstat(y_true, y_pred, sample_weight=None):
    """Return normalized Gini on a percentage scale, averaging tied-score Lorenz segments."""
    target, pred, weights = _arrays(y_true, y_pred, sample_weight)
    if (target < 0).any():
        raise ValueError("Powerstat requires nonnegative targets.")
    mask = weights > 0
    target, pred, weights = target[mask], pred[mask], weights[mask]

    def area(column, prediction):
        order = np.argsort(-prediction, kind="mergesort")
        ends = np.r_[np.flatnonzero(np.diff(prediction[order])), len(order) - 1]
        x = np.r_[0, np.cumsum(weights[order])[ends] / weights.sum()]
        y = np.r_[0, np.cumsum(column[order] * weights[order])[ends] / np.dot(column, weights)]
        return np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2) - 0.5

    values = []
    for column, prediction in zip(target.T, pred.T):
        if np.all(column == column[0]):
            raise ValueError("Powerstat is undefined for constant targets.")
        values.append(area(column, prediction) / area(column, column))
    return float(np.mean(values) * 100)
