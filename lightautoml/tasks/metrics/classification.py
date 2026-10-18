"""Probability and label metrics for classification."""

import numpy as np

from sklearn.metrics import hamming_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from .regression import _arrays
from .regression import _weights


def label_metric(y_true, y_pred, sample_weight=None, metric=None, multiclass=False, cutoff=0.5, **kwargs):
    """Evaluate a label metric using argmax or a configurable probability cutoff."""
    if not np.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("cutoff must be between zero and one.")
    pred = np.asarray(y_pred)
    if not np.isfinite(pred).all():
        raise ValueError("Predictions must be finite.")
    labels = pred.argmax(axis=1) if multiclass else (pred >= cutoff).astype(int)
    return float(metric(y_true, labels, sample_weight=sample_weight, **kwargs))


def gini(y_true, y_pred, sample_weight=None):
    """Return binary Gini, equal to twice ROC AUC minus one, on the [-1, 1] scale."""
    return 2 * roc_auc_score(y_true, y_pred, sample_weight=sample_weight) - 1


def hamming_accuracy(y_true, y_pred, sample_weight=None):
    """Return the fraction of correct labels (one minus Hamming loss)."""
    return 1 - hamming_loss(y_true, y_pred, sample_weight=sample_weight)


def expected_calibration_error(y_true, y_pred, sample_weight=None, n_bins=10):
    """Return equal-width L1 ECE, using positive-class probabilities or multiclass top-label confidence."""
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")
    target, pred = np.asarray(y_true), np.asarray(y_pred, dtype=float)
    if pred.ndim == 2 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if pred.ndim not in (1, 2) or target.ndim != 1 or len(target) != len(pred) or len(target) == 0:
        raise ValueError("ECE requires one-dimensional targets and one- or two-dimensional predictions.")
    if not np.isfinite(pred).all() or ((pred < 0) | (pred > 1)).any():
        raise ValueError("ECE predictions must be probabilities in [0, 1].")
    if pred.ndim == 2:
        if pred.shape[1] < 2 or not np.allclose(pred.sum(axis=1), 1):
            raise ValueError("Multiclass probabilities must sum to one.")
        if not np.isin(target, np.arange(pred.shape[1])).all():
            raise ValueError("Targets must index probability columns.")
        target, pred = (pred.argmax(axis=1) == target).astype(float), pred.max(axis=1)
    elif not np.isin(target, [0, 1]).all():
        raise ValueError("Binary targets must be zero or one.")
    weights = _weights(sample_weight, len(target))
    # Right-closed bins, including zero in the first and one in the last bin.
    bins = np.searchsorted(np.linspace(0, 1, n_bins + 1)[1:-1], pred, side="left")
    residual = np.bincount(bins, weights=weights * (target - pred), minlength=n_bins)
    return float(np.abs(residual).sum() / weights.sum())


def recall_at_specificity(y_true, y_pred, sample_weight=None, threshold=0.99):
    """Return maximum recall at specificity at least threshold, without ROC interpolation."""
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one.")
    target = np.asarray(y_true)
    weights = _weights(sample_weight, len(target))
    if not np.array_equal(np.unique(target[weights > 0]), [0, 1]):
        raise ValueError("Recall at specificity requires both binary classes with positive weight.")
    fpr, recall, _ = roc_curve(target, y_pred, sample_weight=weights)
    return float(recall[1 - fpr >= threshold].max())


def partial_auc(y_true, y_pred, sample_weight=None, threshold=0.99):
    """Return scikit-learn's standardized partial ROC AUC up to FPR=threshold."""
    return float(roc_auc_score(y_true, y_pred, sample_weight=sample_weight, max_fpr=threshold))


def roc_auc_with_nans(y_true, y_pred, sample_weight=None):
    """Return macro ROC AUC over valid target columns, ignoring missing targets and single-class columns."""
    target, pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if target.ndim != 2 or target.shape != pred.shape or not all(target.shape):
        raise ValueError("Targets and predictions must be matching nonempty two-dimensional arrays.")
    if np.isinf(target).any() or not np.isfinite(pred).all():
        raise ValueError("Only targets may contain NaN; all other values must be finite.")
    if not np.isin(target[~np.isnan(target)], [0, 1]).all():
        raise ValueError("Observed targets must be zero or one.")
    weights = _weights(sample_weight, len(target))
    scores = []
    for column, prediction in zip(target.T, pred.T):
        mask = ~np.isnan(column) & (weights > 0)
        if np.unique(column[mask]).size == 2:
            scores.append(roc_auc_score(column[mask], prediction[mask], sample_weight=weights[mask]))
    if not scores:
        raise ValueError("No target column contains both classes with positive weight.")
    return float(np.mean(scores))


def gini_with_nans(y_true, y_pred, sample_weight=None):
    """Return macro Gini on the [-100, 100] scale, ignoring missing targets and single-class columns."""
    return (2 * roc_auc_with_nans(y_true, y_pred, sample_weight) - 1) * 100


def multilabel_brier_loss(y_true, y_pred, sample_weight=None):
    """Return mean independent-label Brier loss, without normalizing rows to sum to one."""
    target, pred, weights = _arrays(y_true, y_pred, sample_weight)
    if not np.isin(target, [0, 1]).all() or ((pred < 0) | (pred > 1)).any():
        raise ValueError("Brier loss requires binary targets and probabilities in [0, 1].")
    return float(np.average((target - pred) ** 2, axis=0, weights=weights).mean())


def multilabel_ece(y_true, y_pred, sample_weight=None, n_bins=10):
    """Return the mean of independent binary label calibration errors."""
    target, pred, weights = _arrays(y_true, y_pred, sample_weight)
    return float(np.mean([expected_calibration_error(t, p, weights, n_bins) for t, p in zip(target.T, pred.T)]))
