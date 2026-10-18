"""Ranking metrics for a binary sample or a matrix of query-item scores."""

import numpy as np

from sklearn.metrics import average_precision_score
from sklearn.metrics import ndcg_score

from .regression import _arrays
from .regression import _weights


def _count(k, size):
    if k is None:
        return size
    if isinstance(k, (bool, np.bool_)):
        raise ValueError("k must be a positive integer or a fraction in (0, 1].")
    if isinstance(k, (int, np.integer)) and k > 0:
        return min(int(k), size)
    if isinstance(k, (float, np.floating)) and np.isfinite(k) and 0 < k <= 1:
        return max(1, int(size * k))
    raise ValueError("k must be a positive integer or a fraction in (0, 1].")


def _binary_ranking(y_true, y_pred, sample_weight, k):
    target, pred = np.asarray(y_true), np.asarray(y_pred, dtype=float)
    if pred.ndim == 2 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if target.ndim == 2 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 1 or target.shape != pred.shape or not len(target):
        raise ValueError("Ranking requires matching nonempty one-dimensional arrays.")
    if not np.isin(target, [0, 1]).all() or not np.isfinite(pred).all():
        raise ValueError("Ranking requires binary relevance and finite scores.")
    weights = _weights(sample_weight, len(target))
    # Zero-weight observations do not occupy ranks. Ties retain input order.
    mask = weights > 0
    target, pred, weights = target[mask], pred[mask], weights[mask]
    order = np.argsort(-pred, kind="mergesort")
    return target[order], pred[order], weights[order], _count(k, len(order))


def precision_at_k(y_true, y_pred, sample_weight=None, k=100):
    """Return weighted precision in the top k rows; tied scores retain input order."""
    target, _, weights, count = _binary_ranking(y_true, y_pred, sample_weight, k)
    return float(np.average(target[:count], weights=weights[:count]))


def recall_at_k(y_true, y_pred, sample_weight=None, k=100):
    """Return top-k positive weight divided by all positive weight; return zero if none is relevant."""
    target, _, weights, count = _binary_ranking(y_true, y_pred, sample_weight, k)
    relevant = target * weights
    return float(relevant[:count].sum() / relevant.sum()) if relevant.sum() else 0.0


def average_precision_at_k(y_true, y_pred, sample_weight=None, k=100):
    """Return truncated AP, normalized by all relevant weight, including positives outside top k."""
    target, _, weights, count = _binary_ranking(y_true, y_pred, sample_weight, k)
    relevant = target * weights
    if not relevant.sum():
        return 0.0
    precision = np.cumsum(relevant[:count]) / np.cumsum(weights[:count])
    return float(np.dot(precision, relevant[:count]) / relevant.sum())


def average_precision_in_top_k(y_true, y_pred, sample_weight=None, k=100):
    """Return scikit-learn AP evaluated only within the top-k subset (legacy work formula)."""
    target, pred, weights, count = _binary_ranking(y_true, y_pred, sample_weight, k)
    if not target[:count].any():
        return 0.0
    return float(average_precision_score(target[:count], pred[:count], sample_weight=weights[:count]))


def mean_average_precision_at_k(y_true, y_pred, sample_weight=None, k=100, within_top_k=False):
    """Return mean AP@k for query-item matrices; sample weights weight queries, not items."""
    target, pred, weights = _arrays(y_true, y_pred, sample_weight)
    if np.asarray(y_true).ndim != 2:
        raise ValueError("MAP requires a two-dimensional query-item matrix.")
    metric = average_precision_in_top_k if within_top_k else average_precision_at_k
    scores = [metric(row, prediction, k=k) for row, prediction in zip(target, pred)]
    return float(np.average(scores, weights=weights))


def ndcg_at_k(y_true, y_pred, sample_weight=None, k=None):
    """Return scikit-learn NDCG for query-item matrices, averaging over score ties."""
    target, pred, weights = _arrays(y_true, y_pred, sample_weight)
    if np.asarray(y_true).ndim != 2 or target.shape[1] < 2:
        raise ValueError("NDCG requires a query-item matrix with at least two items per query.")
    if (target < 0).any():
        raise ValueError("NDCG relevance must be nonnegative.")
    return float(ndcg_score(target, pred, k=_count(k, target.shape[1]), sample_weight=weights))


def reciprocal_rank_score(y_true, y_pred, sample_weight=None, k=None):
    """Return mean reciprocal rank for binary query-item relevance and score matrices."""
    target, pred, weights = _arrays(y_true, y_pred, sample_weight)
    if np.asarray(y_true).ndim != 2 or not np.isin(target, [0, 1]).all():
        raise ValueError("MRR requires a binary query-item relevance matrix.")
    count = _count(k, target.shape[1])
    order = np.argsort(-pred, axis=1, kind="mergesort")[:, :count]
    relevant = np.take_along_axis(target, order, axis=1)
    scores = np.where(relevant.any(axis=1), 1 / (relevant.argmax(axis=1) + 1), 0)
    return float(np.average(scores, weights=weights))


def mean_reciprocal_rank(y_true, y_pred, sample_weight=None):
    """Return MRR for lists of relevant item IDs and ordered predicted item IDs."""
    if len(y_true) != len(y_pred) or not len(y_true):
        raise ValueError("MRR requires matching nonempty query collections.")
    scores = []
    for relevant, predicted in zip(y_true, y_pred):
        relevant = set(relevant)
        scores.append(next((1 / rank for rank, item in enumerate(predicted, 1) if item in relevant), 0))
    return float(np.average(scores, weights=_weights(sample_weight, len(scores))))
