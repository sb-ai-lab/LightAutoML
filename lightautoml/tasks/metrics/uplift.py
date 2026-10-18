"""Binary-outcome metrics adapted from scikit-uplift 0.5.1 (MIT).

Copyright (c) 2019 Maksim Shevchenko. See ``SCIKIT_UPLIFT_LICENSE`` in this
directory. Adaptations use NumPy cumulative sums and validate undefined
normalizations and empty groups. Existing addon curve conventions are unchanged.
"""

import numpy as np


def _check_groups(name, pred, treated, kwargs):
    if name not in ("uplift_at_k", "weighted_average_uplift"):
        return
    strategy = kwargs["strategy"]
    if strategy not in ("overall", "by_group"):
        raise ValueError("strategy must be 'overall' or 'by_group'.")
    order = np.argsort(pred, kind="mergesort")[::-1]
    groups = treated[order]
    if name == "uplift_at_k":
        k = kwargs["k"]
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, float, np.integer, np.floating)):
            raise ValueError("k must be an integer count or a fraction.")
        fractional = isinstance(k, (float, np.floating))
        if not np.isfinite(k) or k <= 0 or k >= (1 if fractional else len(pred)):
            raise ValueError("k must be smaller than the sample size, or a fraction strictly between zero and one.")
        if strategy == "overall":
            count = int(len(pred) * k) if fractional else int(k)
            if np.unique(groups[:count]).size != 2:
                raise ValueError("Both treatment groups must occur in the top-k selection.")
        else:
            sizes = np.bincount(treated.astype(int), minlength=2)
            counts = (sizes * k).astype(int) if fractional else np.full(2, k)
            if (counts < 1).any() or (counts > sizes).any():
                raise ValueError("k must select at least one observation and fit within each treatment group.")
    else:
        bins = kwargs["bins"]
        if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)) or not 0 < bins < len(pred):
            raise ValueError("bins must be a positive integer smaller than the sample size.")
        if strategy == "overall":
            if any(np.unique(group).size != 2 for group in np.array_split(groups, bins)):
                raise ValueError("Both treatment groups must occur in every score bin.")
        elif np.bincount(treated.astype(int), minlength=2).min() < bins:
            raise ValueError("Every treatment group must have at least as many observations as bins.")
    return order


def _inputs(y_true, y_pred, treatment):
    target, pred, treated = np.asarray(y_true), np.asarray(y_pred), np.asarray(treatment)
    if target.ndim != 1 or target.shape != pred.shape or target.shape != treated.shape or not len(target):
        raise ValueError("Targets, scores and treatment must be matching nonempty one-dimensional arrays.")
    if not np.isin(target, [0, 1]).all() or not np.array_equal(np.unique(treated), [0, 1]):
        raise ValueError("Uplift metrics require binary targets and both treatment groups encoded as 0 and 1.")
    if not np.isfinite(pred).all():
        raise ValueError("Uplift scores must be finite.")
    return target.astype(float), pred.astype(float), treated.astype(float)


def _curve(target, pred, treated, qini):
    order = np.argsort(pred, kind="mergesort")[::-1]
    target, pred, treated = target[order], pred[order], treated[order]
    ends = np.r_[np.flatnonzero(np.diff(pred)), len(pred) - 1]
    count = ends + 1
    count_t = np.cumsum(treated, dtype=float)[ends]
    count_c = count - count_t
    response_t = np.cumsum(target * treated, dtype=float)[ends]
    response_c = np.cumsum(target * (1 - treated), dtype=float)[ends]
    rate_c = np.divide(response_c, count_c, out=np.zeros_like(response_c), where=count_c != 0)
    if qini:
        values = response_t - rate_c * count_t
    else:
        rate_t = np.divide(response_t, count_t, out=np.zeros_like(response_t), where=count_t != 0)
        values = (rate_t - rate_c) * count
    return np.r_[0, count], np.r_[0, values]


def _normalized_area(actual, perfect):
    def area(points):
        x, y = points
        return np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2)

    baseline = perfect[0][-1] * perfect[1][-1] / 2
    denominator = area(perfect) - baseline
    if denominator <= 0:
        raise ValueError("Uplift metric is undefined: the perfect curve has no positive area above the baseline.")
    return float((area(actual) - baseline) / denominator)


def uplift_at_k(y_true, y_pred, treatment, k=0.5, strategy="overall"):
    """Return scikit-uplift top-k treatment minus control response rate."""
    target, pred, treated = _inputs(y_true, y_pred, treatment)
    order = _check_groups("uplift_at_k", pred, treated, {"k": k, "strategy": strategy})
    target, treated = target[order], treated[order]
    if strategy == "overall":
        count = int(k * len(target)) if isinstance(k, (float, np.floating)) else int(k)
        target, treated = target[:count], treated[:count]
        return float(target[treated == 1].mean() - target[treated == 0].mean())
    groups = [target[treated == group] for group in (0, 1)]
    counts = [int(k * len(group)) if isinstance(k, (float, np.floating)) else int(k) for group in groups]
    return float(groups[1][: counts[1]].mean() - groups[0][: counts[0]].mean())


def uplift_auc_score(y_true, y_pred, treatment):
    """Return scikit-uplift normalized uplift AUC for binary outcomes."""
    target, pred, treated = _inputs(y_true, y_pred, treatment)
    control_responders = np.sum((target == 1) & (treated == 0))
    treated_nonresponders = np.sum((target == 0) & (treated == 1))
    summand = target if control_responders > treated_nonresponders else treated
    ideal = 2 * (target == treated) + summand
    return _normalized_area(_curve(target, pred, treated, False), _curve(target, ideal, treated, False))


def qini_auc_score(y_true, y_pred, treatment, negative_effect=True):
    """Return scikit-uplift normalized Qini AUC for binary outcomes."""
    target, pred, treated = _inputs(y_true, y_pred, treatment)
    if not isinstance(negative_effect, (bool, np.bool_)):
        raise ValueError("negative_effect must be a boolean.")
    actual = _curve(target, pred, treated, True)
    if negative_effect:
        perfect = _curve(target, target * (2 * treated - 1), treated, True)
    else:
        endpoint = actual[1][-1]
        if endpoint <= 0:
            raise ValueError("Qini without negative effects requires positive overall uplift.")
        perfect = np.array([0, endpoint, len(target)]), np.array([0, endpoint, endpoint])
    return _normalized_area(actual, perfect)


def weighted_average_uplift(y_true, y_pred, treatment, strategy="overall", bins=10):
    """Return treatment-count-weighted average uplift over score percentile bins."""
    target, pred, treated = _inputs(y_true, y_pred, treatment)
    order = _check_groups("weighted_average_uplift", pred, treated, {"strategy": strategy, "bins": bins})
    target, treated = target[order], treated[order]
    if strategy == "overall":
        indices = np.array_split(np.arange(len(target)), bins)
        controls = [target[idx][treated[idx] == 0] for idx in indices]
        treatments = [target[idx][treated[idx] == 1] for idx in indices]
    else:
        controls = np.array_split(target[treated == 0], bins)
        treatments = np.array_split(target[treated == 1], bins)
    effects = [t.mean() - c.mean() for t, c in zip(treatments, controls)]
    return float(np.average(effects, weights=[len(t) for t in treatments]))
