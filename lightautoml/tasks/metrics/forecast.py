"""Evaluation of rolling-origin forecasts against a single observed time series."""

import numpy as np

from .regression import _weights


def _rolling_error(y_true, y_pred, sample_weight, start_indices, squared):
    target = np.asarray(y_true, dtype=float)
    if target.ndim != 1 or not len(target) or not np.isfinite(target).all():
        raise ValueError("Targets must be a nonempty finite one-dimensional series.")
    forecasts = list(y_pred)
    if not forecasts:
        raise ValueError("At least one forecast is required.")
    weights = _weights(sample_weight, len(forecasts))
    starts = np.arange(len(forecasts)) if start_indices is None else np.asarray(start_indices)
    if starts.shape != (len(forecasts),) or not np.issubdtype(starts.dtype, np.integer) or (starts < 0).any():
        raise ValueError("start_indices must contain one nonnegative integer per forecast.")
    errors, used_weights = [], []
    horizon = None
    for forecast, start, weight in zip(forecasts, starts, weights):
        forecast = np.asarray(forecast, dtype=float)
        # The work format has NaN placeholders before the first forecast origin.
        if np.isnan(forecast).all():
            if horizon is not None:
                raise ValueError("Missing forecasts are allowed only before the first forecast.")
            continue
        if forecast.ndim != 1 or not forecast.size or not np.isfinite(forecast).all():
            raise ValueError("Each forecast must be a finite nonempty horizon vector.")
        if horizon is None:
            horizon = len(forecast)
        if len(forecast) != horizon:
            raise ValueError("All forecasts must have the same horizon.")
        if start + horizon > len(target) or weight == 0:
            continue
        residual = target[start : start + horizon] - forecast
        errors.append(np.mean(residual ** 2 if squared else np.abs(residual)))
        used_weights.append(weight)
    if not errors:
        raise ValueError("No complete positive-weight forecast horizon overlaps the observed series.")
    return float(np.average(errors, weights=used_weights))


def mean_mae(y_true, y_pred, sample_weight=None, start_indices=None):
    """Return mean horizon MAE; forecast row i starts at i, or at start_indices[i] if supplied."""
    return _rolling_error(y_true, y_pred, sample_weight, start_indices, False)


def mean_mse(y_true, y_pred, sample_weight=None, start_indices=None):
    """Return mean horizon MSE; skip leading NaN placeholders and incomplete trailing horizons."""
    return _rolling_error(y_true, y_pred, sample_weight, start_indices, True)
