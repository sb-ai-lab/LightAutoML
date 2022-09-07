"""."""

from typing import Callable

import numpy as np


def infer_gib(metric: Callable) -> bool:
    """Infer greater is better from metric.

    Args:
        metric: Score or loss function.

    Returns:
        ```True``` if grater is better.

    Raises:
        AssertionError: If there is no way to order the predictions.

    """
    label = np.array([0, 1])
    pred = np.array([0.1, 0.9])

    g_val = metric(label, pred)
    b_val = metric(label, pred[::-1])

    assert g_val != b_val, "Cannot infer greater is better from metric." " Should be set manually."

    return g_val > b_val


def infer_gib_multiclass(metric: Callable) -> bool:
    """Infer greater is better from metric.

    Args:
        metric: Metric function. It must take two
          arguments y_true, y_pred.

    Returns:
        ```True``` if grater is better.

    Raises:
        AssertionError: If there is no way to order the predictions.

    """
    label = np.array([0, 1, 2])
    pred = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])

    g_val = metric(label, pred)
    b_val = metric(label, pred[::-1])

    assert g_val != b_val, "Cannot infer greater is better from metric. " "Should be set manually."

    return g_val > b_val
