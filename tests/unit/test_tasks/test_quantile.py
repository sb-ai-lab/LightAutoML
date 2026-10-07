import numpy as np
import pytest
import torch

from sklearn.metrics import mean_pinball_loss

from lightautoml.tasks.common_metric import mean_quantile_error
from lightautoml.tasks.losses.torch import torch_quantile


@pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
def test_quantile_metric_matches_pinball_loss(q):
    # under-prediction is weighted by q and over-prediction by 1 - q,
    # as in sklearn and in the `alpha` of the gradient boosting losses
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=100)
    y_pred = rng.normal(size=100)

    np.testing.assert_allclose(mean_quantile_error(y_true, y_pred, q=q), mean_pinball_loss(y_true, y_pred, alpha=q))


def test_quantile_metric_best_constant_is_the_quantile():
    rng = np.random.default_rng(0)
    y = rng.normal(size=20000)
    grid = np.linspace(-3, 3, 601)
    for q in (0.1, 0.9):
        errors = [mean_quantile_error(y, np.full_like(y, c), q=q) for c in grid]
        assert abs(grid[np.argmin(errors)] - np.quantile(y, q)) < 0.05


@pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
def test_torch_quantile_matches_quantile_metric(q):
    rng = np.random.default_rng(1)
    y_true = rng.normal(size=100)
    y_pred = rng.normal(size=100)

    loss = torch_quantile(torch.tensor(y_true), torch.tensor(y_pred), q=q)
    np.testing.assert_allclose(loss.item(), mean_quantile_error(y_true, y_pred, q=q))
