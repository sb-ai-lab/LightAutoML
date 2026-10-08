import numpy as np
import pytest
import torch

import lightautoml.utils.logging  # noqa: F401 - register the logging levels used by Task

from lightautoml.dataset.np_pd_dataset import NumpyDataset
from lightautoml.dataset.roles import NumericRole
from lightautoml.tasks import Task
from lightautoml.tasks.common_metric import mean_quantile_error
from lightautoml.tasks.losses.torch import torch_quantile


@pytest.mark.parametrize("q, expected", [(0.1, 0.55), (0.5, 0.75), (0.9, 0.95)])
def test_quantile_metric_matches_pinball_loss(q, expected):
    y_true = np.array([0.0, 0.0], dtype=np.float32)
    y_pred = np.array([-2.0, 1.0], dtype=np.float32)

    assert mean_quantile_error(y_true, y_pred, q=q) == pytest.approx(expected)


@pytest.mark.parametrize("q, expected", [(0.1, 0.375), (0.5, 0.875), (0.9, 1.375)])
def test_weighted_quantile_metric(q, expected):
    y_true = np.array([0.0, 0.0], dtype=np.float32)
    y_pred = np.array([-2.0, 1.0], dtype=np.float32)
    sample_weight = np.array([3.0, 1.0], dtype=np.float32)

    assert mean_quantile_error(y_true, y_pred, sample_weight, q=q) == pytest.approx(expected)


def test_quantile_metric_best_constant_is_the_quantile():
    rng = np.random.default_rng(0)
    y = rng.normal(size=20000)
    grid = np.linspace(-3, 3, 601)
    for q in (0.1, 0.9):
        errors = [mean_quantile_error(y, np.full_like(y, c), q=q) for c in grid]
        assert abs(grid[np.argmin(errors)] - np.quantile(y, q)) < 0.05


@pytest.mark.parametrize("shape", [(2,), (2, 1)])
@pytest.mark.parametrize("q, expected, weighted_expected", [(0.1, 0.55, 0.375), (0.5, 0.75, 0.875), (0.9, 0.95, 1.375)])
@pytest.mark.parametrize("weighted", [False, True])
def test_torch_quantile_matches_pinball_loss(shape, q, expected, weighted_expected, weighted):
    y_true = torch.tensor([0.0, 0.0]).reshape(shape)
    y_pred = torch.tensor([-2.0, 1.0]).reshape(shape)
    sample_weight = torch.tensor([3.0, 1.0]) if weighted else None

    loss = torch_quantile(y_true, y_pred, sample_weight, q=q)

    assert loss.ndim == 0
    assert loss.item() == pytest.approx(weighted_expected if weighted else expected)


@pytest.mark.parametrize("shape", [(2,), (2, 1)])
@pytest.mark.parametrize("weighted, expected", [(False, [-0.45, 0.05]), (True, [-0.675, 0.025])])
def test_torch_quantile_gradient(shape, weighted, expected):
    y_true = torch.tensor([0.0, 0.0]).reshape(shape)
    y_pred = torch.tensor([-1.0, 1.0]).reshape(shape).requires_grad_()
    sample_weight = torch.tensor([3.0, 1.0]) if weighted else None

    loss = torch_quantile(y_true, y_pred, sample_weight, q=0.9)
    loss.backward()

    np.testing.assert_allclose(y_pred.grad.numpy().reshape(-1), expected, rtol=1e-6)


@pytest.mark.parametrize("shape", [(2,), (2, 1)])
def test_torch_quantile_exact_predictions(shape):
    y_true = torch.tensor([-2.0, 1.0]).reshape(shape)
    y_pred = y_true.clone().requires_grad_()

    loss = torch_quantile(y_true, y_pred, q=0.9)
    loss.backward()

    assert loss.item() == pytest.approx(0.0)
    np.testing.assert_array_equal(y_pred.grad.numpy(), np.zeros(shape))


@pytest.mark.parametrize("q, expected", [(0.1, 0.55), (0.5, 0.75), (0.9, 0.95)])
def test_quantile_task_passes_q_to_metric_and_torch_loss(q, expected):
    task = Task("reg", loss="quantile", loss_params={"q": q}, metric="quantile")
    y_true = np.array([0.0, 0.0], dtype=np.float32)
    y_pred = np.array([-2.0, 1.0], dtype=np.float32)
    dataset = NumpyDataset(y_pred[:, None], ["prediction"], NumericRole(), task, target=y_true)

    assert task.metric_params == {"q": q}
    assert not task.greater_is_better
    assert task.metric_func(y_true, y_pred) == pytest.approx(expected)
    assert task.get_dataset_metric()(dataset) == pytest.approx(-expected)
    assert task.losses["torch"].loss(torch.from_numpy(y_true), torch.from_numpy(y_pred)).item() == pytest.approx(
        expected
    )
