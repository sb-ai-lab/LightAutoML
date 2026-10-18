import numpy as np
import pytest

from lightautoml.tasks.metrics import get_metric
from lightautoml.tasks.metrics.forecast import mean_mae
from lightautoml.tasks.metrics.forecast import mean_mse
from lightautoml.tasks.metrics.uplift import qini_auc_score
from lightautoml.tasks.metrics.uplift import uplift_at_k
from lightautoml.tasks.metrics.uplift import weighted_average_uplift


def test_rolling_forecasts_and_explicit_origins():
    target = np.arange(1, 7)
    forecasts = [np.nan, np.nan, [2, 3], [3, 6], [5, 5], [6, 7]]
    # Origins 2, 3, 4 are complete; the last horizon exceeds the observed series.
    assert mean_mae(target, forecasts) == pytest.approx((1 + 1 + 0.5) / 3)
    assert mean_mse(target, forecasts) == pytest.approx((1 + 1 + 0.5) / 3)
    assert mean_mae(target, forecasts[2:5], start_indices=[2, 3, 4]) == pytest.approx(5 / 6)
    assert mean_mse(target, [[1, 4], [5, 7]], start_indices=[0, 4], sample_weight=[1, 3]) == pytest.approx(0.875)
    assert get_metric("mean_mae", task="timeseries", start_indices=[0])(target, [[1, 2]]) == 0


@pytest.mark.parametrize(
    "forecasts,kwargs",
    [
        ([np.nan, np.nan], {}),
        ([[1, np.nan]], {}),
        ([[1, 2], [1]], {}),
        ([[1, 2]], {"start_indices": [-1]}),
        ([[1, 2]], {"start_indices": [0.5]}),
        ([[1, 2]], {"start_indices": [10]}),
        ([[1, 2], np.nan], {}),
    ],
)
def test_forecast_validation(forecasts, kwargs):
    with pytest.raises(ValueError):
        mean_mae([1, 2, 3], forecasts, **kwargs)


@pytest.mark.parametrize(
    "name,params,expected",
    [
        ("qini_auc_score", {}, 0.022838178235905525),
        ("qini_auc_score", {"negative_effect": False}, 0.04186999343249346),
        ("uplift_auc_score", {}, -0.003000485908649014),
        ("uplift_at_k", {"k": 0.5, "strategy": "overall"}, 0.5),
        ("uplift_at_k", {"k": 0.5, "strategy": "by_group"}, 0.5),
        ("weighted_average_uplift", {"bins": 4, "strategy": "overall"}, 0.5),
        ("weighted_average_uplift", {"bins": 4, "strategy": "by_group"}, 0.5),
    ],
)
def test_uplift_metrics_match_scikit_uplift(name, params, expected):
    # Reference values generated with scikit-uplift 0.5.1 and scikit-learn 1.7.2.
    target = np.tile([1, 0, 1, 1, 0, 0, 1, 0], 4)
    treatment = np.tile([1, 0], 16)
    pred = np.linspace(1, 0, 32)
    assert get_metric(name, task="uplift", **params)(target, pred, treatment) == pytest.approx(expected)


@pytest.mark.parametrize(
    "metric,params",
    [
        (uplift_at_k, {"k": 1}),
        (uplift_at_k, {"k": 0.01, "strategy": "by_group"}),
        (weighted_average_uplift, {"bins": 3}),
        (weighted_average_uplift, {"bins": 3, "strategy": "by_group"}),
        (qini_auc_score, {}),
    ],
)
def test_undefined_uplift_metrics_raise_without_runtime_warnings(metric, params):
    with pytest.raises(ValueError):
        metric([0, 0, 0, 0], [0.9, 0.8, 0.7, 0.6], [1, 1, 0, 0], **params)


def test_uplift_score_ties_and_perfect_ranking():
    target, treatment = [1, 0, 0, 1], [1, 0, 1, 0]
    assert qini_auc_score(target, [1, 1, 1, 1], treatment) == 0
    assert qini_auc_score(target, [1, 0, 0, -1], treatment) == 1
    with pytest.raises(ValueError, match="positive overall uplift"):
        qini_auc_score(target, [1, 0, 0, -1], treatment, negative_effect=False)
