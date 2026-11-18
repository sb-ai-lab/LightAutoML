"""Device selection and CPU inference for CatBoost multioutput models."""

from unittest.mock import Mock

import catboost as cb
import numpy as np
import pytest

from lightautoml.dataset.np_pd_dataset import NumpyDataset
from lightautoml.dataset.roles import CategoryRole
from lightautoml.dataset.roles import NumericRole
from lightautoml.ml_algo.boost_cb import BoostCB
from lightautoml.tasks import Task


def _dataset(task_name, categorical=False):
    rng = np.random.RandomState(42)
    data = rng.normal(size=(24, 3)).astype(np.float32)
    target = data[:, :2] if task_name == "multi:reg" else (data[:, :2] > 0).astype(np.float32)
    roles = {name: NumericRole() for name in ["a", "b", "c"]}
    if categorical:
        data[:, 2] = np.arange(24) % 4
        roles["c"] = CategoryRole(np.int32, label_encoded=True)
    return NumpyDataset(data, list(roles), roles, task=Task(task_name), target=target)


@pytest.mark.parametrize(
    "task_name,version,expected",
    [
        ("multi:reg", "0.26.1", "CPU"),
        ("multi:reg", "1.2", "CPU"),
        ("multi:reg", "1.2.1rc1", "CPU"),
        ("multi:reg", "1.2.1", "GPU"),
        ("multi:reg", "1.2.10", "GPU"),
        ("multilabel", "1.1.1", "CPU"),
        ("multilabel", "1.2rc1", "CPU"),
        ("multilabel", "1.2", "GPU"),
        ("multilabel", "1.10.0", "GPU"),
        ("multilabel", "1.2.10", "GPU"),
    ],
)
def test_requested_gpu_reaches_catboost_when_supported(task_name, version, expected, monkeypatch, caplog):
    dataset = _dataset(task_name)
    algo = BoostCB(default_params={"task_type": "GPU", "devices": "0", "num_trees": 3})
    algo.task = dataset.task
    estimator = Mock()
    estimator.predict.return_value = np.zeros((24, 2))
    factory = Mock(return_value=estimator)
    monkeypatch.setattr(cb, "CatBoost", factory)
    monkeypatch.setattr(cb, "__version__", version)
    algo.fit_predict_single_fold(dataset, dataset)
    params = factory.call_args.args[0]
    assert params["task_type"] == expected
    assert params["devices"] == "0"
    assert algo.params["task_type"] == "GPU"
    estimator.fit.assert_called_once()
    if expected == "CPU":
        assert "training on CPU" in caplog.text
        assert version in caplog.text
    else:
        assert "training on CPU" not in caplog.text


@pytest.mark.parametrize("task_name", ["multi:reg", "multilabel"])
@pytest.mark.parametrize("categorical", [False, True])
def test_explicit_cpu_request_stays_on_cpu(task_name, categorical, caplog):
    dataset = _dataset(task_name, categorical)
    original = dataset.data.copy()
    algo = BoostCB(default_params={"task_type": "CPU", "num_trees": 3, "thread_count": 1})
    algo.task = dataset.task
    model, predictions = algo.fit_predict_single_fold(dataset, dataset)
    assert model.get_all_params()["task_type"] == "CPU"
    np.testing.assert_array_equal(dataset.data, original)
    assert bool(model.get_cat_feature_indices()) == categorical
    assert predictions.shape == (24, 2)
    assert np.isfinite(predictions).all()
    assert "training on CPU" not in caplog.text


def test_gpu_training_errors_are_not_hidden_by_cpu_retry(monkeypatch):
    dataset = _dataset("multi:reg")
    algo = BoostCB(default_params={"task_type": "GPU"})
    algo.task = dataset.task
    estimator = Mock()
    estimator.fit.side_effect = cb.CatBoostError("GPU out of memory")
    factory = Mock(return_value=estimator)
    monkeypatch.setattr(cb, "CatBoost", factory)
    monkeypatch.setattr(cb, "__version__", "1.2.10")
    with pytest.raises(cb.CatBoostError, match="GPU out of memory"):
        algo.fit_predict_single_fold(dataset, dataset)
    factory.assert_called_once()
    assert factory.call_args.args[0]["task_type"] == "GPU"
    estimator.fit.assert_called_once()
