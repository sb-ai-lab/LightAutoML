"""Offline multioutput training and serialization regression tests."""

import pickle

import numpy as np
import pandas as pd
import pytest

from sklearn.metrics import roc_auc_score

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task


@pytest.mark.parametrize("advanced_roles", [False, True])
@pytest.mark.parametrize("model", ["linear_l2", "cb", "rf"])
@pytest.mark.parametrize("task_name", ["multi:reg", "multilabel"])
def test_multitask_fit_predict_and_pickle(model, task_name, advanced_roles):
    rng = np.random.RandomState(42)
    data = pd.DataFrame({"x": rng.normal(size=48), "category": ["a", "b", "c"] * 16})
    data["first"] = 2 * data.x
    data["second"] = -data.x - 3
    task = Task(task_name)
    if task_name == "multilabel":
        data["first"] = pd.Series(np.where(data.x > 0, "yes", "no"), dtype="string")
        data["second"] = pd.Series(np.where(np.arange(48) % 2, "on", "off"), dtype="category")
        task = Task(task_name, metric=roc_auc_score, greater_is_better=True)
    automl = TabularAutoML(
        task=task,
        timeout=60,
        cpu_limit=1,
        gpu_ids="",
        debug=True,
        general_params={"use_algos": [[model]]},
        reader_params={"cv": 2, "advanced_roles": advanced_roles},
        selection_params={"mode": 0},
        linear_l2_params={"default_params": {"cs": [1.0], "max_iter": 10}},
        cb_params={"default_params": {"num_trees": 3, "max_depth": 2}},
        rf_params={"default_params": {"n_estimators": 3, "max_features": 1.0, "n_jobs": 1}, "freeze_defaults": True},
    )
    oof = automl.fit_predict(data, roles={"target": ["second", "first"]}, verbose=0)
    features = data.drop(columns=["first", "second"])
    prediction = automl.predict(features)
    restored = pickle.loads(pickle.dumps(automl)).predict(features)
    assert oof.shape == prediction.shape == (48, 2)
    assert np.isfinite(oof.data).all()
    assert np.isfinite(prediction.data).all()
    np.testing.assert_allclose(restored.data, prediction.data)
    if task_name == "multilabel":
        assert ((prediction.data >= 0) & (prediction.data <= 1)).all()
