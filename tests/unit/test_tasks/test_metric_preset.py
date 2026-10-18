import pickle

import numpy as np
import pandas as pd
import pytest

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task


@pytest.mark.parametrize("algorithm", ["lgb", "xgb", "cb", "mlp", "linear_l2"])
def test_tabular_preset_selects_models_with_parameterized_metric(algorithm):
    rng = np.random.RandomState(42)
    data = pd.DataFrame(rng.normal(size=(128, 4)), columns=["a", "b", "c", "d"])
    data["target"] = (data["a"] + data["b"] > 0).astype(int)
    data["weight"] = rng.uniform(1, 2, len(data))
    task = Task("binary", metric="precision@k", metric_params={"k": 0.5})
    automl = TabularAutoML(
        task=task,
        timeout=30,
        cpu_limit=2,
        gpu_ids="",
        debug=True,
        general_params={"use_algos": [[algorithm]]},
        reader_params={"cv": 2, "random_state": 42, "advanced_roles": False},
        selection_params={"mode": 0},
        lgb_params={"default_params": {"num_trees": 5, "min_data_in_leaf": 5}, "freeze_defaults": True},
        xgb_params={"default_params": {"n_estimators": 5}, "freeze_defaults": True},
        cb_params={"default_params": {"num_trees": 5, "allow_writing_files": False}, "freeze_defaults": True},
        nn_params={
            "n_epochs": 1,
            "bs": 32,
            "num_workers": 0,
            "path_to_save": None,
            "hidden_size": [8],
            "freeze_defaults": True,
        },
    )
    valid = data.iloc[96:]
    prediction = automl.fit_predict(data.iloc[:96], valid_data=valid, roles={"target": "target", "weights": "weight"})
    assert prediction.shape == (32, 1)
    assert np.isfinite(prediction.data).all()
    score = task.metric_func(valid["target"], prediction.data[:, 0], sample_weight=valid["weight"])
    assert 0 <= score <= 1
    restored = pickle.loads(pickle.dumps(automl))
    np.testing.assert_allclose(restored.predict(valid).data, automl.predict(valid).data, atol=1e-6)
