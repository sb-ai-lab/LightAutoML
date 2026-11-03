"""Small offline end-to-end checks across supported dependency stacks."""

import pickle

import numpy as np
import pandas as pd
import pytest

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task


@pytest.mark.parametrize("model", ["xgb", "mlp", "autoint", "fttransformer", "tabm"])
@pytest.mark.parametrize("task", ["binary", "reg", "multiclass"])
def test_mixed_features_and_pickle_roundtrip(model, task):
    rng = np.random.RandomState(42)
    data = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
    data["category"] = pd.Series(np.tile(["red", "green", "blue"], 20), dtype="string")
    data["target"] = np.arange(60) % (3 if task == "multiclass" else 2)
    if task == "reg":
        data["target"] = data["a"] * 2 + rng.normal(size=60) * 0.1
    automl = TabularAutoML(
        task=Task(task),
        timeout=60,
        cpu_limit=1,
        gpu_ids="",
        debug=True,
        general_params={"use_algos": [[model]]},
        reader_params={"cv": 2, "advanced_roles": False},
        xgb_params={"default_params": {"n_estimators": 3, "early_stopping_rounds": 2, "nthread": 1}},
        nn_params={
            "n_epochs": 1,
            "bs": 8,
            "num_workers": 0,
            "path_to_save": None,
            "freeze_defaults": True,
            "embedding_size": 4,
            "cont_embedder": "piecewise" if model == "tabm" else "cont",
            "backbone_params": {"k": 2, "d_block": 8, "n_blocks": 1},
        },
    )
    oof = automl.fit_predict(data, roles={"target": "target", "category": ["category"]}, verbose=0).data
    features = data.drop(columns="target")
    expected = automl.predict(features).data
    restored = pickle.loads(pickle.dumps(automl)).predict(features).data
    assert oof.shape == expected.shape == (60, 3 if task == "multiclass" else 1)
    assert np.isfinite(oof).all()
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(restored, expected)
