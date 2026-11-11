"""Exercise failed Optuna trials through complete tabular presets."""

import pickle

import numpy as np
import pandas as pd
import pytest

from optuna.trial import TrialState

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.tasks import Task


@pytest.mark.parametrize("kind", ["lgb", "xgb", "cb"])
@pytest.mark.parametrize("task_name", ["binary", "reg", "multiclass"])
def test_tabular_preset_survives_first_failed_trial(kind, task_name, monkeypatch):
    original_get_objective = OptunaTuner._get_objective
    studies = []

    def get_objective(self, **kwargs):
        objective = original_get_objective(self, **kwargs)
        studies.append(self.study)

        def fail_first_trial(trial):
            if trial.number == 0:
                raise RuntimeError("Simulated failure of one training trial")
            return objective(trial)

        return fail_first_trial

    monkeypatch.setattr(OptunaTuner, "_get_objective", get_objective)
    rng = np.random.RandomState(42)
    data = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
    data["category"] = pd.Series(np.tile(["red", "green", "blue"], 20), dtype="string")
    data["target"] = np.arange(60) % (3 if task_name == "multiclass" else 2)
    if task_name == "reg":
        data["target"] = data["a"] * 2 + data["b"]

    automl = TabularAutoML(
        task=Task(task_name),
        timeout=60,
        cpu_limit=1,
        gpu_ids="",
        general_params={"use_algos": [[kind + "_tuned"]]},
        reader_params={"cv": 2, "advanced_roles": False},
        selection_params={"mode": 0},
        tuning_params={"max_tuning_iter": 3, "max_tuning_time": 30},
        lgb_params={"default_params": {"num_trees": 4, "early_stopping_rounds": 2}, "freeze_defaults": True},
        xgb_params={"default_params": {"n_estimators": 4, "early_stopping_rounds": 2}, "freeze_defaults": True},
        cb_params={"default_params": {"num_trees": 4, "od_wait": 2}, "freeze_defaults": True},
    )
    oof = automl.fit_predict(data, roles={"target": "target", "category": ["category"]}, verbose=1).data
    features = data.drop(columns="target")
    predicted = automl.predict(features).data
    restored = pickle.loads(pickle.dumps(automl)).predict(features).data

    assert len(studies) == 1
    assert [trial.state for trial in studies[0].trials] == [
        TrialState.FAIL,
        TrialState.COMPLETE,
        TrialState.COMPLETE,
    ]
    assert oof.shape == predicted.shape == (60, 3 if task_name == "multiclass" else 1)
    assert np.isfinite(oof).all()
    assert np.isfinite(predicted).all()
    np.testing.assert_allclose(restored, predicted)
