"""Real boosting and tuning through uplift meta-learners."""

import copy
import pickle

import numpy as np
import pandas as pd
import pytest

from lightautoml.addons.uplift.metalearners import RLearner
from lightautoml.addons.uplift.metalearners import SLearner
from lightautoml.addons.uplift.metalearners import T2Learner
from lightautoml.addons.uplift.metalearners import TDLearner
from lightautoml.addons.uplift.metalearners import TLearner
from lightautoml.addons.uplift.metalearners import XLearner
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task


def small_automl(task, algorithm):
    return TabularAutoML(
        task=task if isinstance(task, Task) else Task(task),
        timeout=60,
        cpu_limit=1,
        gpu_ids="",
        general_params={"use_algos": [[algorithm]]},
        reader_params={"cv": 2, "advanced_roles": False},
        selection_params={"mode": 0},
        tuning_params={"max_tuning_iter": 2, "max_tuning_time": 10},
        cb_params={"default_params": {"num_trees": 8, "max_depth": 3}, "freeze_defaults": True},
        xgb_params={"default_params": {"n_estimators": 8, "max_depth": 3}, "freeze_defaults": True},
    )


@pytest.mark.parametrize("algorithm", ["cb", "cb_tuned", "xgb", "xgb_tuned"])
@pytest.mark.parametrize("task", ["binary", "reg"])
@pytest.mark.parametrize("kind", ["s", "t", "t2", "td", "x", "r"])
@pytest.mark.parametrize("categorical_treatment", [False, True])
def test_categorical_uplift_training_and_pickle(kind, task, algorithm, categorical_treatment):
    rng = np.random.RandomState(42)
    n = 120
    data = pd.DataFrame({"x": rng.normal(size=n), "category": pd.Categorical(np.tile(["a", "b", "c"], 40))})
    data["w"] = np.tile([0, 1, 1, 0], 30)
    y = data.x + (data.category == "b") + data.w * (data.x > 0) + rng.normal(size=n)
    data["y"] = (y > 0.5).astype(int) if task == "binary" else y
    if categorical_treatment:
        data["w"] = data.w.astype("category")
    original = data.copy(deep=True)
    roles = {"target": "y", "treatment": "w", "category": ["category"]}
    saved_roles = copy.deepcopy(roles)

    def outcome():
        return small_automl(task, algorithm)

    if kind == "s":
        model = SLearner(learner=outcome())
    elif kind in ["t", "t2", "td"]:
        klass = {"t": TLearner, "t2": T2Learner, "td": TDLearner}[kind]
        kwargs = {"n_uplift_iterator_folds": 2} if kind == "t2" else {}
        model = klass(treatment_learner=outcome(), control_learner=outcome(), **kwargs)
    elif kind == "x":
        model = XLearner(
            base_task=task if isinstance(task, Task) else Task(task),
            outcome_learners=[outcome()],
            effect_learners=[small_automl("reg", algorithm)],
            propensity_learner=small_automl("binary", algorithm),
        )
    else:
        model = RLearner(
            mean_outcome_learner=outcome(),
            effect_learner=small_automl("reg", algorithm),
            propensity_learner=small_automl("binary", algorithm),
        )
    model.fit(data, roles)
    test = data.drop(columns=["y", "w"]).iloc[:15].copy()
    test["category"] = pd.Categorical(["unseen"] * len(test))
    predictions = model.predict(test)
    restored = pickle.loads(pickle.dumps(model)).predict(test)
    for prediction, reloaded in zip(predictions, restored):
        if prediction is not None:
            assert prediction.shape == (len(test),)
            assert np.isfinite(prediction).all()
            np.testing.assert_allclose(prediction, reloaded)
    pd.testing.assert_frame_equal(data, original)
    assert roles == saved_roles


@pytest.mark.parametrize("algorithm", ["cb", "cb_tuned", "xgb", "xgb_tuned"])
@pytest.mark.parametrize("task_name", ["binary", "reg"])
def test_tx_selects_and_rebuilds_trained_model(task_name, algorithm):
    from lightautoml.addons.uplift.base import AutoUpliftTX
    from lightautoml.addons.uplift.base import BaseLearnerWrapper

    rng = np.random.RandomState(19)
    data = pd.DataFrame({"x": rng.normal(size=160), "category": pd.Categorical(np.tile(["a", "b"], 80))})
    data["w"] = np.tile([0, 0, 1, 1], 40)
    data["y"] = data.x + data.w * (data.category == "a") + rng.normal(size=160)
    if task_name == "binary":
        data["y"] = (data.y > 0).astype(int)
    wrapper = BaseLearnerWrapper("boost", small_automl, {"algorithm": algorithm})
    auto = AutoUpliftTX(Task(task_name), baselearners=[wrapper], metric="qini", random_state=19)
    roles = {"target": "y", "treatment": "w", "category": ["category"]}
    auto.fit(data, roles, verbose=1)
    train, holdout, treatment, target = auto._prepare_data(data, roles)
    prediction = auto.predict(holdout)[0]
    expected_score = auto.calculate_metric(target, prediction, treatment)
    assert np.isfinite(prediction).all()
    assert expected_score == pytest.approx(max(auto._metalearner_metrics.values()))
    saved = pickle.loads(pickle.dumps(auto))
    np.testing.assert_allclose(saved.predict(holdout)[0], prediction)
    stage_count = sum(map(len, auto._trained_stage_baselearners.values()))
    auto.fit(data, roles)
    assert sum(map(len, auto._trained_stage_baselearners.values())) == stage_count
    np.testing.assert_allclose(auto.predict(holdout)[0], prediction)
