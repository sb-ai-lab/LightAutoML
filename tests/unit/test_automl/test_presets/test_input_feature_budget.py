import pickle

import numpy as np
import pandas as pd
import pytest

from sklearn.metrics import roc_auc_score

from lightautoml.addons.uplift.metalearners import TLearner
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.dataset.roles import NumericRole
from lightautoml.tasks import Task


def frame():
    rng = np.random.RandomState(19)
    data = pd.DataFrame(rng.normal(size=(900, 6)), columns=["risk", "signal", "keep", "x", "y", "z"])
    data["category"] = np.where(data.x > 0, "red", "blue")
    data["arm"] = rng.randint(0, 2, len(data))
    data["target_risk"] = (np.where(data.arm, data.risk, data.signal) + rng.normal(size=len(data)) * 0.25 > 0).astype(
        int
    )
    data["weight"] = np.where(data.risk > 0, 5.0, 1.0)
    return data


def automl(limit=None, levels=None, mode=0):
    return TabularAutoML(
        Task("binary"),
        cpu_limit=1,
        gpu_ids="",
        timeout=120,
        debug=True,
        general_params={"use_algos": levels or [["lgb"]], "skip_conn": True},
        reader_params={"cv": 3, "advanced_roles": False, "n_jobs": 1},
        selection_params={"mode": mode, "max_input_features": limit},
        lgb_params={
            "default_params": {"num_threads": 1, "num_trees": 20, "early_stopping_rounds": 3},
            "freeze_defaults": True,
        },
    )


@pytest.mark.parametrize("mode", [0, 2])
def test_stacking_budget_and_serialized_input_schema(mode):
    data = frame().drop(columns="arm")
    roles = {"target": "target_risk", "weights": "weight", NumericRole(force_input=True): "keep"}
    model = automl(3, [["lgb", "linear_l2"], ["lgb"]], mode)
    model.fit_predict(data, roles=roles)
    assert len(model.levels) == 2
    assert len(model.levels[0]) == 2
    model = pickle.loads(pickle.dumps(model))
    used = model.reader.used_features
    assert "keep" in used and len(used) <= 3
    assert {"target_risk", "weight"}.isdisjoint(used)
    np.testing.assert_array_equal(model.predict(data).data, model.predict(data[used[::-1]]).data)


@pytest.mark.parametrize("limit", [True, 0, 1.5])
def test_global_budget_validation(limit):
    with pytest.raises(ValueError, match="positive integer"):
        automl(limit).fit_predict(frame(), roles={"target": "target_risk"})


def test_global_budget_conflict():
    with pytest.raises(ValueError, match="Forced features"):
        automl(1).fit_predict(frame(), roles={"target": "target_risk", NumericRole(force_input=True): ["risk", "keep"]})


def test_tlearner_shared_inputs_and_training_roles():
    data = frame()
    model = TLearner(automl(), automl(), max_input_features=2)
    model.fit(
        data,
        roles={
            "target": "target_risk",
            "treatment": "arm",
            "weights": "weight",
            "category": ["category"],
            "numeric": ["risk", "signal", "x"],
        },
    )
    model = pickle.loads(pickle.dumps(model))
    used = set(model.treatment_learner.reader.used_features) | set(model.control_learner.reader.used_features)
    assert used == {"risk", "signal"}
    assert model.control_learner.reader.used_array_attrs["weights"] == "weight"
    for full, short in zip(model.predict(data), model.predict(data[sorted(used)])):
        np.testing.assert_array_equal(full, short)


@pytest.mark.parametrize("levels", [[["lgb"]], [["lgb"], ["lgb"]]])
def test_accurate_permutation_csv_weights_substrings_and_stacking(tmp_path, levels):
    data = frame().drop(columns="arm")
    model = automl(levels=levels)
    holdout = data.iloc[600:].reset_index(drop=True).copy()
    model.fit_predict(data.iloc[:600], roles={"target": "target_risk", "weights": "weight"})
    before = holdout.copy()
    scores = model.get_feature_scores("accurate", holdout, random_state=29).set_index("Feature").Importance
    assert set(scores.index) == set(model.collect_used_feats())
    baseline = model.predict(holdout).data.ravel()
    permuted = holdout.copy()
    permuted["risk"] = np.random.RandomState(29).permutation(holdout.risk)
    changed = model.predict(permuted).data.ravel()

    def metric(pred):
        return roc_auc_score(holdout.target_risk, pred, sample_weight=holdout.weight)

    assert scores["risk"] == pytest.approx(metric(baseline) - metric(changed))
    pd.testing.assert_frame_equal(holdout, before)
    path = tmp_path / "holdout.csv"
    holdout.to_csv(path, index=False)
    other = model.get_feature_scores("accurate", str(path), random_state=29).set_index("Feature").Importance
    pd.testing.assert_series_equal(scores.sort_index(), other.sort_index())


def test_utilized_configurations_share_raw_budget(tmp_path):
    from lightautoml.automl.presets.tabular_presets import TabularUtilizedAutoML

    data = frame().drop(columns="arm")
    cfg = str(tmp_path / "preset.yml")
    TabularAutoML.get_config(cfg)
    model = TabularUtilizedAutoML(
        Task("binary"),
        timeout=120,
        cpu_limit=1,
        gpu_ids="",
        configs_list=[cfg, cfg],
        max_runs_per_config=1,
        drop_last=False,
        outer_blender_max_nonzero_coef=0,
        general_params={"use_algos": [["lgb"]]},
        reader_params={"cv": 3, "advanced_roles": False},
        selection_params={"mode": 0, "max_input_features": 2},
        lgb_params={"default_params": {"num_trees": 12}, "freeze_defaults": True},
    )
    model.fit_predict(data, roles={"target": "target_risk", "weights": "weight"})
    inputs = set()
    for pipe in model.outer_pipes:
        for learners in pipe.ml_algos[0].models:
            for learner in learners:
                inputs.update(learner.reader.used_features)
    assert 0 < len(inputs) <= 2
    np.testing.assert_array_equal(model.predict(data).data, model.predict(data[sorted(inputs)]).data)


def test_budget_with_separate_labeled_csv_validation(tmp_path):
    data = frame().drop(columns="arm")
    train, valid = data.iloc[:600], data.iloc[600:]
    path = tmp_path / "validation.csv"
    valid.to_csv(path, index=False)
    model = automl(2)
    oof = model.fit_predict(train, valid_data=str(path), roles={"target": "target_risk", "weights": "weight"})
    assert oof.shape == (len(valid), 1)
    assert len(model.reader.used_features) == 2
    np.testing.assert_allclose(oof.data, model.predict(valid[model.reader.used_features]).data)
