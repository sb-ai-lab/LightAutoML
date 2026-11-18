"""Regression tests for uplift training and model selection."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from lightautoml.addons.uplift.base import AutoUplift
from lightautoml.addons.uplift.base import AutoUpliftTX
from lightautoml.addons.uplift.base import BaseLearnerWrapper
from lightautoml.addons.uplift.base import MetaLearnerStage
from lightautoml.addons.uplift.metalearners import SLearner
from lightautoml.addons.uplift.metalearners import T2Learner
from lightautoml.addons.uplift.metalearners import XLearner
from lightautoml.automl.base import AutoML
from lightautoml.tasks import Task
from lightautoml.validation.np_iterators import UpliftIterator


class RecordingLearner(AutoML):
    """Record the public training contract without training a backend."""

    def __init__(self, task=None, value=0.5):
        self.reader = SimpleNamespace(task=task or Task("reg"))
        self.value = value
        self.calls = []
        self.inputs = []

    def fit_predict(self, data, roles, train_features=None, cv_iter=None, verbose=0):
        assert train_features is None
        self.calls.append((data.copy(), roles, cv_iter, verbose))
        return SimpleNamespace(data=np.full((len(data), 1), self.value))

    def predict(self, data):
        self.inputs.append(data.copy())
        return SimpleNamespace(data=np.full((len(data), 1), self.value))


@pytest.fixture()
def uplift_frame():
    return pd.DataFrame({"x": np.arange(40), "w": np.tile([0, 1], 20), "y": np.arange(40) / 10})


@pytest.mark.parametrize("mode", [False, True])
def test_uplift_iterator_validates_requested_group_and_all_folds(mode):
    treatment = np.tile([0, 1], 20)
    iterator = UpliftIterator(treatment, np.arange(40), mode, Task("reg"), n_folds=4)
    folds = list(iterator)
    assert len(folds) == 4
    validation = np.concatenate([valid for _, valid in folds])
    np.testing.assert_array_equal(np.sort(validation), np.flatnonzero(treatment == mode))
    for train, valid in folds:
        assert not np.intersect1d(train, valid).size
        assert len(train) + len(valid) == len(treatment)
        assert np.all(np.isin(np.flatnonzero(treatment != mode), train))
    with pytest.raises(IndexError):
        iterator[4]
    with pytest.raises(IndexError):
        iterator[-1]


def test_t2_public_fit_and_categorical_treatment_inference(uplift_frame):
    # Boolean replacement at inference would turn category labels '0'/'1'
    # into unseen 'False'/'True' labels.
    uplift_frame["w"] = uplift_frame.w.astype("category")
    treated, control = RecordingLearner(), RecordingLearner()
    learner = T2Learner(treated, control, n_uplift_iterator_folds=2)
    learner.fit(uplift_frame, {"target": "y", "treatment": "w"}, verbose=2)
    learner.predict(uplift_frame.drop(columns=["w", "y"]))
    assert treated.calls[0][3] == control.calls[0][3] == 2
    assert set(treated.inputs[0].w.astype(str)) == {"1"}
    assert set(control.inputs[0].w.astype(str)) == {"0"}


@pytest.mark.parametrize("propensity", [0.0, 0.1, 0.8, 1.0])
def test_xlearner_weights_follow_imputation_group(propensity, uplift_frame):
    learner = XLearner(
        base_task=Task("reg"),
        propensity_learner=RecordingLearner(Task("binary"), propensity),
        outcome_learners=[RecordingLearner(value=2), RecordingLearner(value=5)],
        effect_learners=[RecordingLearner(value=10), RecordingLearner(value=20)],
    )
    learner._is_fitted = True
    uplift, treated, control = learner.predict(uplift_frame)
    np.testing.assert_allclose(uplift, propensity * 10 + (1 - propensity) * 20)
    np.testing.assert_allclose(treated - control, 3)
    auto = AutoUpliftTX(Task("reg"))
    stage_predictions = {
        ("outcome_treatment", "effect_control"): SimpleNamespace(pred=np.full(40, 10)),
        ("outcome_control", "effect_treatment"): SimpleNamespace(pred=np.full(40, 20)),
        ("propensity",): SimpleNamespace(pred=np.full(40, propensity)),
    }
    np.testing.assert_allclose(auto._metalearner_predict("XLearner", stage_predictions), uplift)


def test_tx_metric_receives_outcomes_and_treatment_in_correct_order(uplift_frame, monkeypatch):
    auto = AutoUpliftTX(Task("reg"))
    monkeypatch.setattr(auto, "_generate_stage_baselearner_candidates", lambda: iter(()))
    metric = Mock()
    monkeypatch.setattr(auto, "_calculate_metalearners_metrics", metric)
    monkeypatch.setattr(auto, "_set_best_metalearner", lambda: None)
    _, _, expected_treatment, expected_target = auto._prepare_data(uplift_frame, {"target": "y", "treatment": "w"})
    auto.fit(uplift_frame, {"target": "y", "treatment": "w"})
    np.testing.assert_array_equal(metric.call_args.args[0], expected_target)
    np.testing.assert_array_equal(metric.call_args.args[1], expected_treatment)


def test_tx_passes_verbose_by_keyword(uplift_frame):
    auto = AutoUpliftTX(Task("reg"))
    wrapper = BaseLearnerWrapper("record", RecordingLearner, {})
    stage = MetaLearnerStage("outcome_control")
    auto._evaluate(((stage, wrapper),), uplift_frame, uplift_frame, {"target": "y", "treatment": "w"}, verbose=2)
    trained = auto._trained_stage_baselearners[stage.full_name()][0].trained_model
    assert trained.calls[0][3] == 2


def test_regression_imbalance_candidates_keep_regression_task(uplift_frame):
    auto = AutoUplift(Task("reg"), add_dd_candidates=True)
    auto._calculate_tabular_time()
    uplift_frame["w"] = (uplift_frame.x < 5).astype(int)
    candidates = auto._generate_data_depend_uplift_candidates(uplift_frame, {"target": "y", "treatment": "w"})
    assert candidates
    for candidate in candidates:
        assert all(wrapper.params["task"].name == "reg" for wrapper in candidate.params["outcome_learners"])


def test_explicit_holdout_generates_candidates_from_training_data(uplift_frame, monkeypatch):
    auto = AutoUplift(Task("reg"), add_dd_candidates=True)
    generate = Mock()
    monkeypatch.setattr(auto, "_generate_uplift_candidates", generate)
    auto.fit(
        train_data=uplift_frame.iloc[:30], test_data=uplift_frame.iloc[30:], roles={"target": "y", "treatment": "w"}
    )
    pd.testing.assert_frame_equal(generate.call_args.args[0], uplift_frame.iloc[:30])


@pytest.mark.parametrize("klass", [SLearner, T2Learner])
def test_treatment_is_retained_by_feature_selection(klass, uplift_frame):
    learner = RecordingLearner()
    model = klass(learner=learner) if klass is SLearner else klass(treatment_learner=learner)
    # Inspect the training contract; a selector must not discard treatment
    # just because its marginal association with the outcome is weak.
    model._check_timer = lambda: None
    if klass is T2Learner:
        model.control_learner = RecordingLearner()
    roles = {"target": "y", "treatment": "w"}
    model.fit(uplift_frame, roles)
    forced = [columns for role, columns in learner.calls[0][1].items() if getattr(role, "force_input", False)]
    assert "w" in forced
    assert roles == {"target": "y", "treatment": "w"}


def test_rlearner_stabilizes_extreme_propensities(uplift_frame):
    from lightautoml.addons.uplift.metalearners import RLearner

    effect = RecordingLearner()
    model = RLearner(base_task=Task("reg"), effect_learner=effect)
    data = uplift_frame.iloc[:6].copy()
    propensities = np.array([0, 1, model._epsi, 1 - model._epsi, 0.2, 0.8])
    mean_outcome = np.zeros(6)
    model._fit_effect_learner(data, {"target": "y", "treatment": "w"}, propensities, mean_outcome)
    transformed, roles, _, _ = effect.calls[0]
    assert len(transformed) == len(data)
    assert np.isfinite(transformed.y).all()
    assert (transformed["__WEIGHTS__"] > 0).all()
    denominator = data.w.to_numpy() - np.clip(propensities, model._epsi, 1 - model._epsi)
    np.testing.assert_allclose(transformed.y * denominator, data.y)
    np.testing.assert_allclose(transformed["__WEIGHTS__"], denominator ** 2)


def test_tx_unequal_stage_candidate_counts(uplift_frame, monkeypatch):
    auto = AutoUpliftTX(Task("reg"), metalearners=["TLearner"])
    short = BaseLearnerWrapper("short", RecordingLearner, {})
    long = BaseLearnerWrapper("long", RecordingLearner, {})
    stages = {("outcome_control",): [short], ("outcome_treatment",): [short, long]}
    monkeypatch.setattr(auto, "_set_stage_baselearners", lambda: stages)
    generated = list(auto._generate_stage_baselearner_candidates())
    assert len(generated) == 3
    assert all(wrapper is not None for stage_info in generated for _, wrapper in stage_info)


def test_tx_only_trains_requested_metalearner_stages():
    wrapper = BaseLearnerWrapper("record", RecordingLearner, {})
    auto = AutoUpliftTX(Task("reg"), baselearners=[wrapper], metalearners=["TLearner"])
    assert set(auto._set_stage_baselearners()) == {("outcome_control",), ("outcome_treatment",)}


def test_tx_refit_discards_old_models_and_scores(uplift_frame, monkeypatch):
    auto = AutoUpliftTX(Task("reg"))
    auto._trained_stage_baselearners[("outcome_control",)].append(object())
    auto._metalearner_metrics["old"] = 1
    monkeypatch.setattr(auto, "_generate_stage_baselearner_candidates", lambda: iter(()))
    monkeypatch.setattr(auto, "_calculate_metalearners_metrics", lambda *args: None)
    monkeypatch.setattr(auto, "_set_best_metalearner", lambda: None)
    auto.fit(uplift_frame, {"target": "y", "treatment": "w"})
    assert not auto._trained_stage_baselearners
    assert not auto._metalearner_metrics


def test_tx_effect_model_matches_its_fitted_outcome_parent():
    from lightautoml.addons.uplift.base import TrainedStageBaseLearner

    auto = AutoUpliftTX(Task("reg"))
    first = BaseLearnerWrapper("first", RecordingLearner, {})
    second = BaseLearnerWrapper("second", RecordingLearner, {})
    effect = BaseLearnerWrapper("effect", RecordingLearner, {})
    for stage in ["outcome_control", "outcome_treatment", "propensity"]:
        task = Task("binary") if stage == "propensity" else Task("reg")
        auto._trained_stage_baselearners[(stage,)].append(
            TrainedStageBaseLearner(second, None, RecordingLearner(task, 0.5), np.array([0.5]))
        )
    for stage in [("outcome_treatment", "effect_control"), ("outcome_control", "effect_treatment")]:
        auto._trained_stage_baselearners[stage] = [
            TrainedStageBaseLearner(effect, first, RecordingLearner(value=10), np.array([10])),
            TrainedStageBaseLearner(effect, second, RecordingLearner(value=20), np.array([20])),
        ]
    names = {stage: "effect" if len(stage) == 2 else "second" for stage in auto._trained_stage_baselearners}
    model = auto._init_metalearner("XLearner", names)
    np.testing.assert_allclose(model.predict(pd.DataFrame({"x": [0]}))[0], 20)


def test_tx_filters_explicit_stages_to_requested_metalearners():
    wrapper = BaseLearnerWrapper("record", RecordingLearner, {})
    supplied = {(stage,): [wrapper] for stage in ["outcome_control", "outcome_treatment", "propensity"]}
    auto = AutoUpliftTX(Task("reg"), baselearners=supplied, metalearners=["TLearner"])
    assert set(auto._set_stage_baselearners()) == {("outcome_control",), ("outcome_treatment",)}


def test_completed_model_timeout_is_logged(uplift_frame, caplog):
    model = SLearner(learner=RecordingLearner(), timeout=-1)
    model.fit(uplift_frame, {"target": "y", "treatment": "w"})
    assert "SLearner is trained, but time limit exceeded." in caplog.text
    assert model._is_fitted
