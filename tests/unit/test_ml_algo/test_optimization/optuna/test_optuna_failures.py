"""Keep successful models when individual optimization trials fail."""

import logging

from unittest.mock import Mock

import numpy as np
import optuna
import pytest

from optuna.trial import TrialState

from lightautoml.dataset.np_pd_dataset import NumpyDataset
from lightautoml.dataset.roles import NumericRole
from lightautoml.ml_algo.boost_cb import BoostCB
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.boost_xgb import BoostXGB
from lightautoml.ml_algo.tuning import optuna as optuna_module
from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.tasks import Task
from lightautoml.validation.base import HoldoutIterator
from lightautoml.validation.np_iterators import FoldsIterator


@pytest.fixture()
def dataset():
    rng = np.random.RandomState(42)
    data = rng.normal(size=(60, 3))
    target = np.arange(60) % 2
    data[:, 0] += target * 2
    return NumpyDataset(
        data,
        features=["a", "b", "c"],
        roles=NumericRole(),
        task=Task("binary"),
        target=target,
        folds=(np.arange(60) // 2) % 2,
    )


@pytest.fixture()
def progress(monkeypatch):
    bar = Mock()
    monkeypatch.setattr(optuna_module, "tqdm", Mock(return_value=bar))
    monkeypatch.setattr(optuna_module, "get_stdout_level", lambda: logging.INFO)
    return bar


def make_model(kind, fail_trials=()):
    if kind == "lgb":
        model = BoostLGBM(
            default_params={"num_trees": 4, "early_stopping_rounds": 2, "num_threads": 1, "verbose_eval": 0}
        )
        parameter, valid, invalid = "num_leaves", 4, 1
    elif kind == "xgb":
        model = BoostXGB(
            default_params={"n_estimators": 4, "early_stopping_rounds": 2, "nthread": 1, "verbose_eval": False}
        )
        parameter, valid, invalid = "max_depth", 2, -1
    else:
        model = BoostCB(default_params={"num_trees": 4, "od_wait": 2, "thread_count": 1, "verbose_eval": False})
        parameter, valid, invalid = "max_depth", 2, -1

    def search_space(trial, estimated_n_trials, suggested_params):
        params = dict(suggested_params)
        params[parameter] = trial.suggest_int(parameter, valid, valid)
        if trial.number in fail_trials:
            params[parameter] = invalid
        return params

    model.optimization_search_space = search_space
    return model


@pytest.mark.parametrize("kind", ["lgb", "xgb", "cb"])
@pytest.mark.parametrize("level", [logging.ERROR, logging.INFO, logging.INFO2])
def test_first_failed_trial_does_not_abort_pipeline(kind, level, dataset, progress, monkeypatch):
    monkeypatch.setattr(optuna_module, "get_stdout_level", lambda: level)
    tuner = OptunaTuner(n_trials=3, timeout=30)
    pipeline = MLPipeline([(make_model(kind, fail_trials=(0,)), tuner)])
    pipeline.debug = True

    result = pipeline.fit_predict(FoldsIterator(dataset))

    assert [trial.state for trial in tuner.study.trials] == [
        TrialState.FAIL,
        TrialState.COMPLETE,
        TrialState.COMPLETE,
    ]
    assert len(pipeline.ml_algos) == 1
    assert len(pipeline.ml_algos[0].models) == 2
    assert result.shape == (60, 1)
    assert np.isfinite(result.data).all()
    assert np.isfinite(pipeline.predict(dataset).data).all()
    if level != logging.ERROR:
        assert progress.update.call_count == 3
        progress.close.assert_called_once()


def test_failure_limit_keeps_the_best_completed_trial(dataset, progress):
    tuner = OptunaTuner(n_trials=5, timeout=30, fail_tolerance=0.1)
    pipeline = MLPipeline([(make_model("xgb", fail_trials=(1, 2, 3, 4)), tuner)])
    pipeline.debug = True

    result = pipeline.fit_predict(FoldsIterator(dataset))

    assert [trial.state for trial in tuner.study.trials] == [TrialState.COMPLETE, TrialState.FAIL]
    assert tuner.best_params == tuner.study.best_params
    assert np.isfinite(result.data).all()
    progress.close.assert_called_once()


@pytest.mark.parametrize("fit_on_holdout", [False, True])
@pytest.mark.parametrize("tolerance", [0.5, 1.0])
def test_all_failed_trials_fall_back_to_initial_parameters(dataset, progress, fit_on_holdout, tolerance):
    model = make_model("xgb", fail_trials=(0, 1, 2))
    tuner = OptunaTuner(n_trials=3, timeout=30, fit_on_holdout=fit_on_holdout, fail_tolerance=tolerance)
    pipeline = MLPipeline([(model, tuner)])
    pipeline.debug = True

    result = pipeline.fit_predict(FoldsIterator(dataset))

    assert len(tuner.study.trials) == (2 if tolerance == 0.5 else 3)
    assert all(trial.state == TrialState.FAIL for trial in tuner.study.trials)
    assert tuner.best_params == model.default_params
    assert np.isfinite(result.data).all()
    assert len(pipeline.ml_algos[0].models) == 2
    progress.close.assert_called_once()


@pytest.mark.parametrize("outcome", ["pruned", "nan", "no_trials"])
def test_no_completed_trials_do_not_require_best_trial(dataset, progress, outcome, monkeypatch):
    tuner = OptunaTuner(n_trials=0 if outcome == "no_trials" else 2, timeout=30, fail_tolerance=1.0)

    def objective(trial):
        if outcome == "pruned":
            raise optuna.TrialPruned()
        return float("nan")

    monkeypatch.setattr(tuner, "_get_objective", lambda **kwargs: objective)
    iterator = HoldoutIterator(dataset[:40], dataset[40:])
    model, result = tuner.fit(make_model("xgb"), iterator)

    assert model.is_fitted
    assert np.isfinite(result.data).all()
    progress.close.assert_called_once()


def test_keyboard_interrupt_propagates_and_closes_progress(dataset, progress, monkeypatch):
    tuner = OptunaTuner(n_trials=2, timeout=30)

    def objective(trial):
        raise KeyboardInterrupt()

    monkeypatch.setattr(tuner, "_get_objective", lambda **kwargs: objective)

    with pytest.raises(KeyboardInterrupt):
        tuner.fit(make_model("xgb"), FoldsIterator(dataset))

    progress.close.assert_called_once()
