"""Regressions for boosting training, class dimensions, and sparse inputs."""

import pickle

import numpy as np
import pandas as pd
import pytest

from optuna.trial import TrialState
from scipy import sparse

from lightautoml.dataset.np_pd_dataset import NumpyDataset
from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import CategoryRole
from lightautoml.dataset.roles import NumericRole
from lightautoml.ml_algo.boost_cb import BoostCB
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.boost_xgb import BoostXGB
from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.tasks import Task
from lightautoml.tasks.losses.lgb import LGBLoss
from lightautoml.validation.base import HoldoutIterator
from lightautoml.validation.np_iterators import FoldsIterator


def make_model(kind):
    if kind == "lgb":
        return BoostLGBM(
            default_params={
                "num_trees": 5,
                "early_stopping_rounds": 2,
                "num_threads": 1,
                "min_data_in_leaf": 2,
                "verbose_eval": 0,
            }
        )
    if kind == "xgb":
        return BoostXGB(
            default_params={"n_estimators": 5, "early_stopping_rounds": 2, "nthread": 1, "verbose_eval": False}
        )
    return BoostCB(default_params={"num_trees": 5, "od_wait": 2, "thread_count": 1, "verbose_eval": False})


@pytest.mark.parametrize("kind", ["lgb", "xgb", "cb"])
@pytest.mark.parametrize("missing_from", ["train", "validation"])
def test_missing_holdout_class_keeps_probability_columns(kind, missing_from):
    rng = np.random.RandomState(42)
    dataset = NumpyDataset(
        rng.normal(size=(90, 3)), ["a", "b", "c"], NumericRole(), Task("multiclass"), target=np.arange(90) % 3
    )
    train, valid = dataset[:60], dataset[60:]
    if missing_from == "train":
        train = train[train.target != 2]
    else:
        valid = valid[valid.target != 2]
    model = make_model(kind)

    result = model.fit_predict(HoldoutIterator(train, valid))
    oof = result.data
    predicted = pickle.loads(pickle.dumps(model)).predict(valid).data

    assert model.n_classes == 3
    assert oof.shape == predicted.shape == (len(valid), 3)
    assert np.isfinite(oof).all()
    np.testing.assert_allclose(oof.sum(axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(predicted, oof, atol=1e-6)
    assert np.isfinite(model.score(result))


def test_lightgbm_uses_custom_objective():
    calls = []

    def objective(predictions, train):
        calls.append(len(predictions))
        return np.zeros_like(predictions), np.ones_like(predictions)

    task = Task("reg", loss={"lgb": LGBLoss(objective)}, metric="mse")
    data = np.arange(120, dtype=np.float32).reshape(60, 2)
    dataset = NumpyDataset(data, ["a", "b"], NumericRole(), task, target=np.linspace(10, 20, 60))
    model = make_model("lgb")

    predicted = model.fit_predict(HoldoutIterator(dataset[:40], dataset[40:])).data

    assert calls
    np.testing.assert_allclose(predicted, 0)


@pytest.mark.parametrize("kind", ["lgb", "xgb"])
def test_builtin_multiclass_f1_objective_and_metric(kind):
    rng = np.random.RandomState(42)
    task = Task("multiclass", loss="f1", metric="f1_macro")
    dataset = NumpyDataset(rng.normal(size=(90, 3)), ["a", "b", "c"], NumericRole(), task, target=np.arange(90) % 3)
    model = make_model(kind)
    valid = dataset[60:]

    result = model.fit_predict(HoldoutIterator(dataset[:60], valid))

    assert result.shape == (30, 3)
    assert np.isfinite(result.data).all()
    np.testing.assert_allclose(result.data.sum(axis=1), 1, atol=1e-6)
    assert np.isfinite(model.score(result))
    np.testing.assert_allclose(pickle.loads(pickle.dumps(model)).predict(valid).data, result.data)


@pytest.mark.parametrize("has_missing", [False, True])
def test_catboost_tunes_sparse_inputs_without_densifying(has_missing, monkeypatch):
    rng = np.random.RandomState(42)
    data = rng.normal(size=(60, 3))
    data[np.abs(data) < 1] = 0
    if has_missing:
        data[[0, 2], 0] = np.nan
    dataset = NumpyDataset(
        data,
        ["a", "b", "c"],
        NumericRole(),
        Task("binary"),
        target=np.arange(60) % 2,
        folds=(np.arange(60) // 2) % 2,
    ).to_csr()
    original = dataset.data.copy()

    original_toarray = sparse.csr_matrix.toarray

    def forbid_dense(matrix, *args, **kwargs):
        if matrix.shape[1] == dataset.shape[1]:
            raise AssertionError("Sparse training must not allocate a dense feature matrix")
        return original_toarray(matrix, *args, **kwargs)

    monkeypatch.setattr(sparse.csr_matrix, "toarray", forbid_dense)
    tuner = OptunaTuner(n_trials=2, timeout=30)
    pipeline = MLPipeline([(make_model("cb"), tuner)])
    pipeline.debug = True

    result = pipeline.fit_predict(FoldsIterator(dataset))

    assert [trial.state for trial in tuner.study.trials] == [TrialState.COMPLETE, TrialState.COMPLETE]
    assert ("nan_mode" in tuner.study.best_params) == has_missing
    assert np.isfinite(result.data).all()
    assert np.isfinite(pipeline.predict(dataset).data).all()
    np.testing.assert_equal(dataset.data.data, original.data)
    np.testing.assert_array_equal(dataset.data.indices, original.indices)
    np.testing.assert_array_equal(dataset.data.indptr, original.indptr)


def test_catboost_encoded_categories_preserve_input_without_pandas_warnings():
    frame = pd.DataFrame({"category": np.arange(60) % 3, "number": np.arange(60, dtype=np.float32)})
    original = frame.copy(deep=True)
    dataset = PandasDataset(
        frame,
        {"category": CategoryRole(np.int32, label_encoded=True), "number": NumericRole()},
        Task("binary"),
        target=pd.Series(np.arange(60) % 2),
    )
    model = make_model("cb")

    result = model.fit_predict(HoldoutIterator(dataset[:40], dataset[40:]))

    assert model.models[0].get_cat_feature_indices() == [0]
    assert np.isfinite(result.data).all()
    assert np.isfinite(model.predict(dataset).data).all()
    pd.testing.assert_frame_equal(frame, original)
