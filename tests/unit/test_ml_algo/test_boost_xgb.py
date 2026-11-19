"""Regressions for non-finite XGBoost features in dense and sparse datasets."""

import pickle

import numpy as np
import pandas as pd
import pytest
import xgboost

from scipy import sparse

from lightautoml.dataset.np_pd_dataset import NumpyDataset
from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import NumericRole
from lightautoml.ml_algo.boost_xgb import BoostXGB
from lightautoml.ml_algo.boost_xgb import _sanitize_features
from lightautoml.tasks import Task
from lightautoml.validation.base import HoldoutIterator
from lightautoml.validation.np_iterators import FoldsIterator


@pytest.mark.parametrize("sparse_input", [False, True])
def test_sanitize_features_preserves_input_and_finite_values(sparse_input):
    data = np.array([[0, np.inf, -np.inf], [1e300, -1e300, np.nan], [1, -2, 3]], dtype=np.float64)
    original = sparse.csr_matrix(data) if sparse_input else data.copy()
    result = _sanitize_features(original)
    actual = result.toarray() if sparse_input else result
    expected = np.array([[0, np.nan, np.nan], [np.nan, np.nan, np.nan], [1, -2, 3]])
    np.testing.assert_equal(actual, expected)
    np.testing.assert_equal(original.toarray() if sparse_input else original, data)
    if sparse_input:
        assert result.format == original.format
        np.testing.assert_equal(result.indices, original.indices)
        np.testing.assert_equal(result.indptr, original.indptr)


@pytest.mark.parametrize("sparse_input", [False, True])
def test_fit_and_predict_with_nonfinite_features(sparse_input):
    rng = np.random.RandomState(42)
    data = rng.normal(size=(40, 3))
    data[:5, 0] = [np.inf, -np.inf, 1e300, -1e300, np.nan]
    dataset = NumpyDataset(
        data,
        features=["a", "b", "c"],
        roles=NumericRole(np.float64),
        task=Task("binary"),
        target=(data[:, 1] > 0).astype(np.float32),
        folds=np.arange(40) % 2,
    )
    if sparse_input:
        dataset = dataset.to_csr()
    model = BoostXGB(default_params={"n_estimators": 3, "early_stopping_rounds": 2, "nthread": 1})
    oof = model.fit_predict(FoldsIterator(dataset)).data
    predictions = model.predict(dataset).data
    assert oof.shape == predictions.shape == (40, 1)
    assert np.isfinite(oof).all()
    assert np.isfinite(predictions).all()
    np.testing.assert_equal(dataset.data.toarray() if sparse_input else dataset.data, data)


def test_clean_features_do_not_allocate_a_copy():
    data = np.array([[0.0, 1.0, np.nan]], dtype=np.float32)
    assert _sanitize_features(data) is data


def test_pandas_sanitization_preserves_feature_names_for_prediction():
    rng = np.random.RandomState(42)
    frame = pd.DataFrame(rng.normal(size=(40, 2)), columns=["a", "b"], index=np.arange(100, 140))
    dataset = PandasDataset(
        frame,
        roles={name: NumericRole(np.float64) for name in frame},
        task=Task("binary"),
        target=(frame["b"] > 0).astype(np.float32),
        folds=pd.Series(np.arange(40) % 2, index=frame.index),
    )
    model = BoostXGB(default_params={"n_estimators": 3, "early_stopping_rounds": 2, "nthread": 1})
    model.fit_predict(FoldsIterator(dataset))
    dataset.data.iloc[0, 0] = np.inf
    sanitized = _sanitize_features(dataset.data)
    pd.testing.assert_index_equal(sanitized.columns, dataset.data.columns)
    pd.testing.assert_index_equal(sanitized.index, dataset.data.index)
    assert np.isfinite(model.predict(dataset).data).all()
    assert np.isinf(dataset.data.iloc[0, 0])


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "float64[pyarrow]"])
def test_nullable_pandas_features_fit_and_predict(dtype):
    if "pyarrow" in dtype:
        pytest.importorskip("pyarrow")
    frame = pd.DataFrame({"a": [0, None, 2, 3] * 10, "b": np.arange(40)}, dtype=dtype)
    assert _sanitize_features(frame) is frame
    if dtype != "Int64":
        frame.iloc[[0, 2, 4, 6], 0] = [np.inf, -np.inf, 1e300, -1e300]
    original = frame.copy(deep=True)
    expected = frame.copy(deep=True)
    if dtype != "Int64":
        expected.iloc[[0, 2, 4, 6], 0] = pd.NA
    pd.testing.assert_frame_equal(_sanitize_features(frame), expected)
    pd.testing.assert_frame_equal(frame, original)

    dataset = PandasDataset(
        frame,
        roles={name: NumericRole(frame[name].dtype) for name in frame},
        task=Task("binary"),
        target=pd.Series(np.tile([0, 1, 1, 0], 10), dtype=np.float32),
        folds=pd.Series(np.arange(40) % 2),
    )
    model = BoostXGB(default_params={"n_estimators": 3, "early_stopping_rounds": 2, "nthread": 1})
    oof = model.fit_predict(FoldsIterator(dataset)).data
    predictions = model.predict(dataset).data
    assert oof.shape == predictions.shape == (40, 1)
    assert np.isfinite(oof).all()
    assert np.isfinite(predictions).all()
    pd.testing.assert_frame_equal(dataset.data, original)


@pytest.mark.parametrize("early_stopping_rounds", [3, 100])
def test_predictions_use_best_iteration_even_without_early_stop(early_stopping_rounds):
    rng = np.random.RandomState(42)
    data = rng.normal(size=(120, 2)).astype(np.float32)
    target = data[:, 0] * 3
    target[80:] *= -1
    dataset = NumpyDataset(data, ["a", "b"], NumericRole(), Task("reg"), target=target)
    model = BoostXGB(
        default_params={
            "n_estimators": 12,
            "early_stopping_rounds": early_stopping_rounds,
            "learning_rate": 0.5,
            "max_depth": 2,
            "nthread": 1,
            "verbose_eval": False,
        }
    )
    valid = dataset[80:]

    oof = model.fit_predict(HoldoutIterator(dataset[:80], valid)).data[:, 0]
    booster = model.models[0]
    matrix = xgboost.DMatrix(valid.data)
    best = booster.predict(matrix, iteration_range=(0, booster.best_iteration + 1))

    assert booster.best_iteration + 1 < booster.num_boosted_rounds()
    assert not np.allclose(booster.predict(matrix), best)
    np.testing.assert_allclose(oof, best)
    np.testing.assert_allclose(model.predict(valid).data[:, 0], best)
    np.testing.assert_allclose(pickle.loads(pickle.dumps(model)).predict(valid).data[:, 0], best)
    best_importance = booster[: booster.best_iteration + 1].get_score(importance_type="gain")
    np.testing.assert_allclose(
        model.get_features_score().reindex(["a", "b"]),
        [best_importance.get("f0", 0), best_importance.get("f1", 0)],
    )


@pytest.mark.parametrize("pandas_input", [False, True])
def test_feature_importance_includes_unused_features_in_original_names(pandas_input):
    target = np.tile([0, 1], 40).astype(np.float32)
    data = np.column_stack([np.ones(80), target, np.zeros(80)])
    dataset = NumpyDataset(
        data, ["f1", "signal", "f0"], NumericRole(), Task("binary"), target=target, folds=(np.arange(80) // 2) % 2
    )
    if pandas_input:
        dataset = dataset.to_pandas()
    model = BoostXGB(default_params={"n_estimators": 4, "early_stopping_rounds": 2, "max_depth": 1, "nthread": 1})
    model.fit_predict(FoldsIterator(dataset))

    importance = model.get_features_score()

    assert len(importance) == 3
    assert set(importance.index) == set(dataset.features)
    assert importance.index[0] == "signal"
    assert importance["signal"] > 0
    assert importance["f1"] == importance["f0"] == 0


@pytest.mark.filterwarnings("ignore:.*booster=gblinear.*deprecated:UserWarning")
def test_linear_booster_prediction_does_not_slice_tree_iterations():
    data = np.arange(120, dtype=np.float32).reshape(60, 2)
    dataset = NumpyDataset(data, ["a", "b"], NumericRole(), Task("reg"), target=data[:, 0])
    model = BoostXGB(default_params={"booster": "gblinear", "n_estimators": 4, "nthread": 1, "verbose_eval": False})

    result = model.fit_predict(HoldoutIterator(dataset[:40], dataset[40:]))

    assert np.isfinite(result.data).all()
    np.testing.assert_allclose(model.predict(dataset[40:]).data, result.data)
