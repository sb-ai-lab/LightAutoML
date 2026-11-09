"""Regressions for non-finite XGBoost features in dense and sparse datasets."""

import numpy as np
import pandas as pd
import pytest

from scipy import sparse

from lightautoml.dataset.np_pd_dataset import NumpyDataset
from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import NumericRole
from lightautoml.ml_algo.boost_xgb import BoostXGB
from lightautoml.ml_algo.boost_xgb import _sanitize_features
from lightautoml.tasks import Task
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
