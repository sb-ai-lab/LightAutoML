"""Multilabel target validation and encoding."""

import numpy as np
import pandas as pd
import pytest

from lightautoml.reader.base import PandasToPandasReader
from lightautoml.tasks import Task


@pytest.mark.parametrize("dtype", ["object", "string", "category"])
def test_string_targets_are_encoded_without_mutating_input(dtype):
    data = pd.DataFrame(
        {
            "feature": np.arange(12),
            "a": pd.Series(["yes", "no"] * 6, dtype=dtype),
            "b": pd.Series(["off", "on", "off"] * 4, dtype=dtype),
        }
    )
    original = data.copy(deep=True)
    reader = PandasToPandasReader(Task("multilabel"), cv=2, advanced_roles=False)
    train = reader.fit_read(data, roles={"target": ["b", "a"]})
    valid = reader.read(data, add_array_attrs=True)
    assert list(train.target.columns) == ["b", "a"]
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in train.target.dtypes)
    np.testing.assert_array_equal(train.target, valid.target)
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("values", [["a", "b", "c"] * 4, [2, 3, 4] * 4, ["only"] * 12, ["a", "b", None] * 4])
def test_invalid_multilabel_targets_are_rejected(values):
    reader = PandasToPandasReader(Task("multilabel"), cv=2, advanced_roles=False)
    with pytest.raises(AssertionError):
        reader.fit_read(pd.DataFrame({"x": np.arange(12), "target": values}), roles={"target": "target"})


@pytest.mark.parametrize("dtype", ["string", "category"])
def test_validation_labels_use_fitted_mapping(dtype):
    train = pd.DataFrame({"x": np.arange(12), "a": ["no", "yes"] * 6, "b": ["off", "on"] * 6}, dtype=object)
    reader = PandasToPandasReader(Task("multilabel"), cv=2, advanced_roles=False)
    reader.fit_read(train, roles={"target": ["a", "b"]})
    valid = pd.DataFrame(
        {"x": [5, 6], "a": pd.Series(["yes", "no"], dtype=dtype), "b": pd.Series(["off", "on"], dtype=dtype)}
    )
    valid.index = [17, 19]
    original = valid.copy(deep=True)
    result = reader.read(valid, add_array_attrs=True)
    np.testing.assert_array_equal(result.target, [[1, 0], [0, 1]])
    assert list(result.target.index) == [17, 19]
    pd.testing.assert_frame_equal(valid, original)
