"""Support pandas extension dtypes used by current pandas releases."""

import warnings

import numpy as np
import pandas as pd
import pytest

from lightautoml.reader.base import PandasToPandasReader
from lightautoml.tasks import Task
from lightautoml.transformers.categorical import OrdinalEncoder


@pytest.mark.parametrize("dtype", ["object", "string"])
def test_explicit_categorical_role_accepts_string_dtype(dtype):
    data = pd.DataFrame(
        {
            "category": pd.Series(["a", "b", "c", "a"] * 10, dtype=dtype),
            "numeric": np.arange(40, dtype=np.float32),
            "target": [0, 1] * 20,
        }
    )
    reader = PandasToPandasReader(Task("binary"), cv=2, advanced_roles=False)
    dataset = reader.fit_read(data, roles={"target": "target", "category": ["category"]})
    assert dataset.roles["category"].name == "Category"
    assert reader.read(data.drop(columns="target")).shape == dataset.shape


def test_ordinal_encoding_with_arrow_strings_has_no_deprecated_rank_call():
    pytest.importorskip("pyarrow")
    data = pd.DataFrame({"category": pd.Series(["a", "b", "c"] * 10, dtype="string[pyarrow]")})
    data["target"] = np.arange(30) % 2
    reader = PandasToPandasReader(Task("binary"), cv=2, advanced_roles=False)
    dataset = reader.fit_read(data, roles={"target": "target", "category": ["category"]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        encoded = OrdinalEncoder().fit_transform(dataset).data
    assert encoded.shape == (30, 1)
