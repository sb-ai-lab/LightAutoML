"""Pandas dataset index normalization without deprecated inplace arguments."""

import numpy as np
import pandas as pd

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import NumericRole
from lightautoml.tasks import Task


def test_dataset_normalizes_index_without_mutating_input():
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0]}, index=[9, 7, 5])
    original = frame.copy(deep=True)
    target = pd.Series([0.0, 1.0, 0.0], index=frame.index)

    dataset = PandasDataset(frame, {"a": NumericRole(np.float64)}, Task("binary"), target=target)

    pd.testing.assert_frame_equal(dataset.data, original.reset_index(drop=True))
    pd.testing.assert_frame_equal(frame, original)
    np.testing.assert_array_equal(dataset[[2, 0]].target, [0.0, 0.0])
