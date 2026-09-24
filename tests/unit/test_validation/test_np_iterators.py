import numpy as np

from lightautoml.tasks import Task
from lightautoml.validation.np_iterators import UpliftIterator


def test_uplift_iterator_uses_python_bool_cast():
    treatment = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    target = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)

    iterator = UpliftIterator(
        treatment_col=treatment,
        target=target,
        mode=True,
        task=Task("binary"),
        n_folds=2,
    )

    assert iterator.constant_idx.tolist() == [1, 3, 5, 7]
    assert iterator.splitted_idx.tolist() == [0, 2, 4, 6]
