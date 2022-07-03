"""Validation utils."""

from typing import Callable
from typing import Optional
from typing import Union
from typing import cast

from ..dataset.base import LAMLDataset
from ..dataset.np_pd_dataset import CSRSparseDataset
from ..dataset.np_pd_dataset import NumpyDataset
from ..dataset.np_pd_dataset import PandasDataset
from .base import DummyIterator
from .base import HoldoutIterator
from .base import TrainValidIterator
from .np_iterators import get_numpy_iterator


NpDataset = Union[CSRSparseDataset, NumpyDataset, PandasDataset]


def create_validation_iterator(
    train: LAMLDataset,
    valid: Optional[LAMLDataset] = None,
    n_folds: Optional[int] = None,
    cv_iter: Optional[Callable] = None,
) -> TrainValidIterator:
    """Creates train-validation iterator.

    If train is one of common datasets types
    (``PandasDataset``, ``NumpyDataset``, ``CSRSparseDataset``)
    the :func:`~lightautoml.validation.np_iterators.get_numpy_iterator`
    will be used.
    Else if validation dataset is defined,
    the holdout-iterator will be used.
    Else the dummy iterator will be used.

    Args:
        train: Dataset to train.
        valid: Optional dataset for validate.
        n_folds: maximum number of folds to iterate.
          If ``None`` - iterate through all folds.
        cv_iter: Takes dataset as input and return
          an iterator of indexes of train/valid for train dataset.

    Returns:
        New iterator.

    """
    if type(train) in [PandasDataset, NumpyDataset, CSRSparseDataset]:
        train = cast(NpDataset, train)
        valid = cast(NpDataset, valid)
        iterator = get_numpy_iterator(train, valid, n_folds, cv_iter)

    else:
        if valid is not None:
            iterator = HoldoutIterator(train, valid)
        else:
            iterator = DummyIterator(train)

    return iterator
