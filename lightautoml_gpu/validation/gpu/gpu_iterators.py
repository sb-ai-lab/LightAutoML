"""Tabular iterators (GPU version)."""

from typing import Optional, Tuple, Union, cast

import cupy as cp
import numpy as np

from lightautoml_gpu.dataset.gpu.gpu_dataset import CudfDataset
from lightautoml_gpu.dataset.gpu.gpu_dataset import CupyDataset
from lightautoml_gpu.dataset.gpu.gpu_dataset import DaskCudfDataset

from lightautoml_gpu.validation.base import CustomIdxs
from lightautoml_gpu.validation.base import CustomIterator
from lightautoml_gpu.validation.base import DummyIterator
from lightautoml_gpu.validation.base import HoldoutIterator
from lightautoml_gpu.validation.base import TrainValidIterator

from lightautoml_gpu.validation.np_iterators import FoldsIterator

GpuDataset = Union[CupyDataset, CudfDataset, DaskCudfDataset]


class HoldoutIteratorGPU(HoldoutIterator):
    """Iterator for classic holdout - just predefined train and valid samples (GPU version requires indexing).

    Create iterator.

    Args:
        train: Dataset of train data.
        valid: Dataset of valid data.
    """

    def __init__(self, train: GpuDataset, valid: GpuDataset):

        self.train = train
        self.valid = valid

    def __len__(self) -> Optional[int]:
        """Get 1 len.

        Returns:
            1

        """
        return 1

    def __iter__(self) -> "HoldoutIteratorGPU":
        """Simple iterable object.

        Returns:
            Iterable object for train validation dataset.

        """

        return iter([(None, self.train, self.valid)])

    def __getitem__(self, number: int):
        if number >= 1:
            raise IndexError("index out of range")

        return None, self.train, self.valid


class FoldsIteratorGPU(TrainValidIterator):
    """Classic cv iterator.

    Folds should be defined in Reader, based on cross validation method.

    Creates iterator (GPU version).

    Args:
        train: Dataset for folding.
        n_folds: Number of folds.

    """

    def __init__(self, train: GpuDataset, n_folds: Optional[int] = None):

        assert hasattr(
            train, "folds"
        ), "Folds in dataset should be defined to make folds iterator."

        self.train = train

        max_folds = train.folds.max()
        if type(train) == DaskCudfDataset:
            max_folds = max_folds.compute()
        self.n_folds = max_folds + 1
        if n_folds is not None:
            self.n_folds = min(self.n_folds, n_folds)

    def __len__(self) -> int:
        """Get len of iterator.

        Returns:
            Number of folds.

        """
        return self.n_folds

    def __iter__(self) -> "FoldsIterator":
        """Set counter to 0 and return self.

        Returns:
            Iterator for folds.

        """
        self._curr_idx = 0
        return self

    def __getitem__(self, number: int) -> Tuple[np.ndarray, GpuDataset, GpuDataset]:
        if number >= self.n_folds:
            raise IndexError("index out of range")

        val_idx = self.train.folds == number
        if type(self.train) == CudfDataset:
            val_idx = val_idx.values
        elif type(self.train) == DaskCudfDataset:
            val_idx = val_idx.compute().values

        tr_idx = cp.logical_not(val_idx)
        idx = cp.arange(self.train.shape[0])
        tr_idx, val_idx = idx[tr_idx], idx[val_idx]
        if type(self.train) == DaskCudfDataset:
            tr_idx = tr_idx.get()
            val_idx = val_idx.get()

        train, valid = self.train[tr_idx], self.train[val_idx]

        return val_idx, cast(GpuDataset, train), cast(GpuDataset, valid)

    def __next__(self) -> Tuple[np.ndarray, GpuDataset, GpuDataset]:
        """Define how to get next object.

        Returns:
            Mask for current fold, train dataset, validation dataset.

        """

        if self._curr_idx == self.n_folds:
            raise StopIteration

        val_idx = self.train.folds == self._curr_idx

        if type(self.train) == CudfDataset:
            val_idx = val_idx.values
        elif type(self.train) == DaskCudfDataset:
            val_idx = val_idx.compute().values

        tr_idx = cp.logical_not(val_idx)
        idx = cp.arange(self.train.shape[0])
        tr_idx, val_idx = idx[tr_idx], idx[val_idx]
        if type(self.train) == DaskCudfDataset:
            tr_idx = tr_idx.get()
            val_idx = val_idx.get()

        train, valid = self.train[tr_idx], self.train[val_idx]
        self._curr_idx += 1
        return val_idx, train, valid

    def get_validation_data(self) -> GpuDataset:
        """Just return train dataset.

        Returns:
            Whole train dataset.

        """
        return self.train

    def convert_to_holdout_iterator(self) -> HoldoutIteratorGPU:
        """Convert iterator to hold-out-iterator.

        Fold 0 is used for validation, everything else is used for training.

        Returns:
            new hold-out-iterator.

        """
        val_idx = self.train.folds == 0
        if type(self.train) == CudfDataset:
            val_idx = val_idx.values
        elif type(self.train) == DaskCudfDataset:
            val_idx = val_idx.compute().values

        tr_idx = cp.logical_not(val_idx)
        idx = cp.arange(self.train.shape[0])

        tr_idx, val_idx = idx[tr_idx], idx[val_idx]
        if type(self.train) == DaskCudfDataset:
            tr_idx = tr_idx.get()
            val_idx = val_idx.get()

        train, valid = self.train[tr_idx], self.train[val_idx]

        return HoldoutIteratorGPU(train, valid)


def get_gpu_iterator(
    train: GpuDataset,
    valid: Optional[GpuDataset] = None,
    n_folds: Optional[int] = None,
    iterator: Optional[CustomIdxs] = None,
) -> Union[
    FoldsIteratorGPU,
    HoldoutIteratorGPU,
    HoldoutIterator,
    CustomIterator,
    DummyIterator,
]:
    """Get iterator for gpu dataset.

    If valid is defined, other parameters are ignored.
    Else if iterator is defined n_folds is ignored.

    Else if n_folds is defined iterator will be created by folds index.
    Else ``DummyIterator`` - (train, train) will be created.

    Args:
        train: ``LAMLDataset`` to train.
        valid: Optional ``LAMLDataset`` for validate.
        n_folds: maximum number of folds to iterate.
          If ``None`` - iterate through all folds.
        iterator: Takes dataset as input and return an iterator
          of indexes of train/valid for train dataset.

    Returns:
        new train-validation iterator.

    """
    if valid is not None:
        train_valid = HoldoutIterator(train, valid)
    elif iterator is not None:
        train_valid = CustomIterator(train, iterator)
    elif train.folds is not None:
        train_valid = FoldsIteratorGPU(train, n_folds)
    else:
        train_valid = DummyIterator(train)

    return train_valid
