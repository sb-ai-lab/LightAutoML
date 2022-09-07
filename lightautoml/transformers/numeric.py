"""Numeric features transformers."""

from typing import Union

import numpy as np

from ..dataset.base import LAMLDataset
from ..dataset.np_pd_dataset import NumpyDataset
from ..dataset.np_pd_dataset import PandasDataset
from ..dataset.roles import CategoryRole
from ..dataset.roles import NumericRole
from .base import LAMLTransformer


# type - something that can be converted to pandas dataset
NumpyTransformable = Union[NumpyDataset, PandasDataset]


def numeric_check(dataset: LAMLDataset):
    """Check if all passed vars are categories.

    Args:
        dataset: Dataset to check.

    Raises:
        AssertionError: If there is non number role.

    """
    roles = dataset.roles
    features = dataset.features
    for f in features:
        assert roles[f].name == "Numeric", "Only numbers accepted in this transformer"


class NaNFlags(LAMLTransformer):
    """Create NaN flags."""

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "nanflg"

    def __init__(self, nan_rate: float = 0.005):
        """

        Args:
            nan_rate: Nan rate cutoff.

        """
        self.nan_rate = nan_rate

    def fit(self, dataset: NumpyTransformable):
        """Extract nan flags.

        Args:
            dataset: Pandas or Numpy dataset of categorical features.

        Returns:
            self.

        """
        # set transformer names and add checks
        for check_func in self._fit_checks:
            check_func(dataset)
        # set transformer features

        # convert to accepted dtype and get attributes
        dataset = dataset.to_numpy()
        data = dataset.data
        # fit ...
        ds_nan_rate = np.isnan(data).mean(axis=0)
        self.nan_cols = [name for (name, nan_rate) in zip(dataset.features, ds_nan_rate) if nan_rate > self.nan_rate]
        self._features = list(self.nan_cols)

        return self

    def transform(self, dataset: NumpyTransformable) -> NumpyDataset:
        """Transform - extract null flags.

        Args:
            dataset: Pandas or Numpy dataset of categorical features.

        Returns:
            Numpy dataset with encoded labels.

        """
        # checks here
        super().transform(dataset)
        # convert to accepted dtype and get attributes
        dataset = dataset.to_numpy()
        nans = dataset[:, self.nan_cols].data

        # transform
        new_arr = np.isnan(nans).astype(np.float32)

        # create resulted
        output = dataset.empty().to_numpy()
        output.set_data(new_arr, self.features, NumericRole(np.float32))

        return output


class FillnaMedian(LAMLTransformer):
    """Fillna with median."""

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "fillnamed"

    def fit(self, dataset: NumpyTransformable):
        """Estimate medians.

        Args:
            dataset: Pandas or Numpy dataset of categorical features.

        Returns:
            self.

        """
        # set transformer names and add checks
        super().fit(dataset)
        # set transformer features

        # convert to accepted dtype and get attributes
        dataset = dataset.to_numpy()
        data = dataset.data

        self.meds = np.nanmedian(data, axis=0)
        self.meds[np.isnan(self.meds)] = 0

        return self

    def transform(self, dataset: NumpyTransformable) -> NumpyDataset:
        """Transform - fillna with medians.

        Args:
            dataset: Pandas or Numpy dataset of categorical features.

        Returns:
            Numpy dataset with encoded labels.

        """
        # checks here
        super().transform(dataset)
        # convert to accepted dtype and get attributes
        dataset = dataset.to_numpy()
        data = dataset.data
        # transform
        data = np.where(np.isnan(data), self.meds, data)

        # create resulted
        output = dataset.empty().to_numpy()
        output.set_data(data, self.features, NumericRole(np.float32))

        return output


class FillInf(LAMLTransformer):
    """Fill inf with nan to handle as nan value."""

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "fillinf"

    def transform(self, dataset: NumpyTransformable) -> NumpyDataset:
        """Replace inf to nan.

        Args:
            dataset: Pandas or Numpy dataset of categorical features.

        Returns:
            Numpy dataset with encoded labels.

        """
        # checks here
        super().transform(dataset)
        # convert to accepted dtype and get attributes
        dataset = dataset.to_numpy()
        data = dataset.data
        # transform

        data = np.where(np.isinf(data), np.nan, data)

        # create resulted
        output = dataset.empty().to_numpy()
        output.set_data(data, self.features, NumericRole(np.float32))

        return output


class LogOdds(LAMLTransformer):
    """Convert probs to logodds."""

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "logodds"

    def transform(self, dataset: NumpyTransformable) -> NumpyDataset:
        """Transform - convert num values to logodds.

        Args:
            dataset: Pandas or Numpy dataset of categorical features.

        Returns:
            Numpy dataset with encoded labels.

        """
        # checks here
        super().transform(dataset)
        # convert to accepted dtype and get attributes
        dataset = dataset.to_numpy()
        data = dataset.data
        # transform
        # TODO: maybe np.exp and then cliping and logodds?
        data = np.clip(data, 1e-7, 1 - 1e-7)
        data = np.log(data / (1 - data))

        # create resulted
        output = dataset.empty().to_numpy()
        output.set_data(data, self.features, NumericRole(np.float32))

        return output


class StandardScaler(LAMLTransformer):
    """Classic StandardScaler."""

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "scaler"

    def fit(self, dataset: NumpyTransformable):
        """Estimate means and stds.

        Args:
            dataset: Pandas or Numpy dataset of categorical features.

        Returns:
            self.

        """
        # set transformer names and add checks
        super().fit(dataset)
        # set transformer features

        # convert to accepted dtype and get attributes
        dataset = dataset.to_numpy()
        data = dataset.data

        self.means = np.nanmean(data, axis=0)
        self.stds = np.nanstd(data, axis=0)
        # Fix zero stds to 1
        self.stds[(self.stds == 0) | np.isnan(self.stds)] = 1

        return self

    def transform(self, dataset: NumpyTransformable) -> NumpyDataset:
        """Scale test data.

        Args:
            dataset: Pandas or Numpy dataset of numeric features.

        Returns:
            Numpy dataset with encoded labels.

        """
        # checks here
        super().transform(dataset)
        # convert to accepted dtype and get attributes
        dataset = dataset.to_numpy()
        data = dataset.data

        # transform
        data = (data - self.means) / self.stds

        # create resulted
        output = dataset.empty().to_numpy()
        output.set_data(data, self.features, NumericRole(np.float32))

        return output


class QuantileBinning(LAMLTransformer):
    """Discretization of numeric features by quantiles."""

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "qntl"

    def __init__(self, nbins: int = 10):
        """

        Args:
            nbins: maximum number of bins.

        """
        self.nbins = nbins

    def fit(self, dataset: NumpyTransformable):
        """Estimate bins borders.

        Args:
            dataset: Pandas or Numpy dataset of numeric features.

        Returns:
            self.

        """
        # set transformer names and add checks
        super().fit(dataset)
        # set transformer features

        # convert to accepted dtype and get attributes
        dataset = dataset.to_numpy()
        data = dataset.data

        sl = np.isnan(data)
        grid = np.linspace(0, 1, self.nbins + 1)[1:-1]

        self.bins = []

        for n in range(data.shape[1]):
            q = np.quantile(data[:, n][~sl[:, n]], q=grid)
            q = np.unique(q)
            self.bins.append(q)

        return self

    def transform(self, dataset: NumpyTransformable) -> NumpyDataset:
        """Apply bin borders.

        Args:
            dataset: Pandas or Numpy dataset of numeric features.

        Returns:
            Numpy dataset with encoded labels.

        """
        # checks here
        super().transform(dataset)
        # convert to accepted dtype and get attributes
        dataset = dataset.to_numpy()
        data = dataset.data

        # transform
        sl = np.isnan(data)

        new_data = np.zeros(data.shape, dtype=np.int32)

        for n, b in enumerate(self.bins):
            new_data[:, n] = np.searchsorted(b, np.where(sl[:, n], np.inf, data[:, n])) + 1

        new_data = np.where(sl, 0, new_data)

        # create resulted
        output = dataset.empty().to_numpy()
        output.set_data(new_data, self.features, CategoryRole(np.int32, label_encoded=True))

        return output
