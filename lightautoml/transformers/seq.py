"""Generate sequential features."""

from typing import List
from typing import Union

import numpy as np

from ..dataset.np_pd_dataset import CSRSparseDataset
from ..dataset.np_pd_dataset import NumpyDataset
from ..dataset.np_pd_dataset import PandasDataset
from ..dataset.roles import NumericRole
from ..dataset.seq_np_pd_dataset import SeqNumpyPandasDataset
from ..transformers.base import LAMLTransformer


# type - something that can be converted to pandas dataset
NumpyTransformable = Union[NumpyDataset, PandasDataset]
NumpyCSR = Union[NumpyDataset, CSRSparseDataset]


class SeqLagTransformer(LAMLTransformer):
    """LAG.

    Flattens different features

    Args:
        n_lags: number of lags to compute.

    """

    _fit_checks = ()
    _transform_checks = ()
    _fname_prefix = "lag"

    @property
    def features(self) -> List[str]:
        """Features list."""
        return self._features

    def __init__(self, n_lags: int = 10):
        self.n_lags = n_lags

    def fit(self, dataset):
        """Fit algorithm on seq dataset.

        Args:
            dataset: NumpyDataset.

        """

        sample_data = dataset.to_sequence([0]).data

        # convert to accepted dtype and get attributes
        self.n_lags = np.minimum(self.n_lags, sample_data.data.shape[1])

        feats = []
        for feat in dataset.features:
            feats.extend([self._fname_prefix + f"_{i}" + "__" + feat for i in reversed(range(self.n_lags))])
        self._features = list(feats)
        return self

    def transform(self, dataset) -> NumpyDataset:
        """Transform input seq dataset to normal numpy representation.

        Args:
            dataset: seq.

        Returns:
            Numpy dataset with lag features.

        """
        # checks here
        super().transform(dataset)
        # convert to accepted dtype and get attributes

        data = dataset.to_sequence().data[:, -self.n_lags :, :]

        params = {}
        for attribute in dataset._array_like_attrs:
            _data = []
            _d = getattr(dataset, attribute).values
            for row in np.arange(len(dataset)):
                _data.append(_d[dataset.idx[row]][-1])
            _data = np.array(_data)
            params[attribute] = _data
        # transform
        data = np.moveaxis(data, 1, 2).reshape(len(data), -1)
        # create resulted
        return NumpyDataset(data, self.features, NumericRole(np.float32), **params)


class SeqNumCountsTransformer(LAMLTransformer):
    """NC."""

    _fit_checks = ()
    _transform_checks = ()
    _fname_prefix = "numcount"

    @property
    def features(self) -> List[str]:
        """Features list."""
        return self._features

    def __init__(self):
        pass

    def fit(self, dataset):
        """Fit algorithm on seq dataset.

        Args:
            dataset: NumpyDataset.

        """
        # set transformer names and add checks
        # for check_func in self._fit_checks:
        #    check_func(dataset)
        # set transformer features

        feats = [self._fname_prefix + "__" + dataset.name]

        self._features = list(feats)
        return self

    def transform(self, dataset) -> NumpyDataset:
        """Transform input seq dataset to normal numpy representation.

        Args:
            dataset: seq.

        Returns:
            Numpy dataset with len of the sequence.

        """
        # checks here
        super().transform(dataset)
        # convert to accepted dtype and get attributes
        data = dataset.apply_func((slice(None)), len).reshape(-1, 1)
        # transform

        # print('name', dataset.name)
        # print('scheme', dataset.scheme)
        # create resulted
        return NumpyDataset(data, self.features, NumericRole(np.float32))


class SeqStatisticsTransformer(LAMLTransformer):
    """SSF."""

    _fit_checks = ()
    _transform_checks = ()
    _fname_prefix = "stat"

    @property
    def features(self) -> List[str]:
        """Features list."""
        return self._features

    def __init__(self):
        pass

    def fit(self, dataset):
        """Fit algorithm on seq dataset.

        Args:
            dataset: NumpyDataset.

        """

        feats = []
        for feat in dataset.features:
            feats.extend([self._fname_prefix + "__" + dataset.name + "__" + feat])
        self._features = list(feats)
        return self

    def transform(self, dataset) -> NumpyDataset:
        """Transform input seq dataset to normal numpy representation.

        Args:
            dataset: seq.

        Returns:
            Numpy dataset with last known feature in the sequence.

        """
        # checks here
        super().transform(dataset)
        # convert to accepted dtype and get attributes
        data = dataset.apply_func((slice(None)), self._get_last)

        return NumpyDataset(data, self.features, NumericRole(np.float32))

    @staticmethod
    def _std(x):
        return np.std(x, axis=0)

    @staticmethod
    def _get_last(x):
        if len(x) > 0:
            return x[-1, :]
        else:
            return np.array([np.NaN] * x.shape[-1])


class GetSeqTransformer(LAMLTransformer):
    """LAG."""

    _fit_checks = ()
    _transform_checks = ()
    _fname_prefix = "seq"

    @property
    def features(self) -> List[str]:
        """Features list."""
        return self._features

    def __init__(self, name="seq"):

        self.name = name

    def fit(self, dataset):
        """Fit algorithm on seq dataset.

        Args:
            dataset: NumpyDataset.

        """
        # set transformer names and add checks
        # for check_func in self._fit_checks:
        #    check_func(dataset)
        # set transformer features
        sample_data = dataset.seq_data.get(self.name)
        # convert to accepted dtype and get attributes

        feats = []
        self.roles = {}
        for feat, role in sample_data.roles.items():
            # feats.extend([self._fname_prefix + '__' + feat])
            feats.extend([feat])
            self.roles[feats[-1]] = role
        self._features = list(feats)
        return self

    def transform(self, dataset) -> NumpyDataset:
        """Transform input seq dataset to normal numpy representation.

        Args:
            dataset: seq.

        Returns:
            Numpy dataset with lag features.

        """
        # checks here
        super().transform(dataset)
        # convert to accepted dtype and get attributes

        dataset = dataset.seq_data.get(self.name)
        data = dataset.data
        data.columns = self._features
        kwargs = {}
        for attr in dataset._array_like_attrs:
            kwargs[attr] = dataset.__dict__[attr]

        # create resulted
        result = SeqNumpyPandasDataset(
            data=data,
            features=self.features,
            roles=self.roles,
            idx=dataset.idx,
            name=dataset.name,
            scheme=dataset.scheme,
            **kwargs,
        )

        return result
