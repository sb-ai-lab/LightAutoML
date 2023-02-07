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
        lags: int (number of lags to compute), list / np.ndarray (numbers of certain lags to compute).

    """

    _fit_checks = ()
    _transform_checks = ()
    _fname_prefix = "lag"

    @property
    def features(self) -> List[str]:
        """Features list."""
        return self._features

    @staticmethod
    def get_attributes(dataset):
        params = {}
        for attribute in dataset._array_like_attrs:
            _data = []
            _d = getattr(dataset, attribute).values
            for row in np.arange(len(dataset)):
                _data.append(_d[dataset.idx[row]][-1])
            _data = np.array(_data)
            params[attribute] = _data
        return params

    def __init__(self, lags: Union[int, List[int], np.ndarray[int]] = 10):
        if isinstance(lags, list):
            self.lags = np.array(lags)
        if isinstance(lags, int):
            self.lags = np.arange(lags)

    def fit(self, dataset):
        """Fit algorithm on seq dataset.

        Args:
            dataset: NumpyDataset.

        Returns:
            Fitted transformer.

        """
        sample_data = dataset.to_sequence([0]).data  # (1 observation, history, features)

        # convert to accepted dtype and get attributes
        # leave only correct lags (less than number of observations in sample_data)
        self.current_correct_lags = self.lags.copy()[self.lags < sample_data.data.shape[1]]

        feats = []
        for feat in dataset.features:
            feats.extend([self._fname_prefix + f"_{i}" + "__" + feat for i in reversed(self.current_correct_lags)])
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
        data_seq = dataset.to_sequence().data
        data = data_seq[:, (data_seq.shape[1] - 1) - self.current_correct_lags[::-1], :]

        params = self.get_attributes(dataset)

        # transform
        data = np.moveaxis(data, 1, 2).reshape(len(data), -1)

        # create resulted
        return NumpyDataset(data, self.features, NumericRole(np.float32), **params)


class DiffTransformer(SeqLagTransformer, LAMLTransformer):
    """Diff.

    Args:
        diffs: int (number of diffs to compute), list / np.ndarray (numbers of certain diffs to compute).
        diffs = 0 means no diff (y_t), 1 means diff between last and previous (y_t-y_{t-1}), 2 means diff between last and previous to previous (y_t-t_{t-2}), etc.

    """

    _fit_checks = ()
    _transform_checks = ()
    _fname_prefix = "diff"

    @property
    def features(self) -> List[str]:
        """Features list."""
        return self._features

    def __init__(self, diffs: Union[int, List[int], np.ndarray[int]] = 10, flag_del_0_diff=False):
        SeqLagTransformer.__init__(self, lags=diffs)
        self.flag_del_0_diff = flag_del_0_diff  # if True, we need to drop diff with number 0 to avoid column duplication

    def fit(self, dataset):
        if self.flag_del_0_diff:
            self.lags = self.lags[self.lags > 0]  # drop diff=0
        SeqLagTransformer.fit(self, dataset)

    def transform(self, dataset) -> NumpyDataset:
        # checks here
        LAMLTransformer.transform(self, dataset)

        # convert to accepted dtype and get attributes
        data_seq = dataset.to_sequence().data
        data_seq_t = data_seq[:, (data_seq.shape[1] - 1) - np.array([0]), :]

        params = self.get_attributes(dataset)

        # transform
        if 0 in self.current_correct_lags:
            data_seq_diffs = data_seq[:, (data_seq.shape[1] - 1) - self.current_correct_lags[::-1][:-1], :]
            data_seq_diffs = np.concatenate(
                (data_seq_diffs, np.zeros((data_seq_diffs.shape[0], 1, data_seq_diffs.shape[2]))), axis=1)
        else:
            data_seq_diffs = data_seq[:, (data_seq.shape[1] - 1) - self.current_correct_lags[::-1], :]

        data_t = np.moveaxis(data_seq_t, 1, 2)[:, :, ::-1].reshape(len(data_seq_t), -1)
        data_t_diffs = np.moveaxis(data_seq_diffs, 1, 2).reshape(len(data_seq_diffs), -1)
        final_t = np.repeat(data_t, len(self.current_correct_lags), axis=1) - data_t_diffs

        # create resulted
        return NumpyDataset(final_t, self.features, NumericRole(np.float32), **params)


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

        Returns:
            Self.

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

        Returns:
            Self.

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

        Returns:
            Self.

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
