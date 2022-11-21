"""Generate sequential features."""

from typing import List
from typing import Union

from copy import copy

import numpy as np
import cupy as cp
import cudf
import dask_cudf

from lightautoml.dataset.gpu.gpu_dataset import CupyDataset
from lightautoml.dataset.gpu.gpu_dataset import CudfDataset
from lightautoml.dataset.gpu.gpu_dataset import DaskCudfDataset
from lightautoml.dataset.roles import NumericRole
from lightautoml.dataset.gpu.gpu_dataset import SeqCudfDataset
from lightautoml.dataset.gpu.gpu_dataset import SeqDaskCudfDataset
from lightautoml.transformers.base import LAMLTransformer
from lightautoml.transformers.seq import SeqLagTransformer
from lightautoml.transformers.seq import SeqNumCountsTransformer
from lightautoml.transformers.seq import SeqStatisticsTransformer
from lightautoml.transformers.seq import GetSeqTransformer

# type - something that can be converted to pandas dataset
CupyTransformable = Union[CupyDataset, CudfDataset, DaskCudfDataset]

class SeqNumCountsTransformerGPU(SeqNumCountsTransformer):
    """NC."""

    _fit_checks = ()
    _transform_checks = ()
    _fname_prefix = "numcount_gpu"

    def __init__(self):
        super().__init__()

    def fit(self, dataset):
        """Fit algorithm on seq dataset.

        Args:
            dataset: NumpyDataset.

        Returns:
            Self.

        """
        super().fit(dataset)
        self._roles = dict(((x, NumericRole(np.float32)) for x in self._features))
        return self

    def transform(self, dataset):
        """Transform input seq dataset to normal numpy representation.

        Args:
            dataset: seq.

        Returns:
            Numpy dataset with len of the sequence.

        """
        # checks here
        super(SeqNumCountsTransformerGPU.__bases__[0], self).transform(dataset)
        # convert to accepted dtype and get attributes

        #for calculating counts this method is long on gpu
        #data = dataset.apply_func((slice(None)), len)#.reshape(-1, 1)
        #instead you can just calculate length of the idx
        data = cudf.DataFrame([len(x) for x in dataset.idx], columns=self._features)

        # create resulted
        if isinstance(dataset.data, cudf.DataFrame):
            dat_t =  CudfDataset
        elif isinstance(dataset.data, dask_cudf.DataFrame):
            dat_t = DaskCudfDataset
            data = dask_cudf.from_cudf(data, npartitions=dataset.data.npartitions)
        else:
            raise NotImplementedError
        # create resulted
        #data = data.rename(columns=dict(zip(data.columns,
        #                                    self._features)))
        return dat_t(data, roles = self._roles)


class SeqStatisticsTransformerGPU(SeqStatisticsTransformer):
    """SSF."""

    _fit_checks = ()
    _transform_checks = ()
    _fname_prefix = "stat_gpu"

    def __init__(self):
        super().__init__()

    def fit(self, dataset):
        """Fit algorithm on seq dataset.

        Args:
            dataset: NumpyDataset.

        Returns:
            Self.

        """
        super().fit(dataset)
        self._roles = dict(((x, NumericRole(np.float32)) for x in self._features))
        return self

    def transform(self, dataset):
        """Transform input seq dataset to normal numpy representation.

        Args:
            dataset: seq.

        Returns:
            Numpy dataset with last known feature in the sequence.

        """
        # checks here
        super(SeqStatisticsTransformerGPU.__bases__[0], self).transform(dataset)
        # convert to accepted dtype and get attributes
        
        #for calculating _get_last this method is long on gpu
        #data = dataset.apply_func((slice(None)), self._get_last)
        #instead you can just rearange the idx and then get the frame
        temp = copy(dataset.idx)
        dataset.idx = np.array([[x[-1]] for x in dataset.idx])
        data = dataset.get_first_frame().data
        dataset.idx = copy(temp)

        if isinstance(data, cudf.DataFrame):
            dat_t =  CudfDataset
        elif isinstance(data, dask_cudf.DataFrame):
            dat_t = DaskCudfDataset
        else:
            raise NotImplementedError
        # create resulted
        data = data.rename(columns=dict(zip(data.columns,
                                            self._features)))

        return dat_t(data, roles = self._roles)

    @staticmethod
    def _std(x):
        return x.std()

    @staticmethod
    def _get_last(x):
        if len(x) > 0:
            return x[-1, :]
        else:
            return cp.array([np.NaN] * x.shape[-1])


class GetSeqTransformerGPU(GetSeqTransformer):
    """LAG."""

    _fit_checks = ()
    _transform_checks = ()
    _fname_prefix = "seq_gpu"

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def transform(self, dataset):
        """Transform input seq dataset to normal numpy representation.

        Args:
            dataset: seq.

        Returns:
            Numpy dataset with lag features.

        """
        # checks here
        super(GetSeqTransformerGPU.__bases__[0], self).transform(dataset)
        # convert to accepted dtype and get attributes

        dataset = dataset.seq_data.get(self.name)
        data = dataset.data
        data.columns = self._features
        kwargs = {}
        for attr in dataset._array_like_attrs:
            kwargs[attr] = dataset.__dict__[attr]

        if isinstance(data, cudf.DataFrame):
            dat_t =  SeqCudfDataset
        elif isinstance(data, dask_cudf.DataFrame):
            dat_t = SeqDaskCudfDataset
        else:
            raise NotImplementedError
        # create resulted
        result = dat_t(
            data=data,
            features=self.features,
            roles=self.roles,
            idx=dataset.idx,
            name=dataset.name,
            scheme=dataset.scheme,
            **kwargs,
        )

        return result
