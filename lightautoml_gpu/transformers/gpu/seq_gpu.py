"""Generate sequential features (GPU version)."""

from typing import Union

from copy import copy, deepcopy

import numpy as np
import cupy as cp
import cudf
import dask_cudf

from lightautoml_gpu.dataset.gpu.gpu_dataset import CupyDataset
from lightautoml_gpu.dataset.gpu.gpu_dataset import CudfDataset
from lightautoml_gpu.dataset.gpu.gpu_dataset import DaskCudfDataset
from lightautoml_gpu.dataset.roles import NumericRole
from lightautoml_gpu.dataset.gpu.gpu_dataset import SeqCudfDataset
from lightautoml_gpu.dataset.gpu.gpu_dataset import SeqDaskCudfDataset
from lightautoml_gpu.transformers.seq import SeqNumCountsTransformer
from lightautoml_gpu.transformers.seq import SeqStatisticsTransformer
from lightautoml_gpu.transformers.seq import GetSeqTransformer

GpuSeqDataset = Union[SeqCudfDataset, SeqDaskCudfDataset]
GpuDataset = Union[CupyDataset, CudfDataset, DaskCudfDataset]


class SeqNumCountsTransformerGPU(SeqNumCountsTransformer):
    """NC."""

    _fit_checks = ()
    _transform_checks = ()
    _fname_prefix = "numcount"

    def __init__(self):
        super().__init__()

    def to_cpu(self):
        """Move the class properties to CPU and change class to CPU counterpart for CPU inference.

        Returns:
            self
        """
        features = deepcopy(self._features)
        roles = self._roles
        self.__class__ = SeqNumCountsTransformer
        self._features = features
        self._roles = roles
        return self

    def fit(self, dataset: GpuSeqDataset):
        """Fit algorithm on seq dataset.

        Args:
            dataset: CupyDataset, CudfDataset or DaskCudfDataset.

        Returns:
            Self.

        """
        super().fit(dataset)
        self._roles = dict(((x, NumericRole(np.float32)) for x in self._features))
        return self

    def transform(self, dataset: GpuSeqDataset) -> GpuDataset:
        """Transform input seq dataset to normal numpy representation.

        Args:
            dataset: seq.

        Returns:
            GPU dataset with len of the sequence.

        """
        # checks here
        super(SeqNumCountsTransformerGPU.__bases__[0], self).transform(dataset)
        # convert to accepted dtype and get attributes

        # for calculating counts this method is long on gpu
        # data = dataset.apply_func((slice(None)), len)#.reshape(-1, 1)
        # instead you can just calculate length of the idx
        data = cudf.DataFrame([len(x) for x in dataset.idx], columns=self._features)

        # create resulted
        if isinstance(dataset.data, cudf.DataFrame):
            dat_t = CudfDataset
        elif isinstance(dataset.data, dask_cudf.DataFrame):
            dat_t = DaskCudfDataset
            data = dask_cudf.from_cudf(data, npartitions=dataset.data.npartitions)
        else:
            raise NotImplementedError
        # create resulted
        # data = data.rename(columns=dict(zip(data.columns,
        #                                    self._features)))
        return dat_t(data, roles=self._roles)


class SeqStatisticsTransformerGPU(SeqStatisticsTransformer):
    """SSF."""

    _fit_checks = ()
    _transform_checks = ()
    _fname_prefix = "stat"

    def __init__(self):
        super().__init__()

    def to_cpu(self):
        """Move the class properties to CPU and change class to CPU counterpart for CPU inference.

        Returns:
            self
        """
        features = deepcopy(self._features)
        roles = self._roles
        self.__class__ = SeqStatisticsTransformer
        self._features = features
        self._roles = roles
        return self

    def fit(self, dataset: GpuSeqDataset):
        """Fit algorithm on seq dataset.

        Args:
            dataset: CupyDataset, CudfDataset or DaskCudfDataset.

        Returns:
            Self.

        """
        super().fit(dataset)
        self._roles = dict(((x, NumericRole(np.float32)) for x in self._features))
        return self

    def transform(self, dataset: GpuSeqDataset) -> GpuDataset:
        """Transform input seq dataset to normal numpy representation.

        Args:
            dataset: seq.

        Returns:
            GPU dataset with last known feature in the sequence.

        """
        # checks here
        super(SeqStatisticsTransformerGPU.__bases__[0], self).transform(dataset)

        # for calculating _get_last this method is long on gpu
        # data = dataset.apply_func((slice(None)), self._get_last)
        # instead you can just rearange the idx and then get the frame
        temp = copy(dataset.idx)
        dataset.idx = np.array([[x[-1]] if len(x) > 0 else [np.nan] for x in dataset.idx])
        data = dataset.get_first_frame().data
        dataset.idx = copy(temp)

        if isinstance(data, cudf.DataFrame):
            dat_t = CudfDataset
        elif isinstance(data, dask_cudf.DataFrame):
            dat_t = DaskCudfDataset
        else:
            raise NotImplementedError
        # create resulted
        data = data.rename(columns=dict(zip(data.columns,
                                            self._features)))

        return dat_t(data, roles=self._roles)

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
    _fname_prefix = "seq"

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def to_cpu(self):
        """Move the class properties to CPU and change class to CPU counterpart for CPU inference.

        Returns:
            self
        """
        features = deepcopy(self._features)
        roles = self.roles
        name = self.name
        self.__class__ = GetSeqTransformer
        self._features = features
        self.roles = roles
        self.name = name
        return self

    def transform(self, dataset: GpuSeqDataset) -> GpuSeqDataset:
        """Transform input seq dataset to normal numpy representation.

        Args:
            dataset: seq.

        Returns:
            GPU seq dataset with lag features.

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
            dat_t = SeqCudfDataset
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
