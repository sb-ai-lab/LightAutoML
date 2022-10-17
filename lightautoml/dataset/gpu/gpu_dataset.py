"""Internal representation of dataset in cudf formats."""

from copy import copy  # , deepcopy
from typing import Any
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TypeVar
from typing import Union

import numpy as np
import pandas as pd
import cupy as cp
import dask_cudf
import cudf
from cupyx.scipy import sparse as sparse_cupy

from dask_cudf.core import DataFrame as DataFrame_dask
from dask_cudf.core import Series as Series_dask

from cudf.core.dataframe import DataFrame
from cudf.core.series import Series

from lightautoml.tasks.base import Task
from lightautoml.dataset.base import IntIdx
from lightautoml.dataset.base import LAMLDataset
from lightautoml.dataset.np_pd_dataset import NumpyDataset
from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.np_pd_dataset import CSRSparseDataset
from lightautoml.dataset.base import RolesDict
from lightautoml.dataset.base import array_attr_roles
from lightautoml.dataset.base import valid_array_attributes
from lightautoml.dataset.roles import ColumnRole
from lightautoml.dataset.roles import DropRole
from lightautoml.dataset.roles import NumericRole

NpFeatures = Union[Sequence[str], str, None]
NpRoles = Union[Sequence[ColumnRole], ColumnRole, RolesDict, None]
DenseSparseArray = Union[cp.ndarray, sparse_cupy.csr_matrix]
FrameOrSeries = Union[DataFrame, Series]
FrameOrSeries_dask = Union[DataFrame_dask, Series_dask]
Dataset = TypeVar('Dataset', bound=LAMLDataset)


class CupyDataset(NumpyDataset):
    """Dataset that contains info in cp.ndarray format."""

    _dataset_type = "CupyDataset"

    def __init__(self, data: Optional[DenseSparseArray],
                 features: NpFeatures = (), roles: NpRoles = None,
                 task: Optional[Task] = None, **kwargs: np.ndarray):
        """Create dataset from numpy/cupy arrays.

        Args:
            data: 2d array of features.
            features: Features names.
            roles: Roles specifier.
            task: Task specifier.
            **kwargs: Named attributes like target, group etc ..

        Note:
            For different type of parameter feature there is different behavior:

                - list, should be same len as data.shape[1]
                - None - automatic set names like feat_0, feat_1 ...
                - Prefix - automatic set names like Prefix_0, Prefix_1 ...

            For different type of parameter feature there is different behavior:

                - list, should be same len as data.shape[1].
                - None - automatic set NumericRole(np.float32).
                - ColumnRole - single role.
                - dict.

        """
        self._initialize(task, **kwargs)
        for k in kwargs:
            self.__dict__[k] = cp.asarray(kwargs[k])
        if data is not None:
            self.set_data(data, features, roles)

    def _check_dtype(self):
        """Check if dtype in ``.set_data`` is ok and cast if not.

        Raises:
            AttributeError: If there is non-numeric type in dataset.

        """
        dtypes = list(set([i.dtype for i in self.roles.values()]))
        self.dtype = cp.find_common_type(dtypes, [])

        for f in self.roles:
            self._roles[f].dtype = self.dtype

        assert cp.issubdtype(self.dtype, cp.number), \
            'Support only numeric types in Cupy dataset.'

        if self.data.dtype != self.dtype:
            self.data = self.data.astype(self.dtype)

    def set_data(
            self,
            data: DenseSparseArray,
            features: NpFeatures = (),
            roles: NpRoles = None
    ):
        """Inplace set data, features, roles for empty dataset.

        Args:
            data: 2d cp.array of features.
            features: features names.
            roles: Roles specifier.

        Note:
            For different type of parameter feature there is different behavior:

                - List, should be same len as data.shape[1]
                - None - automatic set names like feat_0, feat_1 ...
                - Prefix - automatic set names like Prefix_0, Prefix_1 ...

            For different type of parameter feature there is different behavior:

                - List, should be same len as data.shape[1].
                - None - automatic set NumericRole(cp.float32).
                - ColumnRole - single role.
                - dict.

        """
        assert data is None or type(data) is cp.ndarray, "Cupy dataset support only cp.ndarray features"
        super(CupyDataset.__bases__[0], self).set_data(data, features, roles)
        self._check_dtype()

    @staticmethod
    def _hstack(datasets: Sequence[cp.ndarray]) -> cp.ndarray:
        """Concatenate function for cupy arrays.

        Args:
            datasets: Sequence of cp.ndarray.

        Returns:
            Stacked features array.

        """
        return cp.hstack([data for data in datasets if len(data) > 0])

    def to_numpy(self) -> NumpyDataset:
        """Convert to numpy.

        Returns:
            Numpy dataset
        """

        assert all([self.roles[x].name == 'Numeric'
                    for x in self.features]), \
            'Only numeric data accepted in numpy dataset'

        data = None if self.data is None else cp.asnumpy(self.data)

        roles = self.roles
        features = self.features
        # target and etc ..
        params = dict(((x, cp.asnumpy(self.__dict__[x])) \
                       for x in self._array_like_attrs))
        task = self.task

        return NumpyDataset(data, features, roles, task, **params)

    def to_cupy(self) -> 'CupyDataset':
        """Empty method to convert to cupy.

        Returns:
            Same CupyDataset.
        """

        return self

    def to_pandas(self) -> PandasDataset:
        """Convert to PandasDataset.

        Returns:
            Same dataset in PandasDataset format.
        """

        return self.to_numpy().to_pandas()

    def to_csr(self) -> CSRSparseDataset:
        """Convert to csr.

        Returns:
            Same dataset in CSRSparseDatatset format.
        """

        return self.to_numpy().to_csr()

    def to_cudf(self) -> 'CudfDataset':
        """Convert to CudfDataset.

        Returns:
            Same dataset in CudfDataset format.
        """
        # check for empty case
        data = None if self.data is None else cudf.DataFrame()
        if data is not None:
            data_gpu = cudf.DataFrame()
            for i, col in enumerate(self.features):
                data_gpu[col] = cudf.Series(self.data[:, i], nan_as_null=False)
            data = data_gpu
        roles = self.roles
        # target and etc ..
        params = dict(((x, cudf.Series(self.__dict__[x])) \
                       for x in self._array_like_attrs))
        task = self.task

        return CudfDataset(data, roles, task, **params)

    def to_daskcudf(self, nparts: int = 1, index_ok: bool = True) -> 'DaskCudfDataset':
        """Convert dataset to daskcudf.

        Returns:
            Same dataset in DaskCudfDataset format

        """
        return self.to_cudf().to_daskcudf(nparts, index_ok)

    def to_sparse_gpu(self) -> 'CupySparseDataset':
        """Convert to cupy-based csr.
        Returns:
            Same dataset in CupySparseDataset format (CSR).
        """
        assert all([self.roles[x].name == 'Numeric' for x in self.features]), \
            'Only numeric data accepted in sparse dataset'
        data = None if self.data is None else sparse_cupy.csr_matrix(self.data)

        roles = self.roles
        features = self.features
        # target and etc ..
        params = dict(((x, self.__dict__[x]) for x in self._array_like_attrs))
        task = self.task

        return CupySparseDataset(data, features, roles, task, **params)

    @staticmethod
    def from_dataset(dataset: Dataset) -> 'CupyDataset':
        """Convert random dataset to cupy.

        Returns:
            Cupy dataset.

        """
        return dataset.to_cupy()


class CupySparseDataset(CupyDataset):
    """Dataset that contains sparse features on GPU and cp.ndarray targets."""

    _dataset_type = 'CupySparseDataset'

    @staticmethod
    def _get_cols(data: Any, k: Any):
        """Not implemented."""
        raise NotImplementedError

    @staticmethod
    def _set_col(data: Any, k: Any, val: Any):
        """Not implemented."""
        raise NotImplementedError

    def to_pandas(self) -> Any:
        """Not implemented."""
        raise NotImplementedError

    def to_cupy(self) -> 'CupyDataset':
        """Convert to CupyDataset.
        Returns:
            CupyDataset.
        """
        # check for empty
        data = None if self.data is None else self.data.toarray()
        assert type(data) is cp.ndarray, "Data conversion failed! Check types of datasets."
        roles = self.roles
        features = self.features
        # target and etc ..
        params = dict(((x, self.__dict__[x]) for x in self._array_like_attrs))
        task = self.task

        return CupyDataset(data, features, roles, task, **params)

    @property
    def shape(self) -> Tuple[Optional[int], Optional[int]]:
        """Get size of 2d feature matrix.
        Returns:
            tuple of 2 elements.
        """
        rows, cols = None, None
        try:
            rows, cols = self.data.shape
        except TypeError:
            if len(self._array_like_attrs) > 0:
                rows = len(self.__dict__[self._array_like_attrs[0]])
        return rows, cols

    @staticmethod
    def _hstack(datasets: Sequence[Union[sparse_cupy.csr_matrix, cp.ndarray]]) -> sparse_cupy.csr_matrix:
        """Concatenate function for sparse and numpy arrays.
        Args:
            datasets: Sequence of csr_matrix or np.ndarray.
        Returns:
            Sparse matrix.
        """
        return sparse_cupy.hstack(datasets, format='csr')

    def __init__(self, data: Optional[DenseSparseArray], features: NpFeatures = (), roles: NpRoles = None,
                 task: Optional[Task] = None, **kwargs: np.ndarray):
        """Create dataset from csr_matrix.
        Args:
            data: csr_matrix of features.
            features: Features names.
            roles: Roles specifier.
            task: Task specifier.
            **kwargs: Named attributes like target, group etc ..
        Note:
            For different type of parameter feature there is different behavior:
                - list, should be same len as data.shape[1]
                - None - automatic set names like feat_0, feat_1 ...
                - Prefix - automatic set names like Prefix_0, Prefix_1 ...
            For different type of parameter feature there is different behavior:
                - list, should be same len as data.shape[1].
                - None - automatic set NumericRole(cp.float32).
                - ColumnRole - single role.
                - dict.
        """
        self._initialize(task, **kwargs)
        if data is not None:
            self.set_data(data, features, roles)

    def set_data(self, data: DenseSparseArray, features: NpFeatures = (), roles: NpRoles = None):
        """Inplace set data, features, roles for empty dataset.
        Args:
            data: csr_matrix of features.
            features: features names.
            roles: Roles specifier.
        Note:
            For different type of parameter feature there is different behavior:
                - list, should be same len as data.shape[1]
                - None - automatic set names like feat_0, feat_1 ...
                - Prefix - automatic set names like Prefix_0, Prefix_1 ...
            For different type of parameter feature there is different behavior:
                - list, should be same len as data.shape[1].
                - None - automatic set NumericRole(cp.float32).
                - ColumnRole - single role.
                - dict.
        """
        assert data is None or type(data) is sparse_cupy.csr_matrix, \
            'CSRSparseDataset support only csr_matrix features'
        LAMLDataset.set_data(self, data, features, roles)
        self._check_dtype()

    @staticmethod
    def from_dataset(dataset: Dataset) -> 'CSRSparseDataset':
        """Convert dataset to sparse dataset.
        Returns:
            Dataset in sparse form.
        """
        assert type(dataset) in DenseSparseArray, 'Only Numpy/Cupy based datasets can be converted to sparse datasets!'
        return dataset.to_sparse_gpu()


class CudfDataset(PandasDataset):
    """Dataset that contains `cudf.core.dataframe.DataFrame` features and
       ` cudf.core.series.Series` targets."""

    _dataset_type = 'CudfDataset'

    def __init__(self, data: Optional[DataFrame] = None,
                 roles: Optional[RolesDict] = None, task: Optional[Task] = None,
                 **kwargs: Series):
        """Create dataset from `cudf.core.dataframe.DataFrame` and
           ` cudf.core.series.Series`

        Args:
            data: Table with features.
            features: features names.
            roles: Roles specifier.
            task: Task specifier.
            **kwargs: Series, array like attrs target, group etc...

        """
        if roles is None:
            roles = {}
        # parse parameters
        # check if target, group etc .. defined in roles
        for f in roles:
            for k, r in zip(valid_array_attributes, array_attr_roles):
                if roles[f].name == r:
                    kwargs[k] = data[f].reset_index(drop=True)
                    roles[f] = DropRole()
        self._initialize(task, **kwargs)
        if data is not None:
            self.set_data(data, None, roles)

    @property
    def roles(self) -> RolesDict:
        """Roles dict."""
        return copy(self._roles)

    @roles.setter
    def roles(self, val: NpRoles):
        """Define how to set roles.

        Args:
            val: Roles.

        Note:
            There is different behavior for different type of val parameter:

                - `List` - should be same len as ``data.shape[1]``.
                - `None` - automatic set ``NumericRole(np.float32)``.
                - ``ColumnRole`` - single role for all.
                - ``dict``.

        """
        if type(val) is dict:
            self._roles = dict(((x, val[x]) for x in self.features))
        elif type(val) is list:
            self._roles = dict(zip(self.features, val))
        else:
            role = NumericRole(np.float32) if val is None else val
            self._roles = dict(((x, role) for x in self.features))

    def set_data(self, data: DataFrame, features: None, roles: RolesDict):
        """Inplace set data, features, roles for empty dataset.

        Args:
            data: Table with features.
            features: `None`, just for same interface.
            roles: Dict with roles.

        """
        super(CudfDataset.__bases__[0], self).set_data(data, features, roles)
        self._check_dtype()

    def _check_dtype(self):
        """Check if dtype in .set_data is ok and cast if not."""
        date_columns = []

        self.dtypes = {}
        for f in self.roles:
            if self.roles[f].name == 'Datetime':
                date_columns.append(f)
            else:
                self.dtypes[f] = self.roles[f].dtype

        self.data = self.data.astype(self.dtypes)

        # handle dates types
        self.data = self._convert_datetime(self.data, date_columns)

        for i in date_columns:
            self.dtypes[i] = np.datetime64

    def _convert_datetime(self, data: DataFrame,
                          date_cols: List[str]) -> DataFrame:
        """Convert the listed columns of the DataFrame to DateTime type
           according to the defined roles.

        Args:
            data: Table with features.
            date_cols: Table column names that need to be converted.

        Returns:
            Data converted to datetime format from roles.

        """
        for i in date_cols:
            dt_role = self.roles[i]
            if not data.dtypes[i] is np.datetime64:
                if dt_role.unit is None:
                    data[i] = cudf.to_datetime(data[i], format=dt_role.format,
                                               origin=dt_role.origin, cache=True)
                else:
                    data[i] = cudf.to_datetime(data[i], format=dt_role.format,
                                               unit=dt_role.unit,
                                               origin=dt_role.origin, cache=True)
        return data

    @staticmethod
    def _hstack(datasets: Sequence[DataFrame]) -> DataFrame:
        """Define how to concat features arrays.

        Args:
            datasets: Sequence of tables.

        Returns:
            concatenated table.

        """
        return cudf.concat(datasets, axis=1)

    @staticmethod
    def _get_rows(data: DataFrame, k: IntIdx) -> FrameOrSeries:
        """Define how to get rows slice.

        Args:
            data: Table with data.
            k: Sequence of `int` indexes or `int`.

        Returns:
            Sliced rows.

        """

        return data.iloc[k]

    @staticmethod
    def _get_cols(data: DataFrame, k: IntIdx) -> FrameOrSeries:
        """Define how to get cols slice.

        Args:
            data: Table with data.
            k: Sequence of `int` indexes or `int`

        Returns:
           Sliced cols.

        """
        return data.iloc[:, k]

    @classmethod
    def _get_2d(cls, data: DataFrame, k: Tuple[IntIdx, IntIdx]) -> FrameOrSeries:
        """Define 2d slice of table.

        Args:
            data: Table with data.
            k: Sequence of `int` indexes or `int`.

        Returns:
            2d sliced table.

        """
        rows, cols = k
        return data.iloc[rows, cols]

    @staticmethod
    def _set_col(data: DataFrame, k: int, val: Union[Series, np.ndarray]):
        """Inplace set column value to `cudf.DataFrame`.

        Args:
            data: Table with data.
            k: Column index.
            val: Values to set.

        """
        data.iloc[:, k] = val

    def to_cupy(self) -> CupyDataset:
        """Convert to class:`NumpyDataset`.

        Returns:
            Same dataset in class:`NumpyDataset` format.

        """
        # check for empty
        data = None if self.data is None else cp.asarray(self.data.fillna(cp.nan).values)
        roles = self.roles
        features = self.features
        # target and etc ..
        params = dict(((x, self.__dict__[x].values) \
                       for x in self._array_like_attrs))
        task = self.task

        return CupyDataset(data, features, roles, task, **params)

    def to_numpy(self) -> NumpyDataset:
        """Convert to class:`NumpyDataset`.

        Returns:
            Same dataset in class:`NumpyDataset` format.

        """

        self.to_cupy().to_numpy()

    def to_pandas(self) -> PandasDataset:
        """Convert dataset to pandas.

        Returns:
            Same dataset in PandasDataset format

        """
        data = self.data.to_pandas()
        roles = self.roles
        task = self.task

        params = dict(((x, pd.Series(cp.asnumpy(self.__dict__[x].values))) \
                       for x in self._array_like_attrs))

        return PandasDataset(data, roles, task, **params)

    def to_cudf(self) -> 'CudfDataset':
        """Empty method to return self

        Returns:
            self
        """

        return self

    def to_sparse_gpu(self) -> 'CupySparseDataset':

        return self.to_cupy().to_sparse_gpu()

    def to_daskcudf(self, nparts: int = 1, index_ok=True) -> 'DaskCudfDataset':
        """Convert dataset to daskcudf.

        Returns:
            Same dataset in DaskCudfDataset format

        """
        data = None
        if self.data is not None:
            data = dask_cudf.from_cudf(self.data, npartitions=nparts)
        roles = self.roles
        task = self.task

        params = dict(((x, dask_cudf.from_cudf(self.__dict__[x],
                                               npartitions=nparts)) \
                       for x in self._array_like_attrs))

        return DaskCudfDataset(data, roles, task, index_ok=index_ok, **params)

    @staticmethod
    def from_dataset(dataset: Dataset) -> 'CudfDataset':
        """Convert random dataset (if it has .to_cudf() member) to cudf dataset.

        Returns:
            Converted to cudf dataset.

        """
        return dataset.to_cudf()


class DaskCudfDataset(CudfDataset):
    """Dataset that contains `dask_cudf.core.DataFrame` features and
       `dask_cudf.Series` targets."""

    _dataset_type = 'DaskCudfDataset'

    def __init__(self, data: Optional[DataFrame_dask] = None,
                 roles: Optional[RolesDict] = None,
                 task: Optional[Task] = None,
                 index_ok: bool = False,
                 **kwargs: Series_dask):
        """Dataset that contains `dask_cudf.core.DataFrame` and
           `dask_cudf.core.Series` target

        Args:
            data: Table with features.
            features: features names.
            roles: Roles specifier.
            task: Task specifier.
            **kwargs: Series, array like attrs target, group etc...

        """
        if roles is None:
            roles = {}
        # parse parameters
        # check if target, group etc .. defined in roles
        for f in roles:
            for k, r in zip(valid_array_attributes, array_attr_roles):
                if roles[f].name == r:
                    kwargs[k] = data[f]
                    roles[f] = DropRole()
        if not index_ok:
            size = len(data.index)
            data['index'] = data.index
            mapping = dict(zip(data.index.compute().values_host, np.arange(size)))
            data['index'] = data['index'].map(mapping).persist()
            data = data.set_index('index', drop=True, sorted=True)
            data = data.persist()
            for val in kwargs:
                col_name = kwargs[val].name
                kwargs[val] = kwargs[val].reset_index(drop=False)
                kwargs[val]['index'] = kwargs[val]['index'].map(mapping).persist()
                kwargs[val] = kwargs[val].set_index('index', drop=True, sorted=True)[col_name]

        self._initialize(task, **kwargs)
        if data is not None:
            self.set_data(data, None, roles)

    def _check_dtype(self):
        """Check if dtype in .set_data is ok and cast if not."""
        date_columns = []
        self.dtypes = {}
        for f in self.roles:
            if self.roles[f].name == 'Datetime':
                date_columns.append(f)
            else:
                self.dtypes[f] = self.roles[f].dtype

        self.data = self.data.astype(self.dtypes).persist()
        # handle dates types

        self.data = self.data.map_partitions(self._convert_datetime,
                                             date_columns, meta=self.data).persist()

        for i in date_columns:
            self.dtypes[i] = np.datetime64

    @staticmethod
    def _get_rows(data: DataFrame_dask, k) -> FrameOrSeries_dask:
        """Define how to get rows slice.

        Args:
            data: Table with data.
            k: Sequence of `int` indexes or `int`.

        Returns:
            Sliced rows.

        """

        return data.loc[k].persist()

    def to_cudf(self) -> CudfDataset:
        """Convert to class:`CudfDataset`.

        Returns:
            Same dataset in class:`CudfDataset` format.
        """
        data = None
        if self.data is not None:
            data = self.data.compute()
        roles = self.roles
        task = self.task

        params = dict(((x, self.__dict__[x].compute()) \
                       for x in self._array_like_attrs))
        return CudfDataset(data, roles, task, **params)

    def to_numpy(self) -> 'NumpyDataset':
        """Convert to class:`NumpyDataset`.

        Returns:
            Same dataset in class:`NumpyDataset` format.

        """

        return self.to_cudf().to_numpy()

    def to_cupy(self) -> 'CupyDataset':
        """Convert dataset to cupy.

        Returns:
            Same dataset in CupyDataset format

        """

        return self.to_cudf().to_cupy()

    def to_sparse_gpu(self) -> 'CupySparseDataset':

        return self.to_cupy().to_sparse_gpu()

    def to_pandas(self) -> 'PandasDataset':
        """Convert dataset to pandas.

        Returns:
            Same dataset in PandasDataset format

        """

        return self.to_cudf().to_pandas()

    def to_daskcudf(self, npartitions: int = 1, index_ok: bool = True) -> 'DaskCudfDataset':
        """Empty method to return self

        Returns:
            self
        """

        return self

    @staticmethod
    def _hstack(datasets: Sequence[DataFrame_dask]) -> DataFrame_dask:
        """Define how to concat features arrays.

        Args:
            datasets: Sequence of tables.

        Returns:
            concatenated table.

        """
        cols = []
        res_datasets = []
        for i, data in enumerate(datasets):
            if data is not None:
                cols.extend(data.columns)
                res_datasets.append(data)

        res = dask_cudf.concat(res_datasets, axis=1)
        mapper = dict(zip(np.arange(len(cols)), cols))
        res = res.rename(columns=mapper)
        return res

    @staticmethod
    def from_dataset(dataset: 'DaskCudfDataset', npartitions: int = 1, index_ok: bool = True) -> 'DaskCudfDataset':
        """Convert DaskCudfDataset to DaskCudfDataset
        (for now, later we add  , npartitionsto_daskcudf() to other classes
        using from_pandas and from_cudf.

        Returns:
            Converted to pandas dataset.

        """
        return dataset.to_daskcudf(npartitions, index_ok=index_ok)

    @property
    def shape(self) -> Tuple[Optional[int], Optional[int]]:
        """Get size of 2d feature matrix.

        Returns:
            Tuple of 2 elements.

        """
        rows, cols = self.data.shape[0].compute(), len(self.features)
        return rows, cols
