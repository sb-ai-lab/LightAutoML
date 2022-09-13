"""Internal representation of dataset in numpy, pandas and csr formats."""

import warnings

from copy import copy
from copy import deepcopy
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TypeVar
from typing import Union

import numpy as np
import pandas as pd

from pandas import DataFrame
from pandas import Series
from scipy import sparse

from ..tasks.base import Task
from .base import IntIdx
from .base import LAMLColumn
from .base import LAMLDataset
from .base import RolesDict
from .base import valid_array_attributes
from .np_pd_dataset import CSRSparseDataset
from .np_pd_dataset import NumpyDataset
from .np_pd_dataset import PandasDataset
from .roles import ColumnRole


NpFeatures = Union[Sequence[str], str, None]
NpRoles = Union[Sequence[ColumnRole], ColumnRole, RolesDict, None]
DenseSparseArray = Union[np.ndarray, sparse.csr_matrix]
FrameOrSeries = Union[DataFrame, Series]
Dataset = TypeVar("Dataset", bound=LAMLDataset)
RowSlice = Optional[Union[Sequence[int], Sequence[bool]]]
ColSlice = Optional[Union[Sequence[str], str]]


class SeqNumpyPandasDataset(PandasDataset):
    """Sequential Dataset, that contains info in pd.DataFrame format."""

    _dataset_type = "SeqNumpyPandasDataset"

    def _check_dtype(self):
        """Check if dtype in .set_data is ok and cast if not."""
        date_columns = []

        self.dtypes = {}
        for f in self.roles:
            if self.roles[f].name == "Datetime":
                date_columns.append(f)
            else:
                self.dtypes[f] = self.roles[f].dtype

        self.data = self.data.astype(self.dtypes)
        self.data.reset_index(drop=True, inplace=True)
        # do we need to reset_index ?? If yes - drop for Series attrs too
        # case to check - concat pandas dataset and from numpy to pandas dataset
        # TODO: Think about reset_index here
        # self.data.reset_index(inplace=True, drop=True)

        # handle dates types
        for i in date_columns:
            dt_role = self.roles[i]
            if not (self.data.dtypes[i] is np.datetime64):
                self.data[i] = pd.to_datetime(
                    self.data[i], format=dt_role.format, unit=dt_role.unit, origin=dt_role.origin, cache=True
                )

            self.dtypes[i] = np.datetime64

    @property
    def idx(self) -> Any:
        """Get idx attribute.

        Returns:
            Any, array like or ``None``.

        """
        return self._idx

    @idx.setter
    def idx(self, val: Any):
        """Set idx array or ``None``.

        Args:
            val: Some idx or ``None``.

        """
        self._idx = val

    def _initialize(self, task: Optional[Task], **kwargs: Any):
        """Initialize empty dataset with task and array like attributes.

        Args:
            task: Task name for dataset.
            **kwargs: 1d arrays like attrs like target, group etc.

        """
        assert all([x in valid_array_attributes for x in kwargs]), "Unknown array attribute. Valid are {0}".format(
            valid_array_attributes
        )

        self.task = task
        # here we set target and group and so ...
        self._array_like_attrs = []
        for k in kwargs:
            self._array_like_attrs.append(k)
            self.__dict__[k] = kwargs[k]

        # checks for valid values in target, groups ...
        for check in self._init_checks:
            check(self)

        # set empty attributes
        self._idx = None
        self._data = None
        self._features = []
        self._roles = {}

    def __init__(
        self,
        data: Optional[DenseSparseArray],
        features: NpFeatures = (),
        roles: NpRoles = None,
        idx: List = (),
        task: Optional[Task] = None,
        name: Optional[str] = "seq",
        scheme: Optional[dict] = None,
        **kwargs: np.ndarray
    ):
        """Create dataset from `pd.DataFrame` and `pd.Series`.

        Args:
            data: Table with features.
            features: features names.
            roles: Roles specifier.
            idx: sequential indexes. Each element consists of corresponding sequence in data table.
            task: Task specifier.
            name: name of currnet dataset.
            scheme: dict of relations of current dataset with others.
            **kwargs: Series, array like attrs target, group etc...

        """
        self.name = name
        if scheme is not None:
            self.scheme = scheme
        else:
            self.scheme = {}

        self._initialize(task, **kwargs)
        if data is not None:
            self.set_data(data, roles, idx)

    def set_data(self, data: DenseSparseArray, roles: NpRoles = None, idx: Optional[List] = None):
        """Inplace set data, features, roles for empty dataset.

        Args:
            data: 2d array like or ``None``.
            roles: roles dict.
            idx: list.

        """
        # assert data is None or type(data) is np.ndarray, 'Numpy dataset support only np.ndarray features'
        super().set_data(data, None, roles)
        if idx is None:
            idx = np.arange(len(data)).reshape(-1, 1)
        self.idx = idx
        self._check_dtype()

    def __len__(self):
        return len(self.idx)

    def _get_cols_idx(self, columns: Union[Sequence[str], str]) -> Union[Sequence[int], int]:
        """Get numeric index of columns by column names.

        Args:
            columns: sequence of columns of single column.

        Returns:
            sequence of int indexes or single int.

        """
        if type(columns) is str:
            idx = self.data.columns.get_loc(columns)

        else:
            idx = self.data.columns.get_indexer(columns)

        return idx

    def __getitem__(self, k: Tuple[RowSlice, ColSlice]) -> Union["LAMLDataset", LAMLColumn]:
        """Select a subset of dataset.

        Define how to slice a dataset
        in way ``dataset[[1, 2, 3...], ['feat_0', 'feat_1'...]]``.
        Default behavior based on ``._get_cols``, ``._get_rows``, ``._get_2d``.

        Args:
            k: First element optional integer columns indexes,
              second - optional feature name or list of features names.

        """
        # TODO: Maybe refactor this part?
        if type(k) is tuple:
            rows, cols = k
            if isinstance(cols, str):
                cols = [cols]
        else:
            rows = k
            cols = None

        is_slice = False
        if isinstance(rows, slice):
            is_slice = True

        rows = [rows] if isinstance(rows, int) else np.arange(self.__len__()) if isinstance(rows, slice) else rows
        temp_idx = self.idx[rows]
        rows = []
        idx_new = []
        _c = 0
        for i in temp_idx:
            rows.extend(list(i))
            idx_new.append(list(np.arange(len(i)) + _c))
            _c += len(i)
        idx_new = np.array(idx_new, dtype=object)

        rows = np.array(sorted(list(set(rows))))

        if is_slice:
            idx_new = self.idx
            rows = np.arange(len(self.data))
        else:
            warnings.warn(
                "Resulted sequential dataset may have different structure. It's not recommended to slice new dataset"
            )

        # case when columns are defined
        if cols is not None:
            idx = self._get_cols_idx(cols)
            data = self._get_2d(self.data, (rows, idx))

            # case of multiple columns - return LAMLDataset
            roles = dict(((x, self.roles[x]) for x in self.roles if x in cols))
        else:
            roles = self.roles
            data = self._get_rows(self.data, rows)

        # case when rows are defined
        if rows is None:
            dataset = self.empty()
        else:
            dataset = copy(self)
            params = dict(((x, self._get_rows(self.__dict__[x], rows)) for x in self._array_like_attrs))
            dataset._initialize(self.task, **params)

        dataset.set_data(data, roles, idx=idx_new)

        return dataset

    def to_sequence(self, k: Tuple[RowSlice, ColSlice] = None) -> Union["LAMLDataset", LAMLColumn]:
        """Select a subset of dataset and transform it to sequence.

        Define how to slice a dataset
        in way ``dataset[[1, 2, 3...], ['feat_0', 'feat_1'...]]``.
        Default behavior based on ``._get_cols``, ``._get_rows``, ``._get_2d``.

        Args:
            k: First element optional integer columns indexes,
              second - optional feature name or list of features names.

        Returns:
            Numpy Dataset with new sequential dimension

        """
        self._check_dtype()
        if k is None:
            k = slice(None, None, None)

        # TODO: Maybe refactor this part?
        if type(k) is tuple:
            rows, cols = k
            if isinstance(cols, str):
                cols = [cols]
        else:
            rows = k
            cols = None

        rows = [rows] if isinstance(rows, int) else np.arange(self.__len__()) if isinstance(rows, slice) else rows

        # case when columns are defined
        if cols is not None:
            idx = self._get_cols_idx(cols)

            # case when seqs have different shape, return array with arrays
            if len(self.idx.shape) == 1:
                data = []
                _d = self.data.iloc[:, idx].values
                for row in rows:
                    data.append(_d[self.idx[row]])
                data = np.array(data, dtype=object)
            else:
                data = self._get_3d(self.data, (self.idx[rows], idx))

            # case of multiple columns - return LAMLDataset
            roles = dict(((x, self.roles[x]) for x in self.roles if x in cols))
            features = [x for x in cols if x in set(self.features)]
        else:
            roles, features = self.roles, self.features

            if len(self.idx.shape) == 1:
                data = []
                _d = self.data.values
                for row in rows:
                    data.append(_d[self.idx[row]])
                data = np.array(data, dtype=object)
            else:
                data = self._get_3d(self.data, (self.idx[rows], self._get_cols_idx(self.data.columns)))

        # case when rows are defined
        if rows is None:
            dataset = NumpyDataset(None, features, deepcopy(roles), task=self.task)
        else:
            dataset = NumpyDataset(data, features, deepcopy(roles), task=self.task)

        return dataset

    def apply_func(self, k: Tuple[RowSlice, ColSlice] = None, func: Callable = None) -> np.ndarray:
        """Apply function to each sequence.

        Args:
            k: First element optional integer columns indexes,
              second - optional feature name or list of features names.
            func: any callable function

        Returns:
            output np.ndarray

        """
        self._check_dtype()
        if k is None:
            k = slice(None, None, None)

        # TODO: Maybe refactor this part?
        if type(k) is tuple:
            rows, cols = k
            if isinstance(cols, str):
                cols = [cols]
        else:
            rows = k
            cols = None

        rows = [rows] if isinstance(rows, int) else np.arange(self.__len__()) if isinstance(rows, slice) else rows

        # case when columns are defined
        if cols is not None:
            idx = self._get_cols_idx(cols)

            # case when seqs have different shape, return array with arrays
            data = []
            _d = self.data.iloc[:, idx].values
            for row in rows:
                data.append(func(_d[self.idx[row]]))
            data = np.array(data)
        else:
            data = []
            _d = self.data.values
            for row in rows:
                data.append(func(_d[self.idx[row]]))
            data = np.array(data)

        return data

    @classmethod
    def _get_2dT(cls, data: np.ndarray, k: Tuple[IntIdx, IntIdx]) -> np.ndarray:
        """Get 2d slice.

        Args:
            data: Data.
            k: Tuple of integer sequences.

        Returns:
            2d slice.

        """
        rows, cols = k

        return data[rows][:, cols]

    @classmethod
    def _get_3d(cls, data: np.ndarray, k: Tuple[IntIdx, IntIdx]) -> np.ndarray:
        """Get 3d slice.

        Args:
            data: Data.
            k: Tuple of integer sequences.

        Returns:
            3d slice.

        """
        rows, cols = k

        return data.iloc[:, cols].values[rows]

    def to_csr(self) -> "CSRSparseDataset":
        """Convert to csr.

        Returns:
            Same dataset in CSRSparseDatatset format.

        """
        raise NotImplementedError

    def to_numpy(self) -> "NumpyDataset":
        """Convert to class:`NumpyDataset`.

        Returns:
            Same dataset in class:`NumpyDataset` format without sequential features.

        """
        # check for empty
        data = None if self.data is None else self.data.values
        roles = self.roles
        features = self.features
        # target and etc ..
        params = dict(((x, self.__dict__[x].values) for x in self._array_like_attrs))
        task = self.task

        return NumpyDataset(data, features, roles, task, **params)

    def to_pandas(self) -> "PandasDataset":
        """Convert to plain PandasDataset.

        Returns:
            Same dataset in PandasDataset format without sequential features.

        """
        # check for empty case
        data = None if self.data is None else DataFrame(self.data, columns=self.features)
        roles = self.roles
        # target and etc ..
        params = dict(((x, Series(self.__dict__[x])) for x in self._array_like_attrs))
        task = self.task

        return PandasDataset(data, roles, task, **params)

    @classmethod
    def concat(cls, datasets: Sequence["LAMLDataset"]) -> "LAMLDataset":
        """Concat multiple dataset.

        Default behavior - takes empty dataset from datasets[0]
        and concat all features from others.

        Args:
            datasets: Sequence of datasets.

        Returns:
            Concated dataset.

        """
        for check in cls._concat_checks:
            check(datasets)

        idx = datasets[0].idx
        dataset = datasets[0].empty()
        data = []
        features = []
        roles = {}

        atrs = set(dataset._array_like_attrs)
        for ds in datasets:
            data.append(ds.data)
            features.extend(ds.features)
            roles = {**roles, **ds.roles}
            for atr in ds._array_like_attrs:
                if atr not in atrs:
                    dataset._array_like_attrs.append(atr)
                    dataset.__dict__[atr] = ds.__dict__[atr]
                    atrs.update({atr})

        data = cls._hstack(data)
        dataset.set_data(data, roles, idx=idx)

        return dataset
