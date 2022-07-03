from copy import copy
from typing import Sequence, Any, Tuple, Union, Optional, List, cast, Dict, Set

import pandas as pd
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F, Column
from pyspark.sql.session import SparkSession

from lightautoml.dataset.base import LAMLDataset, IntIdx, RowSlice, ColSlice, LAMLColumn, RolesDict, \
    valid_array_attributes, array_attr_roles
from lightautoml.dataset.np_pd_dataset import PandasDataset, NumpyDataset, NpRoles
from lightautoml.dataset.roles import ColumnRole, NumericRole, DropRole
from lightautoml.spark import VALIDATION_COLUMN
from lightautoml.spark.dataset.roles import NumericVectorOrArrayRole
from lightautoml.spark.utils import warn_if_not_cached, SparkDataFrame
from lightautoml.tasks import Task


class SparkDataset(LAMLDataset):
    """
    Implements a dataset that uses a ``pyspark.sql.DataFrame`` internally, stores some internal state (features, roles, ...) and provide methods to work with dataset.
    """
    _init_checks = ()
    _data_checks = ()
    _concat_checks = ()
    _dataset_type = "SparkDataset"

    ID_COLUMN = "_id"

    def empty(self) -> "SparkDataset":

        dataset = cast(SparkDataset, super().empty())

        return dataset

    @classmethod
    def concatenate(cls, datasets: Sequence["SparkDataset"]) -> "SparkDataset":
        """
        Concat multiple datasets by joining their internal ``pyspark.sql.DataFrame``
        using inner join on special hidden '_id' column
        Args:
            datasets: spark datasets to be joined

        Returns:
            a joined dataset, containing features (and columns too) from all datasets
            except containing only one _id column
        """
        assert len(datasets) > 0, "Cannot join an empty list of datasets"

        # requires presence of hidden "_id" column in each dataset
        # that should be saved across all transformations
        features = [feat for ds in datasets for feat in ds.features]
        roles = {col: role for ds in datasets for col, role in ds.roles.items()}
        curr_sdf = datasets[0].data

        for ds in datasets[1:]:
            curr_sdf = curr_sdf.join(ds.data, cls.ID_COLUMN)

        curr_sdf = curr_sdf.select(datasets[0].data[cls.ID_COLUMN], *features)

        output = datasets[0].empty()
        output.set_data(curr_sdf, features, roles)

        return output

    def __init__(self,
                 data: SparkDataFrame,
                 roles: Optional[RolesDict],
                 task: Optional[Task] = None,
                 **kwargs: Any):

        if "target" in kwargs:
            assert isinstance(kwargs["target"], str), "Target should be a str representing column name"
            self._target_column: str = kwargs["target"]
        else:
            self._target_column = None

        self._folds_column = None
        if "folds" in kwargs and kwargs["folds"] is not None:
            assert isinstance(kwargs["folds"], str), "Folds should be a str representing column name"
            self._folds_column: str = kwargs["folds"]
        else:
            self._folds_column = None

        self._validate_dataframe(data)

        self._data = None
        self._service_columns: Set[str] = {
            self.ID_COLUMN,
            self.target_column,
            self.folds_column,
            VALIDATION_COLUMN
        }

        roles = roles if roles else dict()

        # currently only target is supported
        # adapted from PandasDataset
        for f in roles:
            for k, r in zip(valid_array_attributes, array_attr_roles):
                if roles[f].name == r:
                    roles[f] = DropRole()

        super().__init__(data, None, roles, task, **kwargs)

    @property
    def spark_session(self):
        return SparkSession.getActiveSession()

    @property
    def data(self) -> SparkDataFrame:
        return self._data

    @data.setter
    def data(self, val: SparkDataFrame) -> None:
        self._data = val

    @property
    def features(self) -> List[str]:
        """Get list of features.

        Returns:
            list of features.

        """
        return [c for c in self.data.columns if c not in self._service_columns] \
            if self.data else []

    @features.setter
    def features(self, val: None):
        """Ignore setting features.

        Args:
            val: ignored.

        """
        pass
        # raise NotImplementedError("The operation is not supported")

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
        elif val:
            role = cast(ColumnRole, val)
            self._roles = dict(((x, role) for x in self.features))
        else:
            raise ValueError()

    @property
    def shape(self) -> Tuple[Optional[int], Optional[int]]:
        warn_if_not_cached(self.data)
        return self.data.count(), len(self.features)

    @property
    def target_column(self) -> Optional[str]:
        return self._target_column

    @property
    def folds_column(self) -> Optional[str]:
        return self._folds_column

    @property
    def service_columns(self) -> List[str]:
        return [sc for sc in self._service_columns if sc in self.data.columns]

    def __repr__(self):
        return f"SparkDataset ({self.data})"

    def __getitem__(self, k: Tuple[RowSlice, ColSlice]) -> Union["LAMLDataset", LAMLColumn]:
        rslice, clice = k

        if isinstance(clice, str):
            clice = [clice]

        assert all(c in self.features for c in clice), \
            f"Not all columns presented in the dataset.\n" \
            f"Presented: {self.features}\n" \
            f"Asked for: {clice}"

        present_svc_cols = [c for c in self.service_columns]
        sdf = cast(SparkDataFrame, self.data.select(*present_svc_cols, *clice))
        roles = {c: self.roles[c] for c in clice}

        output = self.empty()
        output.set_data(sdf, clice, roles)

        return output
        # raise NotImplementedError(f"The method is not supported by {self._dataset_type}")

    def __setitem__(self, k: str, val: Any):
        raise NotImplementedError(f"The method is not supported by {self._dataset_type}")

    def _validate_dataframe(self, sdf: SparkDataFrame) -> None:
        assert self.ID_COLUMN in sdf.columns, \
            f"No special unique row id column (the column name: {self.ID_COLUMN}) in the spark dataframe"
        # assert kwargs["target"] in data.columns, \
        #     f"No target column (the column name: {kwargs['target']}) in the spark dataframe"

    def _materialize_to_pandas(self) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Series], Dict[str, ColumnRole]]:
        sdf = self.data

        def expand_if_vec_or_arr(col, role) -> Tuple[List[Column], ColumnRole]:
            if not isinstance(role, NumericVectorOrArrayRole):
                return [col], role
            vrole = cast(NumericVectorOrArrayRole, role)

            def to_array(column):
                if vrole.is_vector:
                    return vector_to_array(column)
                return column

            arr = [
                to_array(F.col(col))[i].alias(vrole.feature_name_at(i))
                for i in range(vrole.size)
            ]

            return arr, NumericRole(dtype=vrole.dtype)

        arr_cols = (expand_if_vec_or_arr(c, self.roles[c]) for c in self.features)
        all_cols_and_roles = {c: role for c_arr, role in arr_cols for c in c_arr}
        all_cols = [scol for scol, _ in all_cols_and_roles.items()]

        if self.target_column is not None:
            all_cols.append(self.target_column)

        if self.folds_column is not None:
            all_cols.append(self.folds_column)

        sdf = sdf.orderBy(SparkDataset.ID_COLUMN).select(*all_cols)
        all_roles = {c: all_cols_and_roles[c] for c in sdf.columns if c not in self.service_columns}

        data = sdf.toPandas()

        df = pd.DataFrame(data=data.to_dict())

        if self.target_column is not None:
            target_series = df[self.target_column]
            df = df.drop(self.target_column, 1)
        else:
            target_series = None

        if self.folds_column is not None:
            folds_series = df[self.folds_column]
            df = df.drop(self.folds_column, 1)
        else:
            folds_series = None

        return df, target_series, folds_series, all_roles

    def set_data(self,
                 data: SparkDataFrame,
                 features: List[str],
                 roles: NpRoles = None):
        """Inplace set data, features, roles for empty dataset.

        Args:
            data: Table with features.
            features: `ignored, always None. just for same interface.
            roles: Dict with roles.
            dependencies: spark dataframes that should be uncached when this spark dataframe has been materialized
        """
        self._validate_dataframe(data)
        super().set_data(data, None, roles)

    def to_pandas(self) -> PandasDataset:
        data, target_data, folds_data, roles = self._materialize_to_pandas()

        task = Task(self.task.name) if self.task else None
        kwargs = dict()
        if target_data is not None:
            kwargs['target'] = target_data
        if folds_data is not None:
            kwargs['folds'] = folds_data
        pds = PandasDataset(data=data, roles=roles, task=task, **kwargs)

        return pds

    def to_numpy(self) -> NumpyDataset:
        data, target_data, folds_data, roles = self._materialize_to_pandas()

        try:
            target = self.target
            if isinstance(target, pd.Series):
                target = target.to_numpy()
            elif isinstance(target, SparkDataFrame):
                target = target.toPandas().to_numpy()
        except AttributeError:
            target = None

        try:
            folds = self.folds
            if isinstance(folds, pd.Series):
                folds = folds.to_numpy()
            elif isinstance(folds, SparkDataFrame):
                folds = folds.toPandas().to_numpy()
        except AttributeError:
            folds = None

        return NumpyDataset(
            data=data.to_numpy(),
            features=list(data.columns),
            roles=roles,
            task=self.task,
            target=target,
            folds=folds
        )

    @staticmethod
    def _hstack(datasets: Sequence[Any]) -> Any:
        raise NotImplementedError("Unsupported operation for this dataset type")

    @staticmethod
    def _get_cols(data, k: IntIdx) -> Any:
        raise NotImplementedError("Unsupported operation for this dataset type")

    @staticmethod
    def from_dataset(dataset: "LAMLDataset") -> "LAMLDataset":
        assert isinstance(dataset, SparkDataset), "Can only convert from SparkDataset"
        return dataset
