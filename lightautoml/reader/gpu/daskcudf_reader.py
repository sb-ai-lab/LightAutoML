"""Dask_cudf reader."""

import logging
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import TypeVar
from typing import Union

from torch.cuda import device_count
import cudf
import cupy as cp
import dask.dataframe as dd
import dask_cudf
import numpy as np
import pandas as pd
from dask_cudf.core import DataFrame
from dask_cudf.core import Series

from lightautoml.dataset.gpu.gpu_dataset import CudfDataset
from lightautoml.dataset.gpu.gpu_dataset import DaskCudfDataset
from lightautoml.dataset.roles import ColumnRole
from lightautoml.dataset.roles import DropRole
from lightautoml.reader.utils import set_sklearn_folds
from lightautoml.tasks import Task

from .cudf_reader import CudfReader

logger = logging.getLogger(__name__)

# roles, how it's passed to automl
RoleType = TypeVar("RoleType", bound=ColumnRole)
RolesDict = Dict[str, RoleType]

# how user can define roles
UserDefinedRole = Optional[Union[str, RoleType]]

UserDefinedRolesDict = Dict[UserDefinedRole, Sequence[str]]
UserDefinedRolesSequence = Sequence[UserDefinedRole]
UserRolesDefinition = Optional[
    Union[UserDefinedRole, UserDefinedRolesDict, UserDefinedRolesSequence]
]


class DaskCudfReader(CudfReader):
    """
    Reader to convert :class:`~dask_cudf.core.DataFrame` to
        AutoML's :class:`~lightautoml.dataset.daskcudf_dataset.DaskCudfDataset`.
    Stages:

        - Drop obviously useless features.
        - Convert roles dict from user format to automl format.
        - Simple role guess for features without input role.
        - Create cv folds.
        - Create initial PandasDataset.
        - Optional: advanced guessing of role and handling types.

    """

    def __init__(
        self,
        task: Task,
        compute: bool = True,
        index_ok: bool = False,
        npartitions: int = 1,
        *args: Any,
        **kwargs: Any
    ):
        """

        Args:
            compute: if True reader transfers sample to a signle GPU before guessing roles.
            index_ok: if data is already indexed
            npartitions: number of partitions

        """
        self.npartitions = npartitions
        self.index_ok = index_ok
        super().__init__(task, *args, **kwargs)

    def _prepare_data_and_target(self, train_data, **kwargs):
        if isinstance(train_data, (pd.DataFrame, pd.Series)):
            train_data = cudf.from_pandas(train_data, nan_as_null=False)
            train_data = dask_cudf.from_cudf(train_data, npartitions=self.npartitions)
            kwargs["target"] = train_data[self.target]

        elif isinstance(train_data, (cudf.DataFrame, cudf.Series)):
            train_data = dask_cudf.from_cudf(train_data, npartitions=self.npartitions)
            kwargs["target"] = train_data[self.target]

        elif isinstance(train_data, (dask_cudf.DataFrame, dask_cudf.Series)):
            pass

        elif isinstance(train_data, (dd.DataFrame, dd.Series)):
            train_data = train_data.map_partitions(
                cudf.DataFrame.from_pandas,
                nan_as_null=False,
                meta=cudf.DataFrame(columns=train_data.columns),
            ).persist()
            kwargs["target"] = train_data[self.target]

        else:
            raise NotImplementedError("Input data type is not supported")

        kwargs["target"] = self._create_target(kwargs["target"])

        return train_data.persist(), kwargs

    def fit_read(
        self,
        train_data: DataFrame,
        features_names: Any = None,
        roles: UserDefinedRolesDict = None,
        roles_parsed: bool = False,
        **kwargs: Any
    ) -> DaskCudfDataset:
        """Get dataset with initial feature selection.

        Args:
            train_data: Input data.
            features_names: Ignored. Just to keep signature.
            roles: Dict of features roles in format
              ``{RoleX: ['feat0', 'feat1', ...], RoleY: 'TARGET', ....}``.
            **kwargs: Can be used for target/group/weights.

        Returns:
            Dataset with selected features.

        """

        logger.info("Train data shape: {}".format(train_data.shape))
        parsed_roles, kwargs = self._prepare_roles_and_kwargs(
            roles, train_data, roles_parsed=roles_parsed, **kwargs
        )
        train_data, kwargs = self._prepare_data_and_target(train_data, **kwargs)
        # get subsample if it needed
        subsample = train_data
        train_len = len(subsample)

        if self.samples is not None and self.samples < train_len:
            frac = self.samples / train_len
            subsample = subsample.sample(frac=frac, random_state=42)

        else:
            subsample = subsample.persist()

        # infer roles
        for feat in subsample.columns:
            if not roles_parsed:
                assert isinstance(feat, str), (
                    "Feature names must be string,"
                    " find feature name: {}, with type: {}".format(feat, type(feat))
                )
                if feat in parsed_roles:
                    r = parsed_roles[feat]
                    # handle datetimes
                    if r.name == "Datetime":
                        self._try_datetime(subsample[feat].compute(), r)
                    # replace default category dtype for numeric roles dtype
                    # if cat col dtype is numeric
                    if r.name == "Category":
                        # default category role
                        cat_role = self._get_default_role_from_str("category")
                        # check if role with dtypes was exactly defined
                        try:
                            flg_default_params = feat in roles["category"]
                        except KeyError:
                            flg_default_params = False

                        if (
                            flg_default_params
                            and not np.issubdtype(cat_role.dtype, np.number)
                            and np.issubdtype(subsample.dtypes[feat], np.number)
                        ):
                            r.dtype = self._get_default_role_from_str("numeric").dtype
                else:
                    # if no - infer
                    cur_feat = subsample[feat].compute()

                    if self._is_ok_feature(cur_feat):
                        r = self._guess_role(cur_feat)
                    else:
                        r = DropRole()
            # set back
            else:
                try:
                    r = parsed_roles[feat]
                except KeyError:
                    r = DropRole()
            if r.name != "Drop":
                self._roles[feat] = r
                self._used_features.append(feat)
            else:
                self._dropped_features.append(feat)
        assert len(self.used_features) > 0, "All features are excluded for some reasons"

        if self.cv is not None:
            folds = set_sklearn_folds(
                self.task,
                kwargs["target"],
                cv=self.cv,
                random_state=self.random_state,
                group=None if "group" not in kwargs else kwargs["group"],
            )
            kwargs["folds"] = folds

        ngpus = device_count()
        train_len = len(train_data)
        sub_size = int(1. / ngpus * train_len)
        idx = np.random.RandomState(self.random_state).permutation(train_len)[:sub_size]
        dataset = None
        if self.advanced_roles:
            computed_kwargs = {}
            for item in kwargs:
                computed_kwargs[item] = kwargs[item].loc[idx].compute()
            subsample = train_data[self.used_features].loc[idx].compute()
            dataset = CudfDataset(
                data=subsample, roles=self.roles, task=self.task, **computed_kwargs
            )
            new_roles = self.advanced_roles_guess(dataset, manual_roles=parsed_roles)
            droplist = [
                x
                for x in new_roles
                if new_roles[x].name == "Drop" and not self._roles[x].force_input
            ]
            self.upd_used_features(remove=droplist)
            self._roles = {x: new_roles[x] for x in new_roles if x not in droplist}
            dataset = DaskCudfDataset(
                train_data[self.used_features],
                self.roles,
                index_ok=self.index_ok,
                task=self.task,
                **kwargs
            )
        else:
            dataset = DaskCudfDataset(
                data=train_data[self.used_features],
                roles=self.roles,
                index_ok=self.index_ok,
                task=self.task,
                **kwargs
            )

        return dataset

    def _create_target(self, target: Series):
        """Validate target column and create class mapping is needed

        Args:
            target: Column with target values.

        Returns:
            Transformed target.

        """
        self.class_mapping = None

        if (self.task.name == "binary") or (self.task.name == "multiclass"):
            # expect binary or multiclass here
            target, self.class_mapping = self.check_class_target(target)

        elif self.task.name == "multilabel":
            self.class_mapping = {}

            for col in target.columns:
                target_col, class_mapping = self.check_class_target(target[col])
                self.class_mapping[col] = class_mapping
                target[col] = target_col.values

            self._n_classes = len(target.columns) * 2

        else:
            assert not target.isna().any().compute().any(), "Nan in target detected"
        return target.persist()

    def check_class_target(self, target):

        cnts = target.value_counts(dropna=False).compute()
        assert not cnts.index.isna().any(), "Nan in target detected"
        unqiues = cnts.index.values
        srtd = cp.sort(unqiues)
        self._n_classes = len(unqiues)
        # case - target correctly defined and no mapping
        if (cp.arange(srtd.shape[0]) == srtd).all():

            assert srtd.shape[0] > 1, "Less than 2 unique values in target"
            if self.task.name == "binary":
                assert (
                    srtd.shape[0] == 2
                ), "Binary task and more than 2 values in target"
            return target.persist(), None

            # case - create mapping
            class_mapping = {n: x for (x, n) in enumerate(cp.asnumpy(unqiues))}
            return target.map(class_mapping).astype(np.int32).persist(), class_mapping

    def read(
        self, data: DataFrame, features_names: Any = None, add_array_attrs: bool = False
    ) -> DaskCudfDataset:
        """Read dataset with fitted metadata.

        Args:
            data: Data.
            features_names: Not used.
            add_array_attrs: Additional attributes, like
              target/group/weights/folds.

        Returns:
            Dataset with new columns.

        """

        if isinstance(data, (pd.DataFrame, pd.Series)):
            data = cudf.from_pandas(data, nan_as_null=False)
            data = dask_cudf.from_cudf(data, npartitions=self.npartitions)

        elif isinstance(data, (cudf.DataFrame, cudf.Series)):

            data = dask_cudf.from_cudf(data, npartitions=self.npartitions)

        elif isinstance(data, (dask_cudf.DataFrame, dask_cudf.Series)):
            pass

        elif isinstance(data, (dd.DataFrame, dd.Series)):
            data = data.map_partitions(
                cudf.DataFrame.from_pandas,
                nan_as_null=False,
                meta=cudf.DataFrame(columns=data.columns),
            ).persist()

        else:
            raise NotImplementedError("Input data type is not supported")

        kwargs = {}
        if add_array_attrs:
            for array_attr in self.used_array_attrs:
                col_name = self.used_array_attrs[array_attr]
                try:
                    val = data[col_name]
                except KeyError:
                    continue

                if array_attr == "target" and self.class_mapping is not None:
                    kwargs[array_attr] = val.map_partitions(
                        self._apply_class_mapping, col_name, meta=val
                    ).persist()

        dataset = DaskCudfDataset(
            data[self.used_features], roles=self.roles, task=self.task, **kwargs
        )

        return dataset
