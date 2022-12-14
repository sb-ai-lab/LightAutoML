"""Cudf reader."""

import logging
from copy import deepcopy
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import TypeVar
from typing import Union
from typing import Tuple
from typing import Mapping
from typing import cast

import cudf
import cupy as cp
import dask.dataframe as dd
import dask_cudf
import numpy as np
import pandas as pd
from cudf.core.dataframe import DataFrame
from cudf.core.series import Series

from lightautoml_gpu.dataset.base import array_attr_roles
from lightautoml_gpu.dataset.base import valid_array_attributes
from lightautoml_gpu.dataset.gpu.gpu_dataset import CudfDataset
from lightautoml_gpu.dataset.roles import CategoryRole
from lightautoml_gpu.dataset.roles import ColumnRole
from lightautoml_gpu.dataset.roles import DatetimeRole
from lightautoml_gpu.dataset.roles import DropRole
from lightautoml_gpu.dataset.roles import NumericRole

from lightautoml_gpu.dataset.utils import roles_parser
from lightautoml_gpu.reader.base import PandasToPandasReader
from lightautoml_gpu.reader.guess_roles import calc_category_rules
from lightautoml_gpu.reader.guess_roles import calc_encoding_rules
from lightautoml_gpu.reader.guess_roles import rule_based_cat_handler_guess

from lightautoml_gpu.tasks import Task

from ..utils import set_sklearn_folds

from .guess_roles_gpu import get_category_roles_stat_gpu
from .guess_roles_gpu import get_null_scores_gpu
from .guess_roles_gpu import get_numeric_roles_stat_gpu
from .guess_roles_gpu import rule_based_roles_guess_gpu

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


class CudfReader(PandasToPandasReader):
    """
    Reader to convert :class:`~cudf.core.DataFrame` to
    AutoML's :class:`~lightautoml_gpu.dataset.cp_cudf_dataset.CudfDataset`.
    Stages:

        - Drop obviously useless features.
        - Convert roles dict from user format to automl format.
        - Simple role guess for features without input role.
        - Create cv folds.
        - Create initial PandasDataset.
        - Optional: advanced guessing of role and handling types.

    """

    def __init__(self, task: Task, device_num: int = 0, n_targets=100, *args: Any, **kwargs: Any):
        """

        Args:
            device_num: ID of GPU

        """
        super().__init__(task, *args, **kwargs)
        self.device_num = device_num
        self.n_targets = n_targets

    def _prepare_roles_and_kwargs(
        self, roles, train_data, roles_parsed: bool = False, **kwargs
    ):

        if roles is None:
            roles = {}

        parsed_roles = roles if roles_parsed else roles_parser(roles)
        # transform str role definition to automl ColumnRole
        attrs_dict = dict(zip(array_attr_roles, valid_array_attributes))

        for feat in parsed_roles:
            r = parsed_roles[feat]
            if isinstance(r, str):
                # get default role params if defined
                r = self._get_default_role_from_str(r)

            # check if column is defined like target/group/weight etc ...
            if r.name in attrs_dict:
                if ((self.task.name == "multi:reg") or (self.task.name == "multilabel")) and (
                    attrs_dict[r.name] == "target"
                ):
                    if attrs_dict[r.name] in kwargs:
                        kwargs[attrs_dict[r.name]].append(feat)
                        self._used_array_attrs[attrs_dict[r.name]].append(feat)
                    else:
                        kwargs[attrs_dict[r.name]] = [feat]
                        self._used_array_attrs[attrs_dict[r.name]] = [feat]
                else:
                    self._used_array_attrs[attrs_dict[r.name]] = feat
                    kwargs[attrs_dict[r.name]] = train_data[feat]
                r = DropRole()

            # add new role
            parsed_roles[feat] = r

        assert "target" in kwargs, "Target should be defined"
        if self.task.name in ["multi:reg", "multilabel"]:
            kwargs["target"] = train_data[kwargs["target"]]
        self.target = kwargs["target"].name if isinstance(kwargs["target"], (pd.Series, cudf.Series, dask_cudf.Series, dd.Series)) else kwargs["target"].columns

        return parsed_roles, kwargs

    def _prepare_data_and_target(self, train_data, **kwargs):

        if isinstance(train_data, (pd.DataFrame, pd.Series)):
            train_data = cudf.from_pandas(train_data, nan_as_null=False)
            for col in train_data.columns:
                if pd.api.types.is_bool_dtype(train_data[col]):
                    train_data[col] = train_data[col].astype(cp.float32).fillna(cp.nan)
            kwargs["target"] = train_data[self.target]

        elif isinstance(train_data, (cudf.DataFrame, cudf.Series)):
            pass

        elif isinstance(train_data, (dask_cudf.DataFrame, dask_cudf.Series)):
            train_data = train_data.compute()
            kwargs["target"] = train_data[self.target]

        elif isinstance(train_data, (dd.DataFrame, dd.Series)):
            train_data = train_data.map_partitions(cudf.DataFrame.from_pandas).compute()
            kwargs["target"] = train_data[self.target]

        else:
            raise NotImplementedError("Input data type is not supported")

        kwargs["target"] = self._create_target(kwargs["target"])

        return train_data, kwargs

    def fit_read(
        self,
        train_data: DataFrame,
        features_names: Any = None,
        roles: UserDefinedRolesDict = None,
        roles_parsed: bool = False,
        **kwargs: Any
    ) -> CudfDataset:
        """Get dataset with initial feature selection.

        Args:
            train_data: Input data.
            features_names: Ignored. Just to keep signature.
            roles: Dict of features roles in format
              ``{RoleX: ['feat0', 'feat1', ...], RoleY: 'TARGET', ....}``.
            roles_parsed: True if roles are already parsed into reader format
              ``{RoleX: feat0, RoleY: feat1, RoleX: feat2, ...}
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
        if self.samples is not None and self.samples < subsample.shape[0]:
            subsample = subsample.sample(self.samples, axis=0, random_state=42)
        # infer roles
        for feat in subsample.columns:
            assert isinstance(feat, str), (
                "Feature names must be string,"
                " find feature name: {}, with type: {}".format(feat, type(feat))
            )
            if feat in parsed_roles:
                r = parsed_roles[feat]
                # handle datetimes

                if r.name == "Datetime":
                    # try if it's ok to infer date with given params
                    self._try_datetime(subsample[feat], r)

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
                if self._is_ok_feature(subsample[feat]):
                    r = self._guess_role(subsample[feat])

                else:
                    r = DropRole()

            # set back
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

        # get dataset
        dataset = CudfDataset(
            train_data[self.used_features], self.roles, task=self.task, **kwargs
        )
        if self.advanced_roles:
            new_roles = self.advanced_roles_guess(dataset, manual_roles=parsed_roles)

            droplist = [
                x
                for x in new_roles
                if new_roles[x].name == "Drop" and not self._roles[x].force_input
            ]

            self.upd_used_features(remove=droplist)
            self._roles = {x: new_roles[x] for x in new_roles if x not in droplist}
            dataset = CudfDataset(
                train_data[self.used_features], self.roles, task=self.task, **kwargs
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
            assert not target.isna().values.any(), "Nan in target detected"
        return target

    def check_class_target(self, target) -> Tuple[cudf.Series, Optional[Union[Mapping, Dict[str, Mapping]]]]:
        """Validate target values."""
        target = cudf.Series(target)
        cnts = target.value_counts(dropna=False)
        assert not cnts.index.isna().any(), "Nan in target detected"
        unqiues = cnts.index.values
        srtd = cp.sort(unqiues)
        self._n_classes = len(unqiues)
        # case - target correctly defined and no mapping
        if (cp.arange(srtd.shape[0]) == srtd).all():

            assert srtd.shape[0] > 1, "Less than 2 unique values in target"
        if (self.task.name == "binary") or (self.task.name == "multilabel"):
            assert srtd.shape[0] == 2, "Binary task and more than 2 values in target"
            return target, None

        # case - create mapping
        class_mapping = {n: x for (x, n) in enumerate(cp.asnumpy(unqiues))}
        return target.map(class_mapping).astype(cp.int32), class_mapping

    def _guess_role(self, feature: Series) -> RoleType:
        """Try to infer role, simple way.

        If convertable to float -> number.
        Else if convertable to datetime -> datetime.
        Else category.

        Args:
            feature: Column from dataset.

        Returns:
            Feature role.

        """
        num_dtype = self._get_default_role_from_str("numeric").dtype
        date_format = self._get_default_role_from_str("datetime").format
        # check if feature is number
        try:
            _ = feature.astype(num_dtype)
            try:
                _ = feature.dt
                return DatetimeRole(np.datetime64, date_format=date_format)
            except AttributeError:
                pass
            return NumericRole(num_dtype)
        except ValueError:
            pass
        except TypeError:
            pass

        try:
            _ = cast(
                cudf.Series,
                cudf.to_datetime(
                    feature, infer_datetime_format=False, format=date_format
                ),
            ).dt
            return DatetimeRole(np.datetime64, date_format=date_format)
        except (AttributeError, ValueError):
            pass

        return CategoryRole(object)

    def read(
        self, data: DataFrame, features_names: Any = None, add_array_attrs: bool = False
    ) -> CudfDataset:
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

        elif isinstance(data, (cudf.DataFrame, cudf.Series)):
            pass

        elif isinstance(data, (dask_cudf.DataFrame, dask_cudf.Series)):
            data = data.compute()

        elif isinstance(data, (dd.DataFrame, dd.Series)):
            data = data.map_partitions(cudf.DataFrame.from_pandas).compute()

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
                    val = self._apply_class_mapping(val, data.index, col_name)
                kwargs[array_attr] = val

        dataset = CudfDataset(
            data[self.used_features], roles=self.roles, task=self.task, **kwargs
        )

        return dataset

    def advanced_roles_guess(
        self, dataset: CudfDataset, manual_roles: Optional[RolesDict] = None
    ) -> RolesDict:
        """Advanced roles guess over user's definition and reader's simple guessing.

        Strategy - compute feature's NormalizedGini
        for different encoding ways and calc stats over results.
        Role is inferred by comparing performance stats with manual rules.
        Rule params are params of roles guess in init.
        Defaults are ok in general case.

        Args:
            dataset: Input CudfDataset.
            manual_roles: Dict of user defined roles.

        Returns:
            Dict.

        """
        if (self.task.name == "multi:reg") or (self.task.name == "multilabel"):
            if dataset.target.shape[1] > self.n_targets:
                dataset.target = dataset.target[dataset.target.std().sort_values(ascending=False).iloc[:self.n_targets].index.values_host]
        if manual_roles is None:
            manual_roles = {}
        top_scores = []
        new_roles_dict = dataset.roles

        advanced_roles_params = deepcopy(self.advanced_roles_params)
        drop_co = advanced_roles_params.pop("drop_score_co")
        # guess roles nor numerics
        stat = get_numeric_roles_stat_gpu(
            dataset,
            manual_roles=manual_roles,
            random_state=self.random_state,
            subsample=self.samples,
            n_jobs=self.n_jobs,
        )
        if len(stat) > 0:
            # upd stat with rules

            stat = calc_encoding_rules(stat, **advanced_roles_params)
            new_roles_dict = {**new_roles_dict, **rule_based_roles_guess_gpu(stat)}
            top_scores.append(stat["max_score"])
        # # # guess categories handling type
        stat, dtypes = get_category_roles_stat_gpu(
            dataset,
            random_state=self.random_state,
            subsample=self.samples,
            n_jobs=self.n_jobs,
        )

        if len(stat) > 0:
            # upd stat with rules

            stat = calc_category_rules(stat)
            stat['dtype'] = dtypes
            new_roles_dict = {
                **new_roles_dict,
                **rule_based_cat_handler_guess(stat)
            }
            top_scores.append(stat["max_score"])
        # # get top scores of feature
        if len(top_scores) > 0:
            top_scores = pd.concat(top_scores, axis=0)

            null_scores = get_null_scores_gpu(
                dataset,
                top_scores.index.values,
                random_state=self.random_state,
                subsample=self.samples,
            )
            top_scores = pd.concat([null_scores, top_scores], axis=1).max(axis=1)

            rejected = list(top_scores[top_scores < drop_co].index.values)
            logger.info("Feats was rejected during automatic roles guess: {0}".format(rejected))
            new_roles_dict = {**new_roles_dict, **{x: DropRole() for x in rejected}}
        return new_roles_dict

    def _is_ok_feature(self, feature: Series) -> bool:
        """Check if column is filled well to be a feature.

        Args:
            feature: Column from dataset.

        Returns:
            ``True`` if nan ratio and freqency are not high.

        """
        if feature.isnull().mean() >= self.max_nan_rate:
            return False
        if (
            feature.value_counts().values[0] / feature.shape[0]
        ) >= self.max_constant_rate:
            return False
        return True

    def _try_datetime(self, feature: Series, r: RoleType):
        """See if the feature can be formatted to datetime according to the role.

        Args:
            feature: Column from dataset.
            r: Role that holds datetime format.

        Returns:
           ``True`` if datetime format is ok for feature.

        """
        try:
            if r.unit is None:
                _ = cudf.to_datetime(feature, format=r.format, origin=r.origin)
            else:
                _ = cudf.to_datetime(
                    feature, format=r.format, origin=r.origin, unit=r.unit
                )
        except ValueError:
            raise ValueError(
                "Looks like given datetime parsing params are not correctly defined"
            )

    def _apply_class_mapping(self, feature: Series, col_name: str) -> Series:
        """Create new columns with remaped values
           according to self.class_mapping property.

        Args:
            feature: Column from dataset.
            data_index: Indices for rows.
            col_name: name of the created feature.

        Returns:
            New remapped feature.

        """
        val = cudf.Series(
            feature.map(self.class_mapping).values, index=feature.index, name=col_name
        )
        return val

    def to_cpu(self, **kwargs):
        """Move the class properties to CPU and change class to CPU counterpart for CPU inference.

        Returns:
            self
        """
        task_cpu = deepcopy(self.task)
        task_cpu.device = 'cpu'
        cpu_reader = PandasToPandasReader(
            task=task_cpu,
            samples=self.samples,
            max_nan_rate=self.max_nan_rate,
            max_constant_rate=self.max_constant_rate,
            cv=self.cv,
            random_state=self.random_state,
            roles_params=self.roles_params,
            n_jobs=self.n_jobs,
            **kwargs)
        cpu_reader.class_mapping = self.class_mapping
        cpu_reader._dropped_features = self.dropped_features
        cpu_reader._used_features = self.used_features
        cpu_reader._used_array_attrs = self.used_array_attrs
        cpu_reader._roles = self.roles
        return cpu_reader
