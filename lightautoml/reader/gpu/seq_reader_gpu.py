from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TypeVar
from typing import Union
from typing import cast

import numpy as np
import pandas as pd
import cudf
import dask.dataframe as dd
import dask_cudf

from copy import deepcopy

from torch.cuda import device_count

from lightautoml.dataset.base import array_attr_roles
from lightautoml.dataset.base import valid_array_attributes
from lightautoml.dataset.gpu.gpu_dataset import CudfDataset
from lightautoml.dataset.gpu.gpu_dataset import DaskCudfDataset
from lightautoml.dataset.roles import CategoryRole
from lightautoml.dataset.roles import ColumnRole
from lightautoml.dataset.roles import DatetimeRole
from lightautoml.dataset.roles import DropRole
from lightautoml.dataset.roles import NumericRole
from lightautoml.dataset.gpu.gpu_dataset import SeqCudfDataset
from lightautoml.dataset.gpu.gpu_dataset import SeqDaskCudfDataset
from lightautoml.dataset.utils import roles_parser
from lightautoml.tasks import Task
from lightautoml.reader.guess_roles import (
    calc_category_rules,
    calc_encoding_rules,
    rule_based_cat_handler_guess,
)
from .guess_roles_gpu import (
    get_category_roles_stat_gpu,
    get_null_scores_gpu,
    get_numeric_roles_stat_gpu,
    rule_based_roles_guess_gpu,
)
from ..utils import set_sklearn_folds

from .seq_gpu import TopIndGPU, IDSIndGPU
from .cudf_reader import CudfReader
from .daskcudf_reader import DaskCudfReader

from lightautoml.reader.base import DictToPandasSeqReader

from ..base import attrs_dict

class DictToCudfSeqReader(CudfReader):

    def __init__(self, task: Task, seq_params = None, *args: Any, **kwargs: Any):
        """

        Args:
            device_num: ID of GPU

        """
        super().__init__(task, *args, **kwargs)
        self.target = None
        if seq_params is not None:
            self.seq_params = seq_params
        else:
            self.seq_params = {
                "seq": {"case": "next_values", "params": {"n_target": 7, "history": 7, "step": 1, "from_last": True}}
            }
        self.ti = {}
        self.meta = {}

    def create_ids(self, seq_data, plain_data, dataset_name):
        """Calculate ids for different seq tasks."""
        if self.seq_params[dataset_name]["case"] == "next_values":
            self.ti[dataset_name] = TopIndGPU(
                scheme=self.seq_params[dataset_name].get("scheme", None),
                roles=self.meta[dataset_name]["roles"],
                **self.seq_params[dataset_name]["params"]
            )
            self.ti[dataset_name].read(seq_data, plain_data)

        elif self.seq_params[dataset_name]["case"] == "ids":
            self.ti[dataset_name] = IDSIndGPU(
                scheme=self.seq_params[dataset_name].get("scheme", None), **self.seq_params[dataset_name]["params"]
            )
            self.ti[dataset_name].read(seq_data, plain_data)

    def parse_seq(self, seq_dataset, plain_data, dataset_name, parsed_roles, roles):

        subsample = seq_dataset
        if self.samples is not None and self.samples < subsample.shape[0]:
            subsample = subsample.sample(self.samples, axis=0, random_state=42)

        seq_roles = {}
        seq_features = []
        kwargs = {}
        used_array_attrs = {}
        for feat in seq_dataset.columns:
            assert isinstance(
                feat, str
            ), "Feature names must be string," " find feature name: {}, with type: {}".format(feat, type(feat))

            if feat in parsed_roles:
                r = parsed_roles[feat]
                if type(r) == str:
                    # get default role params if defined
                    r = self._get_default_role_from_str(r)
                # handle datetimes

                if r.name == "Datetime" or r.name == "Date":
                    # try if it's ok to infer date with given params
                    self._try_datetime(subsample[feat], r)

                # replace default category dtype for numeric roles dtype if cat col dtype is numeric
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

                if r.name == "Target":
                    r = self._get_default_role_from_str("numeric")

            else:
                # if no - infer
                if self._is_ok_feature(subsample[feat]):
                    r = self._guess_role(subsample[feat])

                else:
                    r = DropRole()

            parsed_roles[feat] = r

            if r.name in attrs_dict:
                if attrs_dict[r.name] in ["target"]:
                    pass
                else:
                    kwargs[attrs_dict[r.name]] = seq_dataset[feat]
                    used_array_attrs[attrs_dict[r.name]] = feat
                    r = DropRole()

            # set back
            if r.name != "Drop":
                self._roles[feat] = r
                self._used_features.append(feat)
                seq_roles[feat] = r
                seq_features.append(feat)
            else:
                self._dropped_features.append(feat)

        assert len(seq_features) > 0, "All features are excluded for some reasons"
        self.meta[dataset_name] = {}
        self.meta[dataset_name]["roles"] = seq_roles
        self.meta[dataset_name]["features"] = seq_features
        self.meta[dataset_name]["attributes"] = used_array_attrs

        self.create_ids(seq_dataset, plain_data, dataset_name)
        self.meta[dataset_name].update(
            **{
                "seq_idx_data": self.ti[dataset_name].create_data(seq_dataset, plain_data=plain_data),
                "seq_idx_target": self.ti[dataset_name].create_target(seq_dataset, plain_data=plain_data),
            }
        )

        if self.meta[dataset_name]["seq_idx_target"] is not None:
            assert len(self.meta[dataset_name]["seq_idx_data"]) == len(
                self.meta[dataset_name]["seq_idx_target"]
            ), "Time series ids don`t match"

        seq_dataset = SeqCudfDataset(
            data=seq_dataset[self.meta[dataset_name]["features"]],
            features=self.meta[dataset_name]["features"],
            roles=self.meta[dataset_name]["roles"],
            idx=self.meta[dataset_name]["seq_idx_target"]
            if self.meta[dataset_name]["seq_idx_target"] is not None
            else self.meta[dataset_name]["seq_idx_data"],
            name=dataset_name,
            scheme=self.seq_params[dataset_name].get("scheme", None),
            **kwargs
        )
        return seq_dataset, parsed_roles

    def _check_data(self, x):
        if isinstance(x, (pd.DataFrame, pd.Series)):
            return cudf.from_pandas(x, nan_as_null=False)
        elif isinstance(x, (cudf.DataFrame, cudf.Series)):
            return x
        elif isinstance(x, (dd.DataFrame, dd.Series)):
            return x.map_partitions(cudf.DataFrame.from_pandas).compute()
        elif isinstance(x, (dask_cudf.DataFrame, dask_cudf.Series)):
            return x.compute()
        else:
            raise NotImplementedError

    def fit_read(
        self, train_data: Dict, features_names: Any = None, roles = None, **kwargs: Any
    ):
        """Get dataset with initial feature selection.

        Args:
            train_data: Input data in dict format.
            features_names: Ignored. Just to keep signature.
            roles: Dict of features roles in format ``{RoleX: ['feat0', 'feat1', ...], RoleY: 'TARGET', ....}``.
            **kwargs: Can be used for target/group/weights.

        Returns:
            Dataset with selected features.

        """
        # logger.info('Train data shape: {}'.format(train_data.shape))

        if roles is None:
            roles = {}
        # transform roles from user format {RoleX: ['feat0', 'feat1', ...], RoleY: 'TARGET', ....}
        # to automl format {'feat0': RoleX, 'feat1': RoleX, 'TARGET': RoleY, ...}

        plain_data, seq_data = train_data.get("plain", None), train_data.get("seq", None)
        plain_features = set(plain_data.columns) if plain_data is not None else {}
        plain_data = self._check_data(plain_data)
        parsed_roles = roles_parser(roles)
        # transform str role definition to automl ColumnRole

        seq_datasets = {}
        if seq_data is not None:
            for dataset_name, dataset in seq_data.items():
                dataset = self._check_data(dataset)
                seq_dataset, parsed_roles = self.parse_seq(dataset, plain_data, dataset_name, parsed_roles, roles)
                seq_datasets[dataset_name] = seq_dataset
        else:
            seq_datasets = None

        for feat in parsed_roles:
            r = parsed_roles[feat]
            if type(r) == str:
                # get default role params if defined
                r = self._get_default_role_from_str(r)
            # check if column is defined like target/group/weight etc ...
            if feat in plain_features:
                if r.name in attrs_dict:
                    # defined in kwargs is rewrited.. TODO: Maybe raise warning if rewrited?

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
                        kwargs[attrs_dict[r.name]] = plain_data.loc[:, feat]
                    r = DropRole()

                # add new role
                parsed_roles[feat] = r
        # add target from seq dataset to plain
        for seq_name, values in self.meta.items():
            if values["seq_idx_target"] is not None:
                dat = seq_datasets[seq_name].get_first_frame((slice(None),
                                                roles["target"])).data
                #DON"T FORGET TO MAKE A COMPARISON HER WITH CPU VER.
                kwargs["target"] = dat[dat.columns[0]].astype(float)

                break

        assert "target" in kwargs, "Target should be defined"
        if isinstance(kwargs["target"], list):
            kwargs["target"] = plain_data.loc[:, kwargs["target"]]

        self.target = kwargs["target"].name if type(kwargs["target"]) == cudf.Series else kwargs["target"].columns
        kwargs["target"] = self._create_target(kwargs["target"])

        # TODO: Check target and task
        # get subsample if it needed
        if plain_data is not None:

            subsample = plain_data
            if self.samples is not None and self.samples < subsample.shape[0]:
                subsample = subsample.sample(self.samples, axis=0, random_state=42)

            # infer roles
            for feat in subsample.columns:
                assert isinstance(
                    feat, str
                ), "Feature names must be string," " find feature name: {}, with type: {}".format(feat, type(feat))
                if feat in parsed_roles:
                    r = parsed_roles[feat]
                    # handle datetimes
                    if r.name == "Datetime":
                        self._try_datetime(subsample[feat], r)

                    # replace default category dtype for numeric roles dtype if cat col dtype is numeric
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

            assert (
                len(set(self.used_features) & set(subsample.columns)) > 0
            ), "All features are excluded for some reasons"

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
        self.plain_used_features = sorted(list(set(self.used_features) & set(plain_features)))
        self.plain_roles = {key: value for key, value in self._roles.items() if key in set(self.plain_used_features)}
        dataset = CudfDataset(
            plain_data[self.plain_used_features] if plain_data is not None else cudf.DataFrame(),
            self.plain_roles,
            task=self.task,
            **kwargs
        )
        if self.advanced_roles:
            new_roles = self.advanced_roles_guess(dataset, manual_roles=parsed_roles)
            droplist = [x for x in new_roles if new_roles[x].name == "Drop" and not self._roles[x].force_input]
            self.upd_used_features(remove=droplist)
            self._roles = {x: new_roles[x] for x in new_roles if x not in droplist}
            self.plain_used_features = sorted(list(set(self.used_features) & set(plain_features)))
            self.plain_roles = {
                key: value for key, value in self._roles.items() if key in set(self.plain_used_features)
            }
            dataset = CudfDataset(
                plain_data[self.plain_used_features] if plain_data is not None else cudf.DataFrame(),
                self.plain_roles,
                task=self.task,
                **kwargs
            )

        for seq_name, values in self.meta.items():
            seq_datasets[seq_name].idx = values["seq_idx_data"]

        dataset.seq_data = seq_datasets

        return dataset

    def read(self, data, features_names: Any = None, add_array_attrs: bool = False):
        """Read dataset with fitted metadata.

        Args:
            data: Data.
            features_names: Not used.
            add_array_attrs: Additional attributes, like target/group/weights/folds.

        Returns:
            Dataset with new columns.

        """
        plain_data, seq_data = data.get("plain", None), data.get("seq", None)
        plain_data = self._check_data(plain_data)
        seq_datasets = {}
        if seq_data is not None:
            for dataset_name, dataset in seq_data.items():
                dataset = self._check_data(dataset)
                test_idx = self.ti[dataset_name].create_test(dataset, plain_data=plain_data)
                kwargs = {}
                columns = set(dataset.columns)
                for role, col in self.meta[dataset_name]["attributes"].items():
                    if col in columns:
                        kwargs[role] = dataset[col]

                seq_dataset = SeqCudfDataset(
                    data=dataset[self.meta[dataset_name]["features"]],
                    features=self.meta[dataset_name]["features"],
                    roles=self.meta[dataset_name]["roles"],
                    idx=test_idx,
                    name=dataset_name,
                    scheme=self.seq_params[dataset_name].get("scheme", None),
                    **kwargs
                )

                seq_datasets[dataset_name] = seq_dataset
        else:
            seq_datasets = None

        kwargs = {}
        if add_array_attrs:
            for array_attr in self.used_array_attrs:
                col_name = self.used_array_attrs[array_attr]
                try:
                    val = plain_data[col_name]
                except KeyError:
                    continue

                if array_attr == "target" \
                    and self.class_mapping is not None:
                    val = self._apply_class_mapping(val, 
                                        plain_data.index, col_name)
                kwargs[array_attr] = val

        dataset = CudfDataset(
            plain_data[self.plain_used_features] if plain_data is not None else cudf.DataFrame(),
            self.plain_roles,
            task=self.task,
            **kwargs
        )

        dataset.seq_data = seq_datasets
        return dataset

    def to_cpu(self, **kwargs):
        task_cpu = deepcopy(self.task)
        task_cpu.device = 'cpu'
        cpu_reader = DictToPandasSeqReader(
            task=task_cpu,
            samples=self.samples,
            max_nan_rate=self.max_nan_rate,
            max_constant_rate=self.max_constant_rate,
            cv=self.cv,
            random_state=self.random_state,
            roles_params=self.roles_params,
            n_jobs=self.n_jobs,
            seq_params = self.seq_params,
            **kwargs)
        cpu_reader.class_mapping = self.class_mapping
        cpu_reader._dropped_features = self.dropped_features
        cpu_reader._used_features = self.used_features
        cpu_reader._used_array_attrs = self.used_array_attrs
        cpu_reader._roles = self.roles

        cpu_reader.plain_used_features = self.plain_used_features
        cpu_reader.plain_roles = self.plain_roles
        cpu_reader.meta = self.meta
        cpu_reader.ti = {}
        for elem in self.ti:
            cpu_reader.ti[elem] = self.ti[elem].to_cpu()
        return cpu_reader

class DictToDaskCudfSeqReader(DaskCudfReader):

    def __init__(self, task: Task, seq_params = None, *args: Any, **kwargs: Any):
        """

        Args:
            device_num: ID of GPU

        """
        super().__init__(task, *args, **kwargs)
        self.target = None
        if seq_params is not None:
            self.seq_params = seq_params
        else:
            self.seq_params = {
                "seq": {"case": "next_values", "params": {"n_target": 7, "history": 7, "step": 1, "from_last": True}}
            }
        self.ti = {}
        self.meta = {}

    def create_ids(self, seq_data, plain_data, dataset_name):
        """Calculate ids for different seq tasks."""
        if self.seq_params[dataset_name]["case"] == "next_values":
            self.ti[dataset_name] = TopIndGPU(
                scheme=self.seq_params[dataset_name].get("scheme", None),
                roles=self.meta[dataset_name]["roles"],
                **self.seq_params[dataset_name]["params"]
            )
            self.ti[dataset_name].read(seq_data, plain_data)

        elif self.seq_params[dataset_name]["case"] == "ids":
            self.ti[dataset_name] = IDSIndGPU(
                scheme=self.seq_params[dataset_name].get("scheme", None), **self.seq_params[dataset_name]["params"]
            )
            self.ti[dataset_name].read(seq_data, plain_data)

    def parse_seq(self, seq_dataset, plain_data, dataset_name, parsed_roles, roles):

        sampl = seq_dataset
        if self.samples is not None and self.samples < len(sampl):
            sampl = sampl.sample(frac=float(self.samples/len(sampl)),
                                 random_state=self.random_state).persist()

        seq_roles = {}
        seq_features = []
        kwargs = {}
        used_array_attrs = {}
        for feat in seq_dataset.columns:
            assert isinstance(
                feat, str
            ), "Feature names must be string," " find feature name: {}, with type: {}".format(feat, type(feat))

            if feat in parsed_roles:
                r = parsed_roles[feat]
                if type(r) == str:
                    # get default role params if defined
                    r = self._get_default_role_from_str(r)
                # handle datetimes

                if r.name == "Datetime" or r.name == "Date":
                    # try if it's ok to infer date with given params
                    self._try_datetime(sampl[feat].compute(), r)

                # replace default category dtype for numeric roles dtype if cat col dtype is numeric
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
                        and np.issubdtype(sampl.dtypes[feat], np.number)
                    ):
                        r.dtype = self._get_default_role_from_str("numeric").dtype

                if r.name == "Target":
                    r = self._get_default_role_from_str("numeric")

            else:
                # if no - infer
                cur_feat = sampl[feat].compute()
                if self._is_ok_feature(cur_feat):
                    r = self._guess_role(cur_feat)
                else:
                    r = DropRole()

            parsed_roles[feat] = r

            if r.name in attrs_dict:
                if attrs_dict[r.name] in ["target"]:
                    pass
                else:
                    kwargs[attrs_dict[r.name]] = seq_dataset[feat]
                    used_array_attrs[attrs_dict[r.name]] = feat
                    r = DropRole()

            # set back
            if r.name != "Drop":
                self._roles[feat] = r
                self._used_features.append(feat)
                seq_roles[feat] = r
                seq_features.append(feat)
            else:
                self._dropped_features.append(feat)

        assert len(seq_features) > 0, "All features are excluded for some reasons"
        self.meta[dataset_name] = {}
        self.meta[dataset_name]["roles"] = seq_roles
        self.meta[dataset_name]["features"] = seq_features
        self.meta[dataset_name]["attributes"] = used_array_attrs

        self.create_ids(seq_dataset, plain_data, dataset_name)
        self.meta[dataset_name].update(
            **{
                "seq_idx_data": self.ti[dataset_name].create_data(seq_dataset, plain_data=plain_data),
                "seq_idx_target": self.ti[dataset_name].create_target(seq_dataset, plain_data=plain_data),
            }
        )

        if self.meta[dataset_name]["seq_idx_target"] is not None:
            assert len(self.meta[dataset_name]["seq_idx_data"]) == len(
                self.meta[dataset_name]["seq_idx_target"]
            ), "Time series ids don`t match"

        seq_dataset = SeqDaskCudfDataset(
            data=seq_dataset[self.meta[dataset_name]["features"]],
            features=self.meta[dataset_name]["features"],
            roles=self.meta[dataset_name]["roles"],
            idx=self.meta[dataset_name]["seq_idx_target"]
            if self.meta[dataset_name]["seq_idx_target"] is not None
            else self.meta[dataset_name]["seq_idx_data"],
            name=dataset_name,
            scheme=self.seq_params[dataset_name].get("scheme", None),
            **kwargs
        )
        return seq_dataset, parsed_roles

    def _check_data(self, x):
        if isinstance(x, (pd.DataFrame, pd.Series)):
            x = cudf.from_pandas(x, nan_as_null=False)
            x = dask_cudf.from_cudf(x, npartitions=self.npartitions)
            return x
        elif isinstance(x, (cudf.DataFrame, cudf.Series)):
            return dask_cudf.from_cudf(x, npartitions=self.npartitions)
        elif isinstance(x, (dd.DataFrame, dd.Series)):
            return x.map_partitions(cudf.DataFrame.from_pandas,
                       nan_as_null=False,
                       meta=cudf.DataFrame(columns=train_data.columns),
                       ).persist()
        elif isinstance(x, (dask_cudf.DataFrame, dask_cudf.Series)):
            return x
        else:
            raise NotImplementedError

    def fit_read(
        self, train_data: Dict, features_names: Any = None, roles = None, **kwargs: Any
    ):
        """Get dataset with initial feature selection.

        Args:
            train_data: Input data in dict format.
            features_names: Ignored. Just to keep signature.
            roles: Dict of features roles in format ``{RoleX: ['feat0', 'feat1', ...], RoleY: 'TARGET', ....}``.
            **kwargs: Can be used for target/group/weights.

        Returns:
            Dataset with selected features.

        """
        # logger.info('Train data shape: {}'.format(train_data.shape))

        if roles is None:
            roles = {}
        # transform roles from user format {RoleX: ['feat0', 'feat1', ...], RoleY: 'TARGET', ....}
        # to automl format {'feat0': RoleX, 'feat1': RoleX, 'TARGET': RoleY, ...}

        plain_data, seq_data = train_data.get("plain", None), train_data.get("seq", None)
        plain_features = set(plain_data.columns) if plain_data is not None else {}
        plain_data = self._check_data(plain_data)
        parsed_roles = roles_parser(roles)
        # transform str role definition to automl ColumnRole

        seq_datasets = {}
        if seq_data is not None:
            for dataset_name, dataset in seq_data.items():
                dataset = self._check_data(dataset)
                seq_dataset, parsed_roles = self.parse_seq(dataset, plain_data, dataset_name, parsed_roles, roles)
                seq_datasets[dataset_name] = seq_dataset
        else:
            seq_datasets = None

        for feat in parsed_roles:
            r = parsed_roles[feat]
            if type(r) == str:
                # get default role params if defined
                r = self._get_default_role_from_str(r)
            # check if column is defined like target/group/weight etc ...
            if feat in plain_features:
                if r.name in attrs_dict:
                    # defined in kwargs is rewrited.. TODO: Maybe raise warning if rewrited?

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
                        kwargs[attrs_dict[r.name]] = plain_data.loc[:, feat]
                    r = DropRole()

                # add new role
                parsed_roles[feat] = r
        # add target from seq dataset to plain
        for seq_name, values in self.meta.items():
            if values["seq_idx_target"] is not None:
                dat = seq_datasets[seq_name].get_first_frame((slice(None),
                                                roles["target"])).data
                #DON"T FORGET TO MAKE A COMPARISON HER WITH CPU VER.
                kwargs["target"] = dat[dat.columns[0]].astype(float)

                break

        assert "target" in kwargs, "Target should be defined"
        if isinstance(kwargs["target"], list):
            kwargs["target"] = plain_data.loc[:, kwargs["target"]]

        self.target = kwargs["target"].name if type(kwargs["target"]) == dask_cudf.Series else kwargs["target"].columns
        kwargs["target"] = self._create_target(kwargs["target"])

        # TODO: Check target and task
        # get subsample if it needed
        if plain_data is not None:

            sampl = plain_data
            if self.samples is not None and self.samples < len(sampl):
                sampl = sampl.sample(frac=float(self.samples/len(sampl)),
                                 random_state=42).persist()
            # infer roles
            for feat in sampl.columns:
                assert isinstance(
                    feat, str
                ), "Feature names must be string," " find feature name: {}, with type: {}".format(feat, type(feat))
                if feat in parsed_roles:
                    r = parsed_roles[feat]
                    # handle datetimes
                    if r.name == "Datetime":
                        self._try_datetime(sampl[feat].compute(), r)

                    # replace default category dtype for numeric roles dtype if cat col dtype is numeric
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
                            and np.issubdtype(sampl.dtypes[feat], np.number)
                        ):
                            r.dtype = self._get_default_role_from_str("numeric").dtype

                else:
                    # if no - infer
                    cur_feat = sampl[feat].compute()
                    
                    if self._is_ok_feature(cur_feat):
                        r = self._guess_role(cur_feat)
                    else:
                        r = DropRole()

                # set back
                if r.name != "Drop":
                    self._roles[feat] = r
                    self._used_features.append(feat)
                else:
                    self._dropped_features.append(feat)

            assert (
                len(set(self.used_features) & set(sampl.columns)) > 0
            ), "All features are excluded for some reasons"

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
        self.plain_used_features = sorted(list(set(self.used_features) & set(plain_features)))
        self.plain_roles = {key: value for key, value in self._roles.items() if key in set(self.plain_used_features)}

        ngpus = device_count()
        train_len = len(plain_data)
        sub_size = int(1./ngpus*train_len)
        idx = np.random.RandomState(self.random_state).permutation(train_len)[:sub_size]
        computed_kwargs = {}
        for item in kwargs:
            computed_kwargs[item] = kwargs[item].loc[idx].compute()
        dataset = CudfDataset(
            plain_data[self.plain_used_features].loc[idx].compute() if plain_data is not None else cudf.DataFrame(),
            roles = self.plain_roles,
            task=self.task,
            **computed_kwargs
        )
        if self.advanced_roles:
            new_roles = self.advanced_roles_guess(dataset, manual_roles=parsed_roles)
            droplist = [x for x in new_roles if new_roles[x].name == "Drop" and not self._roles[x].force_input]
            self.upd_used_features(remove=droplist)
            self._roles = {x: new_roles[x] for x in new_roles if x not in droplist}
            self.plain_used_features = sorted(list(set(self.used_features) & set(plain_features)))
            self.plain_roles = {
                key: value for key, value in self._roles.items() if key in set(self.plain_used_features)
            }
            dataset = DaskCudfDataset(
                plain_data[self.plain_used_features] if plain_data is not None else dask_cudf.DataFrame(),
                self.plain_roles,
                index_ok = self.index_ok,
                task = self.task,
                **kwargs
            )

        for seq_name, values in self.meta.items():
            seq_datasets[seq_name].idx = values["seq_idx_data"]

        dataset.seq_data = seq_datasets

        return dataset

    def read(self, data, features_names: Any = None, add_array_attrs: bool = False):
        """Read dataset with fitted metadata.

        Args:
            data: Data.
            features_names: Not used.
            add_array_attrs: Additional attributes, like target/group/weights/folds.

        Returns:
            Dataset with new columns.

        """
        plain_data, seq_data = data.get("plain", None), data.get("seq", None)
        plain_data = self._check_data(plain_data)
        seq_datasets = {}
        if seq_data is not None:
            for dataset_name, dataset in seq_data.items():
                dataset = self._check_data(dataset)
                test_idx = self.ti[dataset_name].create_test(dataset, plain_data=plain_data)
                kwargs = {}
                columns = set(dataset.columns)
                for role, col in self.meta[dataset_name]["attributes"].items():
                    if col in columns:
                        kwargs[role] = dataset[col]

                seq_dataset = SeqDaskCudfDataset(
                    data=dataset[self.meta[dataset_name]["features"]],
                    features=self.meta[dataset_name]["features"],
                    roles=self.meta[dataset_name]["roles"],
                    idx=test_idx,
                    name=dataset_name,
                    scheme=self.seq_params[dataset_name].get("scheme", None),
                    **kwargs
                )

                seq_datasets[dataset_name] = seq_dataset
        else:
            seq_datasets = None

        kwargs = {}
        if add_array_attrs:
            for array_attr in self.used_array_attrs:
                col_name = self.used_array_attrs[array_attr]
                try:
                    val = plain_data[col_name]
                except KeyError:
                    continue

                if array_attr == "target" \
                    and self.class_mapping is not None:

                    kwargs[array_attr] = val.map_partitions(
                        self._apply_class_mapping, col_name, meta=val
                    ).persist()
                else:
                    kwargs[array_attr] = val

        dataset = DaskCudfDataset(
            plain_data[self.plain_used_features] if plain_data is not None else cudf.DataFrame(),
            self.plain_roles,
            task=self.task,
            **kwargs
        )

        dataset.seq_data = seq_datasets
        return dataset
