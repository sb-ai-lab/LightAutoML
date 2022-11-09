"""Hybrid reader."""

import logging
import os
from typing import Any, Union

import numpy as np
import torch
from joblib import Parallel, delayed

from lightautoml.dataset.gpu.gpu_dataset import CudfDataset, DaskCudfDataset
from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import DropRole
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.tasks import Task

from .cudf_reader import CudfReader
from .daskcudf_reader import DaskCudfReader

logger = logging.getLogger(__name__)

LAMLDataset = Union[CudfDataset, PandasDataset, DaskCudfDataset]


class HybridReader(CudfReader):
    """
    Reader to convert :class:`~cudf.core.DataFrame` to
    AutoML's :class:`~lightautoml.dataset.cp_cudf_dataset.CudfDataset`.
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
        num_cpu_readers: int = None,
        num_gpu_readers: int = None,
        gpu_ratio: int = 0.5,
        advanced_roles: bool = True,
        output: str = None,
        index_ok: bool = False,
        npartitions: int = 1,
        compute: bool = False,
        n_jobs: int = 1,
        *args: Any,
        **kwargs: Any
    ):
        """

        Args:
            task: Task object.

        """
        super().__init__(task, *args, **kwargs)
        self.num_cpu_readers = num_cpu_readers
        self.num_gpu_readers = num_gpu_readers
        self.gpu_ratio = gpu_ratio
        self.output = output
        self.advanced_roles = advanced_roles
        self.npartitions = npartitions
        self.index_ok = index_ok
        self.compute = compute
        self.n_jobs = n_jobs

        self.args = args
        self.params = kwargs

        self.final_roles = {}
        self.final_reader = None

    def fit_read(
        self, train_data, features_names: Any = None, roles=None, **kwargs: Any
    ) -> LAMLDataset:
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

        parsed_roles, kwargs = self._prepare_roles_and_kwargs(
            roles, train_data, **kwargs
        )

        if self.num_gpu_readers is None and self.task.device == "mgpu":
            self.num_gpu_readers = torch.cuda.device_count()
        elif self.num_gpu_readers is None and self.task.device == "gpu":
            self.num_gpu_readers = 1

        # self.num_cpu_readers = 0

        if self.num_cpu_readers is None:
            self.num_cpu_readers = min(os.cpu_count() - self.num_gpu_readers, 4)

        if self.num_gpu_readers == 0:
            assert self.num_cpu_readers != 0, "You need at least 1 reader"
            self.gpu_ratio = 0
        elif self.num_cpu_readers == 0:
            self.gpu_ratio = 1

        if self.output is None:
            num_data = train_data.shape[0] * train_data.shape[1]
            if num_data < 1e8 or self.task.device == "gpu":
                self.output = "gpu"
            else:
                self.output = "mgpu"

        train_columns = train_data.columns.difference([self.target])
        num_readers = self.num_gpu_readers + self.num_cpu_readers
        num_features = len(train_columns) - 1
        gpu_num_cols = int(num_features * self.gpu_ratio)
        cpu_num_cols = num_features - gpu_num_cols
        if cpu_num_cols / self.num_cpu_readers < 1:
            self.num_cpu_readers = 0
            gpu_num_cols = num_features
            cpu_num_cols = 0

        single_gpu_num_cols = 0
        single_cpu_num_cols = 0

        if self.num_gpu_readers != 0:
            single_gpu_num_cols = int(gpu_num_cols / self.num_gpu_readers)
        if self.num_cpu_readers != 0:
            single_cpu_num_cols = min(
                int(cpu_num_cols / self.num_cpu_readers), self.num_cpu_readers
            )

        div = []
        for i in range(self.num_gpu_readers):
            div.append((i + 1) * single_gpu_num_cols)
        for i in range(self.num_cpu_readers):
            div.append(gpu_num_cols + (i + 1) * single_cpu_num_cols)

        div = div[:-1]
        idx = np.split(np.arange(num_features), div)
        idx = [x for x in idx if len(x) > 0]
        names = [[train_columns[x] for x in y] for y in idx]
        readers = []
        dev_num = 0

        # assert about max number of gpus here, don't forget
        for i in range(self.num_gpu_readers):
            readers.append(
                CudfReader(
                    self.task,
                    dev_num,
                    *self.args,
                    **self.params,
                    n_jobs=self.n_jobs,
                    advanced_roles=self.advanced_roles
                )
            )
            dev_num += 1

        for i in range(self.num_cpu_readers):
            readers.append(
                PandasToPandasReader(
                    self.task,
                    *self.args,
                    **self.params,
                    n_jobs=self.n_jobs,
                    advanced_roles=self.advanced_roles
                )
            )
        for i, reader in enumerate(readers):
            names[i].append(self.target)

        def call_reader(reader, *args, **kwargs):
            reader.fit_read(*args, **kwargs)
            output_roles = reader.roles
            dropped_features = reader.dropped_features
            used_array_attrs = reader.used_array_attrs
            used_features = reader.used_features
            # for feat in reader.dropped_features:
            #    output_roles[feat] = DropRole()
            return output_roles, dropped_features, used_array_attrs, used_features

        output = []
        if num_readers > 1:
            with Parallel(
                n_jobs=num_readers, prefer="processes", backend="loky", max_nbytes=None
            ) as p:
                output = p(
                    delayed(call_reader)(
                        reader, train_data[name], target=train_data[self.target]
                    )
                    for (reader, name) in zip(readers, names)
                )
                # output = ((call_reader)(reader, train_data[name], target=train_data[self.target]) for (reader, name) in zip(readers, names))
        else:
            output.append(
                call_reader(
                    readers[0], train_data[names[0]], target=train_data[self.target]
                )
            )

        for role, dropped_feat, used_attr, used_feat in output:
            self._roles.update(role)
            for feat in dropped_feat:
                role[feat] = DropRole()
            self.final_roles.update(role)
            self._used_array_attrs.update(used_attr)
            self._dropped_features.extend(dropped_feat)
            self._used_features.extend(used_feat)

        self.final_roles.update({self.target: "target"})

        if self.output == "gpu":
            self.final_reader = CudfReader(
                self.task,
                0,
                *self.args,
                **self.params,
                n_jobs=self.n_jobs,
                advanced_roles=False
            )
        elif self.output == "cpu":
            self.final_reader = PandasToPandasReader(
                self.task, *self.args, **self.params, advanced_roles=False
            )
        elif self.output == "mgpu":
            self.final_reader = DaskCudfReader(
                self.task,
                *self.args,
                **self.params,
                n_jobs=self.n_jobs,
                advanced_roles=False,
                npartitions=self.npartitions,
                index_ok=self.index_ok,
                compute=self.compute
            )

        output = self.final_reader.fit_read(
            train_data, roles=self.final_roles, roles_parsed=True
        )

        return output

    def read(self, data, features_names: Any = None, add_array_attrs: bool = False):

        assert self.final_reader is not None, "reader should be fitted first"

        return self.final_reader.read(data, features_names, add_array_attrs)
