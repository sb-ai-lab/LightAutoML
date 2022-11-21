"""Datetime features transformers."""

from collections import OrderedDict
from typing import List, Optional, Sequence, Union

import cudf
import cupy as cp
import holidays
import numpy as np
import pandas as pd

from lightautoml.dataset.base import LAMLDataset
from lightautoml.dataset.gpu.gpu_dataset import CudfDataset, CupyDataset, DaskCudfDataset
from lightautoml.dataset.roles import CategoryRole, ColumnRole, NumericRole
from lightautoml.transformers.base import LAMLTransformer
from lightautoml.transformers.datetime import date_attrs, datetime_check

DatetimeCompatibleGPU = Union[CudfDataset]
GpuDataset = Union[CupyDataset, CudfDataset, DaskCudfDataset]


class TimeToNumGPU(LAMLTransformer):
    """
    Basic conversion strategy, used in selection one-to-one transformers (GPU version).
    Datetime converted to difference
    with basic_date (``basic_date == '2020-01-01'``).
    """

    basic_time = "2020-01-01"
    basic_interval = "D"

    _fname_prefix = "dtdiff_gpu"
    _fit_checks = (datetime_check,)
    _transform_checks = ()

    def _standardize_date(
        self, data: cudf.DataFrame, mean: np.datetime64, std: np.timedelta64
    ) -> cudf.DataFrame:
        output = (data.astype(int) - mean) / std
        return output

    def _transform_cupy(self, dataset: DatetimeCompatibleGPU) -> CupyDataset:
        """Transform dates to numeric differences with base date (GPU version).

        Args:
            dataset:  Cudf dataset with datetime columns.

        Returns:
            Cupy dataset of numeric features.

        """

        data = dataset.to_cudf().data

        time_diff = cudf.DatetimeIndex(
            pd.date_range(self.basic_time, periods=1, freq="d")
        ).astype(int)[0]

        # bad hardcode, but unit from dataset.roles is None
        timedelta = np.timedelta64(1, self.basic_interval) / np.timedelta64(1, "ns")

        data = self._standardize_date(data, time_diff, timedelta).values

        output = dataset.empty().to_cupy()
        output.set_data(data, self.features, NumericRole(cp.float32))

        return output

    def _transform_daskcudf(self, dataset: DaskCudfDataset) -> DaskCudfDataset:

        data = dataset.data

        time_diff = cudf.DatetimeIndex(
            pd.date_range(self.basic_time, periods=1, freq="d")
        ).astype(int)[0]

        # bad hardcode, but unit from dataset.roles is None
        timedelta = np.timedelta64(1, self.basic_interval) / np.timedelta64(1, "ns")

        data = data.map_partitions(
            self._standardize_date,
            time_diff,
            timedelta,
            meta=cudf.DataFrame(columns=data.columns),
        )

        output = dataset.empty()
        output.set_data(data, self.features, NumericRole(cp.float32))
        return output

    def transform(self, dataset: GpuDataset) -> GpuDataset:
        """Transform dates to numeric differences with base date.

        Args:
            dataset:  Cupy or Cudf or DaskCudf dataset with datetime columns.

        Returns:
            Respective dataset of numeric features.

        """

        super().transform(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self._transform_daskcudf(dataset)
        else:
            return self._transform_cupy(dataset)


class BaseDiffGPU(LAMLTransformer):
    """
    Basic conversion strategy, used in selection one-to-one transformers (GPU version).
    Datetime converted to difference with basic_date.

    """

    basic_interval = "D"

    _fname_prefix = "basediff_gpu"
    _fit_checks = (datetime_check,)
    _transform_checks = ()

    @property
    def features(self) -> List[str]:
        """List of features."""
        return self._features

    def __init__(
        self,
        base_names: Sequence[str],
        diff_names: Sequence[str],
        basic_interval: Optional[str] = "D",
    ):
        """

        Args:
            base_names: Base date names.
            diff_names: Difference date names.
            basic_interval: Time unit.

        """
        self.base_names = base_names
        self.diff_names = diff_names
        self.basic_interval = basic_interval

    def fit(self, dataset: LAMLDataset) -> "LAMLTransformer":
        """Fit transformer and return it's instance (GPU version).

        Args:
            dataset: Dataset to fit on.

        Returns:
            self.

        """

        self._features = []
        for col in self.base_names:
            self._features.extend(
                ["basediff_{0}__{1}".format(col, x) for x in self.diff_names]
            )

        for check_func in self._fit_checks:
            check_func(dataset)
        return self

    def _standardize_date_concat(self, data, std):
        feats_block = []
        for col in self.base_names:

            output = (
                data[self.diff_names].astype(int).values.T - data[col].astype(int).values
            ) / std
            feats_block.append(output.T)

        return cudf.DataFrame(cp.concatenate(feats_block, axis=1), columns=self.features)

    def _transform_cupy(self, dataset: DatetimeCompatibleGPU) -> CupyDataset:

        # convert to accepted format and get attributes
        dataset = dataset.to_cudf()
        data = dataset.data

        # shouldn't hardcode this,
        # should take units from dataset.roles
        # (but its unit is none currently)
        timedelta = np.timedelta64(1, self.basic_interval) / np.timedelta64(1, "ns")

        feats_block = []

        for col in self.base_names:
            output = (
                data[self.diff_names].astype(int).values.T - data[col].astype(int).values
            ) / timedelta
            feats_block.append(output.T)

        # create resulted
        output = dataset.empty().to_cupy()
        output.set_data(
            cp.concatenate(feats_block, axis=1),
            self.features,
            NumericRole(dtype=cp.float32),
        )

        return output

    def _transform_daskcudf(self, dataset: DaskCudfDataset) -> DaskCudfDataset:

        data = dataset.data

        # shouldn't hardcode this,
        # should take units from dataset.roles
        # (but its unit is none currently)
        timedelta = np.timedelta64(1, self.basic_interval) / np.timedelta64(1, "ns")

        data = data.map_partitions(
            self._standardize_date_concat,
            timedelta,
            meta=cudf.DataFrame(columns=self.features),
        )

        output = dataset.empty()
        output.set_data(data, self.features, NumericRole(cp.float32))
        return output

    def transform(self, dataset: GpuDataset) -> GpuDataset:
        """Transform dates to numeric differences with base date (GPU version).

        Args:
            dataset: Cupy or Cudf or DaskCudf dataset with datetime columns.

        Returns:
            Respective dataset numeric features.

        """

        super().transform(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self._transform_daskcudf(dataset)
        else:
            return self._transform_cupy(dataset)


class DateSeasonsGPU(LAMLTransformer):
    """
    Basic conversion strategy, used in selection one-to-one transformers (GPU version).
    Datetime converted to difference with basic_date.
    """

    _fname_prefix = "season_gpu"
    _fit_checks = (datetime_check,)
    _transform_checks = ()

    @property
    def features(self) -> List[str]:
        """List of features names."""
        return self._features

    def __init__(self, output_role: Optional[ColumnRole] = None):
        """

        Args:
            output_role: Which role to assign for input features.

        """
        self.output_role = output_role
        if output_role is None:
            self.output_role = CategoryRole(cp.int32)

    def fit(self, dataset: LAMLDataset) -> "LAMLTransformer":
        """Fit transformer and return it's instance (GPU version).

        Args:
            dataset: LAMLDataset to fit on.

        Returns:
            self.

        """

        for check_func in self._fit_checks:
            check_func(dataset)

        feats = dataset.features
        roles = dataset.roles
        self._features = []
        self.transformations = OrderedDict()

        for col in feats:
            seas = roles[col].seasonality
            self.transformations[col] = seas
            for s in seas:
                self._features.append("season_{0}__{1}".format(s, col))
            if roles[col].country is not None:
                self._features.append("season_hol__{0}".format(col))

        return self

    def _datetime_to_seasons(
        self, data: cudf.DataFrame, roles, _date_attrs
    ) -> cudf.DataFrame:
        new_arr = cp.empty((data.shape[0], len(self._features)), cp.int32)
        n = 0
        for col in data.columns:
            for seas in self.transformations[col]:
                vals = getattr(data[col].dt, _date_attrs[seas]).values.astype(cp.int32)
                new_arr[:, n] = vals
                n += 1

            if roles[col].country is not None:
                # get years
                years = cp.unique(data[col].dt.year)
                hol = holidays.CountryHoliday(
                    roles[col].country,
                    years=years,
                    prov=roles[col].prov,
                    state=roles[col].state,
                )
                new_arr[:, n] = data[col].isin(cudf.Series(pd.Series(hol)))
                n += 1
        return cudf.DataFrame(new_arr, index=data.index, columns=self.features)

    def _transform_cupy(self, dataset: DatetimeCompatibleGPU) -> CupyDataset:

        # convert to accepted format and get attributes
        dataset = dataset.to_cudf()
        df = dataset.data
        roles = dataset.roles

        df = self._datetime_to_seasons(df, roles, date_attrs).values

        output = dataset.empty().to_cupy()
        output.set_data(df, self.features, self.output_role)

        return output

    def _transform_daskcudf(self, dataset: DaskCudfDataset) -> DaskCudfDataset:

        new_arr = dataset.data.map_partitions(
            self._datetime_to_seasons,
            dataset.roles,
            date_attrs,
            meta=cudf.DataFrame(columns=self.features),
        )
        output = dataset.empty()
        output.set_data(new_arr, self.features, self.output_role)
        return output

    def transform(self, dataset: GpuDataset) -> GpuDataset:
        """Transform dates to categories - seasons and holiday flag (GPU version).

        Args:
            dataset: Cupy or Cudf or DaskCudf dataset with datetime columns.

        Returns:
            Respective dataset of numeric features.

        """
        assert isinstance(
            dataset, GpuDataset.__args__
        ), "DateSeasonsGPU works only with CupyDataset, CudfDataset, DaskCudfDataset"

        super().transform(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self._transform_daskcudf(dataset)
        else:
            return self._transform_cupy(dataset)
