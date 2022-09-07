"""Datetime features transformers."""

from collections import OrderedDict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import holidays
import numpy as np

from ..dataset.base import LAMLDataset
from ..dataset.np_pd_dataset import NumpyDataset
from ..dataset.np_pd_dataset import PandasDataset
from ..dataset.roles import CategoryRole
from ..dataset.roles import ColumnRole
from ..dataset.roles import NumericRole
from .base import LAMLTransformer


# type - dataset that is ok with datetime dtypes
DatetimeCompatible = Union[PandasDataset]

date_attrs = {
    "y": "year",
    "m": "month",
    "d": "day",
    "wd": "weekday",
    "hour": "hour",
    "min": "minute",
    "sec": "second",
    "ms": "microsecond",
    "ns": "nanosecond",
}


def datetime_check(dataset: LAMLDataset):
    """Check if all passed vars are datetimes.

    Args:
        dataset: Dataset to check.

    Raises:
        AssertionError: If non-datetime features are present.

    """
    roles = dataset.roles
    features = dataset.features
    for f in features:
        assert roles[f].name == "Datetime", "Only datetimes accepted in this transformer"


class TimeToNum(LAMLTransformer):
    """
    Basic conversion strategy, used in selection one-to-one transformers.
    Datetime converted to difference
    with basic_date (``basic_date == '2020-01-01'``).
    """

    basic_time = "2020-01-01"
    basic_interval = "D"

    _fname_prefix = "dtdiff"
    _fit_checks = (datetime_check,)
    _transform_checks = ()

    def transform(self, dataset: DatetimeCompatible) -> NumpyDataset:
        """Transform dates to numeric differences with base date.

        Args:
            dataset: Numpy or Pandas dataset with datetime columns.

        Returns:
            Numpy dataset of numeric features.

        """
        # checks if exist
        super().transform(dataset)
        # convert to accepted format and get attributes
        dataset = dataset.to_pandas()
        data = dataset.data

        # transform
        roles = NumericRole(np.float32)

        new_arr = ((data - np.datetime64(self.basic_time)) / np.timedelta64(1, self.basic_interval)).values.astype(
            np.float32
        )

        # create resulted
        output = dataset.empty().to_numpy()
        output.set_data(new_arr, self.features, roles)

        return output


class BaseDiff(LAMLTransformer):
    """
    Basic conversion strategy, used in selection one-to-one transformers.
    Datetime converted to difference with basic_date.

    """

    basic_interval = "D"

    _fname_prefix = "basediff"
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
        """Fit transformer and return it's instance.

        Args:
            dataset: Dataset to fit on.

        Returns:
            self.

        """
        self._features = []
        for col in self.base_names:
            self._features.extend(["basediff_{0}__{1}".format(col, x) for x in self.diff_names])

        for check_func in self._fit_checks:
            check_func(dataset)
        return self

    def transform(self, dataset: DatetimeCompatible) -> NumpyDataset:
        """Transform dates to numeric differences with base date.

        Args:
            dataset: Numpy or Pandas dataset with datetime columns.

        Returns:
            NumpyDataset of numeric features.

        """
        # checks if exist
        super().transform(dataset)
        # convert to accepted format and get attributes
        dataset = dataset.to_pandas()
        data = dataset.data[self.diff_names].values
        base_cols = dataset.data[self.base_names]

        feats_block = []

        # transform
        for col in base_cols.columns:
            new_arr = ((data - base_cols[[col]].values) / np.timedelta64(1, self.basic_interval)).astype(np.float32)
            feats_block.append(new_arr)

        feats_block = np.concatenate(feats_block, axis=1)

        # create resulted
        output = dataset.empty().to_numpy()
        output.set_data(feats_block, self.features, NumericRole(dtype=np.float32))

        return output


class DateSeasons(LAMLTransformer):
    """
    Basic conversion strategy, used in selection one-to-one transformers.
    Datetime converted to difference with basic_date.
    """

    _fname_prefix = "season"
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
            self.output_role = CategoryRole(np.int32)

    def fit(self, dataset: LAMLDataset) -> "LAMLTransformer":
        """Fit transformer and return it's instance.

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

    def transform(self, dataset: DatetimeCompatible) -> NumpyDataset:
        """Transform dates to categories - seasons and holiday flag.

        Args:
            dataset: Numpy or Pandas dataset with datetime columns.

        Returns:
            Numpy dataset of numeric features.

        """
        # checks if exist
        super().transform(dataset)
        # convert to accepted format and get attributes
        dataset = dataset.to_pandas()
        df = dataset.data
        roles = dataset.roles

        new_arr = np.empty((df.shape[0], len(self._features)), np.int32)

        n = 0
        for col in dataset.features:
            for seas in self.transformations[col]:
                new_arr[:, n] = getattr(df[col].dt, date_attrs[seas])
                n += 1

            if roles[col].country is not None:
                # get years
                years = np.unique(df[col].dt.year)
                hol = holidays.CountryHoliday(
                    roles[col].country,
                    years=years,
                    prov=roles[col].prov,
                    state=roles[col].state,
                )
                new_arr[:, n] = df[col].isin(hol)
                n += 1

        # create resulted
        output = dataset.empty().to_numpy()
        output.set_data(new_arr, self.features, self.output_role)

        return output
