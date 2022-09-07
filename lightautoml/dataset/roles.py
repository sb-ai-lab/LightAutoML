"""Role contains information about the column, which determines how it is processed."""

from datetime import datetime
from typing import Any
from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np


Dtype = Union[Callable, type, str]


# valid_features_str_names = []


class ColumnRole:
    """Abstract class for column role.

    Role type defines column dtype,
    place of column in dataset and transformers
    and set additional attributes which impacts
    on the way how it's handled.

    """

    dtype = object
    force_input = False
    _name = "Abstract"

    @property
    def name(self) -> str:
        """Get str role name.

        Returns:
            str role name.

        """
        return self._name

    def __repr__(self) -> str:
        """String view of role.

        Returns:
            Representation string.

        """
        params = [(x, self.__dict__[x]) for x in self.__dict__ if x not in ["dtype", "name"]]

        return "{0} role, dtype {1}. Additional params: {2}".format(self.name, self.dtype, params)

    def __hash__(self) -> int:
        """Define how to hash - hash from str view.

        Returns:
            Hashed name of column.

        """
        return hash(self.__repr__())

    def __eq__(self, other: Any) -> bool:
        """Define how to compare - if reprs are equal (hashed).

        Args:
            other: Another :class:`~lightautoml.dataset.roles.ColumnRole`.

        Returns:
            ``True`` if equal.

        """
        return self.__repr__() == other.__repr__()

    @staticmethod
    def from_string(name: str, **kwargs: Any) -> "ColumnRole":
        """Create default params role from string.

        Args:
            name: Role name.

        Returns:
            Corresponding role object.

        """
        name = name.lower()

        if name in ["target"]:
            return TargetRole(**kwargs)

        if name in ["numeric"]:
            return NumericRole(**kwargs)

        if name in ["category"]:
            return CategoryRole(**kwargs)

        if name in ["text"]:
            return TextRole(**kwargs)

        if name in ["datetime"]:
            return DatetimeRole(**kwargs)

        if name in ["base_date"]:
            kwargs = {**{"seasonality": (), "base_date": True}, **kwargs}
            return DatetimeRole(**kwargs)

        if name in ["group"]:
            return GroupRole()

        if name in ["drop"]:
            return DropRole()

        if name in ["id"]:
            kwargs = {**{"encoding_type": "oof", "unknown": 1}, **kwargs}
            return CategoryRole(**kwargs)

        if name in ["folds"]:
            return FoldsRole()

        if name in ["weights"]:
            return WeightsRole()

        if name in ["path"]:
            return PathRole()

        if name in ["treatment"]:
            return TreatmentRole()

        raise ValueError("Unknown string role: {}".format(name))


class NumericRole(ColumnRole):
    """Numeric role."""

    _name = "Numeric"

    def __init__(
        self,
        dtype: Dtype = np.float32,
        force_input: bool = False,
        prob: bool = False,
        discretization: bool = False,
    ):
        """Create numeric role with specific numeric dtype.

        Args:
            dtype: Variable type.
            force_input: Select a feature for training,
              regardless of the selector results.
            prob: If input number is probability.

        """
        self.dtype = dtype
        self.force_input = force_input
        self.prob = prob
        self.discretization = discretization


class CategoryRole(ColumnRole):
    """Category role."""

    _name = "Category"

    def __init__(
        self,
        dtype: Dtype = object,
        encoding_type: str = "auto",
        unknown: int = 5,
        force_input: bool = False,
        label_encoded: bool = False,
        ordinal: bool = False,
    ):
        """Create category role with specific dtype and attrs.

        Args:
            dtype: Variable type.
            encoding_type: Encoding type.
            unknown: Cut-off freq to process rare categories as unseen.
            force_input: Select a feature for training,
              regardless of the selector results.

        Note:
            Valid encoding_type:

                - `'auto'` - default processing
                - `'int'` - encode with int
                - `'oof'` - out-of-fold target encoding
                - `'freq'` - frequency encoding
                - `'ohe'` - one hot encoding

        """
        # TODO: assert dtype is object, 'Dtype for category should be defined' ?
        # assert encoding_type == 'auto', 'For the moment only auto is supported'
        # TODO: support all encodings
        self.dtype = dtype
        self.encoding_type = encoding_type
        self.unknown = unknown
        self.force_input = force_input
        self.label_encoded = label_encoded
        self.ordinal = ordinal


class TextRole(ColumnRole):
    """Text role."""

    _name = "Text"

    def __init__(self, dtype: Dtype = str, force_input: bool = True):
        """Create text role with specific dtype and attrs.

        Args:
            dtype: Variable type.
            force_input: Select a feature for training,
              regardless of the selector results.

        """
        self.dtype = dtype
        self.force_input = force_input


class DatetimeRole(ColumnRole):
    """Datetime role."""

    _name = "Datetime"

    def __init__(
        self,
        dtype: Dtype = np.datetime64,
        seasonality: Optional[Sequence[str]] = ("y", "m", "wd"),
        base_date: bool = False,
        date_format: Optional[str] = None,
        unit: Optional[str] = None,
        origin: Union[str, datetime] = "unix",
        force_input: bool = False,
        base_feats: bool = True,
        country: Optional[str] = None,
        prov: Optional[str] = None,
        state: Optional[str] = None,
    ):
        """Create datetime role with specific dtype and attrs.

        Args:
            dtype: Variable type.
            seasonality: Seasons to extract from date.
              Valid are: 'y', 'm', 'd', 'wd', 'hour',
              'min', 'sec', 'ms', 'ns'.
            base_date: Base date is used to calculate difference
              with other dates, like `age = report_dt - birth_dt`.
            date_format: Format to parse date.
            unit: The unit of the arg denote the unit, pandas like, see more:
              https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html.
            origin: Define the reference date, pandas like, see more:
              https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html.
            force_input: Select a feature for training,
              regardless of the selector results.
            base_feats: To calculate feats on base date.
            country: Datetime metadata to extract holidays.
            prov: Datetime metadata to extract holidays.
            state: Datetime metadata to extract holidays.

        """
        self.dtype = dtype
        self.seasonality = []
        if seasonality is not None:
            self.seasonality = seasonality
        self.base_date = base_date
        self.format = date_format
        self.unit = unit
        self.origin = origin

        self.force_input = force_input
        if self.base_date:
            self.force_input = True
        self.base_feats = base_feats

        self.country = country
        self.prov = prov
        self.state = state


# class MixedRole(ColumnRole):
#     """
#     Mixed role. If exact role extraction is difficult, it goes into both pipelines
#     """


class TargetRole(ColumnRole):
    """Target role."""

    _name = "Target"

    def __init__(self, dtype: Dtype = np.float32):
        """Create target role with specific numeric dtype.

        Args:
            dtype: Dtype of target.

        """
        self.dtype = dtype


class GroupRole(ColumnRole):
    """Group role."""

    _name = "Group"


class DropRole(ColumnRole):
    """Drop role."""

    _name = "Drop"


class WeightsRole(ColumnRole):
    """Weights role."""

    _name = "Weights"


class FoldsRole(ColumnRole):
    """Folds role."""

    _name = "Folds"


class PathRole(ColumnRole):
    """Path role."""

    _name = "Path"


class TreatmentRole(ColumnRole):
    """Uplift Treatment Role."""

    _name = "Treatment"
