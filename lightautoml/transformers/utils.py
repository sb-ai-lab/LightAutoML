"""utils for transformers."""

import numpy as np

from scipy.stats import mode


def get_mode(x):
    """Helper function to calculate mode."""
    return mode(x)[0][0]


class GroupByProcessor:
    """Helper class to calculate group_by features."""

    def __init__(self, keys):
        super().__init__()

        assert keys is not None

        self.index, self.keys_as_int = np.unique(keys, return_inverse=True)
        self.n_keys = max(self.keys_as_int) + 1
        self.set_indices()

    def set_indices(self):
        """Sets indices for keys."""
        self.indices = [[] for i in range(self.n_keys)]
        for i, k in enumerate(self.keys_as_int):
            self.indices[k].append(i)
        self.indices = [np.array(elt) for elt in self.indices]

    def apply(self, functions, vectors):
        """Applies functions to vectors."""
        assert functions is not None
        assert vectors is not None

        if isinstance(functions, list):
            return [[fun(vec[idx].tolist()) for fun, vec in zip(functions, vectors)] for idx in (self.indices)]
        else:
            return [functions(vectors[idx].tolist()) for idx in (self.indices)]


class GroupByFactory:
    """Factory to create group_by classes."""

    @staticmethod
    def get_GroupBy(kind):
        """Uses string identifiers to locate appropriate implementation.

        Args:
            kind: groupby feature transformation class

        Example:
            GroupByFactory.get_GroupBy('delta_mean')

        Returns:
            Object of GroupByBase impementing selected feature.

        Raises:
            ValueError: if identifier is not found.
        """
        assert kind is not None

        available_classes = [
            GroupByNumDeltaMean,
            GroupByNumDeltaMedian,
            GroupByNumMin,
            GroupByNumMax,
            GroupByNumStd,
            GroupByCatMode,
            GroupByCatIsMode,
        ]

        for class_name in available_classes:
            if kind == class_name.class_kind:
                return class_name(
                    class_name.class_kind,
                    class_name.class_fit_func,
                    class_name.class_transform_func,
                )

        raise ValueError(
            f"Unsupported kind: {kind}, available={[class_name.class_kind for class_name in available_classes]}"
        )
        return


class GroupByBase:
    """Base class for all group_by features.

    Note:
        Typically is created from GroupByFactory.

    Args:
            kind (string): Id of group_by feature.
            fit_func (function): function to calculate groups.
            transform_func (function): function to calculate statistics based on fitted groups.

    Example:
        GroupByBase(GroupByNumDeltaMean.class_kind, GroupByNumDeltaMean.class_fit_func, GroupByNumDeltaMean.class_transform_func)

    """

    def __init__(self, kind, fit_func, transform_func):
        super().__init__()

        self.kind = kind
        self.fit_func = fit_func
        self.transform_func = transform_func

        self._dict = None

    def get_dict(self):
        """Gets dict with features' statistics."""
        return self._dict

    def set_dict(self, dict):
        """Sets dict with features' statistics."""
        self._dict = dict

    def fit(self, data, group_by_processor, feature_column):
        """Calculate groups.

        Note:
            GroupByProcessor must be initialiaed before call to this function.

        Args:
            data (dataset): input data to extract ``feature_column``.
            group_by_processor (GroupByProcessor): processor, containig groups.
            feature_column (string): name of column to calculate statistics.

        Returns:
            self

        """
        assert data is not None
        assert group_by_processor is not None
        assert feature_column is not None

        assert self.fit_func is not None

        feature_values = data[feature_column].to_numpy()
        self._dict = dict(
            zip(
                group_by_processor.index,
                group_by_processor.apply(self.fit_func, feature_values),
            )
        )

        assert self._dict is not None

        return self

    def transform(self, data, value):
        """Calculate features statistics.

        Note:
            ``fit`` function must be called before ``transform``.

        Args:
            data (dataset): input data to extract ``value['group_column']`` and ``value['feature_column']``.
            value (dict): column names.

        Returns:
            transformed data

        """
        assert data is not None
        assert value is not None

        assert self.transform_func is not None

        group_values = data[value["group_column"]].to_numpy()
        feature_values = data[value["feature_column"]].to_numpy()
        result = self.transform_func(
            tuple(
                [
                    np.nan_to_num(np.array(np.vectorize(self._dict.get)(group_values), dtype=float)),
                    feature_values,
                ]
            )
        ).reshape(-1, 1)

        assert result is not None
        return result


class GroupByNumDeltaMean(GroupByBase):
    """Groupby delta mean class."""

    class_kind = "delta_mean"
    class_fit_func = np.nanmean

    @staticmethod
    def class_transform_func(values):
        """Get difference between feature values inside group and mean value."""
        return values[1] - values[0]


class GroupByNumDeltaMedian(GroupByBase):
    """Groupby delta median class."""

    class_kind = "delta_median"
    class_fit_func = np.nanmedian

    @staticmethod
    def class_transform_func(values):
        """Get difference between feature values inside group and median value."""
        return values[1] - values[0]


class GroupByNumMin(GroupByBase):
    """Groupby min class."""

    class_kind = "min"
    class_fit_func = np.nanmin

    @staticmethod
    def class_transform_func(values):
        """Get min value inside each group."""
        return values[0]


class GroupByNumMax(GroupByBase):
    """Groupby max class."""

    class_kind = "max"
    class_fit_func = np.nanmax

    @staticmethod
    def class_transform_func(values):
        """Get max value inside each group."""
        return values[0]


class GroupByNumStd(GroupByBase):
    """Groupby std class."""

    class_kind = "std"
    class_fit_func = np.nanstd

    @staticmethod
    def class_transform_func(values):
        """Get std value inside each group."""
        return values[0]


class GroupByCatMode(GroupByBase):
    """Groupby cat mode class."""

    class_kind = "mode"
    class_fit_func = get_mode

    @staticmethod
    def class_transform_func(values):
        """Get category mode inside each group."""
        return values[0]


class GroupByCatIsMode(GroupByBase):
    """Groupby is mode class."""

    class_kind = "is_mode"
    class_fit_func = get_mode

    @staticmethod
    def class_transform_func(values):
        """Check if category value is mode inside each group."""
        return values[0] == values[1]
