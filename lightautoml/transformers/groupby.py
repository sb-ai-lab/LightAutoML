"""GroupBy (categorical/numerical) features transformer."""

from typing import List
from typing import Optional
from typing import Union

import numpy as np

from scipy.stats import mode as get_mode

from ..dataset.base import LAMLDataset
from ..dataset.roles import NumericRole
from .base import LAMLTransformer


_transform_types_numeric = ["delta_median", "delta_mean", "min", "max", "std"]
_transform_types_categorical = ["mode", "is_mode"]


class GroupByTransformer(LAMLTransformer):
    """Transformer, that calculates groupby features.

    Types of group_by transformations:
        - Numerical features:
            - delta_median: Difference with group mode.
            - delta_mean: Difference with group median.
            - min: Group min.
            - max: Group max.
            - std: Group std.
        - Categorical features:
            - mode: Group mode.
            - is_mode: Is current value equal to group mode.

    Attributes:
        features list(str): generated features names.

    """

    _fname_prefix = "grb"

    @property
    def features(self):
        """Features list."""
        return self._features

    def __init__(
        self,
        group_col: str,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        used_transforms: Optional[List[str]] = None,
    ):
        """Initialize transformer.

        Args:
            group_col: Name of categorical variable for grouping.
            numeric_cols: List of numeric variables to calculate groupby with respect to the 'group_column'.
            categorical_cols: List of categorical variables to calculate groupby with respect to the 'group_column'.
            used_transforms: List of used transformation types, for example ["std", "mode", "delta_mean"].
                             If not specified, all available transformations are used.
        """
        # assert set(used_transforms).issubset(_transform_types_numeric + _transform_types_categorical), \
        # f"Only these transformation types supported: {_transform_types_numeric + _transform_types_categorical}"

        super().__init__()
        self.group_col = group_col
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self._feat_idx = self._set_feature_indices()
        self.used_transforms = (
            used_transforms if used_transforms else _transform_types_numeric + _transform_types_categorical
        )
        self.numeric_transforms = [t for t in self.used_transforms if t in _transform_types_numeric]
        self.categorical_transforms = [t for t in self.used_transforms if t in _transform_types_categorical]

    def _set_feature_indices(self):
        feat_idx = dict()
        feat_idx[self.group_col] = 0
        i = 1
        for fc in self.categorical_cols:
            feat_idx[fc] = i
            i += 1
        for fn in self.numeric_cols:
            feat_idx[fn] = i
            i += 1
        return feat_idx

    def fit(self, dataset: LAMLDataset):
        """Fit transformer and return it's instance.

        Args:
            dataset: Dataset to fit on.

        Returns:
            self.

        """
        # checksum
        assert dataset.shape[1] == len(self.categorical_cols) + len(self.numeric_cols) + 1
        self._roles = dataset.roles
        dataset = dataset.to_pandas()

        # list of pairs ('feat_name', 'transform_type')
        self.transformations_list = []
        self.transformations_list.extend([(f, t) for f in self.numeric_cols for t in self.numeric_transforms])
        self.transformations_list.extend([(f, t) for f in self.categorical_cols for t in self.categorical_transforms])
        # transformed feature names
        self._features = [f"{self._fname_prefix}__{self.group_col}__{t}__{f}" for f, t in self.transformations_list]
        self._features_mapping = {self._features[i]: k for i, k in enumerate(self.transformations_list)}

        self._group_ids_dict = self._calculate_group_ids(dataset)
        self._group_stats_dict = self._calculate_group_stats(dataset)

        return self

    def _calculate_group_ids(self, dataset: LAMLDataset) -> dict:
        """Extract unique values from group_col and make a dict with indices corresponding to each value."""
        group_values = dataset.data.iloc[:, self._feat_idx[self.group_col]].to_numpy()
        group_ids_dict = dict()
        for i, k in enumerate(group_values):
            if k not in group_ids_dict:
                group_ids_dict[k] = [i]
            else:
                group_ids_dict[k].append(i)
        return {k: np.array(v) for k, v in group_ids_dict.items()}

    def _calculate_group_stats(self, dataset: LAMLDataset) -> dict:
        """Calculate statistics for each transformed feature, corresponding to each pair (feature, 'transform_type')."""
        group_stats = dict()
        dataset = dataset.to_pandas()
        for feature_name in self._features:
            feat, trans = self._features_mapping[feature_name]
            feature_vals = dataset.data.iloc[:, self._feat_idx[feat]].to_numpy()
            group_stats[feature_name] = {
                k: self._feature_stats(feature_vals[idx], trans) for k, idx in self._group_ids_dict.items()
            }
        return group_stats

    def _feature_stats(self, vals: np.ndarray, trans: str) -> Union[str, int, float]:
        """Calculate statistics for vals vector according to 'trans' type."""
        return getattr(self, trans)(vals)

    def transform(self, dataset: LAMLDataset):
        """Calculate groups statistics.

        Args:
            dataset: Numpy or Pandas dataset with category and numeric columns.

        Returns:
            NumpyDataset of calculated group features (numeric).
        """
        feats_block = []
        dataset = dataset.to_pandas()
        group_vals = dataset.data.iloc[:, self._feat_idx[self.group_col]].to_numpy()
        for feature_name in self._features:
            feat, trans = self._features_mapping[feature_name]
            feature_vals = dataset.data.iloc[:, self._feat_idx[feat]].to_numpy()
            stats_from_fit = np.vectorize(self._group_stats_dict[feature_name].get)(group_vals)
            new_feature_vals = self._transform_one(stats_from_fit, feature_vals, trans)
            feats_block.append(new_feature_vals[:, np.newaxis])
        feats_block = np.concatenate(feats_block, axis=1)
        # create resulted
        output = dataset.empty().to_numpy()
        output.set_data(feats_block, self.features, NumericRole(dtype=np.float32))
        # print(output.shape)
        return output

    def _transform_one(self, stats_from_fit, feature_vals, transform_type):
        """Calculate transformation for one pair (feature, 'transform_type')."""
        if transform_type in ["min", "max", "std", "mode"]:
            return stats_from_fit
        elif transform_type in ["delta_mean", "delta_median"]:
            return feature_vals - stats_from_fit
        elif transform_type == "is_mode":
            return feature_vals == stats_from_fit
        else:
            raise ValueError(f"Unknown transformation type: {transform_type}")

    def delta_median(self, vals: np.ndarray) -> float:
        """Alias for numpy median function. Needs subtraction from feature value to get 'delta_median' transformation."""
        return np.nanmedian(vals)

    def delta_mean(self, vals: np.ndarray) -> float:
        """Alias for numpy mean function. Needs subtraction from feature value to get 'delta_mean' transformation."""
        return np.nanmean(vals)

    def min(self, vals: np.ndarray) -> float:
        """Alias for numpy min function."""
        return np.nanmin(vals)

    def max(self, vals: np.ndarray) -> float:
        """Alias for numpy max function."""
        return np.nanmax(vals)

    def std(self, vals: np.ndarray) -> float:
        """Alias for numpy std function."""
        return np.nanstd(vals)

    def mode(self, vals: np.ndarray) -> float:
        """Calculates mode value for categorical variable."""
        return get_mode(vals, keepdims=True)[0][0]

    def is_mode(self, vals: np.ndarray) -> float:
        """Calculates mode value for categorical variable. Needs comparing from initial feature value."""
        return get_mode(vals, keepdims=True)[0][0]
