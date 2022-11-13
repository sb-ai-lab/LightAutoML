"""GroupBy (categorical/numerical) features transformer."""

import numpy as np

from ..dataset.roles import NumericRole
from ..pipelines.utils import get_columns_by_role
from ..utils.logging import verbosity_to_loglevel
from .base import LAMLTransformer
from .utils import GroupByCatIsMode
from .utils import GroupByCatMode
from .utils import GroupByFactory
from .utils import GroupByNumDeltaMean
from .utils import GroupByNumDeltaMedian
from .utils import GroupByNumMax
from .utils import GroupByNumMin
from .utils import GroupByNumStd
from .utils import GroupByProcessor



class GroupByTransformer(LAMLTransformer):
    """Transformer, that calculates group_by features.

    Types of group_by features:
        - Group by categorical:
            - Numerical features:
                - Difference with group mode.
                - Difference with group median.
                - Group min.
                - Group max.
                - Group std.
            - Categorical features:
                - Group mode.
                - Is current value equal to group mode.

    Attributes:
        features list(str): generated features names.

    """

    _fit_checks = ()
    _transform_checks = ()
    _fname_prefix = "grb"

    @property
    def features(self):
        """Features list."""

        return self._features

    def __init__(self, num_groups=None, use_cat_groups=True, **kwargs):
        """

        Args:
            num_groups (list(str)): IDs of functions to use for numeric features.
            use_cat_groups (boolean): flag to show use for category features.

        """

        super().__init__()

        self.num_groups = (
            num_groups
            if num_groups is not None
            else [
                GroupByNumDeltaMean.class_kind,
                GroupByNumDeltaMedian.class_kind,
                GroupByNumMin.class_kind,
                GroupByNumMax.class_kind,
                GroupByNumStd.class_kind,
            ]
        )
        self.use_cat_groups = use_cat_groups
        self.dicts = {}

    def fit(self, dataset):
        """Fit transformer and return it's instance.

        Args:
            dataset: Dataset to fit on.

        Returns:
            self.

        """

        # set transformer names and add checks
        for check_func in self._fit_checks:
            check_func(dataset)

        # convert to accepted dtype and get attributes
        dataset = dataset.to_pandas()

        # set transformer features
        cat_cols = get_columns_by_role(dataset, "Category")
        num_cols = get_columns_by_role(dataset, "Numeric")
        feats = []
        for group_column in cat_cols:
            group_values = dataset.data[group_column].to_numpy()
            group_by_processor = GroupByProcessor(group_values)

            for feature_column in num_cols:
                for kind in self.num_groups:
                    feature = f"{self._fname_prefix}__{group_column}__{kind}__{feature_column}"
                    self.dicts[feature] = {
                        "group_column": group_column,
                        "feature_column": feature_column,
                        "groups": GroupByFactory.get_GroupBy(kind).fit(
                            data=dataset.data,
                            group_by_processor=group_by_processor,
                            feature_column=feature_column,
                        ),
                        "kind": kind,
                    }
                    feats.append(feature)

            if self.use_cat_groups:
                for feature_column in cat_cols:
                    if group_column != feature_column:
                        kind = GroupByCatMode.class_kind

                        # group results are the same for "cat_mode" and "cat_is_mode"
                        groups_1 = GroupByFactory.get_GroupBy(kind).fit(
                            data=dataset.data,
                            group_by_processor=group_by_processor,
                            feature_column=feature_column,
                        )

                        feature1 = f"{self._fname_prefix}__{group_column}__{kind}__{feature_column}"
                        self.dicts[feature1] = {
                            "group_column": group_column,
                            "feature_column": feature_column,
                            "groups": groups_1,
                            "kind": kind,
                        }

                        kind = GroupByCatIsMode.class_kind

                        # group results are the same for "cat_mode" and "cat_is_mode"
                        groups_2 = GroupByFactory.get_GroupBy(kind)
                        groups_2.set_dict(groups_1.get_dict())

                        feature2 = f"{self._fname_prefix}__{group_column}__{kind}__{feature_column}"
                        self.dicts[feature2] = {
                            "group_column": group_column,
                            "feature_column": feature_column,
                            "groups": groups_2,
                            "kind": kind,
                        }
                        feats.extend([feature1, feature2])

        self._features = feats

        return self

    def transform(self, dataset):
        """Calculate groups statistics.

        Args:
            dataset: Numpy or Pandas dataset with category and numeric columns.

        Returns:
            NumpyDataset of calculated group features (numeric).
        """


        # checks here
        super().transform(dataset)

        # convert to accepted dtype and get attributes
        dataset = dataset.to_pandas()

        # transform
        roles = NumericRole()
        outputs = []

        for feat, value in self.dicts.items():

            new_arr = value["groups"].transform(data=dataset.data, value=value)

            output = dataset.empty().to_numpy()
            output.set_data(new_arr, [feat], roles)
            outputs.append(output)

        # create resulted
        return dataset.empty().to_numpy().concat(outputs)
