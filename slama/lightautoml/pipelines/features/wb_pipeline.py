"""Whitebox features."""

import numpy as np

from ...dataset.np_pd_dataset import PandasDataset
from ...dataset.roles import NumericRole
from ...transformers.base import ColumnsSelector
from ...transformers.base import LAMLTransformer
from ...transformers.base import UnionTransformer
from ..utils import get_columns_by_role
from .base import FeaturesPipeline
from .base import TabularDataFeatures


class WBFeatures(FeaturesPipeline, TabularDataFeatures):
    """Simple WhiteBox pipeline.

    Just handles dates, other are handled inside WhiteBox.

    """

    def create_pipeline(self, train: PandasDataset) -> LAMLTransformer:
        """Create pipeline for WhiteBox.

        Args:
            train: Dataset with train features.

        Returns:
            Transformer.

        """
        others = get_columns_by_role(train, "Category") + get_columns_by_role(train, "Numeric")

        transformer_list = [
            self.get_datetime_diffs(train),
            self.get_datetime_seasons(train, NumericRole(np.float32)),
            ColumnsSelector(others),
        ]

        # final pipeline
        union_all = UnionTransformer([x for x in transformer_list if x is not None])

        return union_all
