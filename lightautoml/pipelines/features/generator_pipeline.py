"""FeatureTools Pipeline for connected tables."""

import logging

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from ...dataset.base import LAMLDataset
from ...transformers.base import SequentialTransformer
from ...transformers.base import UnionTransformer
from ...transformers.generator import FeatureGeneratorTransformer
from .base import FeaturesPipeline


logger = logging.getLogger(__name__)


class FeatureGeneratorPipeline(FeaturesPipeline):
    """Creates pipeline for feature generation.

    Args:
        seq_params: sequence-related params.
        max_gener_features: maximum generated features.
        max_depth: maximum allowed depth of features.
        agg_primitives: list of aggregation primitives.
        trans_primitives: list of transform primitives.
        interesting_values: categorical values if the form of {table_name: {column: [values]}} for feature generation in corresponding slices.
        generate_interesting_values: whether generate feature in slices of unique categories or not.
        per_top_categories: percent of most frequent categories for feature generation in corresponding slices. If number of unique values is less than 10, then the all values are be used.
        sample_size: size of data to make generated feature selection on it.
        n_jobs: number of processes to run in parallel
    """

    def __init__(
        self,
        seq_params: Dict,
        max_gener_features: int = -1,
        max_depth: int = 2,
        agg_primitives: Optional[Tuple[str]] = (
            "entropy",
            "count",
            "mean",
            "std",
            "median",
            "max",
            "sum",
            "num_unique",
            "min",
            "percent_true",
        ),
        trans_primitives: Optional[Tuple[str]] = (
            "hour",
            "month",
            "weekday",
            "is_weekend",
            "day",
            "time_since_previous",
            "week",
            "age",
            "time_since",
        ),
        interesting_values: Dict[str, Dict[str, List[Any]]] = None,
        generate_interesting_values: bool = False,
        per_top_categories: float = 0.25,
        sample_size: int = None,
        n_jobs: int = 1,
    ):
        self.seq_params = seq_params
        self.max_gener_features = max_gener_features
        self.max_depth = max_depth
        self.agg_primitives = agg_primitives
        self.trans_primitives = trans_primitives
        self.interesting_values = interesting_values
        self.generate_interesting_values = generate_interesting_values
        self.per_top_categories = per_top_categories
        self.sample_size = sample_size
        self.n_jobs = n_jobs

        super().__init__()

    def create_pipeline(self, train: LAMLDataset):
        """Create tree pipeline.

        Args:
            train: Dataset with train features.

        Returns:
            Feature Generator transformer.

        """
        transformer_list = list()
        seq_processing = SequentialTransformer(
            [
                FeatureGeneratorTransformer(
                    self.seq_params,
                    self.max_gener_features,
                    self.max_depth,
                    self.agg_primitives,
                    self.trans_primitives,
                    self.interesting_values,
                    self.generate_interesting_values,
                    self.per_top_categories,
                    self.sample_size,
                    self.n_jobs,
                ),
            ]
        )
        transformer_list.append(seq_processing)
        return UnionTransformer(transformer_list)
