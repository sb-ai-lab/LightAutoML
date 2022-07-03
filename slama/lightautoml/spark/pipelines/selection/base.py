"""Base class for selection pipelines."""
from typing import Any

from lightautoml.spark.dataset.base import SparkDataset


class SparkImportanceEstimator:
    """
    Abstract class, that estimates feature importances.
    """

    def __init__(self):
        self.raw_importances = None

    # Change signature here to be compatible with MLAlgo
    def fit(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def get_features_score(self) -> SparkDataset:

        return self.raw_importances
