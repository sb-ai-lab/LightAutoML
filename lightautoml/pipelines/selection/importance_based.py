"""Importance based selectors."""

from typing import Optional
from typing import TypeVar

from lightautoml.validation.base import TrainValidIterator

from ...dataset.base import LAMLDataset
from ...ml_algo.base import MLAlgo
from ..features.base import FeaturesPipeline
from .base import ImportanceEstimator
from .base import SelectionPipeline


ImportanceEstimatedAlgo = TypeVar("ImportanceEstimatedAlgo", bound=ImportanceEstimator)


class ModelBasedImportanceEstimator(ImportanceEstimator):
    """Base class for performing feature selection using model feature importances."""

    def fit(
        self,
        train_valid: Optional[TrainValidIterator] = None,
        ml_algo: Optional[ImportanceEstimatedAlgo] = None,
        preds: Optional[LAMLDataset] = None,
    ):
        """Find the importances of features.

        Args:
            train_valid: dataset iterator.
            ml_algo: ML algorithm used for importance estimation.
            preds: predicted target values.

        """
        assert (
            ml_algo is not None
        ), "ModelBasedImportanceEstimator: raw importances are None and no MLAlgo to calculate them."
        self.raw_importances = ml_algo.get_features_score()


class ImportanceCutoffSelector(SelectionPipeline):
    """Selector based on importance threshold.

    It is important that data which passed to ``.fit``
    should be ok to fit `ml_algo` or preprocessing pipeline should be defined.

    """

    def __init__(
        self,
        feature_pipeline: Optional[FeaturesPipeline],
        ml_algo: MLAlgo,
        imp_estimator: ImportanceEstimator,
        fit_on_holdout: bool = True,
        cutoff: float = 0.0,
    ):
        """

        Args:
            feature_pipeline: Composition of feature transforms.
            ml_algo: Tuple (MlAlgo, ParamsTuner).
            imp_estimator: Feature importance estimator.
            fit_on_holdout: If use the holdout iterator.
            cutoff: Threshold to cut-off features.

        """
        super().__init__(feature_pipeline, ml_algo, imp_estimator, fit_on_holdout)
        self.cutoff = cutoff

    def perform_selection(self, train_valid: Optional[TrainValidIterator] = None):
        """Select features based on cutoff value.

        Args:
            train_valid: Not used.

        """
        imp = self.imp_estimator.get_features_score()
        self.map_raw_feature_importances(imp)
        selected = self.mapped_importances.index.values[self.mapped_importances.values > self.cutoff]
        if len(selected) == 0:
            selected = self.mapped_importances.index.values[:1]
        self._selected_features = list(selected)
