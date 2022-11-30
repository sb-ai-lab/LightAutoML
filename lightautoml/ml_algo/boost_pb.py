"""Wrapped pboost for tabular datasets."""

import logging

import pandas as pd
import numpy as np

from lightautoml.pipelines.selection.base import ImportanceEstimator
from .base import TabularMLAlgo

logger = logging.getLogger(__name__)


class PBPredictor(TabularMLAlgo, ImportanceEstimator):
    """tba
    """

    _name: str = "PB"

    def predict_single_fold(
        self, model, dataset
    ) -> np.ndarray:
        """Predict target values for dataset.

        Args:
            model: Lightgbm object.
            dataset: Test Dataset.

        Return:
            Predicted target values.

        """
        dataset = dataset.to_numpy()
        pred = model.predict(dataset.data)
        return pred

    def get_features_score(self) -> pd.Series:
        """Computes feature importance as mean values of feature importance provided by pyboost per all models.

        Returns:
            Series with feature importances.

        """

        imp = 0
        for model in self.models:
            imp = imp + model.get_feature_importance()

        imp = imp / len(self.models)

        return pd.Series(imp, index=self.features).sort_values(ascending=False)

