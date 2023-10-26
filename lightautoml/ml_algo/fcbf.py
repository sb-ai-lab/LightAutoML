"""Wrapped LightGBM for tabular datasets."""

import logging
import numpy as np
from pandas import Series
from ..pipelines.selection.base import ImportanceEstimator
from ..validation.base import TrainValidIterator
from skfeature.function.information_theoretical_based.FCBF import fcbf

logger = logging.getLogger(__name__)


class FCBF(ImportanceEstimator):

    _name: str = "FCBF"

    def __init__(self):
        super().__init__()
        self.model = fcbf

    def fit(self, train_valid: TrainValidIterator, *args, **kwargs):
        # obtain the feature score on each fold
        self.features = train_valid.features
        self.scores = []
        for n, (idx, train, valid) in enumerate(train_valid):
            idx = self.model(np.nan_to_num(train.data), train.target)
            score = np.zeros(len(self.features))
            score[idx[0]] = 1.0
            self.scores.append(score)

    def get_features_score(self) -> Series:
        """

        Returns:
            Series with feature importances.

        """
        imp = 0
        for score in self.scores:
            imp = imp + score

        imp = imp / len(self.scores)

        return Series(imp, index=self.features).sort_values(ascending=False)
