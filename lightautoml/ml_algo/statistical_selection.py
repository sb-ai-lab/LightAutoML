import logging
import numpy as np
from pandas import Series
from ..pipelines.selection.base import ImportanceEstimator
from ..validation.base import TrainValidIterator

from sklearn.feature_selection import mutual_info_regression


logger = logging.getLogger(__name__)


class MutualInfoRegression(ImportanceEstimator):

    _name: str = "MIR"

    def __init__(self, random_state=42):
        super().__init__()
        self.random_state = random_state
        self.model = mutual_info_regression

    def fit(self, train_valid: TrainValidIterator, *args, **kwargs):
        # obtain the feature score on each fold
        self.scores = []
        self.features = train_valid.features
        for n, (idx, train, valid) in enumerate(train_valid):
            mi = self.model(np.nan_to_num(train.data), train.target, random_state=self.random_state)
            mi /= np.max(mi)
            mi = np.where(mi > 0.8, 1, 0)
            self.scores.append(mi)

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
