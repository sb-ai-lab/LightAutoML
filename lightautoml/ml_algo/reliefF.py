"""Wrapped LightGBM for tabular datasets."""

import logging
import numpy as np
from pandas import Series
from ..pipelines.selection.base import ImportanceEstimator
from ..validation.base import TrainValidIterator
from skrebate import ReliefF as releiff, SURF as surf, SURFstar as surfstar
from skrebate import MultiSURF as multisurf, MultiSURFstar as multisurfstar


logger = logging.getLogger(__name__)


class ReliefF(ImportanceEstimator):

    _name: str = "reliefF"

    def __init__(self, num_obs=1000):
        super().__init__()
        self.num_obs = num_obs
        self.model = releiff(verbose=False, n_jobs=10, n_neighbors=70)

    def fit(self, train_valid: TrainValidIterator, *args, **kwargs):
        # obtain the feature score on each fold
        self.scores = []
        self.features = train_valid.features
        for n, (idx, train, valid) in enumerate(train_valid):
            sampled_idx = np.random.choice(len(train.data), min(self.num_obs, len(train.data)), replace=False)
            self.model.fit(train.data[sampled_idx], train.target[sampled_idx])
            score = self.model.feature_importances_
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


class SURF(ImportanceEstimator):

    _name: str = "SURF"

    def __init__(self, num_obs=1000):
        super().__init__()
        self.num_obs = num_obs
        self.model = surf(verbose=False, n_jobs=10)

    def fit(self, train_valid: TrainValidIterator, *args, **kwargs):
        # obtain the feature score on each fold
        self.scores = []
        self.features = train_valid.features
        for n, (idx, train, valid) in enumerate(train_valid):
            sampled_idx = np.random.choice(len(train.data), min(self.num_obs, len(train.data)), replace=False)
            self.model.fit(train.data[sampled_idx], train.target[sampled_idx])
            score = self.model.feature_importances_
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


class SURFstar(ImportanceEstimator):

    _name: str = "SURFstar"

    def __init__(self):
        super().__init__()
        self.model = surfstar(verbose=False, n_jobs=10)

    def fit(self, train_valid: TrainValidIterator, *args, **kwargs):
        # obtain the feature score on each fold
        self.scores = []
        self.features = train_valid.features
        for n, (idx, train, valid) in enumerate(train_valid):
            self.model.fit(train.data, train.target)
            score = self.model.feature_importances_
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


class MultiSURF(ImportanceEstimator):

    _name: str = "MultiSURF"

    def __init__(self, num_obs=1000):
        super().__init__()
        self.num_obs = num_obs
        self.model = multisurf(verbose=False, n_jobs=10)

    def fit(self, train_valid: TrainValidIterator, *args, **kwargs):
        # obtain the feature score on each fold
        self.scores = []
        self.features = train_valid.features
        for n, (idx, train, valid) in enumerate(train_valid):
            sampled_idx = np.random.choice(len(train.data), min(self.num_obs, len(train.data)), replace=False)
            self.model.fit(train.data[sampled_idx], train.target[sampled_idx])
            score = self.model.feature_importances_
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


class MultiSURFstar(ImportanceEstimator):

    _name: str = "SURFstar"

    def __init__(self):
        super().__init__()
        self.model = multisurfstar(verbose=False, n_jobs=10)

    def fit(self, train_valid: TrainValidIterator, *args, **kwargs):
        # obtain the feature score on each fold
        self.scores = []
        self.features = train_valid.features
        for n, (idx, train, valid) in enumerate(train_valid):
            self.model.fit(train.data, train.target)
            score = self.model.feature_importances_
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
