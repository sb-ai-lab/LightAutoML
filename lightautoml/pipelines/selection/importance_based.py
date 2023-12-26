"""Importance based selectors."""

from typing import Optional
from typing import TypeVar

from lightautoml.validation.base import TrainValidIterator

from ...dataset.base import LAMLDataset
from ...ml_algo.base import MLAlgo
from ..features.base import FeaturesPipeline
from .base import ImportanceEstimator
from .base import SelectionPipeline

import numpy as np
from pandas import Series

from sklearn.feature_selection import mutual_info_regression
from skrebate import ReliefF as releiff
from skrebate import MultiSURF as multisurf
from skfeature.function.information_theoretical_based.FCBF import fcbf


ImportanceEstimatedAlgo = TypeVar("ImportanceEstimatedAlgo", bound=ImportanceEstimator)


class MultiSURF(ImportanceEstimator):

    _name: str = "MultiSURF"

    def __init__(self, num_obs=2000):
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


class ReliefF(ImportanceEstimator):

    _name: str = "reliefF"

    def __init__(self, num_obs=2000):
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

    Args:
        feature_pipeline: Composition of feature transforms.
        ml_algo: Tuple (MlAlgo, ParamsTuner).
        imp_estimator: Feature importance estimator.
        fit_on_holdout: If use the holdout iterator.
        cutoff: Threshold to cut-off features.

    """

    def __init__(
        self,
        feature_pipeline: Optional[FeaturesPipeline],
        ml_algo: MLAlgo,
        imp_estimator: ImportanceEstimator,
        fit_on_holdout: bool = True,
        cutoff: float = 0.0,
    ):
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
