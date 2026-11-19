"""Iterative feature selector."""

import logging

from copy import copy
from copy import deepcopy
from functools import partial
from typing import Optional

import numpy as np

from pandas import Series

from lightautoml.validation.base import DummyIterator
from lightautoml.validation.base import HoldoutIterator
from lightautoml.validation.base import TrainValidIterator

from ...dataset.base import LAMLDataset
from ...dataset.np_pd_dataset import CSRSparseDataset
from ...ml_algo.base import MLAlgo
from ...ml_algo.utils import tune_and_fit_predict
from ..features.base import FeaturesPipeline
from ..utils import map_pipeline_names
from .base import ImportanceEstimator
from .base import PredefinedSelector
from .base import SelectionPipeline
from .input_budget import validate_feature_count


logger = logging.getLogger(__name__)


class NpPermutationImportanceEstimator(ImportanceEstimator):
    """Permutation importance based estimator.

    Importance calculate, using random permutation
    of items in single column for each feature.

    Args:
        random_state: seed for random generation of features permutation.

    """

    def __init__(self, random_state: int = 42):
        super().__init__()
        self.random_state = random_state

    def fit(
        self,
        train_valid: Optional[TrainValidIterator] = None,
        ml_algo: Optional[MLAlgo] = None,
        preds: Optional[LAMLDataset] = None,
    ):
        """Find importances for each feature in dataset.

        Args:
            train_valid: Initial dataset iterator.
            ml_algo: Algorithm.
            preds: Predicted target values for validation dataset.

        """
        rng = np.random.RandomState(self.random_state + 1)
        holdout = isinstance(train_valid, (HoldoutIterator, DummyIterator))
        if holdout:
            valid = train_valid.get_validation_data()
            folds = [(None, valid, ml_algo.predict)]
            if preds is None:
                preds = ml_algo.predict(valid)
        else:
            if preds is None:
                raise ValueError(
                    "Permutation importance for a pre-fitted CV model requires its original OOF predictions"
                )
            # A timed fit may have stopped before the final fold.
            folds = [
                (rows, valid, partial(ml_algo.predict_single_fold, model))
                for model, (rows, _, valid) in zip(ml_algo.models, train_valid)
            ]
        if not folds:
            raise ValueError("Permutation importance requires a fitted model and validation rows")
        folds = [
            (
                rows,
                valid.to_numpy() if isinstance(valid, CSRSparseDataset) else valid,
                predict,
                rng.permutation(valid.shape[0]),
            )
            for rows, valid, predict in folds
        ]

        normal_score = ml_algo.score(preds)
        permutation_importance = {}
        for col in folds[0][1].features:
            if not holdout:
                values = np.zeros_like(preds.data)
                counts = np.zeros((len(values), 1), dtype=np.int32)
            for rows, valid, predict, permutation in folds:
                original = valid[:, col].data.copy()
                try:
                    valid[col] = np.asarray(original)[permutation]
                    prediction = predict(valid)
                finally:
                    valid[col] = original
                if holdout:
                    shuffled = prediction
                else:
                    values[rows] += prediction.reshape(valid.shape[0], -1)
                    counts[rows] += 1
            if not holdout:
                shuffled = copy(preds)
                np.divide(values, counts, out=values, where=counts != 0)
                values[counts[:, 0] == 0] = np.nan
                shuffled.set_data(values, preds.features, preds.roles)
            permutation_importance[col] = normal_score - ml_algo.score(shuffled)

        self.raw_importances = Series(permutation_importance).sort_values(ascending=False)


class NpIterativeFeatureSelector(SelectionPipeline):
    """Select features sequentially using chunks to find the best combination of chunks.

    The general idea of this algorithm is to sequentially
    check groups of features ordered by feature importances and
    if the quality of the model becomes better,
    we select such group, if not - ignore group.

    Args:
        feature_pipeline: Composition of feature transforms.
        ml_algo: Tuple (MlAlgo, ParamsTuner).
        imp_estimator: Feature importance estimator.
        fit_on_holdout: If use the holdout iterator.
        feature_group_size: Chunk size.
        max_features_cnt_in_result: Upper bound on selected input columns, including forced inputs.

    """

    def __init__(
        self,
        feature_pipeline: FeaturesPipeline,
        ml_algo: Optional[MLAlgo] = None,
        imp_estimator: Optional[ImportanceEstimator] = None,
        fit_on_holdout: bool = True,
        feature_group_size: int = 5,
        max_features_cnt_in_result: Optional[int] = None,
    ):
        if not fit_on_holdout:
            logger.info2("This selector only for holdout training. fit_on_holout argument added just to be compatible")

        super().__init__(feature_pipeline, ml_algo, imp_estimator, True)

        validate_feature_count("feature_group_size", feature_group_size)
        if max_features_cnt_in_result is not None:
            validate_feature_count("max_features_cnt_in_result", max_features_cnt_in_result)
        self.feature_group_size = feature_group_size
        self.max_features_cnt_in_result = max_features_cnt_in_result

    def _validate_input(self, dataset):
        forced = sum(role.force_input for role in dataset.roles.values())
        if self.max_features_cnt_in_result is not None and forced > self.max_features_cnt_in_result:
            raise ValueError("Forced features exceed max_features_cnt_in_result")

    def perform_selection(self, train_valid: TrainValidIterator):
        """Evaluate groups of raw input columns, starting with the mandatory inputs."""
        self.map_raw_feature_importances(self.imp_estimator.get_features_score())
        sources = map_pipeline_names(self.in_features, train_valid.features)
        selected = list(self.forced_features)
        mandatory = set(selected)
        ranked = [f for f in self.mapped_importances.index if f not in mandatory]
        limit = self.max_features_cnt_in_result
        if limit is None:
            limit = len(self.in_features)

        def score(inputs):
            if self._empty_algo is None:
                raise ValueError("Iterative selection requires an unfitted model to evaluate candidate subsets")
            inputs = set(inputs)
            columns = [col for col, source in zip(train_valid.features, sources) if source in inputs]
            subset = train_valid.apply_selector(PredefinedSelector(columns))
            model, preds = tune_and_fit_predict(deepcopy(self._empty_algo), self.tuner, subset)
            if model is None:
                raise RuntimeError("Could not fit the model for iterative feature selection")
            return model.score(preds)

        best = score(selected) if selected and ranked and len(selected) < limit else None
        start = 0
        while start < len(ranked) and len(selected) < limit:
            size = min(self.feature_group_size, limit - len(selected))
            group = ranked[start : start + size]
            start += len(group)
            candidate_score = score(selected + group)
            if best is None or candidate_score > best:
                selected.extend(group)
                best = candidate_score

        self._selected_features = selected
        self.mapped_importances = self.mapped_importances[self.mapped_importances.index.isin(selected)]
