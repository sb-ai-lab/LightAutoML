"""Iterative feature selector."""

import logging

from copy import deepcopy
from typing import Optional

import numpy as np

from pandas import Series

from lightautoml.validation.base import TrainValidIterator

from ...dataset.base import LAMLDataset
from ...ml_algo.base import MLAlgo
from ...ml_algo.utils import tune_and_fit_predict
from ..features.base import FeaturesPipeline
from .base import ImportanceEstimator
from .base import PredefinedSelector
from .base import SelectionPipeline


logger = logging.getLogger(__name__)


def _create_chunks_from_list(lst, n):
    """Creates chunks of list.

    Args:
        lst: List of elements.
        n: Size of chunk.

    """
    chunks = []
    for i in range(0, len(lst), n):
        chunks.append(lst[i : i + n])
    return chunks


class NpPermutationImportanceEstimator(ImportanceEstimator):
    """Permutation importance based estimator.

    Importance calculate, using random permutation
    of items in single column for each feature.

    """

    def __init__(self, random_state: int = 42):
        """
        Args:
            random_state: seed for random generation of features permutation.

        """
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

        normal_score = ml_algo.score(preds)
        logger.debug("Normal score = {}".format(normal_score))

        valid_data = train_valid.get_validation_data()
        valid_data = valid_data.to_numpy()

        permutation = np.random.RandomState(seed=self.random_state).permutation(valid_data.shape[0])
        permutation_importance = {}

        for it, col in enumerate(valid_data.features):
            logger.debug("Start processing ({},{})".format(it, col))
            # Save initial column
            save_col = deepcopy(valid_data[:, col])

            # Get current column and shuffle it
            shuffled_col = valid_data[permutation, col]

            # Set shuffled column
            logger.info3("Shuffled column set")
            valid_data[col] = shuffled_col

            # Calculate predict and metric
            logger.info3("Shuffled column set")
            new_preds = ml_algo.predict(valid_data)
            shuffled_score = ml_algo.score(new_preds)
            logger.debug(
                "Shuffled score for col {} = {}, difference with normal = {}".format(
                    col, shuffled_score, normal_score - shuffled_score
                )
            )
            permutation_importance[col] = normal_score - shuffled_score

            # Set normal column back to the dataset
            logger.debug("Normal column set")
            valid_data[col] = save_col

        self.raw_importances = Series(permutation_importance).sort_values(ascending=False)


class NpIterativeFeatureSelector(SelectionPipeline):
    """Select features sequentially using chunks to find the best combination of chunks.

    The general idea of this algorithm is to sequentially
    check groups of features ordered by feature importances and
    if the quality of the model becomes better,
    we select such group, if not - ignore group.
    """

    def __init__(
        self,
        feature_pipeline: FeaturesPipeline,
        ml_algo: Optional[MLAlgo] = None,
        imp_estimator: Optional[ImportanceEstimator] = None,
        fit_on_holdout: bool = True,
        feature_group_size: Optional[int] = 5,
        max_features_cnt_in_result: Optional[int] = None,
    ):
        """

        Args:
            feature_pipeline: Composition of feature transforms.
            ml_algo: Tuple (MlAlgo, ParamsTuner).
            imp_estimator: Feature importance estimator.
            fit_on_holdout: If use the holdout iterator.
            feature_group_size: Chunk size.
            max_features_cnt_in_result: Lower bound of features after selection,
              if it is reached, it will stop.

        """
        if not fit_on_holdout:
            logger.info2("This selector only for holdout training. fit_on_holout argument added just to be compatible")

        super().__init__(feature_pipeline, ml_algo, imp_estimator, True)

        self.feature_group_size = feature_group_size
        self.max_features_cnt_in_result = max_features_cnt_in_result

    def perform_selection(self, train_valid: Optional[TrainValidIterator] = None):
        """Select features iteratively by checking model quality for current selected feats and new group.

        Args:
            train_valid: Iterator for dataset.

        """

        # Calculate or receive permutation importances scores
        imp = self.imp_estimator.get_features_score()

        features_to_check = [x for x in imp.index if x in set(train_valid.features)]

        # Perform iterative selection algo
        chunks = _create_chunks_from_list(features_to_check, self.feature_group_size)
        selected_feats = []
        cnt_without_update = 0
        cur_best_score = None

        for it, chunk in enumerate(chunks):
            if self.max_features_cnt_in_result is not None and len(selected_feats) >= self.max_features_cnt_in_result:
                logger.info3(
                    "We exceeded max_feature_cnt_in_result bound (selected features count = {}). Exiting from iterative algo...".format(
                        len(selected_feats)
                    )
                )
                break
            selected_feats += chunk
            logger.info3("Started iteration {}, chunk = {}, feats to check = {}".format(it, chunk, selected_feats))
            cs = PredefinedSelector(selected_feats)
            selected_cols_iterator = train_valid.apply_selector(cs)
            logger.info3("Features in SCI = {}".format(selected_cols_iterator.features))

            # Create copy of MLAlgo for iterative algo only
            ml_algo_for_iterative, preds = tune_and_fit_predict(
                deepcopy(self._empty_algo), self.tuner, selected_cols_iterator
            )

            cur_score = ml_algo_for_iterative.score(preds)
            logger.debug("Current score = {}, current best score = {}".format(cur_score, cur_best_score))

            if cur_best_score is None or cur_best_score < cur_score:
                logger.info3("Update best score from {} to {}".format(cur_best_score, cur_score))
                cur_best_score = cur_score
                cnt_without_update = 0
            else:
                cnt_without_update += 1
                logger.debug(
                    "Without update for {} steps. Remove last added group {} from selected features...".format(
                        cnt_without_update, chunk
                    )
                )
                selected_feats = selected_feats[: -len(chunk)]
                logger.debug("Selected feats after delete = {}".format(selected_feats))

        logger.debug("Update mapped importance")
        imp = imp[imp.index.isin(selected_feats)]
        self.map_raw_feature_importances(imp)

        selected_feats = list(self.mapped_importances.index)
        logger.info3("Finally selected feats = {}".format(selected_feats))
        self._selected_features = selected_feats
