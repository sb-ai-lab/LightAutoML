"""Wrapped RandomForest for tabular datasets."""

import logging

from copy import copy
from typing import Dict
from typing import Tuple
from typing import Union

import numpy as np

from pandas import Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from ..pipelines.selection.base import ImportanceEstimator
from ..validation.base import TrainValidIterator
from .base import TabularDataset
from .base import TabularMLAlgo
from .tuning.base import Distribution
from .tuning.base import SearchSpace


logger = logging.getLogger(__name__)

RFModel = Union[RandomForestClassifier, RandomForestRegressor]


class RandomForestSklearn(TabularMLAlgo, ImportanceEstimator):
    """Random forest algorigthm from Sklearn.

    default_params: All available parameters listed in lightgbm documentation:
        - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

    freeze_defaults:
        - ``True`` :  params may be rewritten depending on dataset.
        - ``False``:  params may be changed only manually or with tuning.
    timer: :class:`~lightautoml.utils.timer.Timer` instance or ``None``.
    """

    _name: str = "RFSklearn"

    _default_params = {
        "bootstrap": True,
        "ccp_alpha": 0.0,
        "max_depth": None,
        "max_features": "auto",
        "max_leaf_nodes": None,
        "max_samples": None,
        "min_impurity_decrease": 0.0,
        "min_impurity_split": None,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "min_weight_fraction_leaf": 0.0,
        "n_estimators": 250,
        "n_jobs": 4,
        "oob_score": False,
        "random_state": 42,
        "warm_start": False,
    }

    def _infer_params(self) -> dict:
        """Infer parameters for RF.

        Returns:
            Tuple (params, verbose).
        """
        params = copy(self.params)

        # Logging
        if "verbose" not in params:
            root_logger = logging.getLogger()
            level = root_logger.getEffectiveLevel()
            if level in (logging.CRITICAL, logging.ERROR, logging.WARNING):
                params["verbose"] = 0
            else:
                params["verbose"] = 2

        return params

    def init_params_on_input(self, train_valid_iterator: TrainValidIterator) -> dict:
        """Get model parameters depending on dataset parameters.

        Args:
            train_valid_iterator: Classic cv-iterator.

        Returns:
            Parameters of model.
        """
        rows_num = len(train_valid_iterator.train)
        features_num = len(train_valid_iterator.features)
        task = train_valid_iterator.train.task.name
        suggested_params = copy(self.default_params)

        if "criterion" not in suggested_params:
            suggested_params["criterion"] = "mse" if ((task == "reg") or (task == "multi:reg")) else "gini"

        if self.freeze_defaults:
            # if user change defaults manually - keep it
            return suggested_params

        # just for speed training
        if rows_num <= 10000:
            suggested_params["n_estimators"] = 500
        else:
            suggested_params["n_estimators"] = 250

        # say no to overfitting
        if rows_num > 10000:
            suggested_params["min_samples_leaf"] = 8 if ((task == "reg") or (task == "multi:reg")) else 16
        else:
            suggested_params["min_samples_leaf"] = 32 if ((task == "reg") or (task == "multi:reg")) else 64

        # how many features to check
        if features_num > 50:
            suggested_params["max_features"] = "sqrt"
        elif features_num > 10:
            suggested_params["max_features"] = 0.75
        else:
            suggested_params["max_features"] = 1.0

        return suggested_params

    def _get_default_search_spaces(self, suggested_params: Dict, estimated_n_trials: int) -> Dict:
        """Sample hyperparameters from suggested.

        Args:
            trial: Optuna trial object.
            suggested_params: Dict with parameters.
            estimated_n_trials: Maximum number of hyperparameter estimations.

        Returns:
            dict with sampled hyperparameters.

        """
        optimization_search_space = {}

        optimization_search_space["min_samples_leaf"] = SearchSpace(
            Distribution.INTUNIFORM,
            low=1,
            high=256,
        )

        optimization_search_space["max_depth"] = SearchSpace(
            Distribution.INTUNIFORM,
            low=1,
            high=10,
        )

        return optimization_search_space

    def fit_predict_single_fold(self, train: TabularDataset, valid: TabularDataset) -> Tuple[RFModel, np.ndarray]:
        """Implements training and prediction on single fold.

        Args:
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Tuple (model, predicted_values)
        """
        params = self._infer_params()

        task = self.task.name

        if (task == "reg") or (task == "multi:reg"):
            model = RandomForestRegressor(**params)
            model.fit(train.data, train.target, train.weights)
            val_pred = model.predict(valid.data)
        else:
            model = RandomForestClassifier(**params)
            model.fit(train.data, train.target, train.weights)
            val_pred = model.predict_proba(valid.data)
            if task == "binary":
                val_pred = val_pred[:, 1]
            elif task == "multilabel":
                val_pred = np.moveaxis(np.array(val_pred)[:, :, 1], 1, 0)

        metric = self.task.losses["sklearn"].metric_func
        score = metric(valid.target, val_pred, valid.weights)
        logger.info2("Score for RF model: {:5f}".format(score))

        return model, val_pred

    def predict_single_fold(self, model: RFModel, dataset: TabularDataset) -> np.ndarray:
        """Predict target values for dataset.

        Args:
            model: Lightgbm object.
            dataset: Test Dataset.

        Return:
            Predicted target values.
        """
        task = self.task.name
        if (task == "reg") or (task == "multi:reg"):
            pred = model.predict(dataset.data)
        else:
            pred = model.predict_proba(dataset.data)
            if task == "binary":
                pred = pred[:, 1]
            elif task == "multilabel":
                pred = np.moveaxis(np.array(pred)[:, :, 1], 1, 0)

        return pred

    def get_features_score(self) -> Series:
        """Computes feature importance as mean values of feature importance provided by RandomForest per all models.

        Returns:
            Series with feature importances.
        """
        imp = 0
        for model in self.models:
            imp = imp + model.feature_importances_

        imp = imp / len(self.models)

        return Series(imp, index=self.features).sort_values(ascending=False)

    def fit(self, train_valid: TrainValidIterator):
        """Just to be compatible with :class:`~lightautoml.pipelines.selection.base.ImportanceEstimator`.

        Args:
            train_valid: Classic cv-iterator.
        """
        self.fit_predict(train_valid)
