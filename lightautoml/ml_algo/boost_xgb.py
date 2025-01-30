"""Wrapped XGBoost for tabular datasets."""

import logging

from contextlib import redirect_stdout
from copy import copy
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

import xgboost
import numpy as np

from pandas import Series

from ..pipelines.selection.base import ImportanceEstimator
from ..utils.logging import LoggerStream
from ..validation.base import TrainValidIterator
from .base import TabularDataset
from .base import TabularMLAlgo
from .tuning.base import Uniform, Choice


logger = logging.getLogger(__name__)


class BoostXGB(TabularMLAlgo, ImportanceEstimator):
    """Gradient boosting on decision trees from XGBoost library.

    default_params: All available parameters listed in XGBoost documentation:

        - https://xgboost.readthedocs.io/en/stable/parameter.html

    freeze_defaults:

        - ``True`` :  params may be rewritten depending on dataset.
        - ``False``:  params may be changed only manually or with tuning.

    timer: :class:`~lightautoml.utils.timer.Timer` instance or ``None``.

    """

    _name: str = "XGBoost"

    _default_params = {
        "n_estimators": 3000,
        "early_stopping_rounds": 100,
        "seed": 42,
        "verbose_eval": 100,
    }

    def _infer_params(
        self,
    ) -> Tuple[dict, int, int, int, Optional[Callable], Optional[Callable]]:
        """Infer all parameters in lightgbm format.

        Returns:
            Tuple (params, num_trees, early_stopping_rounds, fobj, feval).
            About parameters: https://xgboost.readthedocs.io/en/stable/parameter.html

        """
        # TODO: Check how it works with custom tasks
        params = copy(self.params)
        early_stopping_rounds = params.pop("early_stopping_rounds")
        num_trees = params.pop("n_estimators")
        verbose_eval = params.pop("verbose_eval")

        # get objective params
        loss = self.task.losses["xgb"]
        params["objective"] = loss.fobj_name
        fobj = loss.fobj

        # # get metric params
        params["eval_metric"] = loss.metric_name
        feval = loss.feval

        params["num_class"] = self.n_classes
        # add loss and tasks params if defined
        params = {**params, **loss.fobj_params, **loss.metric_params}

        return params, num_trees, early_stopping_rounds, verbose_eval, fobj, feval

    def init_params_on_input(self, train_valid_iterator: TrainValidIterator) -> dict:
        """Get model parameters depending on dataset parameters.

        Args:
            train_valid_iterator: Classic cv-iterator.

        Returns:
            Parameters of model.

        """
        suggested_params = copy(self.default_params)

        if self.freeze_defaults:
            # if user change defaults manually - keep it
            return suggested_params

        return suggested_params

    def _get_default_search_spaces(self, suggested_params: Dict, estimated_n_trials: int) -> Dict:
        """Sample hyperparameters from suggested.

        Args:
            suggested_params: Dict with parameters.
            estimated_n_trials: Maximum number of hyperparameter estimations.

        Returns:
            dict with sampled hyperparameters.

        """
        optimization_search_space = {}

        optimization_search_space["colsample_bytree"] = Choice(options=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        optimization_search_space["subsample"] = Choice(options=[0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
        optimization_search_space["max_depth"] = Choice(options=[5, 7, 9, 11, 13, 15, 17])
        optimization_search_space["learning_rate"] = Choice(options=[0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02])

        if estimated_n_trials > 30:
            optimization_search_space["min_child_weight"] = Uniform(low=1, high=300, q=1, log=False)

            optimization_search_space["reg_alpha"] = Uniform(
                low=1e-3,
                high=10.0,
                log=True,
            )
            optimization_search_space["reg_lambda"] = Uniform(
                low=1e-3,
                high=10.0,
                log=True,
            )

        return optimization_search_space

    def fit_predict_single_fold(
        self, train: TabularDataset, valid: TabularDataset
    ) -> Tuple[xgboost.Booster, np.ndarray]:
        """Implements training and prediction on single fold.

        Args:
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Tuple (model, predicted_values)

        """
        (
            params,
            num_trees,
            early_stopping_rounds,
            verbose_eval,
            fobj,
            feval,
        ) = self._infer_params()

        train_target, train_weight = self.task.losses["xgb"].fw_func(train.target, train.weights)
        valid_target, valid_weight = self.task.losses["xgb"].fw_func(valid.target, valid.weights)

        dtrain = xgboost.DMatrix(train.data, label=train_target, weight=train_weight)
        dval = xgboost.DMatrix(valid.data, label=valid_target, weight=valid_weight)

        with redirect_stdout(LoggerStream(logger)):
            early_stopping_params = {
                "rounds": early_stopping_rounds,
                "data_name": "valid",
            }

            if feval is not None:
                early_stopping_params["metric_name"] = "Opt_metric"
                early_stopping_params["maximize"] = feval.greater_is_better

            early_stopping = xgboost.callback.EarlyStopping(**early_stopping_params)

            model = xgboost.train(
                params=params,
                dtrain=dtrain,
                verbose_eval=verbose_eval,
                num_boost_round=num_trees,
                evals=[(dval, "valid")],
                obj=fobj,
                custom_metric=feval,
                callbacks=[early_stopping],
            )

        val_pred = model.predict(data=dval)
        val_pred = self.task.losses["xgb"].bw_func(val_pred)

        return model, val_pred

    def predict_single_fold(self, model: xgboost.Booster, dataset: TabularDataset) -> np.ndarray:
        """Predict target values for dataset.

        Args:
            model: Xgboost object.
            dataset: Test Dataset.

        Returns:
            Predicted target values.

        """
        pred = self.task.losses["xgb"].bw_func(model.predict(xgboost.DMatrix(dataset.data)))

        return pred

    def get_features_score(self) -> Series:
        """Computes feature importance as mean values of feature importance provided by xgboost per all models.

        Returns:
            Series with feature importances.

        """
        imp = 0
        for model in self.models:
            imp = imp + model.feature_importance(importance_type="gain")

        imp = imp / len(self.models)

        return Series(imp, index=self.features).sort_values(ascending=False)

    def fit(self, train_valid: TrainValidIterator):
        """Just to be compatible with :class:`~lightautoml.pipelines.selection.base.ImportanceEstimator`.

        Args:
            train_valid: Classic cv-iterator.

        """
        self.fit_predict(train_valid)
