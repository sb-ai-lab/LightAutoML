"""Wrapped Catboost for tabular datasets."""

import logging

from copy import copy
from typing import Callable
from typing import Dict
from typing import Tuple
from typing import Union

import catboost as cb
import numpy as np

from pandas import Series

from ..dataset.np_pd_dataset import CSRSparseDataset
from ..dataset.np_pd_dataset import NumpyDataset
from ..dataset.np_pd_dataset import PandasDataset
from ..pipelines.selection.base import ImportanceEstimator
from ..pipelines.utils import get_columns_by_role
from ..utils.logging import LoggerStream
from ..validation.base import TrainValidIterator
from .base import TabularMLAlgo
from .tuning.base import Distribution
from .tuning.base import SearchSpace


logger = logging.getLogger(__name__)
TabularDataset = Union[NumpyDataset, CSRSparseDataset, PandasDataset]


class BoostCB(TabularMLAlgo, ImportanceEstimator):
    """Gradient boosting on decision trees from catboost library.

    All available parameters listed in CatBoost documentation:

        - https://catboost.ai/docs/concepts/python-reference_parameters-list.html#python-reference_parameters-list

    freeze_defaults:

        - ``True`` :  params may be rewritten depending on dataset.
        - ``False``:  params may be changed only manually or with tuning.

    ``timer``: :class:`~lightautoml.utils.timer.Timer` instance or ``None``.

    """

    _name: str = "CatBoost"

    _default_params = {
        "task_type": "CPU",
        "thread_count": 4,
        "random_seed": 42,
        "num_trees": 3000,
        "learning_rate": 0.03,
        "l2_leaf_reg": 1e-2,
        "bootstrap_type": "Bernoulli",
        # "bagging_temperature": 1,
        "grow_policy": "SymmetricTree",
        "max_depth": 5,
        "min_data_in_leaf": 1,
        "one_hot_max_size": 10,
        "fold_permutation_block": 1,
        "boosting_type": "Plain",
        "boost_from_average": True,
        "od_type": "Iter",
        "od_wait": 100,
        "max_bin": 32,
        "feature_border_type": "GreedyLogSum",
        "nan_mode": "Min",
        # "silent": False,
        "verbose": True,
        "allow_writing_files": False,
    }

    def _infer_params(self) -> Tuple[dict, int, int, Callable, Callable]:
        """Infer all parameters.

        Returns:
            Tuple (params, num_trees, early_stopping_rounds, fobj, feval).

        """

        params = copy(self.params)
        early_stopping_rounds = params.pop("od_wait")
        num_trees = params.pop("num_trees")

        loss = self.task.losses["cb"]
        fobj = loss.fobj_name
        feval = loss.metric_name

        if fobj not in ["RMSE", "LogLoss", "CrossEntropy", "Quantile", "MAE", "MAPE"]:
            params.pop("boost_from_average")

        # TODO: what if parameter of metric occurs in list of parameters of loss -> intersection...

        return params, num_trees, early_stopping_rounds, fobj, feval

    def init_params_on_input(self, train_valid_iterator: TrainValidIterator) -> dict:
        """Get model parameters depending on input dataset parameters.

        Args:
            train_valid_iterator: Classic cv-iterator.

        Returns:
            Parameters of model.

        """

        rows_num = len(train_valid_iterator.train)
        dataset = train_valid_iterator.train
        self.task = train_valid_iterator.train.task

        if train_valid_iterator.train.dataset_type != "CSRSparseDataset":
            self._nan_rate = train_valid_iterator.train.to_pandas().nan_rate()

        try:
            self._le_cat_features = getattr(self, "_le_cat_features")
        except AttributeError:
            self._le_cat_features = get_columns_by_role(dataset, "Category", label_encoded=True)

        try:
            self._text_features = getattr(self, "_text_features")
        except AttributeError:
            self._text_features = get_columns_by_role(dataset, "Text")

        suggested_params = copy(self.default_params)

        if self.freeze_defaults:
            return suggested_params

        init_lr, ntrees, es = 0.01, 2000, 100
        if self.task.name == "binary":
            # TODO: with an increase in the number of columns, the learning rate decreases
            if rows_num <= 6000:
                init_lr = 0.02
                ntrees = 500

            elif rows_num <= 20000:
                init_lr = 0.035
                ntrees = 5000

            elif rows_num <= 50000:
                init_lr = 0.03
                ntrees = 5000

            elif rows_num <= 60000:
                init_lr = 0.05
                ntrees = 2000

            elif rows_num <= 100000:
                init_lr = 0.045
                ntrees = 1500

            elif rows_num <= 150000:
                init_lr = 0.045
                ntrees = 3000

            elif rows_num <= 300000:
                init_lr = 0.045
                ntrees = 2000

            else:
                init_lr = 0.05
                ntrees = 3000

        elif self.task.name == "multiclass":
            init_lr = 0.03
            ntrees = 4000

            if rows_num <= 100000:
                ntrees = 3000

            elif rows_num <= 50000:
                ntrees = 3000

            elif rows_num <= 10000:
                ntrees = 3000

        elif self.task.name == "reg":
            init_lr = 0.05
            ntrees = 2000
            es = 300

        suggested_params["learning_rate"] = init_lr
        suggested_params["num_trees"] = ntrees
        suggested_params["od_wait"] = es

        return suggested_params

    def _get_default_search_spaces(self, suggested_params: Dict, estimated_n_trials: int) -> Dict:
        """Sample hyperparameters from suggested.

        Args:
            trial: Optuna trial object.
            suggested_params: Dict with parameters.
            estimated_n_trials: Maximum number of hyperparameter estimation.

        Returns:
            Dict with sampled hyperparameters.

        """

        optimization_search_space = {}

        try:
            nan_rate = getattr(self, "_nan_rate")
        except AttributeError:
            nan_rate = 0

        optimization_search_space["max_depth"] = SearchSpace(Distribution.INTUNIFORM, low=3, high=7)

        if nan_rate > 0:
            optimization_search_space["nan_mode"] = SearchSpace(Distribution.CHOICE, choices=["Max", "Min"])

        if estimated_n_trials > 20:
            optimization_search_space["l2_leaf_reg"] = SearchSpace(
                Distribution.LOGUNIFORM,
                low=1e-8,
                high=10.0,
            )

            # optimization_search_space['bagging_temperature'] = trial.suggest_loguniform(
            #     name='bagging_temperature',
            #     low=0.01,
            #     high=10.0,
            # )

        if estimated_n_trials > 50:
            optimization_search_space["min_data_in_leaf"] = SearchSpace(Distribution.INTUNIFORM, low=1, high=20)

            # the only case when used this parameter is when categorical columns more than 0
            if len(self._le_cat_features) > 0:
                optimization_search_space["one_hot_max_size"] = SearchSpace(Distribution.INTUNIFORM, low=3, high=10)

        return optimization_search_space

    def _get_pool(self, dataset: TabularDataset):

        try:
            self._le_cat_features = getattr(self, "_le_cat_features")
        except AttributeError:
            self._le_cat_features = get_columns_by_role(dataset, "Category", label_encoded=True)
        self._le_cat_features = self._le_cat_features if self._le_cat_features else None

        try:
            self._text_features = getattr(self, "_text_features")
        except AttributeError:
            self._text_features = get_columns_by_role(dataset, "Text")
        self._text_features = self._text_features if self._text_features else None

        dataset = dataset.to_pandas()
        data = dataset.data
        dtypes = data.dtypes.to_dict()
        if self._le_cat_features:
            dtypes = {**dtypes, **{x: "int" for x in self._le_cat_features}}
        # for future
        if self._text_features:
            dtypes = {**dtypes, **{x: "int" for x in self._text_features}}

        data = data.astype(dtypes)
        if self._le_cat_features:
            # copy was made in prev astype
            data.astype({x: "category" for x in self._le_cat_features}, copy=False)

        if dataset.target is not None:
            target, weights = self.task.losses["cb"].fw_func(dataset.target, dataset.weights)
        else:
            target, weights = dataset.target, dataset.weights

        pool = cb.Pool(
            data,
            label=target,
            weight=weights,
            feature_names=dataset.features,
            cat_features=self._le_cat_features,
            text_features=self._text_features,
        )

        return pool

    def fit_predict_single_fold(self, train: TabularDataset, valid: TabularDataset) -> Tuple[cb.CatBoost, np.ndarray]:
        """Implements training and prediction on single fold.

        Args:
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Tuple (model, predicted_values).

        """
        params, num_trees, early_stopping_rounds, fobj, feval = self._infer_params()

        cb_train = self._get_pool(train)
        cb_valid = self._get_pool(valid)

        model = cb.CatBoost(
            {
                **params,
                **{
                    "num_trees": num_trees,
                    "objective": fobj,
                    "eval_metric": feval,
                    "od_wait": early_stopping_rounds,
                },
            }
        )

        model.fit(cb_train, eval_set=cb_valid, log_cout=LoggerStream(logger, verbose_eval=100))

        val_pred = self._predict(model, cb_valid, params)

        return model, val_pred

    def predict_single_fold(self, model: cb.CatBoost, dataset: TabularDataset) -> np.ndarray:
        """Predict of target values for dataset.

        Args:
            model: CatBoost object.
            dataset: Test dataset.

        Return:
            Predicted target values.

        """

        params = self._infer_params()[0]
        cb_test = self._get_pool(dataset)

        pred = self._predict(model, cb_test, params)

        return pred

    def get_features_score(self) -> Series:
        """Computes feature importance.

        Computes as mean values of feature importance, provided by CatBoost (PredictionValuesChange), per all models.

        Returns:
            Series with feature importances.

        """
        # TODO: Not worked if used text features, there is a ticket in their github
        assert self.is_fitted, "Model must be fitted to compute importance."
        imp = 0
        for model in self.models:
            imp = imp + model.get_feature_importance(
                type="FeatureImportance",
                prettified=False,
                thread_count=self.params["thread_count"],
            )

        imp = imp / len(self.models)

        return Series(imp, index=self.features).sort_values(ascending=False)

    def fit(self, train_valid: TrainValidIterator):
        """Just to be compatible with :class:`~lightautoml.pipelines.selection.base.ImportanceEstimator`.

        Args:
            train_valid: Classic cv-iterator.

        """
        self.fit_predict(train_valid)

    def _predict(self, model: cb.CatBoost, pool: cb.Pool, params):
        pred = None
        if self.task.name == "multiclass":
            pred = model.predict(pool, prediction_type="Probability", thread_count=params["thread_count"])
        elif self.task.name == "binary":
            pred = model.predict(pool, prediction_type="Probability", thread_count=params["thread_count"])[..., 1]
        elif self.task.name == "reg":
            pred = model.predict(pool, thread_count=params["thread_count"])

        pred = self.task.losses["cb"].bw_func(pred)

        return pred
