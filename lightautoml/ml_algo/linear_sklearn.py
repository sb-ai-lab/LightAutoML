"""Linear models for tabular datasets."""

import logging

from copy import copy
from copy import deepcopy
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression

from ..dataset.np_pd_dataset import PandasDataset
from ..validation.base import TrainValidIterator
from .base import TabularDataset
from .base import TabularMLAlgo
from .torch_based.linear_model import TorchBasedLinearEstimator
from .torch_based.linear_model import TorchBasedLinearRegression
from .torch_based.linear_model import TorchBasedLogisticRegression


logger = logging.getLogger(__name__)

LinearEstimator = Union[LogisticRegression, ElasticNet, Lasso]


class LinearLBFGS(TabularMLAlgo):
    """LBFGS L2 regression based on torch.


    default_params:

        - cs: List of regularization coefficients.
        - max_iter: Maximum iterations of L-BFGS.
        - tol: The tolerance for the stopping criteria.
        - early_stopping: Maximum rounds without improving.

    freeze_defaults:

        - ``True`` :  params may be rewrited depending on dataset.
        - ``False``:  params may be changed only manually or with tuning.

    timer: :class:`~lightautoml.utils.timer.Timer` instance or ``None``.

    """

    _name: str = "LinearL2"

    _default_params = {
        "tol": 1e-6,
        "max_iter": 100,
        "cs": [
            1e-5,
            5e-5,
            1e-4,
            5e-4,
            1e-3,
            5e-3,
            1e-2,
            5e-2,
            1e-1,
            5e-1,
            1,
            5,
            10,
            50,
            100,
            500,
            1000,
            5000,
            10000,
            50000,
            100000,
        ],
        "early_stopping": 2,
    }

    def _infer_params(self) -> TorchBasedLinearEstimator:

        params = copy(self.params)
        params["loss"] = self.task.losses["torch"].loss
        params["metric"] = self.task.losses["torch"].metric_func
        if self.task.name in ["binary", "multiclass"]:
            model = TorchBasedLogisticRegression(output_size=self.n_classes, **params)
        elif self.task.name == "reg":
            model = TorchBasedLinearRegression(output_size=1, **params)
        else:
            raise ValueError("Task not supported")

        return model

    def init_params_on_input(self, train_valid_iterator: TrainValidIterator) -> dict:

        suggested_params = copy(self.default_params)
        train = train_valid_iterator.train
        suggested_params["categorical_idx"] = [
            n for (n, x) in enumerate(train.features) if train.roles[x].name == "Category"
        ]

        suggested_params["embed_sizes"] = ()
        if len(suggested_params["categorical_idx"]) > 0:
            suggested_params["embed_sizes"] = (
                train.data[:, suggested_params["categorical_idx"]].max(axis=0).astype(np.int32) + 1
            )

        suggested_params["data_size"] = train.shape[1]

        return suggested_params

    def fit_predict_single_fold(
        self, train: TabularDataset, valid: TabularDataset
    ) -> Tuple[TorchBasedLinearEstimator, np.ndarray]:
        """Train on train dataset and predict on holdout dataset.

        Args:
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Target predictions for valid dataset.

        """
        if type(train) is PandasDataset:
            train = train.to_numpy()
            valid = valid.to_numpy()

        model = self._infer_params()

        model.fit(
            train.data,
            train.target,
            train.weights,
            valid.data,
            valid.target,
            valid.weights,
        )

        val_pred = model.predict(valid.data)

        return model, val_pred

    def predict_single_fold(self, model: TorchBasedLinearEstimator, dataset: TabularDataset) -> np.ndarray:
        """Implements prediction on single fold.

        Args:
            model: Model uses to predict.
            dataset: ``NumpyDataset`` used for prediction.

        Returns:
            Predictions for input dataset.

        """
        pred = model.predict(dataset.data)

        return pred


class LinearL1CD(TabularMLAlgo):
    """Coordinate descent based on sklearn implementation."""

    _name: str = "LinearElasticNet"

    _default_params = {
        "tol": 1e-3,
        "max_iter": 100,
        "cs": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 100000, 1000000],
        "early_stopping": 2,
        "l1_ratios": (1,),
        "solver": "saga",
    }

    def _infer_params(
        self,
    ) -> Tuple[LinearEstimator, Sequence[float], Sequence[float], int]:

        params = copy(self.params)
        l1_ratios = params.pop("l1_ratios")
        early_stopping = params.pop("early_stopping")
        cs = params.pop("cs")

        if self.task.name in ["binary", "multiclass"]:

            if l1_ratios == (1,):
                model = LogisticRegression(warm_start=True, penalty="l1", **params)
            else:
                model = LogisticRegression(warm_start=True, penalty="elasticnet", **params)

        elif self.task.name == "reg":
            params.pop("solver")
            if l1_ratios == (1,):
                model = Lasso(warm_start=True, **params)
            else:
                model = ElasticNet(warm_start=True, **params)

        else:
            raise AttributeError("Task not supported")

        return model, cs, l1_ratios, early_stopping

    def init_params_on_input(self, train_valid_iterator: TrainValidIterator) -> dict:
        """Get model parameters depending on dataset parameters.

        Args:
            train_valid_iterator: Classic cv-iterator.

        Returns:
            Parameters of model.

        """

        suggested_params = copy(self.default_params)
        task = train_valid_iterator.train.task

        assert "sklearn" in task.losses, "Sklearn loss should be defined"

        if task.name == "reg":
            # suggested_params['cs'] = list(map(lambda x: 1 / (2 * x), suggested_params['cs']))
            suggested_params["cs"] = [1 / (2 * i) for i in suggested_params["cs"]]

        return suggested_params

    def _predict_w_model_type(self, model, data):

        if self.task.name == "binary":
            pred = model.predict_proba(data)[:, 1]

        elif self.task.name == "reg":
            pred = model.predict(data)

        elif self.task.name == "multiclass":
            pred = model.predict_proba(data)

        else:
            raise ValueError("Task not suppoted")

        return pred

    def fit_predict_single_fold(
        self, train: TabularDataset, valid: TabularDataset
    ) -> Tuple[LinearEstimator, np.ndarray]:
        """Train on train dataset and predict on holdout dataset.

        Args:
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Target predictions for valid dataset.

        """
        if type(train) is PandasDataset:
            train = train.to_numpy()
            valid = valid.to_numpy()

        _model, cs, l1_ratios, early_stopping = self._infer_params()

        train_target, train_weight = self.task.losses["sklearn"].fw_func(train.target, train.weights)
        valid_target, valid_weight = self.task.losses["sklearn"].fw_func(valid.target, valid.weights)

        model = deepcopy(_model)

        best_score = -np.inf
        best_pred = None
        best_model = None

        metric = self.task.losses["sklearn"].metric_func

        for l1_ratio in sorted(l1_ratios, reverse=True):

            try:
                model.set_params(**{"l1_ratio": l1_ratio})
            except ValueError:
                pass

            model = deepcopy(_model)

            c_best_score = -np.inf
            c_best_pred = None
            c_best_model = None
            es = 0

            for n, c in enumerate(cs):

                try:
                    model.set_params(**{"C": c})
                except ValueError:
                    model.set_params(**{"alpha": c})

                model.fit(train.data, train_target, train_weight)

                if np.allclose(model.coef_, 0):
                    if n == (len(cs) - 1):
                        logger.info2(
                            "All model coefs are 0. Model with l1_ratio {0} is dummy".format(l1_ratio),
                            UserWarning,
                        )
                    else:
                        logger.debug("C = {0} all model coefs are 0".format(c))
                        continue

                pred = self._predict_w_model_type(model, valid.data)
                score = metric(valid_target, pred, valid_weight)

                logger.debug("C = {0}, l1_ratio = {1}, score = {2}".format(c, 1, score))

                # TODO: check about greater and equal
                if score >= c_best_score:
                    c_best_score = score
                    c_best_pred = deepcopy(pred)
                    es = 0
                    c_best_model = deepcopy(model)
                else:
                    es += 1

                if es >= early_stopping:
                    logger.debug("Early stopping..")
                    break

                if self.timer.time_limit_exceeded():
                    logger.info3("Time limit exceeded")
                    break

                # TODO: Think about is it ok to check time inside train loop?
                if (model.coef_ != 0).all():
                    logger.debug("All coefs are nonzero")
                    break

            if c_best_score >= best_score:
                best_score = c_best_score
                best_pred = deepcopy(c_best_pred)
                best_model = deepcopy(c_best_model)

            if self.timer.time_limit_exceeded():
                logger.info3("Time limit exceeded")
                break

        val_pred = self.task.losses["sklearn"].bw_func(best_pred)

        return best_model, val_pred

    def predict_single_fold(self, model: LinearEstimator, dataset: TabularDataset) -> np.ndarray:
        """Implements prediction on single fold.

        Args:
            model: Model uses to predict.
            dataset: Dataset used for prediction.

        Returns:
            Predictions for input dataset.

        """
        pred = self.task.losses["sklearn"].bw_func(self._predict_w_model_type(model, dataset.data))

        return pred
