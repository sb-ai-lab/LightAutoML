"""Wrapped xgboost for tabular datasets."""

import logging
from copy import copy, deepcopy
from typing import Callable, Dict, Optional, Tuple

from dask.distributed import Client
from dask_cuda import LocalCUDACluster

import cudf
import cupy as cp
import dask_cudf
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from torch.cuda import device_count
from xgboost import dask as dxgb

from lightautoml.tasks.base import Task
from copy import deepcopy

from lightautoml.dataset.gpu.gpu_dataset import CudfDataset, DaskCudfDataset
from lightautoml.ml_algo.tuning.base import Uniform
from lightautoml.pipelines.selection.base import ImportanceEstimator
from lightautoml.validation.base import TrainValidIterator

from .base_gpu import TabularDatasetGpu, TabularMLAlgoGPU

from ..boost_xgb import BoostXGB as BoosterCPU

logger = logging.getLogger(__name__)


class BoostXGB(TabularMLAlgoGPU, ImportanceEstimator):
    """Gradient boosting on decision trees from LightGBM library.

    default_params: All available parameters listed in lightgbm documentation:

        - https://lightgbm.readthedocs.io/en/latest/Parameters.html

    freeze_defaults:

        - ``True`` :  params may be rewritten depending on dataset.
        - ``False``:  params may be changed only manually or with tuning.

    timer: :class:`~lightautoml.utils.timer.Timer` instance or ``None``.

    """

    _name: str = "XGB"

    _default_params = {
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor",
        "task": "train",
        "learning_rate": 0.05,
        "max_leaves": 128,
        "max_depth": 0,
        "verbosity": 0,
        "reg_alpha": 1,
        "reg_lambda": 0.0,
        "gamma": 0.0,
        "max_bin": 255,
        "n_estimators": 3000,
        "early_stopping_rounds": 100,
        "random_state": 42,
    }

    def _infer_params(
        self,
    ) -> Tuple[dict, int, int, int, Optional[Callable], Optional[Callable]]:
        """Infer all parameters in lightgbm format.

        Returns:
            Tuple (params, num_trees, early_stopping_rounds, verbose_eval, fobj, feval).
            About parameters: https://lightgbm.readthedocs.io/en/latest/_modules/lightgbm/engine.html

        """
        params = copy(self.params)
        early_stopping_rounds = params.pop("early_stopping_rounds")
        num_trees = params.pop("n_estimators")

        root_logger = logging.getLogger()
        level = root_logger.getEffectiveLevel()

        if level in (logging.CRITICAL, logging.ERROR, logging.WARNING):
            verbose_eval = False
        elif level == logging.INFO:
            verbose_eval = 100
        else:
            verbose_eval = 10

        # get objective params
        loss = self.task.losses["xgb"]
        params["objective"] = loss.fobj_name
        fobj = loss.fobj

        # get metric params
        params["metric"] = loss.metric_name
        feval = loss.feval

        params["num_class"] = 1 if (self.task.name == "multi:reg") or (self.task.name == "multilabel") else self.n_classes
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

        rows_num = len(train_valid_iterator.train)

        task = train_valid_iterator.train.task.name

        suggested_params = copy(self.default_params)

        if self.freeze_defaults:
            # if user change defaults manually - keep it
            return suggested_params

        if task == "reg":
            suggested_params = {"learning_rate": 0.05, "max_leaves": 32}

        if rows_num <= 10000:
            init_lr = 0.01
            ntrees = 3000
            es = 200

        elif rows_num <= 20000:
            init_lr = 0.02
            ntrees = 3000
            es = 200

        elif rows_num <= 100000:
            init_lr = 0.03
            ntrees = 1200
            es = 200
        elif rows_num <= 300000:
            init_lr = 0.04
            ntrees = 2000
            es = 100
        else:
            init_lr = 0.05
            ntrees = 2000
            es = 100

        if rows_num > 300000:
            suggested_params["max_leaves"] = 128 if task == "reg" else 244
        elif rows_num > 100000:
            suggested_params["max_leaves"] = 64 if task == "reg" else 128
        elif rows_num > 50000:
            suggested_params["max_leaves"] = 32 if task == "reg" else 64
            # params['reg_alpha'] = 1 if task == 'reg' else 0.5
        elif rows_num > 20000:
            suggested_params["max_leaves"] = 32 if task == "reg" else 32
            suggested_params["reg_alpha"] = 0.5 if task == "reg" else 0.0
        elif rows_num > 10000:
            suggested_params["max_leaves"] = 32 if task == "reg" else 64
            suggested_params["reg_alpha"] = 0.5 if task == "reg" else 0.2
        elif rows_num > 5000:
            suggested_params["max_leaves"] = 24 if task == "reg" else 32
            suggested_params["reg_alpha"] = 0.5 if task == "reg" else 0.5
        else:
            suggested_params["max_leaves"] = 16 if task == "reg" else 16
            suggested_params["reg_alpha"] = 1 if task == "reg" else 1

        suggested_params["learning_rate"] = init_lr
        suggested_params["n_estimators"] = ntrees
        suggested_params["early_stopping_rounds"] = es

        return suggested_params

    def _get_default_search_spaces(
        self, suggested_params: Dict, estimated_n_trials: int
    ) -> Dict:
        """Sample hyperparameters from suggested.

        Args:
            suggested_params: Dict with parameters.
            estimated_n_trials: Maximum number of hyperparameter estimations.

        Returns:
            dict with sampled hyperparameters.

        """
        optimization_search_space = {}

        optimization_search_space["max_depth"] = Uniform(low=3, high=7, q=1)

        optimization_search_space["max_leaves"] = Uniform(low=16, high=255, q=1)

        if estimated_n_trials > 30:
            optimization_search_space["min_child_weight"] = Uniform(low=1e-8, high=10.0, log=True)

        if estimated_n_trials > 100:
            optimization_search_space["reg_alpha"] = Uniform(low=1e-8, high=10.0, log=True)
            optimization_search_space["reg_lambda"] = Uniform(low=1e-8, high=10.0, log=True)

        return optimization_search_space

    def fit_predict_single_fold(
        self, train: TabularDatasetGpu, valid: TabularDatasetGpu, dev_id: int = 0
    ) -> Tuple[xgb.Booster, np.ndarray]:
        """Implements training and prediction on single fold.

        Args:
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Tuple (model, predicted_values)

        """
        train = train.to_cudf()
        valid = valid.to_cudf()
        train_target = train.target
        train_weights = train.weights
        valid_target = valid.target
        valid_weights = valid.weights
        train_data = train.data
        valid_data = valid.data
        with cp.cuda.Device(dev_id):
            if type(train) == DaskCudfDataset:
                train_target = train_target.compute()
                if train_weights is not None:
                    train_weights = train_weights.compute()
                valid_target = valid_target.compute()
                if valid_weights is not None:
                    valid_weights = valid_weights.compute()
                train_data = train_data.compute()
                valid_data = valid_data.compute()
            elif type(train) == CudfDataset:
                train_target = train_target.copy()
                if train_weights is not None:
                    train_weights = train_weights.copy()
                valid_target = valid_target.copy()
                if valid_weights is not None:
                    valid_weights = valid_weights.copy()
                train_data = train_data.copy()
                valid_data = valid_data.copy()
            elif type(train_target) == cp.ndarray:
                train_target = cp.copy(train_target)
                if train_weights is not None:
                    train_weights = cp.copy(train_weights)
                valid_target = cp.copy(valid_target)
                if valid_weights is not None:
                    valid_weights = cp.copy(valid_weights)
                train_data = cp.copy(train_data)
                valid_data = cp.copy(valid_data)
            else:
                raise NotImplementedError(
                    "given type of input is not implemented:"
                    + str(type(train_target))
                    + "class:"
                    + str(self._name)
                )

        (
            params,
            num_trees,
            early_stopping_rounds,
            verbose_eval,
            fobj,
            feval,
        ) = self._infer_params()
        train_target, train_weight = self.task.losses["xgb"].fw_func(
            train_target, train_weights
        )
        valid_target, valid_weight = self.task.losses["xgb"].fw_func(
            valid_target, valid_weights
        )

        xgb_train = xgb.DMatrix(train_data, label=train_target, weight=train_weight)

        xgb_valid = xgb.DMatrix(valid_data, label=valid_target, weight=valid_weight)
        params["gpu_id"] = dev_id
        model = xgb.train(
            params,
            xgb_train,
            num_boost_round=num_trees,
            evals=[(xgb_train, "train"), (xgb_valid, "valid")],
            obj=fobj,
            feval=feval,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
        )
        val_pred = model.inplace_predict(valid_data)
        val_pred = self.task.losses["xgb"].bw_func(val_pred)

        with cp.cuda.Device(0):
            val_pred = cp.copy(val_pred)
        return model, val_pred

    def predict_single_fold(
        self, model: xgb.Booster, dataset: TabularDatasetGpu
    ) -> np.ndarray:
        """Predict target values for dataset.

        Args:
            model: Lightgbm object.
            dataset: Test Dataset.

        Return:
            Predicted target values.

        """
        dataset_data = dataset.data
        if type(dataset) == DaskCudfDataset:
            dataset_data = dataset_data.compute()

        pred = self.task.losses["xgb"].bw_func(model.inplace_predict(dataset_data))

        return pred

    def get_features_score(self) -> pd.Series:
        """Computes feature importance as mean values of feature importance provided by lightgbm per all models.

        Returns:
            Series with feature importances.

        """

        # FIRST SORT TO FEATURES AND THEN SORT BACK TO IMPORTANCES - BAD
        imp = 0
        for model in self.models:
            val = model.get_score(importance_type="gain")
            sorted_list = [
                0.0 if val.get(i) is None else val.get(i) for i in self.features
            ]
            scores = np.array(sorted_list)
            imp = imp + scores

        imp = imp / len(self.models)

        return pd.Series(imp, index=self.features).sort_values(ascending=False)

    def fit(self, train_valid: TrainValidIterator):
        """Just to be compatible with :class:`~lightautoml.pipelines.selection.base.ImportanceEstimator`.

        Args:
            train_valid: Classic cv-iterator.

        """
        self.fit_predict(train_valid)

    def to_cpu(self):
        print("XGB:", self.__dict__)
        print("XGB model type:", self.models[0].__class__.__name__)
        print("XGB model:", self.models[0].__dict__)
        models = deepcopy(self.models)
        for i in range(len(models)):
            models[i].set_param({"predictor": "cpu_predictor"})

        task = Task(name=self.task._name,
                    device='cpu',
                    metric=self.task.metric_name,
                    greater_is_better=self.task.greater_is_better)

        algo = BoosterCPU()
        algo.models = deepcopy(models)
        algo.task = task

        return algo


class BoostXGB_dask(BoostXGB):
    def __init__(self, client, *args, **kwargs):

        if client is None:
            self.client = Client(LocalCUDACluster(
                                     rmm_managed_memory=True,
                                     protocol='ucx',
                                     enable_nvlink=True,
                                     memory_limit="30GB"))
            self.client.run(cudf.set_allocator, 'managed')
        else:
            self.client = client
        super().__init__(*args, **kwargs)

    def __deepcopy__(self, memo):

        new_inst = type(self).__new__(self.__class__)
        new_inst.client = None

        for k, v in super().__dict__.items():
            if k != "client":
                setattr(new_inst, k, deepcopy(v, memo))
        return new_inst

    def fit_predict_single_fold(
        self, train: DaskCudfDataset, valid: DaskCudfDataset, dev_id: int = 0
    ) -> Tuple[dxgb.Booster, np.ndarray]:
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

        train_target, train_weight = self.task.losses["xgb"].fw_func(
            train.target, train.weights
        )

        valid_target, valid_weight = self.task.losses["xgb"].fw_func(
            valid.target, valid.weights
        )
        if type(train) is not DaskCudfDataset:
            train = train.to_daskcudf(nparts=torch.cuda.device_count())
            valid = valid.to_daskcudf(nparts=torch.cuda.device_count())
            train_target = dask_cudf.from_cudf(
                cudf.DataFrame(train_target),
                #cudf.Series(train_target), 
                npartitions=torch.cuda.device_count()
            )
            valid_target = dask_cudf.from_cudf(
                cudf.DataFrame(valid_target),
                #cudf.Series(valid_target), 
                npartitions=torch.cuda.device_count()
            )
        xgb_train = dxgb.DaskDeviceQuantileDMatrix(
            self.client, train.data, label=train_target, weight=train_weight
        )
        xgb_valid = dxgb.DaskDeviceQuantileDMatrix(
            self.client, valid.data, label=valid_target, weight=valid_weight
        )
        model = dxgb.train(
            self.client,
            params,
            xgb_train,
            num_boost_round=num_trees,
            evals=[(xgb_train, "train"), (xgb_valid, "valid")],
            obj=fobj,
            feval=feval,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
        )
        val_pred = dxgb.inplace_predict(self.client, model, valid.data)
        val_pred = self.task.losses["xgb"].bw_func(val_pred)

        return model, val_pred

    def predict_single_fold(
        self, model: dxgb.Booster, dataset: TabularDatasetGpu
    ) -> np.ndarray:
        """Predict target values for dataset.

        Args:
            model: Lightgbm object.
            dataset: Test Dataset.

        Return:
            Predicted target values.

        """
        if type(dataset) is not DaskCudfDataset:
            dataset = dataset.to_daskcudf(nparts=device_count())
        pred = self.task.losses["xgb"].bw_func(
            dxgb.inplace_predict(self.client, model, dataset.data)
        )

        return pred

    def get_features_score(self) -> pd.Series:
        """Computes feature importance as mean values of feature importance provided by lightgbm per all models.

        Returns:
            Series with feature importances.

        """

        # FIRST SORT TO FEATURES AND THEN SORT BACK TO IMPORTANCES - BAD
        imp = 0
        for model in self.models:
            val = model["booster"].get_score(importance_type="gain")
            sorted_list = [
                0.0 if val.get(i) is None else val.get(i) for i in self.features
            ]
            scores = np.array(sorted_list)
            imp = imp + scores

        imp = imp / len(self.models)

        return pd.Series(imp, index=self.features).sort_values(ascending=False)
