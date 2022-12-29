"""Linear models for tabular datasets (GPU version)."""

import logging
from copy import copy
from typing import Sequence, Tuple, Union

import cudf
import cupy as cp
import dask.array as da
from cuml.dask.linear_model import ElasticNet as daskElasticNet
from cuml.dask.linear_model import Lasso as daskLasso
from cuml.linear_model import ElasticNet, Lasso, LogisticRegression

from lightautoml_gpu.dataset.gpu.gpu_dataset import CupyDataset, DaskCudfDataset
from lightautoml_gpu.tasks.base import Task
from copy import deepcopy

from lightautoml_gpu.ml_algo.torch_based.gpu.linear_model_cupy import (
    TorchBasedLinearEstimator,
    TorchBasedLinearRegression,
    TorchBasedLogisticRegression,
)
from lightautoml_gpu.ml_algo.torch_based.gpu.linear_model_distributed import (
    TorchBasedLinearRegression as TLinR_dask,
)
from lightautoml_gpu.ml_algo.torch_based.gpu.linear_model_distributed import (
    TorchBasedLogisticRegression as TLR_dask,
)

from ..linear_sklearn import LinearLBFGS

from ..torch_based.linear_model import TorchBasedLinearRegression as TLinR_CPU, TorchBasedLogisticRegression as TLR_CPU

from lightautoml_gpu.validation.base import TrainValidIterator

from .base_gpu import TabularDatasetGpu, TabularMLAlgoGPU

logger = logging.getLogger(__name__)

LinearEstimator = Union[LogisticRegression, ElasticNet, Lasso]


class LinearLBFGSGPU(TabularMLAlgoGPU):
    """LBFGS L2 regression based on torch.


    default_params:

        - cs: List of regularization coefficients.
        - max_iter: Maximum iterations of L-BFGS.
        - tol: The tolerance for the stopping criteria.
        - early_stopping: Maximum rounds without improving.

    freeze_defaults:

        - ``True`` :  params may be rewrited depending on dataset.
        - ``False``:  params may be changed only manually or with tuning.

    timer: :class:`~lightautoml_gpu.utils.timer.Timer` instance or ``None``.

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
        params["loss"] = self.task.losses["torch_gpu"].loss
        params["metric"] = self.task.losses["torch_gpu"].metric_func
        model = None
        if self.task.name in ["binary", "multiclass", "multilabel"]:
            if self.task.device == "gpu":
                model = TorchBasedLogisticRegression(
                    output_size=self.n_classes, **params
                )
            elif self.task.device == "mgpu":
                if self.parallel_folds:
                    model = TorchBasedLogisticRegression(
                        output_size=self.n_classes, **params
                    )
                else:
                    model = TLR_dask(
                        output_size=self.n_classes, gpu_ids=self.gpu_ids, **params
                    )
            else:
                raise ValueError("Device not supported")
        elif self.task.name == "reg":
            if self.task.device == "gpu":
                model = TorchBasedLinearRegression(output_size=1, **params)
            elif self.task.device == "mgpu":
                if self.parallel_folds:
                    model = TorchBasedLinearRegression(output_size=1, **params)
                else:
                    model = TLinR_dask(output_size=1, gpu_ids=self.gpu_ids, **params)
            else:
                raise ValueError("Device not supported")
        elif self.task.name == "multi:reg":
            if self.task.device == "gpu":
                model = TorchBasedLinearRegression(output_size=self.n_classes, **params)
            elif self.task.device == "mgpu":
                if self.parallel_folds:
                    model = TorchBasedLinearRegression(output_size=self.n_classes, **params)
                else:
                    model = TLinR_dask(output_size=self.n_classes, gpu_ids=self.gpu_ids, **params)
        else:
            raise ValueError("Task not supported")
        return model

    def init_params_on_input(self, train_valid_iterator: TrainValidIterator) -> dict:

        suggested_params = copy(self.default_params)
        train = train_valid_iterator.train

        suggested_params["categorical_idx"] = {}
        suggested_params["categorical_idx"]["int"] = [
            i
            for i, x in enumerate(train.features)
            if train.roles[x].name == "Category"
        ]
        suggested_params["categorical_idx"]["str"] = [
            x
            for i, x in enumerate(train.features)
            if train.roles[x].name == "Category"
        ]

        suggested_params["embed_sizes"] = ()

        if len(suggested_params["categorical_idx"]["int"]) > 0:
            if type(train) == CupyDataset:
                suggested_params["embed_sizes"] = (
                    cp.asnumpy(
                        train.data[:, suggested_params["categorical_idx"]["int"]]
                        .astype(cp.int32)
                        .max(axis=0)
                    )
                    + 1
                )
            elif type(train) == DaskCudfDataset and self.parallel_folds:
                cat = [x for x in train.features if train.roles[x].name == "Category"]
                suggested_params["embed_sizes"] = (
                    train.data[cat].astype(cp.int32).max(axis=0) + 1
                )
            else:
                suggested_params["embed_sizes"] = (
                    train.data[suggested_params["categorical_idx"]["str"]]
                    .astype(cp.int32)
                    .max(axis=0)
                    + 1
                )
            if type(train) == DaskCudfDataset:
                suggested_params["embed_sizes"] = suggested_params[
                    "embed_sizes"
                ].compute()
        suggested_params["data_size"] = train.shape[1]
        return suggested_params

    def fit_predict_single_fold(
        self, train: TabularDatasetGpu, valid: TabularDatasetGpu, dev_id: int = 0
    ) -> Tuple[TorchBasedLinearEstimator, cp.ndarray]:
        """Train on train dataset and predict on holdout dataset.

        Args:
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Target predictions for valid dataset.

        """
        train_target = train.target
        train_weights = train.weights
        valid_target = valid.target
        valid_weights = valid.weights
        train_data = train.data
        valid_data = valid.data

        if type(train) == DaskCudfDataset and self.parallel_folds:
            train_target = train_target.compute().values
            if train_weights is not None:
                train_weights = train_weights.compute().values
            valid_target = valid_target.compute().values
            if valid_weights is not None:
                valid_weights = valid_weights.compute().values
            train_data = train_data.compute().values
            valid_data = valid_data.compute().values

        model = self._infer_params()
        model.model = model.model.to(f"cuda:{dev_id}")

        model.fit(
            train_data,
            train_target,
            train_weights,
            valid_data,
            valid_target,
            valid_weights,
            dev_id,
        )
        val_pred = model.predict(valid_data, dev_id)
        return model, val_pred

    def predict_single_fold(
        self,
        model: TorchBasedLinearEstimator,
        dataset: TabularDatasetGpu,
        dev_id: int = 0,
    ) -> cp.ndarray:
        """Implements prediction on single fold.

        Args:
            model: Model uses to predict.
            dataset: ``CupyDataset`` used for prediction.

        Returns:
            Predictions for input dataset.

        """
        dataset_data = dataset.data
        if type(dataset) == DaskCudfDataset:
            dataset_data = dataset_data.compute().values

        model.model = model.model.to(f"cuda:{dev_id}")
        pred = model.predict(dataset_data, dev_id)

        return pred

    def to_cpu(self):
        CPU_CL = TLR_CPU if isinstance(self.models[0], (TorchBasedLogisticRegression, TLR_dask)) else TLinR_CPU
        models = [CPU_CL(data_size=self.models[i].data_size,
                         categorical_idx=self.models[i].categorical_idx['int'],
                         embed_sizes=cp.asnumpy(self.models[i].embed_sizes),
                         output_size=self.models[i].output_size,
                         cs=self.models[i].cs) for i in range(len(self.models))]
        for i in range(len(models)):
            models[i].model = deepcopy(self.models[i].model.to_cpu())
            models[i].loss = deepcopy(self.models[i].loss.cpu())
            models[i].metric = None

        algo = LinearLBFGS(default_params=self.default_params,
                           freeze_defaults=self.freeze_defaults,
                           timer=self.timer)
        algo.task = Task(name=self.task._name,
                         device='cpu',
                         loss=self.task.loss,
                         metric=self.task.metric_name,
                         greater_is_better=self.task.greater_is_better)
        algo.models = deepcopy(models)
        algo._features = self._features
        algo._nan_rate = self._nan_rate
        algo._name = self._name
        algo._params = deepcopy(self._params)
        return algo


class LinearL1CDGPU(TabularMLAlgoGPU):
    """Coordinate descent based on cuml implementation."""

    _name: str = "LinearElasticNet"

    _default_params = {
        "tol": 1e-3,
        "max_iter": 100,
        "cs": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 100000, 1000000],
        "early_stopping": 2,
        "l1_ratios": (1,),
        "solver": "qn",
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
                model = LogisticRegression(penalty="l1", **params)
            else:
                model = LogisticRegression(penalty="elasticnet", **params)

        elif self.task.name == "reg":
            params.pop("solver")
            if l1_ratios == (1,):
                model = Lasso(**params)
            else:
                model = ElasticNet(**params)
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
        assert "cuml" in task.losses, "Cuml loss should be defined"

        if task.name == "reg":
            suggested_params["cs"] = [1 / (2 * i) for i in suggested_params["cs"]]

        return suggested_params

    def _predict_w_model_type(self, model, data):

        if self.task.name == "binary":
            pred = None
            res = model.predict_proba(data)
            if type(res) == cudf.DataFrame:
                pred = model.predict_proba(data)[1]
            elif type(res) == cp.ndarray:
                pred = model.predict_proba(data)[:, 1]
        elif self.task.name == "reg":
            pred = model.predict(data)

        elif (self.task.name == "multiclass") or (self.task.name == "multilabel"):
            pred = model.predict_proba(data)

        else:
            raise ValueError("Task not suppoted")

        return pred

    def fit_predict_single_fold(
        self, train: TabularDatasetGpu, valid: TabularDatasetGpu
    ) -> Tuple[LinearEstimator, cp.ndarray]:
        """Train on train dataset and predict on holdout dataset.

        Args:
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Target predictions for valid dataset.

        """
        train_target = train.target
        train_weights = train.weights
        valid_target = valid.target
        valid_weights = valid.weights
        train_data = train.data
        valid_data = valid.data

        if type(train) == DaskCudfDataset:
            train_target = train_target.compute()
            if train_weights is not None:
                train_weights = train_weights.compute()
            valid_target = valid_target.compute()
            if valid_weights is not None:
                valid_weights = valid_weights.compute()
            train_data = train_data.compute()
            valid_data = valid_data.compute()

        _model, cs, l1_ratios, early_stopping = self._infer_params()

        train_target, train_weight = self.task.losses["cuml"].fw_func(
            train_target, train_weights
        )
        valid_target, valid_weight = self.task.losses["cuml"].fw_func(
            valid_target, valid_weights
        )

        model = deepcopy(_model)

        best_score = -cp.inf
        best_pred = None
        best_model = None

        metric = self.task.losses["cuml"].metric_func

        for l1_ratio in sorted(l1_ratios, reverse=True):

            try:
                model.set_params(**{"l1_ratio": l1_ratio})
            except ValueError:
                pass

            model = deepcopy(_model)

            c_best_score = -cp.inf
            c_best_pred = None
            c_best_model = None
            es = 0

            for n, c in enumerate(cs):

                try:
                    model.set_params(**{"C": c})
                except ValueError:
                    model.set_params(**{"alpha": c})

                model.fit(train_data, train_target, train_weight)
                model_coefs = model.coef_
                if type(model_coefs) == cudf.DataFrame:
                    model_coefs = model_coefs.values
                if cp.allclose(model_coefs, 0):
                    if n == (len(cs) - 1):
                        logger.warning(
                            "All model coefs are 0. Model with l1_ratio {0} is dummy".format(
                                l1_ratio
                            ),
                            UserWarning,
                        )
                    else:
                        logger.debug("C = {0} all model coefs are 0".format(c))
                        continue

                pred = self._predict_w_model_type(model, valid_data)
                score = metric(valid_target, pred, valid_weight)

                logger.debug("C = {0}, l1_ratio = {1}, score = {2}".format(c, 1, score))

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
                    logger.info("Time limit exceeded")
                    break

                if (model_coefs != 0).all():
                    logger.debug("All coefs are nonzero")
                    break

            if c_best_score >= best_score:
                best_score = c_best_score
                best_pred = deepcopy(c_best_pred)
                best_model = deepcopy(c_best_model)

            if self.timer.time_limit_exceeded():
                logger.info("Time limit exceeded")
                break
        val_pred = self.task.losses["cuml"].bw_func(best_pred)
        return best_model, val_pred

    def predict_single_fold(
        self, model: LinearEstimator, dataset: TabularDatasetGpu, part_id: int = None
    ) -> cp.ndarray:
        """Implements prediction on single fold.

        Args:
            model: Model uses to predict.
            dataset: Dataset used for prediction.

        Returns:
            Predictions for input dataset.

        """
        dataset_data = dataset.data
        if type(dataset) == DaskCudfDataset:
            assert (
                part_id is not None
            ), "predict_single_fold: partition id should be set if data is distributed"
            dataset_data = dataset_data.compute()

        pred = self.task.losses["cuml"].bw_func(
            self._predict_w_model_type(model, dataset_data)
        )

        return pred


class LinearL1CDMGPU(LinearL1CDGPU):
    def __init__(self, client, *args, **kwargs):
        self.client = client
        super().__init__(*args, **kwargs)

    def __deepcopy__(self, memo):

        new_inst = type(self).__new__(self.__class__)
        new_inst.client = self.client

        for k, v in super().__dict__.items():
            if k != "client":
                setattr(new_inst, k, deepcopy(v, memo))
        return new_inst

    def _infer_params(
        self,
    ) -> Tuple[LinearEstimator, Sequence[float], Sequence[float], int]:

        params = copy(self.params)
        l1_ratios = params.pop("l1_ratios")
        early_stopping = params.pop("early_stopping")
        cs = params.pop("cs")

        if self.task.name == "reg":

            params.pop("solver")
            if l1_ratios == (1,):
                model_class = daskLasso
            else:
                model_class = daskElasticNet

            model = model_class(**params, client=self.client)

        else:

            raise AttributeError("Task {0} not supported".format(self.task.name))

        return model, cs, l1_ratios, early_stopping, params, model_class

    def _predict_w_model_type(self, model, data):

        if self.task.name == "binary":
            pred = model.predict_proba(data)[:, 1]

        elif self.task.name == "reg":
            pred = model.predict(data, delayed=False)
            pred.compute_chunk_sizes()

        elif (self.task.name == "multiclass") or (self.task.name == "multilabel"):
            pred = model.predict_proba(data)

        else:
            raise ValueError("Task not suppoted")

        return pred

    def fit_predict_single_fold(
        self, train: TabularDatasetGpu, valid: TabularDatasetGpu
    ) -> Tuple[LinearEstimator, cp.ndarray]:
        """Train on train dataset and predict on holdout dataset.

        Args:
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Target predictions for valid dataset.

        """

        (
            _model,
            cs,
            l1_ratios,
            early_stopping,
            params,
            _model_class,
        ) = self._infer_params()
        train_target, train_weight = self.task.losses["cuml"].fw_func(
            train.target, train.weights
        )
        valid_target, valid_weight = self.task.losses["cuml"].fw_func(
            valid.target, valid.weights
        )

        model = _model

        best_score = -cp.inf
        best_pred = None
        best_model = None
        best_params = None

        metric = self.task.losses["cuml"].metric_func

        X = (
            train.data.astype(cp.float32)
            .to_dask_array(lengths=True, meta=cp.array((), dtype=cp.float32))
            .persist()
        )
        y = (
            train_target.astype(cp.float32)
            .to_dask_array(lengths=True, meta=cp.array((), dtype=cp.float32))
            .persist()
        )
        valid = (
            valid.data.astype(cp.float32)
            .to_dask_array(lengths=True, meta=cp.array((), dtype=cp.float32))
            .persist()
        )
        valid_target = valid_target.to_dask_array(
            lengths=True, meta=cp.array((), dtype=cp.float32)
        ).persist()

        for l1_ratio in sorted(l1_ratios, reverse=True):

            if l1_ratios != (1,):
                params["l1_ratio"] = l1_ratio
                model = _model_class(**params, client=self.client)

            c_best_score = -cp.inf
            c_best_pred = None
            c_best_params = None
            es = 0

            for n, c in enumerate(cs):

                params["alpha"] = c
                model = _model_class(**params, client=self.client)

                model.fit(X, y)

                if cp.allclose(model.solver.coef_, 0):
                    if n == (len(cs) - 1):
                        logger.warning(
                            "All model coefs are 0. Model with l1_ratio {0} is dummy".format(
                                l1_ratio
                            ),
                            UserWarning,
                        )
                    else:
                        logger.debug("C = {0} all model coefs are 0".format(c))
                        continue
                pred = self._predict_w_model_type(model, valid)

                score = metric(valid_target, pred, valid_weight)

                logger.debug("C = {0}, l1_ratio = {1}, score = {2}".format(c, 1, score))

                if score >= c_best_score:
                    c_best_score = score
                    c_best_pred = deepcopy(pred)
                    es = 0
                    c_best_params = deepcopy(params)
                else:
                    es += 1

                if es >= early_stopping:
                    logger.debug("Early stopping..")
                    break

                if self.timer.time_limit_exceeded():
                    logger.info("Time limit exceeded")
                    break

                if (model.solver.coef_ != 0).all():
                    logger.debug("All coefs are nonzero")
                    break

            if c_best_score >= best_score:
                best_score = c_best_score
                best_pred = deepcopy(c_best_pred)
                best_params = deepcopy(c_best_params)

            if self.timer.time_limit_exceeded():
                logger.info("Time limit exceeded")
                break

        best_model = _model_class(**best_params, client=self.client)
        best_model.fit(X, y)

        val_pred = self.task.losses["cuml"].bw_func(best_pred)
        if type(val_pred) == da.Array:
            pass
        else:
            val_pred = val_pred.values
        return best_model, val_pred
