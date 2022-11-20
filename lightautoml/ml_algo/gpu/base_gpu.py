"""Base classes for machine learning algorithms (GPU version)."""

import logging
from typing import Any, Union, cast

import cudf
import cupy as cp
import dask.dataframe as dd
import dask_cudf
import numpy as np
import torch
from joblib import Parallel, delayed

from lightautoml.dataset.gpu.gpu_dataset import CudfDataset, CupyDataset, DaskCudfDataset
from lightautoml.ml_algo.base import TabularMLAlgo
from lightautoml.validation.base import TrainValidIterator

logger = logging.getLogger(__name__)
TabularDatasetGpu = Union[CupyDataset, CudfDataset, DaskCudfDataset]


class TabularMLAlgo_gpu(TabularMLAlgo):
    """Machine learning algorithms that accepts gpu data as input."""

    _name: str = "TabularAlgo_gpu"

    def __init__(
        self,
        parallel_folds: bool = False,
        gpu_ids: [int] = None,
        *args: Any,
        **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.gpu_ids = gpu_ids
        self.parallel_folds = parallel_folds

    def fit_predict(self, train_valid_iterator: TrainValidIterator) -> CupyDataset:
        """Fit and then predict accordig the strategy that uses train_valid_iterator.

        If item uses more then one time it will
        predict mean value of predictions.
        If the element is not used in training then
        the prediction will be ``cp.nan`` for this item

        Args:
            train_valid_iterator: Classic cv-iterator.

        Returns:
            Dataset with predicted values.

        """

        logger.info("Start fitting {} ...".format(self._name))
        self.timer.start()
        assert self.is_fitted is False, "Algo is already fitted"
        # init params on input if no params was set before
        if self._params is None:
            self.params = self.init_params_on_input(train_valid_iterator)

        # save features names
        self._features = train_valid_iterator.features

        self.task = train_valid_iterator.train.task
        val_data = train_valid_iterator.get_validation_data().empty()
        preds_ds = cast(CupyDataset, val_data.to_cupy())
        outp_dim = 1

        if self.task.name == "multiclass":
            if type(val_data) == DaskCudfDataset:
                outp_dim = int(val_data.target.max().compute() + 1)
            else:
                outp_dim = int(val_data.target.max()) + 1
        elif (self.task.name == "multi:reg") or (self.task.name == "multilabel"):
            if type(val_data) == DaskCudfDataset:
                outp_dim = val_data.target.shape.compute()[1]
            else:
                outp_dim = val_data.target.shape[1]
        # save n_classes to infer params
        self.n_classes = outp_dim

        preds_arr = cp.zeros(
            (train_valid_iterator.get_validation_data().shape[0], outp_dim),
            dtype=cp.float32,
        )
        counter_arr = cp.zeros(
            (train_valid_iterator.get_validation_data().shape[0], 1), dtype=cp.float32
        )

        if self.parallel_folds:

            def perform_iterations(fit_predict_single_fold, train_valid, ind, dev_id):
                (idx, train, valid) = train_valid[ind]
                logger.info(
                    "===== Start working with \x1b[1mfold {}\x1b[0m for \x1b[1m{}\x1b[0m (par) =====".format(
                        ind, self._name
                    )
                )
                model, pred = fit_predict_single_fold(train, valid, dev_id)
                return model, pred

            if self.task.device == "gpu":
                n_parts = 1
            else:
                n_parts = torch.cuda.device_count()

            n_folds = len(train_valid_iterator)
            num_its = int(np.ceil(n_folds / n_parts))

            inds = []
            for i in range(num_its):
                left = n_folds - i * n_parts
                if left > n_parts:
                    inds.append(np.arange(i * n_parts, i * n_parts + n_parts))
                elif left > 0:
                    inds.append(np.arange(i * n_parts, i * n_parts + left))
            # inds = np.array_split(np.arange(n_folds), num_its)
            inds = [x for x in inds if len(x) > 0]

            res = None
            models = []
            preds = []

            # with Parallel(n_jobs=n_parts, prefer='processes',
            #              backend='loky', max_nbytes=None) as p:

            for n in range(num_its):
                self.timer.set_control_point()
                with Parallel(n_jobs=n_parts, prefer="threads") as p:
                    res = p(
                        delayed(perform_iterations)(
                            self.fit_predict_single_fold,
                            train_valid_iterator,
                            ind,
                            device_id,
                        )
                        for (ind, device_id) in zip(inds[n], self.gpu_ids)
                    )

                for elem in res:
                    models.append(elem[0])
                    preds.append(elem[1])
                    del elem

                self.timer.write_run_info()
                if (n + 1) != num_its:
                    if self.timer.time_limit_exceeded():
                        logger.warning(
                            "Time limit exceeded after calculating fold(s) {0}".format(
                                inds[n]
                            )
                        )
                        break

            logger.debug(
                "Time history {0}. Time left {1}".format(
                    self.timer.get_run_results(), self.timer.time_left
                )
            )

            self.models = models
            for n, (idx, _, _) in enumerate(train_valid_iterator):
                if n < len(preds):
                    if isinstance(
                        preds[n],
                        (dask_cudf.DataFrame, dask_cudf.Series, dd.DataFrame, dd.Series),
                    ):
                        preds_arr[idx] += (
                            preds[n]
                            .compute()
                            .values.reshape(preds[n].shape[0].compute(), -1)
                        )
                        counter_arr[idx] += 1
                    else:
                        if isinstance(preds[n], np.ndarray):
                            preds[n] = cp.asarray(preds[n])
                        preds_arr[idx] += preds[n].reshape((preds[n].shape[0], -1))
                        counter_arr[idx] += 1
        else:

            for n, (idx, train, valid) in enumerate(train_valid_iterator):
                logger.info(
                    "===== Start working with \x1b[1mfold {}\x1b[0m for \x1b[1m{}\x1b[0m (orig) =====".format(
                        n, self._name
                    )
                )

                self.timer.set_control_point()
                model, pred = self.fit_predict_single_fold(train, valid)
                self.models.append(model)

                if isinstance(
                    pred,
                    (dask_cudf.DataFrame, dask_cudf.Series, dd.DataFrame, dd.Series),
                ):

                    if idx is not None:
                        preds_arr[idx] += pred.compute().values.reshape(
                            pred.shape[0].compute(), -1
                        )
                        counter_arr[idx] += 1
                    else:
                        preds_arr += pred.compute().values.reshape(
                            pred.shape[0].compute(), -1
                        )
                        counter_arr += 1

                else:
                    if isinstance(pred, np.ndarray):
                        pred = cp.asarray(pred)
                    preds_arr[idx] += pred.reshape((pred.shape[0], -1))
                    counter_arr[idx] += 1

                self.timer.write_run_info()
                if (n + 1) != len(train_valid_iterator):
                    if self.timer.time_limit_exceeded():
                        logger.warning(
                            "Time limit exceeded after calculating fold {0}".format(n)
                        )
                        break

            logger.debug(
                "Time history {0}. Time left {1}".format(
                    self.timer.get_run_results(), self.timer.time_left
                )
            )

        preds_arr /= cp.where(counter_arr == 0, 1, counter_arr)
        preds_arr = cp.where(counter_arr == 0, cp.nan, preds_arr)

        preds_ds = self._set_prediction(preds_ds, preds_arr)
        logger.info("{} fitting and predicting completed".format(self._name))
        return preds_ds

    def predict(self, dataset: TabularDatasetGpu) -> CupyDataset:
        """Mean prediction for all fitted models.

        Args:
            dataset: Dataset used for prediction.

        Returns:
            Dataset with predicted values.

        """

        assert self.models != [], "Should be fitted first."

        """if type(dataset) == DaskCudfDataset:
            preds_ds = dataset.empty()
            preds_arr = None
        else:"""
        preds_ds = dataset.empty().to_cupy()
        preds_arr = None

        for model in self.models:
            pred = self.predict_single_fold(model, dataset)

            if isinstance(
                pred, (dask_cudf.DataFrame, dd.DataFrame, dask_cudf.Series, dd.Series)
            ):
                pred = pred.compute().values
            elif isinstance(pred, (cudf.DataFrame, cudf.Series)):
                pred = pred.values
            elif isinstance(pred, np.ndarray):
                pred = cp.asarray(pred)

            if preds_arr is None:
                preds_arr = pred
            else:
                preds_arr += pred

        preds_arr /= len(self.models)
        preds_arr = preds_arr.reshape((preds_arr.shape[0], -1))
        preds_ds = self._set_prediction(preds_ds, preds_arr)

        return preds_ds
