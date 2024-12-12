#!/usr/bin/env python
# coding: utf-8

"""Building ML pipeline from blocks and fit + predict the pipeline itself."""

import time

import numpy as np
import pandas as pd

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.utils import roles_parser
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.pipelines.features.lgb_pipeline import LGBSimpleFeatures
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.pipelines.selection.importance_based import ImportanceCutoffSelector
from lightautoml.pipelines.selection.importance_based import (
    ModelBasedImportanceEstimator,
)
from lightautoml.validation.np_iterators import FoldsIterator

MAX_SELECTOR_FIT_TIME = 0.5
MAX_PD_DATASET_CREATING_TIME = 0.2
MAX_MLPIPELINE_FIT_PREDICT_TIME = 200
MAX_PREDICT_TIME = 1

FILE_PATH = "examples/data/sampled_app_train.csv"


def test_simple_pipeline(sampled_app_roles, binary_task):
    data = pd.read_csv(
        FILE_PATH,
        usecols=[
            "TARGET",
            "NAME_CONTRACT_TYPE",
            "AMT_CREDIT",
            "NAME_TYPE_SUITE",
            "AMT_GOODS_PRICE",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
        ],
    )

    assert isinstance(data, pd.DataFrame)
    assert "TARGET" in data.columns and "AMT_GOODS_PRICE" in data.columns

    # Fix dates and convert to date type
    data["BIRTH_DATE"] = np.datetime64("2018-01-01") + data["DAYS_BIRTH"].astype(np.dtype("timedelta64[D]"))
    data["EMP_DATE"] = np.datetime64("2018-01-01") + np.clip(data["DAYS_EMPLOYED"], None, 0).astype(
        np.dtype("timedelta64[D]")
    )
    data.drop(["DAYS_BIRTH", "DAYS_EMPLOYED"], axis=1, inplace=True)
    assert "DAY_BIRTH" not in data.columns

    # Create folds
    data["__fold__"] = np.random.randint(0, 5, len(data))

    assert isinstance(data.head(), pd.DataFrame)

    # Set roles for columns
    check_roles = sampled_app_roles

    # create Task
    task = binary_task

    assert task.metric_name == "auc"

    # Creating PandasDataSet

    pd_dataset_timing_list = []
    for _ in range(30):
        start_time = time.time()
        pd_dataset = PandasDataset(data, roles_parser(check_roles), task=task)
        pd_dataset_timing_list.append(time.time() - start_time)

    assert np.mean(pd_dataset_timing_list) < MAX_PD_DATASET_CREATING_TIME, np.mean(pd_dataset_timing_list)

    roles_classes = [object, str, np.float32, np.float32, str, np.datetime64, np.datetime64, object]
    assert all([roles_classes[i] == pd_dataset.roles[role].dtype for i, role in enumerate(pd_dataset.roles)])

    # Feature selection part
    model = BoostLGBM()

    assert not model.is_fitted and model._name == "LightGBM"

    pipe = LGBSimpleFeatures()

    model0 = BoostLGBM(
        default_params={
            "learning_rate": 0.05,
            "num_leaves": 64,
            "seed": 0,
            "num_threads": 5,
        }
    )

    selector_timing_list = []
    for _ in range(30):
        selector_iterator = FoldsIterator(pd_dataset, 1)

        mbie = ModelBasedImportanceEstimator()

        selector = ImportanceCutoffSelector(pipe, model0, mbie, cutoff=10)

        start_time = time.time()
        selector.fit(selector_iterator)

        selector_timing_list.append(time.time() - start_time)

    assert np.mean(selector_timing_list) < MAX_SELECTOR_FIT_TIME, np.mean(selector_timing_list)

    assert isinstance(selector.get_features_score(), pd.Series)

    # Build AutoML pipeline
    pipe = LGBSimpleFeatures()

    params_tuner1 = OptunaTuner(n_trials=10, timeout=300)
    model1 = BoostLGBM(default_params={"learning_rate": 0.05, "num_leaves": 128})

    params_tuner2 = OptunaTuner(n_trials=20, timeout=300)
    model2 = BoostLGBM(default_params={"learning_rate": 0.025, "num_leaves": 64})

    total = MLPipeline(
        [(model1, params_tuner1), (model2, params_tuner2)],
        pre_selection=selector,
        features_pipeline=pipe,
        post_selection=None,
    )

    assert total._ml_algos[0]._name == "Mod_0_LightGBM"
    assert total._ml_algos[1]._name == "Mod_1_LightGBM"

    train_valid = FoldsIterator(pd_dataset)

    # Fit predict using pipeline
    start_time = time.time()
    pred = total.fit_predict(train_valid)

    assert time.time() - start_time < MAX_MLPIPELINE_FIT_PREDICT_TIME

    # Check preds
    assert pred.shape == (10000, 2)

    start_time = time.time()
    train_pred = total.predict(pd_dataset)
    assert time.time() - start_time < MAX_PREDICT_TIME

    assert train_pred.shape == (10000, 2)

    assert isinstance(model1.get_features_score(), pd.Series)
    assert isinstance(model2.get_features_score(), pd.Series)

    assert ((0 <= train_pred.data[:, 1]) & (train_pred.data[:, 1] <= 1)).all()
