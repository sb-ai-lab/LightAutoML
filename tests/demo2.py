#!/usr/bin/env python
# coding: utf-8

import logging
import os
import pickle
import time

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lightautoml.automl.base import AutoML
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.pipelines.features.lgb_pipeline import LGBSimpleFeatures
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.pipelines.selection.permutation_importance_based import (
    NpIterativeFeatureSelector,
)
from lightautoml.pipelines.selection.permutation_importance_based import (
    NpPermutationImportanceEstimator,
)
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.tasks import Task


def test_permutation_importance_based_iterative_selector():
    logging.basicConfig(
        format="[%(asctime)s] (%(levelname)s): %(message)s", level=logging.DEBUG
    )

    logging.debug("Load data...")
    data = pd.read_csv("../examples/data/sampled_app_train.csv")
    logging.debug("Data loaded")

    logging.debug("Features modification from user side...")
    data["BIRTH_DATE"] = (
        np.datetime64("2018-01-01")
        + data["DAYS_BIRTH"].astype(np.dtype("timedelta64[D]"))
    ).astype(str)
    data["EMP_DATE"] = (
        np.datetime64("2018-01-01")
        + np.clip(data["DAYS_EMPLOYED"], None, 0).astype(np.dtype("timedelta64[D]"))
    ).astype(str)

    data["constant"] = 1
    data["allnan"] = np.nan

    data.drop(["DAYS_BIRTH", "DAYS_EMPLOYED"], axis=1, inplace=True)
    logging.debug("Features modification finished")

    logging.debug("Split data...")
    train_data, test_data = train_test_split(
        data, test_size=2000, stratify=data["TARGET"], random_state=13
    )
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    logging.debug(
        "Data splitted. Parts sizes: train_data = {}, test_data = {}".format(
            train_data.shape, test_data.shape
        )
    )

    logging.debug("Create task...")
    task = Task("binary")
    logging.debug("Task created")

    logging.debug("Create reader...")
    reader = PandasToPandasReader(task, cv=5, random_state=1)
    logging.debug("Reader created")

    # selector parts
    logging.debug("Create feature selector")
    model0 = BoostLGBM(
        default_params={
            "learning_rate": 0.05,
            "num_leaves": 64,
            "seed": 42,
            "num_threads": 5,
        }
    )
    pipe0 = LGBSimpleFeatures()
    pie = NpPermutationImportanceEstimator()
    selector = NpIterativeFeatureSelector(
        pipe0, model0, pie, feature_group_size=1, max_features_cnt_in_result=15
    )
    logging.debug("Feature selector created")

    # pipeline 1 level parts
    logging.debug("Start creation pipeline_1...")
    pipe = LGBSimpleFeatures()

    logging.debug("\t ParamsTuner1 and Model1...")
    model1 = BoostLGBM(
        default_params={
            "learning_rate": 0.05,
            "num_leaves": 128,
            "seed": 1,
            "num_threads": 5,
        }
    )
    logging.debug("\t Tuner1 and model1 created")

    logging.debug("\t ParamsTuner2 and Model2...")
    params_tuner2 = OptunaTuner(n_trials=100, timeout=100)
    model2 = BoostLGBM(
        default_params={
            "learning_rate": 0.025,
            "num_leaves": 64,
            "seed": 2,
            "num_threads": 5,
        }
    )
    logging.debug("\t Tuner2 and model2 created")

    logging.debug("\t Pipeline1...")
    pipeline_lvl1 = MLPipeline(
        [model1, (model2, params_tuner2)],
        pre_selection=selector,
        features_pipeline=pipe,
        post_selection=None,
    )
    logging.debug("Pipeline1 created")

    # pipeline 2 level parts
    logging.debug("Start creation pipeline_2...")
    pipe1 = LGBSimpleFeatures()

    logging.debug("\t ParamsTuner and Model...")
    model = BoostLGBM(
        default_params={
            "learning_rate": 0.05,
            "num_leaves": 64,
            "max_bin": 1024,
            "seed": 3,
            "num_threads": 5,
        },
        freeze_defaults=True,
    )
    logging.debug("\t Tuner and model created")

    logging.debug("\t Pipeline2...")
    pipeline_lvl2 = MLPipeline(
        [model], pre_selection=None, features_pipeline=pipe1, post_selection=None
    )
    logging.debug("Pipeline2 created")

    logging.debug("Create AutoML pipeline...")
    automl = AutoML(
        reader,
        [
            [pipeline_lvl1],
            [pipeline_lvl2],
        ],
        skip_conn=False,
    )

    logging.debug("AutoML pipeline created...")

    logging.debug("Start AutoML pipeline fit_predict...")
    start_time = time.time()
    oof_pred = automl.fit_predict(train_data, roles={"target": "TARGET"})
    logging.debug(
        "AutoML pipeline fitted and predicted. Time = {:.3f} sec".format(
            time.time() - start_time
        )
    )

    logging.debug(
        "Feature importances of selector:\n{}".format(selector.get_features_score())
    )

    logging.debug("oof_pred:\n{}\nShape = {}".format(oof_pred, oof_pred.shape))

    logging.debug(
        "Feature importances of top level algorithm:\n{}".format(
            automl.levels[-1][0].ml_algos[0].get_features_score()
        )
    )

    logging.debug(
        "Feature importances of lowest level algorithm - model 0:\n{}".format(
            automl.levels[0][0].ml_algos[0].get_features_score()
        )
    )

    logging.debug(
        "Feature importances of lowest level algorithm - model 1:\n{}".format(
            automl.levels[0][0].ml_algos[1].get_features_score()
        )
    )

    test_pred = automl.predict(test_data)
    logging.debug(
        "Prediction for test data:\n{}\nShape = {}".format(test_pred, test_pred.shape)
    )

    logging.debug("Check scores...")
    logging.debug(
        "OOF score: {}".format(
            roc_auc_score(train_data["TARGET"].values, oof_pred.data[:, 0])
        )
    )
    logging.debug(
        "TEST score: {}".format(
            roc_auc_score(test_data["TARGET"].values, test_pred.data[:, 0])
        )
    )
    logging.debug("Pickle automl")
    with open("automl.pickle", "wb") as f:
        pickle.dump(automl, f)

    logging.debug("Load pickled automl")
    with open("automl.pickle", "rb") as f:
        automl = pickle.load(f)

    logging.debug("Predict loaded automl")
    test_pred = automl.predict(test_data)
    logging.debug(
        "TEST score, loaded: {}".format(
            roc_auc_score(test_data["TARGET"].values, test_pred.data[:, 0])
        )
    )

    os.remove("automl.pickle")
