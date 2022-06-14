#!/usr/bin/env python
# coding: utf-8

import logging
import os
import pickle
import time

from lightautoml.dataset.np_pd_dataset import *
from lightautoml.dataset.roles import *
from lightautoml.dataset.utils import roles_parser
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.pipelines.features.lgb_pipeline import LGBSimpleFeatures
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.pipelines.selection.importance_based import ImportanceCutoffSelector
from lightautoml.pipelines.selection.importance_based import (
    ModelBasedImportanceEstimator,
)
from lightautoml.tasks import Task
from lightautoml.validation.np_iterators import FoldsIterator


logging.basicConfig(
    format="[%(asctime)s] (%(levelname)s): %(message)s", level=logging.DEBUG
)


def test_manual_pipeline():
    # Read data from file
    logging.debug("Read data from file")
    data = pd.read_csv(
        "../examples/data/sampled_app_train.csv",
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

    # Fix dates and convert to date type
    logging.debug("Fix dates and convert to date type")
    data["BIRTH_DATE"] = np.datetime64("2018-01-01") + data["DAYS_BIRTH"].astype(
        np.dtype("timedelta64[D]")
    )
    data["EMP_DATE"] = np.datetime64("2018-01-01") + np.clip(
        data["DAYS_EMPLOYED"], None, 0
    ).astype(np.dtype("timedelta64[D]"))
    data.drop(["DAYS_BIRTH", "DAYS_EMPLOYED"], axis=1, inplace=True)

    # Create folds
    logging.debug("Create folds")
    data["__fold__"] = np.random.randint(0, 5, len(data))

    # Print data head
    logging.debug("Print data head")
    print(data.head())

    # # Set roles for columns
    logging.debug("Set roles for columns")
    check_roles = {
        TargetRole(): "TARGET",
        CategoryRole(dtype=str): ["NAME_CONTRACT_TYPE", "NAME_TYPE_SUITE"],
        NumericRole(np.float32): ["AMT_CREDIT", "AMT_GOODS_PRICE"],
        DatetimeRole(seasonality=["y", "m", "wd"]): ["BIRTH_DATE", "EMP_DATE"],
        FoldsRole(): "__fold__",
    }

    # create Task
    task = Task("binary")
    # # Creating PandasDataSet
    logging.debug("Creating PandasDataset")
    start_time = time.time()
    pd_dataset = PandasDataset(data, roles_parser(check_roles), task=task)
    logging.debug(
        "PandasDataset created. Time = {:.3f} sec".format(time.time() - start_time)
    )

    # # Print pandas dataset feature roles
    logging.debug("Print pandas dataset feature roles")
    roles = pd_dataset.roles
    for role in roles:
        logging.debug("{}: {}".format(role, roles[role]))

    # # Feature selection part
    logging.debug("Feature selection part")
    selector_iterator = FoldsIterator(pd_dataset, 1)
    logging.debug("Selection iterator created")

    model = BoostLGBM()
    pipe = LGBSimpleFeatures()
    logging.debug("Pipe and model created")

    model0 = BoostLGBM(
        default_params={
            "learning_rate": 0.05,
            "num_leaves": 64,
            "seed": 0,
            "num_threads": 5,
        }
    )

    mbie = ModelBasedImportanceEstimator()
    selector = ImportanceCutoffSelector(pipe, model0, mbie, cutoff=10)
    start_time = time.time()
    selector.fit(selector_iterator)
    logging.debug(
        "Feature selector fitted. Time = {:.3f} sec".format(time.time() - start_time)
    )

    logging.debug("Feature selector scores:")
    logging.debug("\n{}".format(selector.get_features_score()))

    # # Build AutoML pipeline
    logging.debug("Start building AutoML pipeline")
    pipe = LGBSimpleFeatures()
    logging.debug("Pipe created")

    params_tuner1 = OptunaTuner(n_trials=10, timeout=300)
    model1 = BoostLGBM(default_params={"learning_rate": 0.05, "num_leaves": 128})
    logging.debug("Tuner1 and model1 created")

    params_tuner2 = OptunaTuner(n_trials=100, timeout=300)
    model2 = BoostLGBM(default_params={"learning_rate": 0.025, "num_leaves": 64})
    logging.debug("Tuner2 and model2 created")

    total = MLPipeline(
        [(model1, params_tuner1), (model2, params_tuner2)],
        pre_selection=selector,
        features_pipeline=pipe,
        post_selection=None,
    )

    logging.debug("Finished building AutoML pipeline")

    # # Create full train iterator
    logging.debug("Full train valid iterator creation")
    train_valid = FoldsIterator(pd_dataset)
    logging.debug("Full train valid iterator created")

    # # Fit predict using pipeline
    logging.debug("Start AutoML pipeline fit_predict")
    start_time = time.time()
    pred = total.fit_predict(train_valid)
    logging.debug(
        "Fit_predict finished. Time = {:.3f} sec".format(time.time() - start_time)
    )

    # # Check preds
    logging.debug("Preds:")
    logging.debug("\n{}".format(pred))
    logging.debug("Preds.shape = {}".format(pred.shape))

    # # Predict full train dataset
    logging.debug("Predict full train dataset")
    start_time = time.time()
    train_pred = total.predict(pd_dataset)
    logging.debug(
        "Predict finished. Time = {:.3f} sec".format(time.time() - start_time)
    )
    logging.debug("Preds:")
    logging.debug("\n{}".format(train_pred))
    logging.debug("Preds.shape = {}".format(train_pred.shape))

    logging.debug("Pickle automl")
    with open("automl.pickle", "wb") as f:
        pickle.dump(total, f)

    logging.debug("Load pickled automl")
    with open("automl.pickle", "rb") as f:
        total = pickle.load(f)

    logging.debug("Predict loaded automl")
    train_pred = total.predict(pd_dataset)
    os.remove("automl.pickle")

    # # Check preds feature names
    logging.debug("Preds features: {}".format(train_pred.features))

    # # Check model feature scores
    logging.debug("Feature scores for model_1:\n{}".format(model1.get_features_score()))
    logging.debug("Feature scores for model_2:\n{}".format(model2.get_features_score()))
