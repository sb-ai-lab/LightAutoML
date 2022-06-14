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
from lightautoml.pipelines.features.lgb_pipeline import LGBSimpleFeatures
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.tasks import Task


def test_different_losses_and_metrics():
    np.random.seed(42)
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

    for task_params, target in zip(
        [
            {"name": "binary"},
            {"name": "binary", "metric": roc_auc_score},
            {"name": "reg", "loss": "mse", "metric": "r2"},
            {"name": "reg", "loss": "rmsle", "metric": "rmsle"},
            {
                "name": "reg",
                "loss": "quantile",
                "loss_params": {"q": 0.9},
                "metric": "quantile",
                "metric_params": {"q": 0.9},
            },
        ],
        ["TARGET", "TARGET", "AMT_CREDIT", "AMT_CREDIT", "AMT_CREDIT"],
    ):
        logging.debug("Create task..")
        task = Task(**task_params)
        logging.debug("Task created")

        logging.debug("Create reader...")
        reader = PandasToPandasReader(task, cv=5, random_state=1)
        logging.debug("Reader created")

        # pipeline 1 level parts
        logging.debug("Start creation pipeline_1...")
        pipe = LGBSimpleFeatures()

        logging.debug("\t ParamsTuner2 and Model2...")
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
            [model2],
            pre_selection=None,  # selector,
            features_pipeline=pipe,
            post_selection=None,
        )
        logging.debug("Pipeline1 created")

        logging.debug("Create AutoML pipeline...")
        automl = AutoML(
            reader,
            [
                [pipeline_lvl1],
            ],
            skip_conn=False,
        )

        logging.debug("AutoML pipeline created...")

        logging.debug("Start AutoML pipeline fit_predict...")
        start_time = time.time()
        oof_pred = automl.fit_predict(train_data, roles={"target": target})
        logging.debug(
            "AutoML pipeline fitted and predicted. Time = {:.3f} sec".format(
                time.time() - start_time
            )
        )

        test_pred = automl.predict(test_data)
        logging.debug(
            "Prediction for test data:\n{}\nShape = {}".format(
                test_pred, test_pred.shape
            )
        )

        logging.debug("Check scores...")
        logging.debug(
            "OOF score: {}".format(
                task.metric_func(train_data[target].values, oof_pred.data[:, 0])
            )
        )
        logging.debug(
            "TEST score: {}".format(
                task.metric_func(test_data[target].values, test_pred.data[:, 0])
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
