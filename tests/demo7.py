#!/usr/bin/env python
# coding: utf-8
import logging
import os
import pickle

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.dataset.roles import DatetimeRole
from lightautoml.tasks import Task


def test_tabular_automl_preset_without_params():
    np.random.seed(42)
    logging.basicConfig(
        format="[%(asctime)s] (%(levelname)s): %(message)s", level=logging.DEBUG
    )

    data = pd.read_csv("../examples/data/sampled_app_train.csv")

    data["BIRTH_DATE"] = (
        np.datetime64("2018-01-01")
        + data["DAYS_BIRTH"].astype(np.dtype("timedelta64[D]"))
    ).astype(str)
    data["EMP_DATE"] = (
        np.datetime64("2018-01-01")
        + np.clip(data["DAYS_EMPLOYED"], None, 0).astype(np.dtype("timedelta64[D]"))
    ).astype(str)

    data["report_dt"] = np.datetime64("2018-01-01")

    data["constant"] = 1
    data["allnan"] = np.nan

    data.drop(["DAYS_BIRTH", "DAYS_EMPLOYED"], axis=1, inplace=True)

    train, test = train_test_split(data, test_size=2000, random_state=42)

    roles = {
        "target": "TARGET",
        DatetimeRole(base_date=True, seasonality=(), base_feats=False): "report_dt",
    }

    task = Task(
        "binary",
    )

    automl = TabularAutoML(
        task=task,
        timeout=3600,
    )
    oof_pred = automl.fit_predict(train, roles=roles)
    test_pred = automl.predict(test)

    not_nan = np.any(~np.isnan(oof_pred.data), axis=1)

    logging.debug("Check scores...")
    print(
        "OOF score: {}".format(
            roc_auc_score(
                train[roles["target"]].values[not_nan], oof_pred.data[not_nan][:, 0]
            )
        )
    )
    print(
        "TEST score: {}".format(
            roc_auc_score(test[roles["target"]].values, test_pred.data[:, 0])
        )
    )
    logging.debug("Pickle automl")
    with open("automl.pickle", "wb") as f:
        pickle.dump(automl, f)

    logging.debug("Load pickled automl")
    with open("automl.pickle", "rb") as f:
        automl = pickle.load(f)

    logging.debug("Predict loaded automl")
    test_pred = automl.predict(test)
    logging.debug(
        "TEST score, loaded: {}".format(
            roc_auc_score(test["TARGET"].values, test_pred.data[:, 0])
        )
    )

    os.remove("automl.pickle")
