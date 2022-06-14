#!/usr/bin/env python
# coding: utf-8
import logging
import os
import pickle
import shutil

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from lightautoml.automl.presets.text_presets import TabularNLPAutoML
from lightautoml.tasks import Task


def test_nlp_preset():
    np.random.seed(42)
    logging.basicConfig(
        format="[%(asctime)s] (%(levelname)s): %(message)s", level=logging.DEBUG
    )

    data = pd.read_csv("../examples/data/avito1k_train.csv")

    train, test = train_test_split(data, test_size=500, random_state=42)

    roles = {
        "target": "deal_probability",
        "group": ["user_id"],
        "text": ["description", "title", "param_1", "param_2", "param_3"],
    }

    task = Task(
        "reg",
    )

    automl = TabularNLPAutoML(task=task, timeout=600)
    oof_pred = automl.fit_predict(train, roles=roles)
    test_pred = automl.predict(test)
    not_nan = np.any(~np.isnan(oof_pred.data), axis=1)

    logging.debug("Check scores...")
    print(
        "OOF score: {}".format(
            mean_squared_error(
                train[roles["target"]].values[not_nan], oof_pred.data[not_nan][:, 0]
            )
        )
    )
    print(
        "TEST score: {}".format(
            mean_squared_error(test[roles["target"]].values, test_pred.data[:, 0])
        )
    )
    print("Pickle automl")

    with open("automl.pickle", "wb") as f:
        pickle.dump(automl, f)

    logging.debug("Load pickled automl")
    with open("automl.pickle", "rb") as f:
        automl = pickle.load(f)

    logging.debug("Predict loaded automl")
    test_pred = automl.predict(test)
    print(
        "TEST score, loaded: {}".format(
            mean_squared_error(test["deal_probability"].values, test_pred.data[:, 0])
        )
    )

    os.remove("automl.pickle")
    shutil.rmtree("./models", ignore_errors=True)


if __name__ == "__main__":
    test_nlp_preset()
