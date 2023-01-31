#!/usr/bin/env python
# coding: utf-8

import shutil

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from lightautoml.automl.presets.text_presets import TabularNLPAutoML
from lightautoml.tasks import Task


np.random.seed(42)

# load and prepare data
data = pd.read_csv("./data/avito1k_train.csv")

train, test = train_test_split(data, test_size=500, random_state=42)

roles = {
    "target": "deal_probability",
    "group": ["user_id"],
    "text": ["description", "title", "param_1", "param_2", "param_3"],
}

# init automl
automl = TabularNLPAutoML(task=Task("reg"), timeout=600)

# run automl
oof_pred = automl.fit_predict(train, roles=roles)

# get predictions
test_pred = automl.predict(test)

# calculate scores
not_nan = np.any(~np.isnan(oof_pred.data), axis=1)
print("OOF score: {}".format(mean_squared_error(train[roles["target"]].values[not_nan], oof_pred.data[not_nan][:, 0])))
print("TEST score: {}".format(mean_squared_error(test[roles["target"]].values, test_pred.data[:, 0])))

shutil.rmtree("./models", ignore_errors=True)
