#!/usr/bin/env python
# coding: utf-8


"""
AutoML with nested CV usage
"""

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.dataset.roles import DatetimeRole
from lightautoml.tasks import Task


np.random.seed(42)

data = pd.read_csv("./data/sampled_app_train.csv")

data["BIRTH_DATE"] = (np.datetime64("2018-01-01") + data["DAYS_BIRTH"].astype(np.dtype("timedelta64[D]"))).astype(str)
data["EMP_DATE"] = (
    np.datetime64("2018-01-01") + np.clip(data["DAYS_EMPLOYED"], None, 0).astype(np.dtype("timedelta64[D]"))
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
    timeout=600,
    general_params={
        "use_algos": [
            [
                "linear_l2",
                "lgb",
            ],
            ["linear_l2", "lgb"],
        ],
        "nested_cv": True,
        "skip_conn": True,
    },
    nested_cv_params={"cv": 5, "n_folds": None},
)

oof_pred = automl.fit_predict(train, roles=roles)
test_pred = automl.predict(test)

not_nan = np.any(~np.isnan(oof_pred.data), axis=1)

print(f"OOF score: {roc_auc_score(train[roles['target']].values[not_nan], oof_pred.data[not_nan][:, 0])}")
print(f"TEST score: {roc_auc_score(test[roles['target']].values, test_pred.data[:, 0])}")
