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
from lightautoml.pipelines.selection.importance_based import ImportanceCutoffSelector
from lightautoml.pipelines.selection.importance_based import (
    ModelBasedImportanceEstimator,
)
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.tasks import Task


# load data
data = pd.read_csv("./examples/data/sampled_app_train.csv")
data["BIRTH_DATE"] = (np.datetime64("2018-01-01") + data["DAYS_BIRTH"].astype(np.dtype("timedelta64[D]"))).astype(
    str
)
data["EMP_DATE"] = (
    np.datetime64("2018-01-01") + np.clip(data["DAYS_EMPLOYED"], None, 0).astype(np.dtype("timedelta64[D]"))
).astype(str)

data["constant"] = 1
data["allnan"] = np.nan

data.drop(["DAYS_BIRTH", "DAYS_EMPLOYED"], axis=1, inplace=True)

# Split data
train_data, test_data = train_test_split(data, test_size=2000, stratify=data["TARGET"], random_state=13)
train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

task = Task("binary")
reader = PandasToPandasReader(task, cv=5, random_state=1)

# Create feature selector that uses importances to cutoff features set
selector = ImportanceCutoffSelector(
    LGBSimpleFeatures(),
        BoostLGBM(
        default_params={
            "learning_rate": 0.05,
            "num_leaves": 64,
            "seed": 42,
            "num_threads": 5,
        }
    ), 
    ModelBasedImportanceEstimator(), cutoff=10
)

# Or you can use iteartive feature selection algorithm:
# selector = NpIterativeFeatureSelector(
#     LGBSimpleFeatures(),
#     BoostLGBM(
#         default_params={
#             "learning_rate": 0.05,
#             "num_leaves": 64,
#             "seed": 42,
#             "num_threads": 5,
#         }
#     ), 
#     NpPermutationImportanceEstimator(), 
#     feature_group_size=1,
#     max_features_cnt_in_result=15
# )

# Build pipeline for 1st level
pipeline_lvl1 = MLPipeline(
    [
        (
            BoostLGBM(
                default_params={
                    "learning_rate": 0.05,
                    "num_leaves": 128,
                    "seed": 1,
                    "num_threads": 5,
                }
            ), 
            OptunaTuner(n_trials=100, timeout=300)
        ),
        BoostLGBM(
            default_params={
                "learning_rate": 0.025,
                "num_leaves": 64,
                "seed": 2,
                "num_threads": 5,
            }
        )
    ],
    pre_selection=selector,
    features_pipeline=LGBSimpleFeatures(),
    post_selection=None,
)

# Build pipeline for 2nd level

pipeline_lvl2 = MLPipeline(
    [
        BoostLGBM(
            default_params={
                "learning_rate": 0.05,
                "num_leaves": 64,
                "max_bin": 1024,
                "seed": 3,
                "num_threads": 5,
            },
            freeze_defaults=True,
        )
    ], 
    pre_selection=None,
    features_pipeline=LGBSimpleFeatures(),
    post_selection=None
)

# Create AutoML with 2 level stacking
automl = AutoML(
    reader,
    [
        [pipeline_lvl1],
        [pipeline_lvl2],
    ],
    skip_conn=False,
)


# Start AutoML training
oof_pred = automl.fit_predict(train_data, roles={"target": "TARGET"})

print(f"Feature importances of selector:\n{selector.get_features_score()}")
print(f"Feature importances of top level algorithm:\n{}".format(automl.levels[-1][0].ml_algos[0].get_features_score()))
print(f"Feature importances of lowest level algorithm - model 0:\n{automl.levels[0][0].ml_algos[0].get_features_score()}")
print(f"Feature importances of lowest level algorithm - model 1:\n{automl.levels[0][0].ml_algos[1].get_features_score()}")

# predict for test data
test_pred = automl.predict(test_data)

print(f"OOF score: {roc_auc_score(train_data["TARGET"].values, oof_pred.data[:, 0]))}")
print(f"TEST score: {roc_auc_score(test_data["TARGET"].values, test_pred.data[:, 0])}")
