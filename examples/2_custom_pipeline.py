#!/usr/bin/env python
# coding: utf-8

"""Building ML pipeline from blocks and fit + predict the pipeline itself."""

import os
import pickle
import time

import numpy as np
import pandas as pd

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import CategoryRole
from lightautoml.dataset.roles import DatetimeRole
from lightautoml.dataset.roles import FoldsRole
from lightautoml.dataset.roles import NumericRole
from lightautoml.dataset.roles import TargetRole
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


# Read data from file
data = pd.read_csv(
    "./data/sampled_app_train.csv",
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

data["BIRTH_DATE"] = np.datetime64("2018-01-01") + data["DAYS_BIRTH"].astype(np.dtype("timedelta64[D]"))
data["EMP_DATE"] = np.datetime64("2018-01-01") + np.clip(data["DAYS_EMPLOYED"], None, 0).astype(
    np.dtype("timedelta64[D]")
)
data.drop(["DAYS_BIRTH", "DAYS_EMPLOYED"], axis=1, inplace=True)

data["__fold__"] = np.random.randint(0, 5, len(data))

# set roles
check_roles = {
    TargetRole(): "TARGET",
    CategoryRole(dtype=str): ["NAME_CONTRACT_TYPE", "NAME_TYPE_SUITE"],
    NumericRole(np.float32): ["AMT_CREDIT", "AMT_GOODS_PRICE"],
    DatetimeRole(seasonality=["y", "m", "wd"]): ["BIRTH_DATE", "EMP_DATE"],
    FoldsRole(): "__fold__",
}

# make reader
pd_dataset = PandasDataset(data, roles_parser(check_roles), task=Task("binary"))

# select features
selector_iterator = FoldsIterator(pd_dataset, 1)

# Create full train iterator
train_valid = FoldsIterator(pd_dataset)

# Create selector
selector = ImportanceCutoffSelector(
    LGBSimpleFeatures(), 
    BoostLGBM(
        default_params={
            "learning_rate": 0.05,
            "num_leaves": 64,
            "seed": 0,
            "num_threads": 5,
        }
    ), 
    ModelBasedImportanceEstimator(), 
    cutoff=10
)

# train our new selector
selector.fit(selector_iterator)
print("Feature selector scores:")
print(f"\n{selector.get_features_score()}")

# Let's build our ML pipeline!
# Our pipeline:
# - select features
# - generate new features from selected features set
# - 2 LGBM models with hyperparameters tuning
total = MLPipeline(
    [
        # model 1
        (
            BoostLGBM(default_params={"learning_rate": 0.05, "num_leaves": 128}),
            OptunaTuner(n_trials=10, timeout=300)
        ), 
        # model 2
        (
            BoostLGBM(default_params={"learning_rate": 0.025, "num_leaves": 64}), 
            OptunaTuner(n_trials=100, timeout=300)
        )
    ],
    pre_selection=selector,
    features_pipeline=LGBSimpleFeatures(),
    post_selection=None,
)

# Fit our pipeline
predictions = total.fit_predict(train_valid) # fit_predict returns OOF predictions for input dataset

# Predict full train dataset
train_pred = total.predict(pd_dataset)

# Check model feature scores
print("Feature scores for model_1:\n{}".format(model1.get_features_score()))
print("Feature scores for model_2:\n{}".format(model2.get_features_score()))
