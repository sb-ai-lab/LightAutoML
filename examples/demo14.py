#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from lightautoml.automl.base import AutoML
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.dataset.roles import CategoryRole
from lightautoml.dataset.roles import NumericRole
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.pipelines.features.lgb_pipeline import LGBAdvancedPipeline
from lightautoml.pipelines.features.lgb_pipeline import LGBSimpleFeatures
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.pipelines.selection.importance_based import ImportanceCutoffSelector
from lightautoml.pipelines.selection.importance_based import (
    ModelBasedImportanceEstimator,
)
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.tasks import Task


################################
# Features:
# - group_by transformer
################################

N_FOLDS = 3  # number of folds for cross-validation inside AutoML
RANDOM_STATE = 42  # fixed random state for various reasons
N_THREADS = 4  # threads cnt for lgbm and linear models
TIMEOUT = 100
USED_COLS = ["SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "CODE_GENDER", "AMT_INCOME_TOTAL", "DAYS_BIRTH"]
TARGET = "TARGET"

# load data
data = pd.read_csv("./data/sampled_app_train.csv")
data = data[USED_COLS]
train, test = train_test_split(data, test_size=2000, random_state=42)

# Using TabularAutoML preset
task = Task("binary")
roles = {
    "target": TARGET,
    CategoryRole(dtype=str): ["NAME_CONTRACT_TYPE", "CODE_GENDER"],
    NumericRole(np.float32): ["AMT_INCOME_TOTAL"],
}

# specify groupby triplets: [("group_col", "feature", "transform_type"),]
groupby_triplets = [
    ("CODE_GENDER", "AMT_INCOME_TOTAL", "max"),
    ("NAME_CONTRACT_TYPE", "CODE_GENDER", "mode"),
    ("NAME_CONTRACT_TYPE", "AMT_INCOME_TOTAL", "delta_mean"),
]

print(f"Try TabularAutoML with the following groupby_triplets:\n{groupby_triplets}")

automl = TabularAutoML(
    task=task,
    timeout=TIMEOUT,
    cpu_limit=N_THREADS,
    reader_params={"n_jobs": N_THREADS, "cv": N_FOLDS, "random_state": RANDOM_STATE},
    general_params={"use_algos": [["lgb"]]},
    gbm_pipeline_params={"use_groupby": True, "groupby_triplets": groupby_triplets},
)
automl.fit_predict(train, roles=roles)

feature_scores = automl.levels[0][0].ml_algos[0].get_features_score()

print(f"Feature importances of BoostLGBM model. Pay attention to groupby features:\n{feature_scores}")

# Custom pipeline with groupby features defined by importance
print("\nTry custom pipeline with groupby features defined by importance:\n")

task = Task("binary")
reader = PandasToPandasReader(task, cv=N_FOLDS, random_state=RANDOM_STATE)
model0 = BoostLGBM(default_params={"learning_rate": 0.1, "num_leaves": 64, "seed": 42, "num_threads": N_THREADS})
pie = ModelBasedImportanceEstimator()
selector = ImportanceCutoffSelector(LGBSimpleFeatures(), model0, pie, cutoff=-9999)


pipe = LGBAdvancedPipeline(
    use_groupby=True, pre_selector=selector, groupby_types=["delta_median", "std"], groupby_top_based_on="importance"
)

model = BoostLGBM(
    default_params={
        "learning_rate": 0.05,
        "num_leaves": 128,
        "seed": 1,
        "num_threads": N_THREADS,
    }
)

pipeline = MLPipeline([model], pre_selection=selector, features_pipeline=pipe, post_selection=None)

automl = AutoML(
    reader,
    [[pipeline]],
    skip_conn=False,
)

oof_pred = automl.fit_predict(
    train,
    roles={"target": TARGET},
)

print(f"Feature used by BoostLGBM model. Pay attention to groupby features:\n{pipe.output_features}")
