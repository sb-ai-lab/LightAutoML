#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lightautoml.automl.base import AutoML
from lightautoml.automl.blend import WeightedBlender
from lightautoml.ml_algo.boost_cb import BoostCB
from lightautoml.ml_algo.linear_sklearn import LinearLBFGS
from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.pipelines.features.lgb_pipeline import LGBAdvancedPipeline
from lightautoml.pipelines.features.lgb_pipeline import LGBSimpleFeatures
from lightautoml.pipelines.features.linear_pipeline import LinearFeatures
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.pipelines.selection.importance_based import ImportanceCutoffSelector
from lightautoml.pipelines.selection.importance_based import (
    ModelBasedImportanceEstimator,
)
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.tasks import Task
from lightautoml.utils.timer import PipelineTimer


# demo of timer, blender and multiclass

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
data["TARGET"] = np.where(np.random.rand(data.shape[0]) > 0.5, 2, data["TARGET"].values)

train, test = train_test_split(data, test_size=2000, random_state=42)
# ======================================================================================
print("Create timer...")
timer = PipelineTimer(600, mode=2)
print("Timer created...")
# ======================================================================================
print("Create selector...")
timer_gbm = timer.get_task_timer("gbm")
feat_sel_0 = LGBSimpleFeatures()
mod_sel_0 = BoostCB(timer=timer_gbm)
imp_sel_0 = ModelBasedImportanceEstimator()
selector_0 = ImportanceCutoffSelector(
    feat_sel_0,
    mod_sel_0,
    imp_sel_0,
    cutoff=0,
)
print("Selector created...")
# ======================================================================================
print("Create gbms...")
feats_gbm_0 = LGBAdvancedPipeline(top_intersections=4, feats_imp=imp_sel_0)
timer_gbm_0 = timer.get_task_timer("gbm")
timer_gbm_1 = timer.get_task_timer("gbm")

gbm_0 = BoostCB(timer=timer_gbm_0, default_params={"devices": "0"})
gbm_1 = BoostCB(timer=timer_gbm_1, default_params={"devices": "0"})

tuner_0 = OptunaTuner(n_trials=10, timeout=10, fit_on_holdout=True)
gbm_lvl0 = MLPipeline(
    [(gbm_0, tuner_0), gbm_1],
    pre_selection=selector_0,
    features_pipeline=feats_gbm_0,
    post_selection=None,
)
print("Gbms created...")
# ======================================================================================
print("Create linear...")
feats_reg_0 = LinearFeatures(output_categories=True, sparse_ohe="auto")

timer_reg = timer.get_task_timer("reg")
reg_0 = LinearLBFGS(timer=timer_reg)

reg_lvl0 = MLPipeline([reg_0], pre_selection=None, features_pipeline=feats_reg_0, post_selection=None)
print("Linear created...")
# ======================================================================================
print("Create reader...")
reader = PandasToPandasReader(
    Task(
        "multiclass",
        metric="crossentropy",
    ),
    samples=None,
    max_nan_rate=1,
    max_constant_rate=1,
    advanced_roles=True,
    drop_score_co=-1,
    n_jobs=1,
)
print("Reader created...")
# ======================================================================================
print("Create blender...")
blender = WeightedBlender()
print("Blender created...")
# ======================================================================================
print("Create AutoML...")
automl = AutoML(
    reader=reader,
    levels=[[gbm_lvl0, reg_lvl0]],
    timer=timer,
    blender=blender,
    skip_conn=False,
)
print("AutoML created...")
# ======================================================================================
print("Fit predict...")
oof_pred = automl.fit_predict(train, roles={"target": "TARGET"})
print("Finished fitting...")

test_pred = automl.predict(test)
print("Prediction for test data:\n{}\nShape = {}".format(test_pred, test_pred.shape))
# ======================================================================================
print("Check scores...")
# use only not nan
not_nan = np.any(~np.isnan(oof_pred.data), axis=1)

print("OOF score: {}".format(log_loss(train["TARGET"].values[not_nan], oof_pred.data[not_nan])))
print("TEST score: {}".format(log_loss(test["TARGET"].values, test_pred.data)))
# ======================================================================================
for dat, df, name in zip([oof_pred, test_pred], [train, test], ["train", "test"]):
    print("Check aucs {0}...".format(name))
    for c in range(3):
        _sc = roc_auc_score((df["TARGET"].values == c).astype(np.float32), dat.data[:, c])
        print("Cl {0} auc score: {1}".format(c, _sc))
# ======================================================================================
