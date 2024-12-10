#!/usr/bin/env python
# coding: utf-8

import numpy as np

from sklearn.metrics import log_loss

from lightautoml.automl.base import AutoML
from lightautoml.automl.blend import WeightedBlender
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
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
from lightautoml.utils.timer import PipelineTimer


def test_lgbm_linear_pipeline(sampled_app_train_test, multiclass_task):

    # demo of timer, blender and multiclass
    np.random.seed(42)
    train, test = sampled_app_train_test
    timer = PipelineTimer(600, mode=2)

    timer_gbm = timer.get_task_timer("gbm")
    feat_sel_0 = LGBSimpleFeatures()
    mod_sel_0 = BoostLGBM(timer=timer_gbm)
    imp_sel_0 = ModelBasedImportanceEstimator()
    selector_0 = ImportanceCutoffSelector(
        feat_sel_0,
        mod_sel_0,
        imp_sel_0,
        cutoff=0,
    )

    feats_gbm_0 = LGBAdvancedPipeline(top_intersections=4, output_categories=True, feats_imp=imp_sel_0)
    timer_gbm_0 = timer.get_task_timer("gbm")
    timer_gbm_1 = timer.get_task_timer("gbm")

    gbm_0 = BoostLGBM(timer=timer_gbm_0)
    gbm_1 = BoostLGBM(timer=timer_gbm_1)

    tuner_0 = OptunaTuner(n_trials=10, timeout=10, fit_on_holdout=True)
    gbm_lvl0 = MLPipeline(
        [(gbm_0, tuner_0), gbm_1],
        pre_selection=selector_0,
        features_pipeline=feats_gbm_0,
        post_selection=None,
    )

    feats_reg_0 = LinearFeatures(output_categories=True, sparse_ohe="auto")

    timer_reg = timer.get_task_timer("reg")
    reg_0 = LinearLBFGS(timer=timer_reg)

    reg_lvl0 = MLPipeline([reg_0], pre_selection=None, features_pipeline=feats_reg_0, post_selection=None)

    reader = PandasToPandasReader(
        multiclass_task,
        samples=None,
        max_nan_rate=1,
        max_constant_rate=1,
        advanced_roles=True,
        drop_score_co=-1,
        n_jobs=1,
    )

    blender = WeightedBlender()

    automl = AutoML(
        reader=reader,
        levels=[[gbm_lvl0, reg_lvl0]],
        timer=timer,
        blender=blender,
        debug=True,
        skip_conn=False,
    )
    oof_pred = automl.fit_predict(train, roles={"target": "TARGET"}, verbose=5)
    test_pred = automl.predict(test)

    not_nan = np.any(~np.isnan(oof_pred.data), axis=1)

    oof_score = log_loss(train["TARGET"].values[not_nan], oof_pred.data[not_nan, :])
    assert oof_score < 1

    test_score = log_loss(test["TARGET"].values, test_pred.data)
    assert test_score < 1
