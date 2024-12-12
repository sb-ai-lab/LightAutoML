#!/usr/bin/env python
# coding: utf-8

"""2 level stacking using AutoML class with different algos on first level including LGBM, Linear and LinearL1."""


import tempfile
from os.path import join as pjoin
import numpy as np

from sklearn.metrics import roc_auc_score

from lightautoml.automl.base import AutoML
from lightautoml.automl.blend import MeanBlender
from lightautoml.dataset.roles import DatetimeRole
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.linear_sklearn import LinearL1CD
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
from lightautoml.pipelines.selection.linear_selector import HighCorrRemoval
from lightautoml.pipelines.selection.permutation_importance_based import (
    NpIterativeFeatureSelector,
)
from lightautoml.pipelines.selection.permutation_importance_based import (
    NpPermutationImportanceEstimator,
)
from lightautoml.reader.base import PandasToPandasReader

from integration_utils import load_and_test_automl

np.random.seed(42)


def test_blending(sampled_app_train_test, binary_task):

    train, test = sampled_app_train_test
    task = binary_task

    feat_sel_0 = LGBSimpleFeatures()
    mod_sel_0 = BoostLGBM()
    imp_sel_0 = ModelBasedImportanceEstimator()
    selector_0 = ImportanceCutoffSelector(feat_sel_0, mod_sel_0, imp_sel_0, cutoff=0)

    feats_gbm_0 = LGBAdvancedPipeline()
    gbm_0 = BoostLGBM()
    gbm_1 = BoostLGBM()
    tuner_0 = OptunaTuner(n_trials=100, timeout=30, fit_on_holdout=True)
    gbm_lvl0 = MLPipeline(
        [(gbm_0, tuner_0), gbm_1],
        pre_selection=selector_0,
        features_pipeline=feats_gbm_0,
        post_selection=None,
    )

    feats_reg_0 = LinearFeatures(output_categories=True)
    reg_0 = LinearLBFGS()
    reg_lvl0 = MLPipeline(
        [reg_0],
        pre_selection=None,
        features_pipeline=feats_reg_0,
        post_selection=HighCorrRemoval(corr_co=1),
    )

    feat_sel_1 = LGBSimpleFeatures()
    mod_sel_1 = BoostLGBM()
    imp_sel_1 = NpPermutationImportanceEstimator()
    selector_1 = NpIterativeFeatureSelector(feat_sel_1, mod_sel_1, imp_sel_1, feature_group_size=1)

    feats_reg_1 = LinearFeatures(output_categories=False)
    reg_1 = LinearL1CD()
    reg_l1_lvl0 = MLPipeline(
        [reg_1],
        pre_selection=selector_1,
        features_pipeline=feats_reg_1,
        post_selection=HighCorrRemoval(),
    )

    feats_reg_2 = LinearFeatures(output_categories=True)
    reg_2 = LinearLBFGS()
    reg_lvl1 = MLPipeline(
        [reg_2],
        pre_selection=None,
        features_pipeline=feats_reg_2,
        post_selection=HighCorrRemoval(corr_co=1),
    )

    reader = PandasToPandasReader(
        task,
        samples=None,
        max_nan_rate=1,
        max_constant_rate=1,
    )

    automl = AutoML(
        reader,
        [
            [gbm_lvl0, reg_lvl0, reg_l1_lvl0],
            [reg_lvl1],
        ],
        skip_conn=False,
        blender=MeanBlender(),
        debug=True,
    )

    roles = {
        "target": "TARGET",
        DatetimeRole(base_date=True, seasonality=(), base_feats=False): "report_dt",
    }

    with tempfile.TemporaryDirectory() as tmpdirname:
        path_to_save = pjoin(tmpdirname, "model.joblib")

        oof_pred = automl.fit_predict(train, roles=roles, verbose=2, path_to_save=path_to_save)

        test_pred = automl.predict(test)

        not_nan = np.any(~np.isnan(oof_pred.data), axis=1)

        oof_score = roc_auc_score(train[roles["target"]].values[not_nan], oof_pred.data[not_nan][:, 0])
        assert oof_score > 0.7

        test_score = roc_auc_score(test[roles["target"]].values, test_pred.data[:, 0])
        assert test_score > 0.7

        target_name = roles["target"]
        load_and_test_automl(
            path_to_save, task=task, score=test_score, pred=test_pred, data=test, target_name=target_name
        )
