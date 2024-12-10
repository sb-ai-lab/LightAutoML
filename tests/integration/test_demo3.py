#!/usr/bin/env python
# coding: utf-8

import numpy as np

from sklearn.metrics import roc_auc_score

from lightautoml.automl.base import AutoML
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.pipelines.features.lgb_pipeline import LGBSimpleFeatures
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.pipelines.selection.base import ComposedSelector
from lightautoml.pipelines.selection.importance_based import ImportanceCutoffSelector
from lightautoml.pipelines.selection.importance_based import (
    ModelBasedImportanceEstimator,
)
from lightautoml.pipelines.selection.permutation_importance_based import (
    NpIterativeFeatureSelector,
)
from lightautoml.pipelines.selection.permutation_importance_based import (
    NpPermutationImportanceEstimator,
)
from lightautoml.reader.base import PandasToPandasReader


def test_pipeline_with_selectors(sampled_app_train_test, binary_task):
    np.random.seed(42)

    train_data, test_data = sampled_app_train_test
    task = binary_task

    reader = PandasToPandasReader(task, cv=5, random_state=1)

    # selector parts
    model01 = BoostLGBM(
        default_params={
            "learning_rate": 0.05,
            "num_leaves": 64,
            "seed": 42,
            "num_threads": 5,
        }
    )

    model02 = BoostLGBM(
        default_params={
            "learning_rate": 0.05,
            "num_leaves": 64,
            "seed": 42,
            "num_threads": 5,
        }
    )
    pipe0 = LGBSimpleFeatures()
    pie = NpPermutationImportanceEstimator()
    pie1 = ModelBasedImportanceEstimator()
    sel1 = ImportanceCutoffSelector(pipe0, model01, pie1, cutoff=0)
    sel2 = NpIterativeFeatureSelector(pipe0, model02, pie, feature_group_size=1, max_features_cnt_in_result=15)
    selector = ComposedSelector([sel1, sel2])

    # pipeline 1 level parts
    pipe = LGBSimpleFeatures()

    params_tuner1 = OptunaTuner(n_trials=100, timeout=100)
    model1 = BoostLGBM(
        default_params={
            "learning_rate": 0.05,
            "num_leaves": 128,
            "seed": 1,
            "num_threads": 5,
        }
    )

    model2 = BoostLGBM(
        default_params={
            "learning_rate": 0.025,
            "num_leaves": 64,
            "seed": 2,
            "num_threads": 5,
        }
    )

    pipeline_lvl1 = MLPipeline(
        [(model1, params_tuner1), model2],
        pre_selection=selector,
        features_pipeline=pipe,
        post_selection=None,
    )

    # pipeline 2 level parts
    pipe1 = LGBSimpleFeatures()

    model = BoostLGBM(
        default_params={
            "learning_rate": 0.05,
            "num_leaves": 64,
            "max_bin": 1024,
            "seed": 3,
            "num_threads": 5,
        }
    )
    pipeline_lvl2 = MLPipeline([model], pre_selection=None, features_pipeline=pipe1, post_selection=None)

    automl = AutoML(
        reader,
        [
            [pipeline_lvl1],
            [pipeline_lvl2],
        ],
        skip_conn=False,
        debug=True,
    )

    oof_pred = automl.fit_predict(train_data, roles={"target": "TARGET"}, verbose=5)

    test_pred = automl.predict(test_data)
    oof_score = roc_auc_score(train_data["TARGET"].values, oof_pred.data[:, 0])
    test_score = roc_auc_score(test_data["TARGET"].values, test_pred.data[:, 0])

    assert oof_score > 0.57
    assert test_score > 0.55
