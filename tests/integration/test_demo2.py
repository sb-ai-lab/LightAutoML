#!/usr/bin/env python
# coding: utf-8

import tempfile
from os.path import join as pjoin

from sklearn.metrics import roc_auc_score

from lightautoml.automl.base import AutoML
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.pipelines.features.lgb_pipeline import LGBSimpleFeatures
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.pipelines.selection.permutation_importance_based import (
    NpIterativeFeatureSelector,
)
from lightautoml.pipelines.selection.permutation_importance_based import (
    NpPermutationImportanceEstimator,
)
from lightautoml.reader.base import PandasToPandasReader


from integration_utils import load_and_test_automl, get_target_name


def test_permutation_importance_based_iterative_selector(sampled_app_train_test, sampled_app_roles, binary_task):

    train_data, test_data = sampled_app_train_test
    target_name = get_target_name(sampled_app_roles)

    task = binary_task

    reader = PandasToPandasReader(task, cv=5, random_state=1)

    # selector parts
    model0 = BoostLGBM(
        default_params={
            "learning_rate": 0.05,
            "num_leaves": 64,
            "seed": 42,
            "num_threads": 5,
        }
    )
    pipe0 = LGBSimpleFeatures()
    pie = NpPermutationImportanceEstimator()
    selector = NpIterativeFeatureSelector(pipe0, model0, pie, feature_group_size=1, max_features_cnt_in_result=15)

    # pipeline 1 level parts
    pipe = LGBSimpleFeatures()

    model1 = BoostLGBM(
        default_params={
            "learning_rate": 0.05,
            "num_leaves": 128,
            "seed": 1,
            "num_threads": 5,
        }
    )

    params_tuner2 = OptunaTuner(n_trials=100, timeout=100)
    model2 = BoostLGBM(
        default_params={
            "learning_rate": 0.025,
            "num_leaves": 64,
            "seed": 2,
            "num_threads": 5,
        }
    )

    pipeline_lvl1 = MLPipeline(
        [model1, (model2, params_tuner2)],
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
        },
        freeze_defaults=True,
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

    with tempfile.TemporaryDirectory() as tmpdirname:
        path_to_save = pjoin(tmpdirname, "model.joblib")
        automl.fit_predict(train_data, roles={"target": "TARGET"}, verbose=0, path_to_save=path_to_save)

        test_pred = automl.predict(test_data)
        test_score = roc_auc_score(test_data["TARGET"].values, test_pred.data[:, 0])
        assert test_score > 0.55

        load_and_test_automl(
            path_to_save, task=task, score=test_score, pred=test_pred, data=test_data, target_name=target_name
        )
