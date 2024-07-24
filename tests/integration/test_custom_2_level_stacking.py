#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import tempfile

from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score

from lightautoml.automl.base import AutoML
from lightautoml.dataset.roles import TargetRole
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.pipelines.features.lgb_pipeline import LGBSimpleFeatures
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.pipelines.selection.importance_based import ImportanceCutoffSelector
from lightautoml.pipelines.selection.importance_based import (
    ModelBasedImportanceEstimator,
)
from lightautoml.reader.base import PandasToPandasReader


def check_pickling(automl, ho_score, task, test, target_name):
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = os.path.join(tmpdirname, "automl.pickle")
        with open(filename, "wb") as f:
            pickle.dump(automl, f)

        with open(filename, "rb") as f:
            automl = pickle.load(f)

        test_pred = automl.predict(test)

        if task.name == "binary":
            ho_score_new = roc_auc_score(test[target_name].values, test_pred.data[:, 0])
        elif task.name == "multiclass":
            ho_score_new = log_loss(test[target_name].map(automl.reader.class_mapping), test_pred.data)
        elif task.name == "reg":
            ho_score_new = mean_squared_error(test[target_name].values, test_pred.data[:, 0])

        assert ho_score == ho_score_new


def get_target_name(roles):
    for key, value in roles.items():
        if (key == "target") or isinstance(key, TargetRole):
            return value


def test_manual_pipeline(sampled_app_train_test, sampled_app_roles, binary_task):

    train, test = sampled_app_train_test
    target_name = get_target_name(sampled_app_roles)

    reader = PandasToPandasReader(binary_task, cv=5, random_state=1)

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
        ModelBasedImportanceEstimator(),
        cutoff=10,
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
                OptunaTuner(n_trials=100, timeout=300),
            ),
            BoostLGBM(
                default_params={
                    "learning_rate": 0.025,
                    "num_leaves": 64,
                    "seed": 2,
                    "num_threads": 5,
                }
            ),
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
        post_selection=None,
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
    oof_predictions = automl.fit_predict(train, roles=sampled_app_roles)

    # predict for test data
    ho_predictions = automl.predict(test)

    oof_score = roc_auc_score(train[target_name].values, oof_predictions.data[:, 0])
    ho_score = roc_auc_score(test[target_name].values, ho_predictions.data[:, 0])

    assert oof_score > 0.67
    assert ho_score > 0.67

    check_pickling(automl, ho_score, binary_task, test, target_name)
